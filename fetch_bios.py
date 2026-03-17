#!/usr/bin/env python3
"""Fetch player bios (bat/bowl style, role, age) for all players in player_innings/."""

import asyncio
import csv
import sys
import time
from pathlib import Path

from cricdata import AsyncCricinfoClient

ROOT = Path(__file__).parent
PLAYER_DIR = ROOT / "player_innings"
OUT = ROOT / "player_bios.csv"
CONCURRENCY = 3
MAX_RETRIES = 4

FIELDS = ["player_id", "name", "bat_style", "bowl_style", "bowl_type",
          "role", "age", "bat_hand"]


async def retry(coro_fn, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt + 1
            print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(wait)


def classify_bowl_type(desc: str) -> str:
    desc = desc.lower()
    if any(k in desc for k in ["fast", "medium", "seam"]):
        return "pace"
    if any(k in desc for k in ["spin", "break", "orthodox", "chinaman"]):
        return "spin"
    return "unknown"


def collect_player_ids() -> dict[str, str]:
    existing = set()
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                existing.add(row["player_id"])

    players = {}
    for p in sorted(PLAYER_DIR.glob("*.csv")):
        pid = p.stem.rsplit("_", 1)[-1]
        if pid in existing:
            continue
        with open(p) as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
            if first:
                players[pid] = first["player_name"]
    return players


async def fetch_bio(ci, sem, pid, name):
    async with sem:
        await asyncio.sleep(0.3)
        try:
            bio = await retry(lambda: ci.player_bio(pid))
        except Exception as e:
            print(f"  WARN {name} ({pid}): {e}", file=sys.stderr, flush=True)
            return None

    bat_styles = bio.get("batStyle") or []
    bowl_styles = bio.get("bowlStyle") or []
    position = bio.get("position") or {}

    bat_desc = bat_styles[0]["description"] if bat_styles else ""
    bowl_desc = bowl_styles[0]["description"] if bowl_styles else ""
    bat_hand = "left" if "left" in bat_desc.lower() else "right" if bat_desc else ""

    return {
        "player_id": pid,
        "name": bio.get("displayName", name),
        "bat_style": bat_desc,
        "bowl_style": bowl_desc,
        "bowl_type": classify_bowl_type(bowl_desc) if bowl_desc else "",
        "role": position.get("name", ""),
        "age": bio.get("age", ""),
        "bat_hand": bat_hand,
    }


async def main():
    players = collect_player_ids()
    if not players:
        print("All bios already cached.", flush=True)
        return

    print(f"Fetching bios for {len(players)} players...", flush=True)

    write_header = not OUT.exists()
    sem = asyncio.Semaphore(CONCURRENCY)

    async with AsyncCricinfoClient() as ci:
        items = list(players.items())
        total = len(items)
        done = 0

        with open(OUT, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()

            for start in range(0, total, 10):
                batch = items[start:start + 10]
                results = await asyncio.gather(
                    *[fetch_bio(ci, sem, pid, name) for pid, name in batch]
                )
                for r in results:
                    if r:
                        writer.writerow(r)
                f.flush()
                done += len(batch)
                print(f"  {done}/{total}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Elapsed: {time.time() - t0:.0f}s")
