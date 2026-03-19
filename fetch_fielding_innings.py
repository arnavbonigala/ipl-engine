#!/usr/bin/env python3
"""Fetch T20 innings-by-innings fielding data for all players."""

import asyncio
import csv
import sys
import time
from pathlib import Path

from cricdata import AsyncCricinfoClient

ROOT = Path(__file__).parent
BIOS_CSV = ROOT / "player_bios.csv"
OUT_DIR = ROOT / "player_fielding_innings"
OUT_DIR.mkdir(exist_ok=True)

CONCURRENCY = 5
MAX_RETRIES = 3


async def retry(coro_fn, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt + 1
            print(f"    Retry {attempt+1}/{retries} after {wait}s: {e}", file=sys.stderr)
            await asyncio.sleep(wait)


async def fetch_one(ci, sem, player_id, name):
    out = OUT_DIR / f"{name.replace(' ', '_')}_{player_id}.csv"
    if out.exists():
        return "cached"

    async with sem:
        await asyncio.sleep(0.15)
        try:
            data = await retry(
                lambda: ci.player_innings(int(player_id), fmt="t20", stat_type="fielding")
            )
        except Exception as e:
            print(f"  WARN {player_id} ({name}): {e}", file=sys.stderr)
            return "error"

    innings = data.get("innings", [])
    if not innings:
        return "empty"

    fields = ["player_id", "start_date", "opposition", "ground",
              "innings", "dismissals", "catches", "stumpings",
              "ct_wk", "ct_fi"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in innings:
            w.writerow({
                "player_id": player_id,
                "start_date": row.get("Start Date", ""),
                "opposition": row.get("Opposition", ""),
                "ground": row.get("Ground", ""),
                "innings": row.get("Inns", ""),
                "dismissals": row.get("Dis", "0"),
                "catches": row.get("Ct", "0"),
                "stumpings": row.get("St", "0"),
                "ct_wk": row.get("Ct Wk", "0"),
                "ct_fi": row.get("Ct Fi", "0"),
            })

    return "ok"


async def main():
    with open(BIOS_CSV) as f:
        players = list(csv.DictReader(f))

    print(f"{len(players)} players total")
    cached = sum(1 for p in players
                 if (OUT_DIR / f"{p['name'].replace(' ', '_')}_{p['player_id']}.csv").exists())
    print(f"  {cached} already cached")

    async with AsyncCricinfoClient() as ci:
        sem = asyncio.Semaphore(CONCURRENCY)
        batch_size = 10
        ok = err = empty = 0
        for start in range(0, len(players), batch_size):
            batch = players[start:start + batch_size]
            results = await asyncio.gather(
                *[fetch_one(ci, sem, p["player_id"], p["name"]) for p in batch]
            )
            for r in results:
                if r == "ok" or r == "cached":
                    ok += 1
                elif r == "empty":
                    empty += 1
                else:
                    err += 1
            done = min(start + batch_size, len(players))
            print(f"  {done}/{len(players)} (ok={ok} empty={empty} err={err})", flush=True)

    print(f"\nDone: {ok} ok, {empty} empty, {err} errors")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Elapsed: {time.time() - t0:.0f}s")
