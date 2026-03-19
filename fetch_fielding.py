#!/usr/bin/env python3
"""Fetch T20 fielding stats for all players in player_bios.csv."""

import asyncio
import csv
import sys
import time
from pathlib import Path

from cricdata import AsyncCricinfoClient

ROOT = Path(__file__).parent
BIOS_CSV = ROOT / "player_bios.csv"
FIELDING_CSV = ROOT / "player_fielding.csv"

CONCURRENCY = 5
MAX_RETRIES = 3

FIELDS = ["player_id", "name", "matches", "innings", "dismissals",
          "catches", "stumpings", "dismissals_per_innings"]


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


async def fetch_player_fielding(ci, sem, player_id, name):
    async with sem:
        await asyncio.sleep(0.15)
        try:
            stats = await retry(
                lambda: ci.player_career_stats(int(player_id), fmt="t20", stat_type="fielding")
            )
        except Exception as e:
            print(f"  WARN {player_id} ({name}): {e}", file=sys.stderr)
            return None

    summary = stats.get("summary", {})
    if not summary or not summary.get("Mat"):
        return None

    matches = int(summary.get("Mat", "0"))
    innings = int(summary.get("Inns", "0"))
    dismissals = int(summary.get("Dis", "0"))
    catches = int(summary.get("Ct", "0"))
    stumpings = int(summary.get("St", "0"))
    dpi = float(summary.get("D/I", "0")) if summary.get("D/I") else (
        dismissals / innings if innings > 0 else 0
    )

    return {
        "player_id": player_id,
        "name": name,
        "matches": matches,
        "innings": innings,
        "dismissals": dismissals,
        "catches": catches,
        "stumpings": stumpings,
        "dismissals_per_innings": round(dpi, 3),
    }


async def main():
    existing = set()
    rows = []
    if FIELDING_CSV.exists():
        with open(FIELDING_CSV) as f:
            for row in csv.DictReader(f):
                existing.add(row["player_id"])
                rows.append(row)
        print(f"Loaded {len(rows)} cached rows from {FIELDING_CSV}")

    with open(BIOS_CSV) as f:
        players = list(csv.DictReader(f))

    to_fetch = [p for p in players if p["player_id"] not in existing]
    print(f"{len(players)} players total, {len(to_fetch)} need fetching")

    async with AsyncCricinfoClient() as ci:
        sem = asyncio.Semaphore(CONCURRENCY)

        batch_size = 10
        for start in range(0, len(to_fetch), batch_size):
            batch = to_fetch[start:start + batch_size]
            results = await asyncio.gather(
                *[fetch_player_fielding(ci, sem, p["player_id"], p["name"])
                  for p in batch]
            )
            for r in results:
                if r is not None:
                    rows.append(r)
                    existing.add(str(r["player_id"]))
            done = min(start + batch_size, len(to_fetch))
            print(f"  {done}/{len(to_fetch)}", flush=True)

    with open(FIELDING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {FIELDING_CSV}")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Elapsed: {time.time() - t0:.0f}s")
