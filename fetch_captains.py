#!/usr/bin/env python3
"""Fetch captain data for all IPL matches using cricdata."""

import asyncio
import csv
import sys
import time
from pathlib import Path

from cricdata import AsyncCricinfoClient

ROOT = Path(__file__).parent
CAPTAINS_CSV = ROOT / "match_captains.csv"

IPL_SERIES = {
    2015: "pepsi-indian-premier-league-2015-791129",
    2016: "indian-premier-league-2016-968923",
    2017: "indian-premier-league-2017-1078425",
    2018: "indian-premier-league-2018-1131611",
    2019: "indian-premier-league-2019-1165643",
    2020: "indian-premier-league-2020-21-1210595",
    2021: "indian-premier-league-2021-1249214",
    2022: "indian-premier-league-2022-1298423",
    2023: "indian-premier-league-2023-1345038",
    2024: "indian-premier-league-2024-1410320",
    2025: "ipl-2025-1449924",
}

CONCURRENCY = 3
MAX_RETRIES = 4

FIELDS = [
    "match_id", "season",
    "captain1_id", "captain1_name", "captain1_team",
    "captain2_id", "captain2_name", "captain2_team",
]


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


async def fetch_match_captains(ci, sem, series_slug, match_obj, year):
    match_id = match_obj["objectId"]
    match_slug = f"{match_obj['slug']}-{match_id}"

    async with sem:
        await asyncio.sleep(0.3)
        try:
            info = await retry(lambda: ci.match_info(series_slug, match_slug))
        except Exception as e:
            print(f"  WARN {match_id}: {e}", file=sys.stderr)
            return None

    captains = info.get("captains", [])
    if len(captains) < 2:
        print(f"  WARN {match_id}: only {len(captains)} captain(s) found", file=sys.stderr)
        return None

    return {
        "match_id": match_id,
        "season": year,
        "captain1_id": captains[0].get("player_id", ""),
        "captain1_name": captains[0].get("name", ""),
        "captain1_team": captains[0].get("team_name", ""),
        "captain2_id": captains[1].get("player_id", ""),
        "captain2_name": captains[1].get("name", ""),
        "captain2_team": captains[1].get("team_name", ""),
    }


async def main():
    existing = set()
    rows = []
    if CAPTAINS_CSV.exists():
        with open(CAPTAINS_CSV) as f:
            for row in csv.DictReader(f):
                existing.add(row["match_id"])
                rows.append(row)
        print(f"Loaded {len(rows)} cached rows from {CAPTAINS_CSV}")

    async with AsyncCricinfoClient() as ci:
        sem = asyncio.Semaphore(CONCURRENCY)

        for year, slug in sorted(IPL_SERIES.items()):
            print(f"\n{year} ({slug})...", flush=True)
            try:
                result = await ci.series_matches(slug)
            except Exception:
                result = await ci.series_fixtures(slug)
            matches = result["content"]["matches"]
            to_fetch = [m for m in matches if str(m["objectId"]) not in existing]
            print(f"  {len(matches)} matches, {len(to_fetch)} need fetching", flush=True)

            batch_size = 5
            for start in range(0, len(to_fetch), batch_size):
                batch = to_fetch[start:start + batch_size]
                results = await asyncio.gather(
                    *[fetch_match_captains(ci, sem, slug, m, year) for m in batch]
                )
                for r in results:
                    if r is not None:
                        rows.append(r)
                        existing.add(str(r["match_id"]))
                done = min(start + batch_size, len(to_fetch))
                print(f"    {done}/{len(to_fetch)}", flush=True)

    rows.sort(key=lambda r: (int(r.get("season", 0)), str(r.get("match_id", ""))))
    with open(CAPTAINS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {CAPTAINS_CSV}")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Elapsed: {time.time() - t0:.0f}s")
