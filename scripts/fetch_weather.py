#!/usr/bin/env python3
"""Fetch weather and match-time data for all IPL matches using cricdata."""

import asyncio
import csv
import sys
import time
from pathlib import Path

from cricdata import AsyncCricinfoClient

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEATHER_CSV = DATA_DIR / "match_weather.csv"

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
    "match_id", "season", "floodlit", "day_type", "start_time_utc",
    "temperature", "humidity", "precipitation", "wind_speed", "cloud_cover",
    "dew_point_approx",
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


def _match_hour_indices(start_time_utc: str) -> list[int]:
    """Return hourly indices (0-23) covering typical match window (~3.5 hours)."""
    try:
        hour = int(start_time_utc[11:13])
        return [h % 24 for h in range(hour, hour + 4)]
    except (ValueError, IndexError):
        return list(range(14, 18))


def _approx_dew_point(temp_c: float, rh_pct: float) -> float:
    """Magnus formula approximation for dew point."""
    import math
    if rh_pct <= 0:
        return temp_c
    a, b = 17.27, 237.7
    gamma = (a * temp_c) / (b + temp_c) + math.log(rh_pct / 100.0)
    return (b * gamma) / (a - gamma)


async def fetch_match_weather(ci, sem, series_slug, match_obj, year):
    match_id = match_obj["objectId"]
    match_slug = f"{match_obj['slug']}-{match_id}"

    async with sem:
        await asyncio.sleep(0.3)
        try:
            info = await retry(lambda: ci.match_info(series_slug, match_slug))
        except Exception as e:
            print(f"  WARN info {match_id}: {e}", file=sys.stderr)
            return None

        await asyncio.sleep(0.3)
        try:
            weather = await retry(lambda: ci.match_weather(series_slug, match_slug))
        except Exception as e:
            print(f"  WARN weather {match_id}: {e}", file=sys.stderr)
            weather = None

    t = info.get("time", {})
    start_time = t.get("startTime", "")
    floodlit = t.get("floodlit", "")
    day_type = t.get("dayType", "")

    temp, hum, precip, wind, cloud = 0.0, 0.0, 0.0, 0.0, 0.0
    if weather and isinstance(weather, dict) and "hourly" in weather:
        hourly = weather["hourly"]
        indices = _match_hour_indices(start_time)
        def _avg(key):
            vals = hourly.get(key, [])
            selected = [vals[i] for i in indices if i < len(vals) and vals[i] is not None]
            return sum(selected) / len(selected) if selected else 0.0

        temp = _avg("temperature_2m")
        hum = _avg("relative_humidity_2m")
        precip = _avg("precipitation")
        wind = _avg("wind_speed_10m")
        cloud = _avg("cloud_cover")

    dew_point = _approx_dew_point(temp, hum) if hum > 0 else 0.0

    return {
        "match_id": match_id,
        "season": year,
        "floodlit": floodlit,
        "day_type": day_type,
        "start_time_utc": start_time,
        "temperature": round(temp, 1),
        "humidity": round(hum, 1),
        "precipitation": round(precip, 2),
        "wind_speed": round(wind, 1),
        "cloud_cover": round(cloud, 1),
        "dew_point_approx": round(dew_point, 1),
    }


async def main():
    existing = set()
    rows = []
    if WEATHER_CSV.exists():
        with open(WEATHER_CSV) as f:
            for row in csv.DictReader(f):
                existing.add(row["match_id"])
                rows.append(row)
        print(f"Loaded {len(rows)} cached rows from {WEATHER_CSV}")

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
                    *[fetch_match_weather(ci, sem, slug, m, year) for m in batch]
                )
                for r in results:
                    if r is not None:
                        rows.append(r)
                        existing.add(str(r["match_id"]))
                done = min(start + batch_size, len(to_fetch))
                print(f"    {done}/{len(to_fetch)}", flush=True)

    rows.sort(key=lambda r: (int(r.get("season", 0)), str(r.get("start_time_utc", ""))))
    with open(WEATHER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {WEATHER_CSV}")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Elapsed: {time.time() - t0:.0f}s")
