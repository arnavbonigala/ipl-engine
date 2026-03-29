"""Discover IPL Kalshi events for a given season.

Usage:
  python -m engine.discover_all          # default: 2026
  python -m engine.discover_all 2025     # specific season
"""

import json
import sys
from pathlib import Path

import urllib.request

from engine.config import KALSHI_API_BASE, KALSHI_IPL_SERIES, KALSHI_ABBREV_TO_TEAM

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
YY = str(SEASON)[-2:]

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "engine" / f"kalshi_events_{SEASON}.json"


def _fetch_json(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ipl-engine/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  fetch error: {e}")
        return None


def discover_all():
    """Fetch all KXIPLGAME events from Kalshi and filter by season."""
    url = f"{KALSHI_API_BASE}/events?series_ticker={KALSHI_IPL_SERIES}&with_nested_markets=true&limit=200"
    data = _fetch_json(url)

    if not data or "events" not in data:
        print("Failed to fetch Kalshi events.")
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)
        return {}

    events = data["events"]
    print(f"Total KXIPLGAME events from Kalshi: {len(events)}")

    results = {}
    skipped = 0
    for ev in events:
        ticker = ev["event_ticker"]
        if not ticker.startswith(f"{KALSHI_IPL_SERIES}-{YY}"):
            skipped += 1
            continue

        markets = ev.get("markets", [])
        if len(markets) != 2:
            continue

        team_map = {}
        total_vol = 0
        for m in markets:
            abbr = m["ticker"].split("-")[-1]
            team_name = KALSHI_ABBREV_TO_TEAM.get(abbr)
            if team_name:
                team_map[team_name] = {
                    "ticker": m["ticker"],
                    "status": m.get("status", ""),
                    "result": m.get("result", ""),
                    "yes_bid": m.get("yes_bid_dollars", ""),
                    "yes_ask": m.get("yes_ask_dollars", ""),
                    "last_price": m.get("last_price_dollars", ""),
                    "volume": m.get("volume_fp", "0"),
                }
            total_vol += float(m.get("volume_fp", "0"))

        if len(team_map) != 2:
            continue

        teams = list(team_map.keys())
        exp = markets[0].get("expected_expiration_time", "")[:10]

        results[ticker] = {
            "event_ticker": ticker,
            "title": ev.get("title", ""),
            "date": exp,
            "team1": teams[0],
            "team2": teams[1],
            "t1_ticker": team_map[teams[0]]["ticker"],
            "t2_ticker": team_map[teams[1]]["ticker"],
            "volume": total_vol,
            "markets": team_map,
        }

        status = markets[0].get("status", "?")
        print(f"  {ticker:45s}  {teams[0]:>30} v {teams[1]:<30}  vol={total_vol:>10.0f}  status={status}")

    print(f"\n{'='*60}")
    print(f"  Season {SEASON}: {len(results)} events found ({skipped} from other seasons)")
    print(f"{'='*60}")

    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {CACHE_FILE}")

    return results


if __name__ == "__main__":
    discover_all()
