"""Kalshi market discovery and price reads (all public, no auth required)."""

import json
import urllib.request
from datetime import datetime

from engine.config import KALSHI_API_BASE, KALSHI_IPL_SERIES, KALSHI_ABBREV_TO_TEAM, TEAM_KEYWORDS


def _fetch_json(url: str) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ipl-engine/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _match_team_in_text(team: str, text: str) -> bool:
    return any(kw in text.lower() for kw in TEAM_KEYWORDS.get(team, []))


def find_ipl_markets(match_date: str | None = None) -> list[dict]:
    """Find active IPL match events on Kalshi.

    Returns list of dicts with keys:
        event_ticker, title, team1, team2, t1_ticker, t2_ticker, volume
    """
    url = f"{KALSHI_API_BASE}/events?series_ticker={KALSHI_IPL_SERIES}&with_nested_markets=true&limit=100"
    data = _fetch_json(url)
    if not data or "events" not in data:
        return []

    results = []
    for ev in data["events"]:
        markets = ev.get("markets", [])
        active_markets = [m for m in markets if m.get("status") in ("active", "open")]
        if not active_markets:
            continue

        if match_date:
            match_dt = datetime.strptime(match_date, "%Y-%m-%d")
            close_str = active_markets[0].get("expected_expiration_time", "")
            if close_str:
                try:
                    exp_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    if abs((exp_dt.date() - match_dt.date()).days) > 3:
                        continue
                except ValueError:
                    pass

        team_map = {}
        total_vol = 0
        for m in active_markets:
            abbr = m["ticker"].split("-")[-1]
            team_name = KALSHI_ABBREV_TO_TEAM.get(abbr)
            if team_name:
                team_map[team_name] = m["ticker"]
            total_vol += float(m.get("volume_fp", "0"))

        if len(team_map) != 2:
            continue

        teams = list(team_map.keys())
        results.append({
            "event_ticker": ev["event_ticker"],
            "title": ev.get("title", ""),
            "team1": teams[0],
            "team2": teams[1],
            "t1_ticker": team_map[teams[0]],
            "t2_ticker": team_map[teams[1]],
            "volume": total_vol,
        })

    return results


def get_market_price(ticker: str) -> float | None:
    """Get the midpoint price for a Kalshi market ticker.

    Computes (yes_bid + yes_ask) / 2.
    """
    data = _fetch_json(f"{KALSHI_API_BASE}/markets/{ticker}")
    if not data or "market" not in data:
        return None

    m = data["market"]
    bid = m.get("yes_bid_dollars")
    ask = m.get("yes_ask_dollars")
    if bid is None or ask is None:
        return None

    try:
        mid = (float(bid) + float(ask)) / 2.0
        return round(mid, 4) if mid > 0 else None
    except (ValueError, TypeError):
        return None


def get_market_data(ticker: str) -> dict | None:
    """Get full market data for a Kalshi market ticker."""
    data = _fetch_json(f"{KALSHI_API_BASE}/markets/{ticker}")
    if not data or "market" not in data:
        return None
    return data["market"]


def get_order_book(ticker: str) -> dict | None:
    """Get the order book for a Kalshi market ticker."""
    data = _fetch_json(f"{KALSHI_API_BASE}/markets/{ticker}/orderbook")
    if not data:
        return None
    return data.get("orderbook_fp")
