"""Polymarket market discovery and price reads."""

import json
import urllib.request
from datetime import date

from engine.config import (
    GAMMA_HOST, POLYMARKET_HOST,
    TEAM_ABBREVS, TEAM_SLUG_NAMES, TEAM_KEYWORDS,
)


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
    """Find IPL match events on Polymarket for a given date (YYYY-MM-DD).

    Returns list of dicts with keys:
        event_id, title, slug, team1, team2,
        t1_token_id, t2_token_id, outcomes
    """
    if match_date is None:
        match_date = date.today().isoformat()

    teams = list(TEAM_ABBREVS.keys())
    found = {}

    for team in teams:
        q = team.replace(" ", "+")
        data = _fetch_json(f"{GAMMA_HOST}/public-search?q={q}&limit_per_type=50")
        if not data or "events" not in data:
            continue
        for ev in data["events"]:
            eid = str(ev.get("id", ""))
            title = ev.get("title", "")
            if eid not in found and "vs" in title.lower():
                found[eid] = ev

    results = []
    for eid, ev in found.items():
        event = _fetch_json(f"{GAMMA_HOST}/events/{eid}")
        if not event or not event.get("markets"):
            continue

        title = event["title"]
        start = event.get("startDate", "")

        # check date proximity
        if match_date and start:
            event_date = start[:10]
            if abs(_day_diff(match_date, event_date)) > 2:
                continue

        mk = event["markets"][0]
        outcomes = json.loads(mk.get("outcomes", "[]"))
        tokens = json.loads(mk.get("clobTokenIds", "[]"))
        if len(outcomes) != 2 or len(tokens) != 2:
            continue

        # identify which teams
        matched_teams = {}
        for team in teams:
            for i, o in enumerate(outcomes):
                if _match_team_in_text(team, o):
                    matched_teams[team] = i

        if len(matched_teams) < 2:
            continue

        team_list = list(matched_teams.keys())
        t1, t2 = team_list[0], team_list[1]
        t1_idx, t2_idx = matched_teams[t1], matched_teams[t2]

        results.append({
            "event_id": eid,
            "title": title,
            "slug": event.get("slug", ""),
            "team1": t1,
            "team2": t2,
            "t1_token_id": tokens[t1_idx],
            "t2_token_id": tokens[t2_idx],
            "outcomes": outcomes,
            "volume": event.get("volume", 0),
        })

    # Also try slug-based lookup for common patterns
    _try_slug_patterns(match_date, results, teams)

    # deduplicate by event_id
    seen = set()
    deduped = []
    for r in results:
        if r["event_id"] not in seen:
            seen.add(r["event_id"])
            deduped.append(r)
    return deduped


def _try_slug_patterns(match_date: str, results: list, teams: list):
    existing_pairs = {(r["team1"], r["team2"]) for r in results}
    existing_pairs |= {(r["team2"], r["team1"]) for r in results}

    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            if (t1, t2) in existing_pairs:
                continue
            a1, a2 = TEAM_ABBREVS[t1], TEAM_ABBREVS[t2]
            s1, s2 = TEAM_SLUG_NAMES[t1], TEAM_SLUG_NAMES[t2]
            slugs = [
                f"ipl-{a1}-{a2}-{match_date}",
                f"ipl-{a2}-{a1}-{match_date}",
                f"{s1}-vs-{s2}",
                f"{s2}-vs-{s1}",
            ]
            for slug in slugs:
                data = _fetch_json(f"{GAMMA_HOST}/events?slug={slug}&limit=1")
                if data and len(data) > 0:
                    ev = data[0]
                    markets = ev.get("markets", [])
                    if not markets:
                        continue
                    mk = markets[0]
                    outcomes = json.loads(mk.get("outcomes", "[]"))
                    tokens = json.loads(mk.get("clobTokenIds", "[]"))
                    if len(outcomes) != 2 or len(tokens) != 2:
                        continue

                    t1_idx = t2_idx = None
                    for idx, o in enumerate(outcomes):
                        if _match_team_in_text(t1, o):
                            t1_idx = idx
                        if _match_team_in_text(t2, o):
                            t2_idx = idx

                    if t1_idx is not None and t2_idx is not None:
                        results.append({
                            "event_id": str(ev["id"]),
                            "title": ev["title"],
                            "slug": slug,
                            "team1": t1,
                            "team2": t2,
                            "t1_token_id": tokens[t1_idx],
                            "t2_token_id": tokens[t2_idx],
                            "outcomes": outcomes,
                            "volume": ev.get("volume", 0),
                        })
                        break


def get_market_price(token_id: str) -> float | None:
    """Get the current midpoint price for a token."""
    data = _fetch_json(f"{POLYMARKET_HOST}/midpoint?token_id={token_id}")
    if data and "mid" in data:
        return float(data["mid"])
    return None


def get_order_book(token_id: str) -> dict | None:
    """Get the order book for a token."""
    data = _fetch_json(f"{POLYMARKET_HOST}/book?token_id={token_id}")
    return data


def _day_diff(d1: str, d2: str) -> int:
    from datetime import datetime
    a = datetime.strptime(d1[:10], "%Y-%m-%d")
    b = datetime.strptime(d2[:10], "%Y-%m-%d")
    return (a - b).days
