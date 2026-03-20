"""Exhaustively discover all 71 IPL 2025 Polymarket events."""

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset.csv"
CACHE_FILE = ROOT / "engine" / "pm_events_2025.json"

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

TEAM_ABBREVS = {
    "Mumbai Indians": ["mi"],
    "Chennai Super Kings": ["csk"],
    "Royal Challengers Bengaluru": ["rcb"],
    "Kolkata Knight Riders": ["kkr"],
    "Rajasthan Royals": ["rr"],
    "Sunrisers Hyderabad": ["srh"],
    "Delhi Capitals": ["dc"],
    "Punjab Kings": ["pbks", "pk"],
    "Lucknow Super Giants": ["lsg"],
    "Gujarat Titans": ["gt"],
}

TEAM_SLUGS = {
    "Mumbai Indians": ["mumbai-indians"],
    "Chennai Super Kings": ["chennai-super-kings"],
    "Royal Challengers Bengaluru": [
        "royal-challengers-bengaluru",
        "royal-challengers-bangalore",
    ],
    "Kolkata Knight Riders": ["kolkata-knight-riders"],
    "Rajasthan Royals": ["rajasthan-royals"],
    "Sunrisers Hyderabad": ["sunrisers-hyderabad"],
    "Delhi Capitals": ["delhi-capitals"],
    "Punjab Kings": ["punjab-kings"],
    "Lucknow Super Giants": ["lucknow-super-giants"],
    "Gujarat Titans": ["gujarat-titans"],
}

TEAM_KEYWORDS = {
    "Mumbai Indians": ["mumbai"],
    "Chennai Super Kings": ["chennai"],
    "Royal Challengers Bengaluru": ["bangalore", "bengaluru", "challengers"],
    "Kolkata Knight Riders": ["kolkata"],
    "Rajasthan Royals": ["rajasthan"],
    "Sunrisers Hyderabad": ["hyderabad", "sunrisers"],
    "Delhi Capitals": ["delhi"],
    "Punjab Kings": ["punjab"],
    "Lucknow Super Giants": ["lucknow"],
    "Gujarat Titans": ["gujarat"],
}


def _curl(url):
    try:
        r = subprocess.run(["curl", "-s", url], capture_output=True, text=True, timeout=15)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def _match_team(text):
    t = text.lower()
    for team, kws in TEAM_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return team
    return None


def _gen_slugs(team1, team2, match_date):
    """Generate all plausible slug patterns for a match."""
    slugs = []
    dt = datetime.strptime(match_date, "%Y-%m-%d")

    for a1 in TEAM_ABBREVS[team1]:
        for a2 in TEAM_ABBREVS[team2]:
            for d in range(-3, 8):
                day = (dt + timedelta(days=d)).strftime("%Y-%m-%d")
                slugs.append(f"ipl-{a1}-{a2}-{day}")
                slugs.append(f"ipl-{a2}-{a1}-{day}")

    for s1 in TEAM_SLUGS[team1]:
        for s2 in TEAM_SLUGS[team2]:
            slugs.append(f"{s1}-vs-{s2}")
            slugs.append(f"{s2}-vs-{s1}")

    for d in range(-1, 3):
        day = (dt + timedelta(days=d)).strftime("%Y-%m-%d")
        slugs.append(f"ipl-dailies-{day}")

    return slugs


def _try_event_slug(slug):
    """Try fetching event(s) by slug. Return list of events."""
    data = _curl(f"{GAMMA}/events?slug={slug}&limit=5")
    if data and isinstance(data, list) and len(data) > 0:
        return data
    return []


def _search_team(team):
    """Search for events mentioning a team."""
    q = team.replace(" ", "+")
    data = _curl(f"{GAMMA}/public-search?q={q}&limit_per_type=100")
    if data and "events" in data:
        return data["events"]
    return []


def _extract_event_info(ev, team1, team2):
    """Extract token IDs from a full event object. Returns dict or None."""
    markets = ev.get("markets", [])
    if not markets:
        return None
    mk = markets[0]
    outcomes = json.loads(mk.get("outcomes", "[]"))
    tokens = json.loads(mk.get("clobTokenIds", "[]"))
    volume = mk.get("volumeNum", 0) or 0
    if len(outcomes) != 2 or len(tokens) != 2:
        return None

    t1_idx = t2_idx = None
    for i, o in enumerate(outcomes):
        m = _match_team(o)
        if m == team1:
            t1_idx = i
        elif m == team2:
            t2_idx = i

    if t1_idx is None or t2_idx is None:
        return None

    return {
        "event_id": str(ev.get("id", "")),
        "title": ev.get("title", ""),
        "slug": ev.get("slug", mk.get("slug", "")),
        "t1_token": tokens[t1_idx],
        "t2_token": tokens[t2_idx],
        "outcomes": outcomes,
        "volume": volume,
    }


def _check_event_has_match(ev_or_market, team1, team2):
    """Check if an event/market mentions both teams."""
    text = (ev_or_market.get("title", "") + " " +
            ev_or_market.get("question", "") + " " +
            " ".join(ev_or_market.get("outcomes", []))).lower()
    has_t1 = any(kw in text for kw in TEAM_KEYWORDS[team1])
    has_t2 = any(kw in text for kw in TEAM_KEYWORDS[team2])
    return has_t1 and has_t2


def find_event(team1, team2, match_date):
    """Find the Polymarket event for a specific match. Returns full event info or None."""

    # Strategy 1: Try direct slug patterns
    for slug in _gen_slugs(team1, team2, match_date):
        events = _try_event_slug(slug)
        for ev in events:
            if _check_event_has_match(ev, team1, team2):
                full = _curl(f"{GAMMA}/events/{ev['id']}")
                if full:
                    info = _extract_event_info(full, team1, team2)
                    if info:
                        return info
            # For dailies events, check nested markets
            for mk in ev.get("markets", []):
                if _check_event_has_match(mk, team1, team2):
                    outcomes = json.loads(mk.get("outcomes", "[]"))
                    tokens = json.loads(mk.get("clobTokenIds", "[]"))
                    if len(outcomes) == 2 and len(tokens) == 2:
                        t1_idx = t2_idx = None
                        for i, o in enumerate(outcomes):
                            m = _match_team(o)
                            if m == team1:
                                t1_idx = i
                            elif m == team2:
                                t2_idx = i
                        if t1_idx is not None and t2_idx is not None:
                            return {
                                "event_id": str(ev.get("id", "")),
                                "title": mk.get("question", ev.get("title", "")),
                                "slug": mk.get("slug", ""),
                                "t1_token": tokens[t1_idx],
                                "t2_token": tokens[t2_idx],
                                "outcomes": outcomes,
                                "volume": mk.get("volumeNum", 0) or 0,
                            }
        time.sleep(0.15)

    # Strategy 2: Search by team name
    for team in [team1, team2]:
        results = _search_team(team)
        for ev in results:
            title = ev.get("title", "")
            start = (ev.get("startDate") or "")[:10]
            if not _check_event_has_match(ev, team1, team2):
                continue
            # Check date is within 2 days
            try:
                ev_dt = datetime.strptime(start, "%Y-%m-%d")
                m_dt = datetime.strptime(match_date, "%Y-%m-%d")
                if abs((ev_dt - m_dt).days) > 2:
                    continue
            except ValueError:
                continue

            eid = str(ev.get("id", ""))
            full = _curl(f"{GAMMA}/events/{eid}")
            if full:
                info = _extract_event_info(full, team1, team2)
                if info:
                    return info
        time.sleep(0.15)

    return None


def discover_all():
    df = pd.read_csv(DATASET)
    matches = df[df["season"] == 2025][["date", "team1", "team2", "match_id"]].sort_values("date")
    print(f"Finding Polymarket events for {len(matches)} IPL 2025 matches...\n")

    results = {}
    found = 0
    missing = []

    for _, row in matches.iterrows():
        date_str = str(row["date"])[:10]
        t1, t2 = row["team1"], row["team2"]
        mid = str(row["match_id"])

        print(f"  {date_str} {t1:>30} v {t2:<30}", end=" ", flush=True)

        info = find_event(t1, t2, date_str)
        if info:
            results[mid] = {
                "match_id": mid,
                "date": date_str,
                "team1": t1,
                "team2": t2,
                **info,
            }
            found += 1
            print(f"FOUND  slug={info['slug'][:50]}", flush=True)
        else:
            missing.append((date_str, t1, t2, mid))
            print("MISSING", flush=True)

    print(f"\n{'='*60}")
    print(f"  Found: {found}/{len(matches)}")
    if missing:
        print(f"  Missing: {len(missing)}")
        for d, t1, t2, mid in missing:
            print(f"    {d} {t1} v {t2} ({mid})")
    print(f"{'='*60}")

    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {CACHE_FILE}")

    return results


if __name__ == "__main__":
    discover_all()
