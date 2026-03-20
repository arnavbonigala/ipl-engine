"""Auto-fetch playing XIs and toss from ESPNcricinfo via cricdata."""

import time

from cricdata import CricinfoClient

from engine.config import SCRAPE_POLL_SECS, TEAM_KEYWORDS

_ci = CricinfoClient()


def _match_team_name(raw_name: str) -> str | None:
    """Map an ESPNCricinfo short team name to our canonical name."""
    raw = raw_name.lower()
    for canonical, keywords in TEAM_KEYWORDS.items():
        if any(kw in raw for kw in keywords):
            return canonical
    return None


def find_ipl_series_slug() -> str | None:
    """Find the current IPL series slug from live/recent matches."""
    for match in _ci.live_matches():
        series = match.get("series", {})
        name = series.get("longName", "") or series.get("name", "")
        if "indian premier league" in name.lower():
            return f"{series['slug']}-{series['objectId']}"
    return None


def get_ipl_fixtures(series_slug: str) -> list[dict]:
    """Get all fixtures for the IPL series."""
    data = _ci.series_fixtures(series_slug)
    matches = data.get("content", {}).get("matches", [])
    results = []
    for m in matches:
        match_slug = f"{m['slug']}-{m['objectId']}"
        teams = m.get("teams", [])
        if len(teams) < 2:
            continue

        t1 = _match_team_name(teams[0].get("team", {}).get("longName", ""))
        t2 = _match_team_name(teams[1].get("team", {}).get("longName", ""))
        ground = m.get("ground", {})

        results.append({
            "match_slug": match_slug,
            "series_slug": series_slug,
            "date": m.get("startDate", "")[:10],
            "team1": t1,
            "team2": t2,
            "venue": ground.get("longName", ground.get("name", "")),
            "city": ground.get("town", {}).get("name", ""),
            "status": m.get("state", ""),
        })
    return results


def get_todays_ipl_matches() -> list[dict]:
    """Get today's IPL matches from live matches endpoint."""
    results = []
    for match in _ci.live_matches():
        series = match.get("series", {})
        name = series.get("longName", "") or series.get("name", "")
        if "indian premier league" not in name.lower():
            continue

        series_slug = f"{series['slug']}-{series['objectId']}"
        match_slug = f"{match['slug']}-{match['objectId']}"
        teams = match.get("teams", [])
        if len(teams) < 2:
            continue

        t1 = _match_team_name(teams[0].get("team", {}).get("longName", ""))
        t2 = _match_team_name(teams[1].get("team", {}).get("longName", ""))
        ground = match.get("ground", {})

        results.append({
            "match_slug": match_slug,
            "series_slug": series_slug,
            "date": match.get("startDate", "")[:10],
            "team1": t1,
            "team2": t2,
            "venue": ground.get("longName", ground.get("name", "")),
            "city": ground.get("town", {}).get("name", ""),
            "status": match.get("state", ""),
        })
    return results


def get_match_details(series_slug: str, match_slug: str) -> dict | None:
    """Fetch toss, venue, and playing XI for a match.

    Returns the dict shape that predictor.predict() expects, or None
    if toss/XIs are not yet available.
    """
    try:
        info = _ci.match_info(series_slug, match_slug)
    except Exception:
        return None

    toss = info.get("toss")
    if not toss or not toss.get("winner_team"):
        return None

    venue = info.get("venue", {})
    toss_winner_raw = toss["winner_team"]
    toss_decision = toss.get("decision", "field")

    # Get playing XI from scorecard (available once toss is done)
    try:
        sc = _ci.match_scorecard(series_slug, match_slug)
    except Exception:
        return None

    if not sc or not isinstance(sc, list) or len(sc) < 1:
        return None

    # Extract player names from scorecard innings
    xi_names = {"team1": [], "team2": []}
    seen = set()
    for innings in sc:
        team_name = innings.get("team", {}).get("longName", "")
        bat = innings.get("inningBatsmen", [])
        bowl = innings.get("inningBowlers", [])
        players = []
        for b in bat:
            name = b.get("player", {}).get("longName", b.get("player", {}).get("name", ""))
            pid = str(b.get("player", {}).get("objectId", ""))
            if pid not in seen and name:
                players.append(name)
                seen.add(pid)
        for b in bowl:
            name = b.get("player", {}).get("longName", b.get("player", {}).get("name", ""))
            pid = str(b.get("player", {}).get("objectId", ""))
            if pid not in seen and name:
                players.append(name)
                seen.add(pid)

        canonical = _match_team_name(team_name)
        if not canonical:
            continue

        key = "team1" if not xi_names["team1"] else "team2"
        xi_names[key] = players

    if not xi_names["team1"] or not xi_names["team2"]:
        return None

    # Determine team1/team2 from the scorecard order
    first_innings_team = sc[0].get("team", {}).get("longName", "")
    team1 = _match_team_name(first_innings_team)
    second_innings_team = sc[1].get("team", {}).get("longName", "") if len(sc) > 1 else ""
    team2 = _match_team_name(second_innings_team)

    if not team1 or not team2:
        return None

    toss_winner = _match_team_name(toss_winner_raw) or toss_winner_raw

    return {
        "team1": team1,
        "team2": team2,
        "venue": venue.get("name", ""),
        "city": venue.get("town", {}).get("name", ""),
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "team1_xi": xi_names["team1"],
        "team2_xi": xi_names["team2"],
    }


def poll_until_toss(series_slug: str, match_slug: str, timeout: int = 7200) -> dict | None:
    """Poll until toss and XIs are available. Returns match details or None on timeout."""
    elapsed = 0
    while elapsed < timeout:
        details = get_match_details(series_slug, match_slug)
        if details:
            return details
        time.sleep(SCRAPE_POLL_SECS)
        elapsed += SCRAPE_POLL_SECS
    return None
