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


IPL_2026_SERIES = "ipl-2026-1510719"


def get_todays_ipl_matches() -> list[dict]:
    """Get today's IPL matches from series_matches (correct objectIds for scorecard)."""
    from datetime import date

    today = date.today().isoformat()
    results = []
    try:
        data = _ci.series_matches(IPL_2026_SERIES)
        matches = data.get("content", {}).get("matches", [])
    except Exception:
        return []

    for m in matches:
        match_date = (m.get("startDate") or "")[:10]
        if match_date != today:
            continue

        stage = m.get("stage", "")
        if stage == "FINISHED":
            continue

        match_slug = f"{m['slug']}-{m['objectId']}"
        teams = m.get("teams", [])
        if len(teams) < 2:
            continue

        t1 = _match_team_name(teams[0].get("team", {}).get("longName", ""))
        t2 = _match_team_name(teams[1].get("team", {}).get("longName", ""))
        ground = m.get("ground", {})

        results.append({
            "match_slug": match_slug,
            "series_slug": IPL_2026_SERIES,
            "date": match_date,
            "team1": t1,
            "team2": t2,
            "venue": ground.get("longName", ground.get("name", "")),
            "city": ground.get("town", {}).get("name", ""),
            "status": stage,
        })
    return results


def get_match_details(series_slug: str, match_slug: str) -> dict | None:
    return _get_match_details(_ci, series_slug, match_slug)


def _get_match_details(ci: CricinfoClient, series_slug: str, match_slug: str) -> dict | None:
    """Fetch toss, venue, and playing XI for a match.

    Uses match_scorecard endpoint which exposes:
      - match.tossWinnerTeamId + tossWinnerChoice for toss
      - content.matchPlayers.teamPlayers for confirmed playing XIs
      - content.notes for impact player substitution options

    Returns the dict shape that predictor.predict() expects, or None
    if toss/XIs are not yet available.
    """
    try:
        sc = ci.match_scorecard(series_slug, match_slug)
    except Exception:
        return None

    if not sc or not isinstance(sc, dict):
        return None

    m = sc.get("match", {})
    content = sc.get("content", {})

    # --- Toss ---
    toss_winner_id = m.get("tossWinnerTeamId")
    toss_choice = m.get("tossWinnerChoice")
    if not toss_winner_id:
        return None

    toss_decision = "bat" if toss_choice == 1 else "field"

    # --- Teams ---
    match_info_teams = m.get("teams", [])
    if len(match_info_teams) < 2:
        return None

    id_to_team = {}
    for t in match_info_teams:
        team_obj = t.get("team", {})
        tid = team_obj.get("id")
        canonical = _match_team_name(team_obj.get("longName", ""))
        if tid and canonical:
            id_to_team[tid] = canonical

    team1_raw = match_info_teams[0].get("team", {}).get("longName", "")
    team2_raw = match_info_teams[1].get("team", {}).get("longName", "")
    team1 = _match_team_name(team1_raw)
    team2 = _match_team_name(team2_raw)

    if not team1 or not team2:
        return None

    toss_winner = id_to_team.get(toss_winner_id)
    if not toss_winner:
        return None

    # --- Playing XIs from matchPlayers.teamPlayers ---
    team_players = (
        content.get("matchPlayers", {}).get("teamPlayers", [])
    )
    if len(team_players) < 2:
        return None

    xi = {}
    for tp in team_players:
        if tp.get("type") != "PLAYING":
            continue
        tp_team = tp.get("team", {})
        canonical = _match_team_name(tp_team.get("longName", ""))
        if not canonical:
            continue
        names = []
        for p in tp.get("players", []):
            player = p.get("player", {})
            name = player.get("longName") or player.get("name", "")
            if name:
                names.append(name)
        if names:
            xi[canonical] = names

    if team1 not in xi or team2 not in xi:
        return None

    # --- Venue ---
    ground = m.get("ground", {})
    venue_name = ground.get("longName") or ground.get("name", "")
    city = ground.get("town", {}).get("name", "")

    # --- Impact player options (from notes) ---
    impact_subs = {}
    notes = content.get("notes", {})
    for group in notes.get("groups", []) if isinstance(notes, dict) else []:
        for note in group.get("notes", []):
            if not isinstance(note, str):
                continue
            low = note.lower()
            if "impact player sub" not in low:
                continue
            for canonical_name in (team1, team2):
                for kw in TEAM_KEYWORDS.get(canonical_name, []):
                    if kw in low:
                        names = note.split(":", 1)[-1].strip()
                        impact_subs[canonical_name] = [
                            n.strip() for n in names.replace(" and ", ",").split(",") if n.strip()
                        ]
                        break

    result = {
        "team1": team1,
        "team2": team2,
        "venue": venue_name,
        "city": city,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "team1_xi": xi[team1],
        "team2_xi": xi[team2],
    }

    if impact_subs:
        result["impact_subs"] = impact_subs

    return result


def poll_until_toss(series_slug: str, match_slug: str, timeout: int = 7200) -> dict | None:
    """Poll until toss and XIs are available. Returns match details or None on timeout."""
    elapsed = 0
    while elapsed < timeout:
        # Fresh client each iteration — CricinfoClient caches scorecards
        # in memory, so reusing _ci would return stale pre-toss data forever.
        ci = CricinfoClient()
        details = _get_match_details(ci, series_slug, match_slug)
        if details:
            return details
        time.sleep(SCRAPE_POLL_SECS)
        elapsed += SCRAPE_POLL_SECS
    return None


def get_live_scores() -> list[dict]:
    """Get live IPL match scores from Cricbuzz (much more real-time than cricdata).

    Scrapes the Cricbuzz live-scores page and parses React Server Component
    data embedded in the HTML for current match state.

    Returns list of dicts with keys:
        team1, team2, t1_score, t2_score, status_text,
        match_state, match_id
    """
    import json
    import re
    import requests

    results = []
    try:
        resp = requests.get(
            "https://www.cricbuzz.com/cricket-match/live-scores",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=8,
        )
        if resp.status_code != 200:
            return results

        chunks = re.findall(
            r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', resp.text, re.DOTALL,
        )

        for chunk in chunks:
            try:
                unesc = chunk.encode().decode("unicode_escape")
            except Exception:
                continue
            if "seriesMatches" not in unesc or "matchId" not in unesc:
                continue

            idx = unesc.find("[{")
            if idx < 0:
                continue
            try:
                decoder = json.JSONDecoder()
                data, _ = decoder.raw_decode(unesc[idx:])
            except Exception:
                continue

            for match_type in data:
                for sm in match_type.get("seriesMatches", []):
                    sw = sm.get("seriesAdWrapper", {})
                    sname = sw.get("seriesName", "")
                    if "indian premier league" not in sname.lower():
                        continue

                    for m in sw.get("matches", []):
                        mi = m.get("matchInfo", {})
                        ms = m.get("matchScore", {})

                        state = mi.get("state", "")
                        if state == "Complete":
                            state = "POST"
                        elif state == "In Progress":
                            state = "LIVE"
                        else:
                            state = "PRE"

                        t1_info = mi.get("team1", {})
                        t2_info = mi.get("team2", {})
                        t1 = _match_team_name(t1_info.get("teamName", ""))
                        t2 = _match_team_name(t2_info.get("teamName", ""))
                        if not t1 or not t2:
                            continue

                        t1_inn = ms.get("team1Score", {}).get("inngs1", {})
                        t2_inn = ms.get("team2Score", {}).get("inngs1", {})

                        def _fmt_score(inn: dict) -> str | None:
                            if not inn:
                                return None
                            r = inn.get("runs")
                            if r is None:
                                return None
                            w = inn.get("wickets", 0)
                            return f"{r}/{w}"

                        def _fmt_overs(inn: dict) -> str | None:
                            if not inn:
                                return None
                            ov = inn.get("overs")
                            if ov is None:
                                return None
                            return f"({ov} ov)"

                        results.append({
                            "team1": t1,
                            "team2": t2,
                            "t1_score": _fmt_score(t1_inn),
                            "t2_score": _fmt_score(t2_inn),
                            "t1_score_info": _fmt_overs(t1_inn),
                            "t2_score_info": _fmt_overs(t2_inn),
                            "status_text": mi.get("status", ""),
                            "match_state": state,
                            "match_id": mi.get("matchId"),
                        })
    except Exception:
        pass
    return results
