"""Feature builders for IPL match prediction.

Every builder takes historical data strictly before the match date and
returns feature dicts keyed with a team prefix (t1_ / t2_).
"""

import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import re

from predictor.normalize import (
    ROOT, DATA_DIR, normalize_team, is_home, OPP_ABBREV, TEAM_HOME_CITIES,
)

MATCHES_DIR = DATA_DIR / "matches"
PLAYER_DIR = DATA_DIR / "player_innings"
BIOS_PATH = DATA_DIR / "player_bios.csv"

_IPL_OPPOSITIONS = set(OPP_ABBREV.keys())


def _is_ipl_innings(inn: dict) -> bool:
    return inn.get("opposition", "") in _IPL_OPPOSITIONS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _mean(vals, default=0.0):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else default


def _std(vals, default=0.0):
    m = _mean(vals)
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return default
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _over_to_balls(over_str):
    """Convert over like '6.3' to 39 balls."""
    try:
        parts = str(over_str).split(".")
        overs = int(parts[0])
        extra = int(parts[1]) if len(parts) > 1 else 0
        return overs * 6 + extra
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Loaders (called once, cached)
# ---------------------------------------------------------------------------

_bbb_cache: dict[str, list[dict]] = {}


def _load_bbb(match_id: str) -> list[dict]:
    if match_id in _bbb_cache:
        return _bbb_cache[match_id]
    path = MATCHES_DIR / f"{match_id}.csv"
    if not path.exists():
        _bbb_cache[match_id] = []
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    _bbb_cache[match_id] = rows
    return rows


_player_innings_cache: dict[str, list[dict]] = {}


def _load_player_innings(player_id: str) -> list[dict]:
    if player_id in _player_innings_cache:
        return _player_innings_cache[player_id]

    for p in PLAYER_DIR.glob(f"*_{player_id}.csv"):
        with open(p) as f:
            rows = list(csv.DictReader(f))
        _player_innings_cache[player_id] = rows
        return rows
    _player_innings_cache[player_id] = []
    return []


_bios_cache: dict[str, dict] | None = None


def load_bios() -> dict[str, dict]:
    global _bios_cache
    if _bios_cache is not None:
        return _bios_cache
    _bios_cache = {}
    if BIOS_PATH.exists():
        with open(BIOS_PATH) as f:
            for row in csv.DictReader(f):
                _bios_cache[row["player_id"]] = row
    return _bios_cache


# ---------------------------------------------------------------------------
# Parse date from player_innings "Start Date" field (e.g. "3 Apr 2007")
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_innings_date(d: str) -> datetime | None:
    parts = d.strip().split()
    if len(parts) != 3:
        return None
    try:
        return datetime(int(parts[2]), _MONTH_MAP.get(parts[1], 0), int(parts[0]))
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# A. Team Form Features
# ---------------------------------------------------------------------------

def team_form_features(
    team: str,
    prior_matches: list[dict],
    prefix: str,
) -> dict:
    """Win rates, NRR, chasing/setting split from team's prior IPL matches."""
    team_matches = [
        m for m in prior_matches
        if m["team1"] == team or m["team2"] == team
    ]

    feats = {}

    def _win_rate(matches):
        if not matches:
            return 0.5
        return sum(1 for m in matches if m["winner"] == team) / len(matches)

    feats[f"{prefix}win_rate_last5"] = _win_rate(team_matches[-5:])
    feats[f"{prefix}win_rate_last10"] = _win_rate(team_matches[-10:])
    feats[f"{prefix}win_rate_season"] = _win_rate(team_matches[-20:])
    feats[f"{prefix}matches_played"] = len(team_matches)

    bat_first = [m for m in team_matches if
                 (m["toss_winner"] == team and m["toss_decision"] == "bat") or
                 (m["toss_winner"] != team and m["toss_decision"] == "field")]
    bat_second = [m for m in team_matches if m not in bat_first]

    feats[f"{prefix}chase_win_rate"] = _win_rate(bat_second[-10:])
    feats[f"{prefix}set_win_rate"] = _win_rate(bat_first[-10:])

    return feats


def head_to_head_features(
    team1: str, team2: str,
    prior_matches: list[dict],
) -> dict:
    h2h = [
        m for m in prior_matches
        if {m["team1"], m["team2"]} == {team1, team2}
    ]
    if not h2h:
        return {"h2h_t1_win_rate": 0.5, "h2h_matches": 0,
                "h2h_recent_t1_win_rate": 0.5}
    t1_wins = sum(1 for m in h2h if m["winner"] == team1)
    recent = h2h[-6:]
    t1_recent_wins = sum(1 for m in recent if m["winner"] == team1)
    return {
        "h2h_t1_win_rate": t1_wins / len(h2h),
        "h2h_matches": len(h2h),
        "h2h_recent_t1_win_rate": t1_recent_wins / len(recent),
    }


# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------

def elo_features(
    team1: str, team2: str,
    prior_matches: list[dict],
    elo_state: dict[str, float] | None = None,
    K: float = 32.0,
    season_decay: float = 0.4,
) -> tuple[dict, dict[str, float]]:
    """Compute ELO ratings from match history.

    Returns (features_dict, updated_elo_state).
    If elo_state is provided, uses it directly (O(1) lookup).
    Otherwise recomputes from scratch (O(N) over prior_matches).
    season_decay: regress toward 1500 by this fraction at season boundaries.
    """
    if elo_state is None:
        elos: dict[str, float] = defaultdict(lambda: 1500.0)
        prev_season = None
        for m in prior_matches:
            s = m.get("season")
            if prev_season is not None and s != prev_season:
                for t in list(elos.keys()):
                    elos[t] = elos[t] * (1 - season_decay) + 1500.0 * season_decay
            prev_season = s
            t1, t2 = m["team1"], m["team2"]
            e1 = 1 / (1 + 10 ** ((elos[t2] - elos[t1]) / 400))
            s1 = 1.0 if m["winner"] == t1 else 0.0
            elos[t1] += K * (s1 - e1)
            elos[t2] += K * ((1 - s1) - (1 - e1))
    else:
        elos = defaultdict(lambda: 1500.0, elo_state)

    e1 = elos[team1]
    e2 = elos[team2]
    diff = e1 - e2
    expected = 1 / (1 + 10 ** (-diff / 400))

    feats = {
        "t1_elo": e1,
        "t2_elo": e2,
        "elo_diff": diff,
        "elo_expected": expected,
    }
    return feats, dict(elos)


def update_elo(elo_state: dict[str, float], match: dict,
               K: float = 32.0, season_decay: float = 0.4,
               prev_season: int | None = None) -> dict[str, float]:
    """Update ELO state after a single match result."""
    elos = defaultdict(lambda: 1500.0, elo_state)
    s = match.get("season")
    if prev_season is not None and s != prev_season:
        for t in list(elos.keys()):
            elos[t] = elos[t] * (1 - season_decay) + 1500.0 * season_decay
    t1, t2 = match["team1"], match["team2"]
    e1 = 1 / (1 + 10 ** ((elos[t2] - elos[t1]) / 400))
    s1 = 1.0 if match["winner"] == t1 else 0.0
    elos[t1] += K * (s1 - e1)
    elos[t2] += K * ((1 - s1) - (1 - e1))
    return dict(elos)


# ---------------------------------------------------------------------------
# Current-season form & momentum
# ---------------------------------------------------------------------------

def current_season_form(
    team: str,
    prior_matches: list[dict],
    season: int,
    prefix: str,
) -> dict:
    """Current season win rate, win streak, and net run rate proxy."""
    season_matches = [
        m for m in prior_matches
        if (m["team1"] == team or m["team2"] == team) and m.get("season") == season
    ]

    if not season_matches:
        return {
            f"{prefix}season_win_rate": 0.5,
            f"{prefix}season_matches": 0,
            f"{prefix}win_streak": 0,
            f"{prefix}loss_streak": 0,
            f"{prefix}season_chase_win_rate": 0.5,
        }

    wins = sum(1 for m in season_matches if m["winner"] == team)
    wr = wins / len(season_matches)

    streak = 0
    for m in reversed(season_matches):
        if m["winner"] == team:
            streak += 1
        else:
            break
    loss_streak = 0
    for m in reversed(season_matches):
        if m["winner"] != team:
            loss_streak += 1
        else:
            break

    chasing = [m for m in season_matches if
               (m["toss_winner"] == team and m["toss_decision"] == "field") or
               (m["toss_winner"] != team and m["toss_decision"] == "bat")]
    chase_wr = sum(1 for m in chasing if m["winner"] == team) / len(chasing) if chasing else 0.5

    return {
        f"{prefix}season_win_rate": wr,
        f"{prefix}season_matches": len(season_matches),
        f"{prefix}win_streak": streak,
        f"{prefix}loss_streak": loss_streak,
        f"{prefix}season_chase_win_rate": chase_wr,
    }


# ---------------------------------------------------------------------------
# B. Phase-specific team features from BBB
# ---------------------------------------------------------------------------

_PHASES = {"powerplay": (0, 6), "middle": (6, 15), "death": (15, 20)}


def _phase_of_ball(over_val) -> str | None:
    try:
        ov = float(over_val)
    except (ValueError, TypeError):
        return None
    ov_int = int(ov)
    for name, (lo, hi) in _PHASES.items():
        if lo <= ov_int < hi:
            return name
    return None


def _team_phase_stats_from_bbb(
    team: str,
    match_row: dict,
    xi: dict[str, str],
) -> dict:
    """Compute phase stats for one match where team participated.

    Returns dict with phase-specific batting and bowling aggregates.
    """
    mid = match_row["match_id"]
    balls = _load_bbb(mid)
    if not balls:
        return {}

    xi_ids = set(xi.keys())

    stats = {}
    for phase_name in _PHASES:
        for role in ("bat", "bowl"):
            stats[f"{phase_name}_{role}_runs"] = 0
            stats[f"{phase_name}_{role}_balls"] = 0
            stats[f"{phase_name}_{role}_wkts"] = 0
            stats[f"{phase_name}_{role}_dots"] = 0
            stats[f"{phase_name}_{role}_boundaries"] = 0
            stats[f"{phase_name}_{role}_extras"] = 0

    for ball in balls:
        phase = _phase_of_ball(ball.get("over"))
        if phase is None:
            continue

        bat_id = ball.get("batsman_id", "")
        bowl_id = ball.get("bowler_id", "")
        runs = int(ball.get("runs", 0))
        play_type = ball.get("play_type", "")
        is_wkt = ball.get("is_wicket") == "True"

        is_batting = bat_id in xi_ids
        is_bowling = bowl_id in xi_ids

        if is_batting:
            stats[f"{phase}_bat_runs"] += runs
            if play_type not in ("wide", "no ball"):
                stats[f"{phase}_bat_balls"] += 1
            if runs == 0 and play_type == "no run":
                stats[f"{phase}_bat_dots"] += 1
            if play_type in ("four", "six"):
                stats[f"{phase}_bat_boundaries"] += 1
            if is_wkt:
                stats[f"{phase}_bat_wkts"] += 1

        if is_bowling:
            stats[f"{phase}_bowl_runs"] += runs
            if play_type not in ("wide", "no ball"):
                stats[f"{phase}_bowl_balls"] += 1
            if runs == 0 and play_type in ("no run",):
                stats[f"{phase}_bowl_dots"] += 1
            if play_type in ("four", "six"):
                stats[f"{phase}_bowl_boundaries"] += 1
            if is_wkt:
                stats[f"{phase}_bowl_wkts"] += 1
            if play_type in ("wide", "no ball"):
                stats[f"{phase}_bowl_extras"] += 1

    return stats


def aggregate_phase_features(
    team: str,
    prior_matches: list[dict],
    all_xis: dict[str, dict],
    team_key: str,
    prefix: str,
    n: int = 10,
) -> dict:
    """Aggregate phase stats over team's last N matches."""
    team_matches = [
        m for m in prior_matches
        if (m["team1"] == team or m["team2"] == team) and m["match_id"] in all_xis
    ][-n:]

    agg = defaultdict(list)
    for m in team_matches:
        xi_data = all_xis[m["match_id"]]
        xi = xi_data[team_key] if m[team_key.replace("_xi", "")] == team else (
            xi_data["team1_xi"] if m["team1"] == team else xi_data["team2_xi"]
        )
        stats = _team_phase_stats_from_bbb(team, m, xi)
        for k, v in stats.items():
            agg[k].append(v)

    feats = {}
    for phase in _PHASES:
        for role in ("bat", "bowl"):
            runs = sum(agg.get(f"{phase}_{role}_runs", [0]))
            balls = sum(agg.get(f"{phase}_{role}_balls", [0]))
            wkts = sum(agg.get(f"{phase}_{role}_wkts", [0]))
            dots = sum(agg.get(f"{phase}_{role}_dots", [0]))
            boundaries = sum(agg.get(f"{phase}_{role}_boundaries", [0]))
            extras = sum(agg.get(f"{phase}_{role}_extras", [0]))

            rr = (runs / balls * 6) if balls > 0 else 0.0
            dot_pct = dots / balls if balls > 0 else 0.0
            bound_pct = boundaries / balls if balls > 0 else 0.0

            feats[f"{prefix}{phase}_{role}_rr"] = rr
            feats[f"{prefix}{phase}_{role}_dot_pct"] = dot_pct
            feats[f"{prefix}{phase}_{role}_bound_pct"] = bound_pct
            if role == "bowl":
                feats[f"{prefix}{phase}_bowl_extras_per_match"] = (
                    extras / len(team_matches) if team_matches else 0
                )
    return feats


# ---------------------------------------------------------------------------
# C. Player aggregate features from career innings
# ---------------------------------------------------------------------------

def _player_batting_form(player_id: str, before_date: str, n: int = 10) -> dict:
    """Recent T20 batting form for a player before a given date."""
    innings = _load_player_innings(player_id)
    cutoff = datetime.strptime(before_date, "%Y-%m-%d")

    recent = []
    for inn in innings:
        d = _parse_innings_date(inn.get("start_date", ""))
        if d is None or d >= cutoff:
            continue
        runs = inn.get("bat_runs", "")
        if runs in ("DNB", "-", "TDNB", "sub", "absent", ""):
            continue
        runs_val = _safe_float(runs.rstrip("*"))
        bf = _safe_float(inn.get("bat_bf", "0"))
        sr = _safe_float(inn.get("bat_sr", "0"))
        pos = _safe_float(inn.get("bat_position", "0"))
        not_out = 1 if inn.get("bat_dismissal") == "not out" else 0
        recent.append({
            "runs": runs_val, "bf": bf, "sr": sr, "pos": pos, "not_out": not_out,
        })

    if not recent:
        return {"bat_runs_avg": 0, "bat_sr_avg": 0, "bat_consistency": 0,
                "bat_pos_avg": 0, "bat_not_out_pct": 0, "bat_experience": 0}

    last_n = recent[-n:]
    return {
        "bat_runs_avg": _mean([i["runs"] for i in last_n]),
        "bat_sr_avg": _mean([i["sr"] for i in last_n if i["sr"] > 0]),
        "bat_consistency": _std([i["runs"] for i in last_n]),
        "bat_pos_avg": _mean([i["pos"] for i in last_n if i["pos"] > 0]),
        "bat_not_out_pct": _mean([i["not_out"] for i in last_n]),
        "bat_experience": len(recent),
    }


def _player_bowling_form(player_id: str, before_date: str, n: int = 10) -> dict:
    innings = _load_player_innings(player_id)
    cutoff = datetime.strptime(before_date, "%Y-%m-%d")
    role = load_bios().get(player_id, {}).get("role", "")

    def _overs_to_float(overs: str) -> float:
        if "." not in overs:
            return _safe_float(overs)
        whole, part = overs.split(".", 1)
        balls = _safe_float(part)
        return _safe_float(whole) + (balls / 6.0)

    recent = []
    for inn in innings:
        d = _parse_innings_date(inn.get("start_date", ""))
        if d is None or d >= cutoff:
            continue
        overs = inn.get("bowl_overs", "")
        if overs in ("DNB", "-", "", "0"):
            continue
        econ = _safe_float(inn.get("bowl_economy", "0"))
        wkts = _safe_float(inn.get("bowl_wickets", "0"))
        recent.append({"econ": econ, "wkts": wkts, "overs": _overs_to_float(str(overs))})

    if not recent:
        return {"bowl_econ_avg": 12.0, "bowl_wkts_avg": 0, "bowl_experience": 0,
                "is_regular_bowler": 0}

    last_n = recent[-n:]
    avg_overs = _mean([i["overs"] for i in last_n])
    is_regular = role in ("Bowler", "Bowling allrounder") or (len(last_n) >= 5 and avg_overs >= 2.0)
    return {
        "bowl_econ_avg": _mean([i["econ"] for i in last_n if i["econ"] > 0]),
        "bowl_wkts_avg": _mean([i["wkts"] for i in last_n]),
        "bowl_experience": len(recent),
        "is_regular_bowler": 1 if is_regular else 0,
    }


def _player_venue_form(player_id: str, ground: str, before_date: str) -> dict:
    """Career batting/bowling at a specific ground before date."""
    innings = _load_player_innings(player_id)
    cutoff = datetime.strptime(before_date, "%Y-%m-%d")

    bat_runs, bat_outs, bowl_runs, bowl_balls, matches = 0, 0, 0, 0, 0
    for inn in innings:
        d = _parse_innings_date(inn.get("start_date", ""))
        if d is None or d >= cutoff:
            continue
        inn_ground = inn.get("ground", "")
        if ground.lower() not in inn_ground.lower() and inn_ground.lower() not in ground.lower():
            continue
        matches += 1
        runs_str = inn.get("bat_runs", "")
        if runs_str not in ("DNB", "-", "TDNB", "sub", "absent", ""):
            r = _safe_float(runs_str.rstrip("*"))
            bat_runs += r
            if inn.get("bat_dismissal") != "not out":
                bat_outs += 1
        overs_str = inn.get("bowl_overs", "")
        if overs_str not in ("DNB", "-", "", "0"):
            bowl_runs += _safe_float(inn.get("bowl_runs", "0"))
            bowl_balls += _over_to_balls(overs_str)

    bat_avg = bat_runs / bat_outs if bat_outs > 0 else bat_runs
    bowl_econ = (bowl_runs / bowl_balls * 6) if bowl_balls > 0 else 0
    return {"venue_bat_avg": bat_avg, "venue_bowl_econ": bowl_econ, "venue_matches": matches}


def _player_opp_form(player_id: str, opponent: str, before_date: str) -> dict:
    """Career batting SR and bowling economy vs a specific opponent."""
    innings = _load_player_innings(player_id)
    cutoff = datetime.strptime(before_date, "%Y-%m-%d")

    mapped_opp = normalize_team(opponent)
    bat_runs, bat_bf, bowl_runs, bowl_balls = 0, 0, 0, 0
    for inn in innings:
        d = _parse_innings_date(inn.get("start_date", ""))
        if d is None or d >= cutoff:
            continue
        opp_raw = inn.get("opposition", "")
        opp_canon = OPP_ABBREV.get(opp_raw, "")
        if opp_canon != mapped_opp:
            continue
        runs_str = inn.get("bat_runs", "")
        if runs_str not in ("DNB", "-", "TDNB", "sub", "absent", ""):
            bat_runs += _safe_float(runs_str.rstrip("*"))
            bat_bf += _safe_float(inn.get("bat_bf", "0"))
        overs_str = inn.get("bowl_overs", "")
        if overs_str not in ("DNB", "-", "", "0"):
            bowl_runs += _safe_float(inn.get("bowl_runs", "0"))
            bowl_balls += _over_to_balls(overs_str)

    bat_sr = (bat_runs / bat_bf * 100) if bat_bf > 0 else 0
    bowl_econ = (bowl_runs / bowl_balls * 6) if bowl_balls > 0 else 0
    return {"opp_bat_sr": bat_sr, "opp_bowl_econ": bowl_econ}


def player_xi_features(
    xi: dict[str, str],
    opponent: str,
    venue_city: str,
    before_date: str,
    prefix: str,
) -> dict:
    """Aggregate player-level features across an XI."""
    bios = load_bios()
    bat_forms = []
    bowl_forms = []
    venue_forms = []
    opp_forms = []

    spin_count = 0
    pace_count = 0
    left_bat = 0
    left_bowl = 0
    specialist_bat = 0
    wk_bat = 0
    bowling_ar = 0
    batting_ar = 0
    allrounder = 0

    for pid, pname in xi.items():
        bf = _player_batting_form(pid, before_date)
        bat_forms.append(bf)

        wf = _player_bowling_form(pid, before_date)
        bowl_forms.append(wf)

        vf = _player_venue_form(pid, venue_city, before_date)
        venue_forms.append(vf)

        of = _player_opp_form(pid, opponent, before_date)
        opp_forms.append(of)

        bio = bios.get(pid, {})
        bt = bio.get("bowl_type", "")
        if bt == "spin":
            spin_count += 1
        elif bt == "pace":
            pace_count += 1
        if bio.get("bat_hand") == "left":
            left_bat += 1
        if "left" in bio.get("bowl_style", "").lower():
            left_bowl += 1

        role = bio.get("role", "")
        if role in ("Opening batter", "Top-order batter", "Middle-order batter", "Batter"):
            specialist_bat += 1
        elif role in ("Wicketkeeper batter", "Wicketkeeper"):
            wk_bat += 1
        elif role == "Bowling allrounder":
            bowling_ar += 1
        elif role == "Batting allrounder":
            batting_ar += 1
        elif role == "Allrounder":
            allrounder += 1

    feats = {}

    # Batting aggregates by position bucket
    top = [f for f in bat_forms if 0 < f["bat_pos_avg"] <= 3]
    mid = [f for f in bat_forms if 3 < f["bat_pos_avg"] <= 6]
    low = [f for f in bat_forms if f["bat_pos_avg"] > 6]

    for bucket, label in [(top, "top"), (mid, "mid"), (low, "low")]:
        feats[f"{prefix}{label}_bat_sr"] = _mean([f["bat_sr_avg"] for f in bucket])
        feats[f"{prefix}{label}_bat_avg"] = _mean([f["bat_runs_avg"] for f in bucket])

    feats[f"{prefix}xi_bat_sr"] = _mean([f["bat_sr_avg"] for f in bat_forms])
    feats[f"{prefix}xi_bat_avg"] = _mean([f["bat_runs_avg"] for f in bat_forms])
    feats[f"{prefix}xi_bat_consistency"] = _mean([f["bat_consistency"] for f in bat_forms])
    feats[f"{prefix}xi_bat_experience"] = sum(f["bat_experience"] for f in bat_forms)
    feats[f"{prefix}xi_not_out_pct"] = _mean(
        [f["bat_not_out_pct"] for f in bat_forms if f["bat_pos_avg"] >= 5]
    )

    # Bowling aggregates
    bowlers = [f for f in bowl_forms if f["is_regular_bowler"]]
    feats[f"{prefix}xi_bowl_econ"] = _mean([f["bowl_econ_avg"] for f in bowlers]) if bowlers else 12.0
    feats[f"{prefix}xi_bowl_wkts"] = _mean([f["bowl_wkts_avg"] for f in bowlers]) if bowlers else 0.0
    feats[f"{prefix}xi_bowl_experience"] = sum(f["bowl_experience"] for f in bowl_forms)
    feats[f"{prefix}frontline_bowlers"] = len(bowlers)
    best_econ = min((f["bowl_econ_avg"] for f in bowlers if f["bowl_econ_avg"] > 0), default=12.0)
    feats[f"{prefix}best_bowler_econ"] = best_econ

    # Venue aggregates
    feats[f"{prefix}venue_bat_avg"] = _mean([f["venue_bat_avg"] for f in venue_forms])
    feats[f"{prefix}venue_bowl_econ"] = _mean(
        [f["venue_bowl_econ"] for f in venue_forms if f["venue_bowl_econ"] > 0]
    )
    feats[f"{prefix}venue_familiarity"] = sum(f["venue_matches"] for f in venue_forms)

    # Opposition aggregates
    feats[f"{prefix}opp_bat_sr"] = _mean(
        [f["opp_bat_sr"] for f in opp_forms if f["opp_bat_sr"] > 0]
    )
    feats[f"{prefix}opp_bowl_econ"] = _mean(
        [f["opp_bowl_econ"] for f in opp_forms if f["opp_bowl_econ"] > 0]
    )

    # Composition
    feats[f"{prefix}spin_count"] = spin_count
    feats[f"{prefix}pace_count"] = pace_count
    feats[f"{prefix}left_bat_count"] = left_bat
    feats[f"{prefix}left_bowl_count"] = left_bowl

    # Role balance
    feats[f"{prefix}specialist_bat"] = specialist_bat
    feats[f"{prefix}wk_bat"] = wk_bat
    feats[f"{prefix}bowling_ar"] = bowling_ar
    feats[f"{prefix}batting_ar"] = batting_ar
    feats[f"{prefix}allrounder"] = allrounder
    feats[f"{prefix}specialist_bowler"] = len([f for f in bowl_forms if f["is_regular_bowler"]]) - bowling_ar

    return feats


# ---------------------------------------------------------------------------
# Boundary size lookup (straight boundary in meters, approximate)
# ---------------------------------------------------------------------------

_BOUNDARY_SIZES = {
    "Mumbai": 66,         # Wankhede
    "Bengaluru": 62,      # Chinnaswamy
    "Kolkata": 75,        # Eden Gardens
    "Delhi": 64,          # Feroz Shah Kotla / Arun Jaitley
    "Chennai": 70,        # Chepauk
    "Hyderabad": 72,      # Rajiv Gandhi / Uppal
    "Jaipur": 68,         # Sawai Mansingh
    "Mohali": 72,         # IS Bindra / PCA
    "New Chandigarh": 72, # Mullanpur
    "Pune": 70,           # MCA
    "Ahmedabad": 78,      # Narendra Modi / Motera
    "Lucknow": 72,        # Ekana
    "Visakhapatnam": 70,  # ACA-VDCA
    "Rajkot": 70,         # Saurashtra
    "Indore": 68,         # Holkar
    "Guwahati": 68,       # Barsapara
    "Dharamsala": 62,     # HPCA
    "Navi Mumbai": 70,    # DY Patil
    "Raipur": 70,         # VNS International
    "Kanpur": 68,         # Green Park
    "Ranchi": 68,         # JSCA
    "Dubai": 72,          # Dubai International
    "Abu Dhabi": 72,      # Zayed / Sheikh Zayed
    "Sharjah": 60,        # Sharjah Cricket Stadium
}

_DEFAULT_BOUNDARY = 70


# ---------------------------------------------------------------------------
# G. Venue features
# ---------------------------------------------------------------------------

def venue_features(
    venue: str,
    city: str,
    prior_matches: list[dict],
) -> dict:
    venue_matches = [
        m for m in prior_matches
        if m.get("city", "") == city or venue.lower() in m.get("venue", "").lower()
    ]

    if not venue_matches:
        return {
            "venue_avg_1st_total": 160.0,
            "venue_chase_win_rate": 0.5,
            "venue_bat_first_win_rate": 0.5,
            "venue_matches_count": 0,
            "venue_boundary_size": _BOUNDARY_SIZES.get(city, _DEFAULT_BOUNDARY),
        }

    # Compute avg 1st innings total from BBB
    first_totals = []
    for m in venue_matches[-30:]:
        balls = _load_bbb(m["match_id"])
        if not balls:
            continue
        inn1_runs = 0
        for b in balls:
            if b.get("innings") == "1":
                inn1_runs = max(inn1_runs, int(b.get("total_runs", 0)))
        if inn1_runs > 0:
            first_totals.append(inn1_runs)

    # Chasing win rate: who won when batting second
    chase_wins = 0
    chase_total = 0
    for m in venue_matches:
        toss_w = m["toss_winner"]
        toss_d = m["toss_decision"]
        if toss_d == "field":
            chasing_team = toss_w
        else:
            chasing_team = m["team1"] if toss_w == m["team2"] else m["team2"]
        chase_total += 1
        if m["winner"] == chasing_team:
            chase_wins += 1

    # Bat-first advantage when toss winner chose to bat
    bat_first_matches = [m for m in venue_matches if m["toss_decision"] == "bat"]
    bat_first_wins = sum(1 for m in bat_first_matches if m["winner"] == m["toss_winner"])

    boundary = _BOUNDARY_SIZES.get(city, _DEFAULT_BOUNDARY)

    return {
        "venue_avg_1st_total": _mean(first_totals, 160.0),
        "venue_chase_win_rate": chase_wins / chase_total if chase_total else 0.5,
        "venue_bat_first_win_rate": bat_first_wins / len(bat_first_matches) if bat_first_matches else 0.5,
        "venue_matches_count": len(venue_matches),
        "venue_boundary_size": boundary,
    }


# ---------------------------------------------------------------------------
# H. Toss features
# ---------------------------------------------------------------------------

def toss_features(match: dict, venue_feats: dict) -> dict:
    toss_is_team1 = 1 if match["toss_winner"] == match["team1"] else 0
    chose_bat = 1 if match["toss_decision"] == "bat" else 0
    chose_field = 1 - chose_bat

    chase_adv = venue_feats.get("venue_chase_win_rate", 0.5)
    toss_venue_interaction = chose_field * chase_adv + chose_bat * (1 - chase_adv)

    return {
        "toss_is_team1": toss_is_team1,
        "toss_chose_bat": chose_bat,
        "toss_venue_interaction": toss_venue_interaction,
    }


# ---------------------------------------------------------------------------
# I. Context features
# ---------------------------------------------------------------------------

def context_features(match: dict, prior_matches: list[dict]) -> dict:
    # Match number within season
    season_matches = [m for m in prior_matches if m["season"] == match["season"]]
    match_num = len(season_matches) + 1

    # Days since last match for each team
    match_date = datetime.strptime(match["date"], "%Y-%m-%d")

    def _days_since(team):
        team_prior = [
            m for m in prior_matches
            if m["team1"] == team or m["team2"] == team
        ]
        if not team_prior:
            return 30
        last = datetime.strptime(team_prior[-1]["date"], "%Y-%m-%d")
        return (match_date - last).days

    return {
        "match_num_in_season": match_num,
        "t1_days_rest": _days_since(match["team1"]),
        "t2_days_rest": _days_since(match["team2"]),
        "t1_is_home": 1 if is_home(match["team1"], match.get("city", "")) else 0,
        "t2_is_home": 1 if is_home(match["team2"], match.get("city", "")) else 0,
    }


# ---------------------------------------------------------------------------
# J. Phase matchup advantages (batting strength vs opposing bowling weakness)
# ---------------------------------------------------------------------------

def phase_matchup_features(feats: dict) -> dict:
    """Cross-team matchups: team1 batting RR vs team2 bowling RR per phase."""
    matchup = {}
    for phase in ("powerplay", "middle", "death"):
        t1_bat = feats.get(f"t1_{phase}_bat_rr", 0)
        t2_bowl = feats.get(f"t2_{phase}_bowl_rr", 0)
        t2_bat = feats.get(f"t2_{phase}_bat_rr", 0)
        t1_bowl = feats.get(f"t1_{phase}_bowl_rr", 0)
        matchup[f"t1_{phase}_matchup"] = t1_bat - t2_bowl
        matchup[f"t2_{phase}_matchup"] = t2_bat - t1_bowl
    return matchup


# ---------------------------------------------------------------------------
# K. Difference features
# ---------------------------------------------------------------------------

def add_diff_features(feats: dict) -> dict:
    """For every t1_X / t2_X pair, add diff_X = t1_X - t2_X."""
    t1_keys = [k for k in feats if k.startswith("t1_")]
    for k in t1_keys:
        t2_key = "t2_" + k[3:]
        if t2_key in feats:
            v1 = feats[k]
            v2 = feats[t2_key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                feats["diff_" + k[3:]] = v1 - v2
    return feats


# ---------------------------------------------------------------------------
# Master: build all features for one match
# ---------------------------------------------------------------------------

def build_match_features(
    match: dict,
    prior_matches: list[dict],
    all_xis: dict[str, dict],
    elo_state: dict[str, float] | None = None,
) -> dict:
    """Build the full feature vector for a single match."""
    mid = match["match_id"]
    feats = {}

    # ELO
    elo_feats, _ = elo_features(match["team1"], match["team2"],
                                prior_matches, elo_state)
    feats.update(elo_feats)

    # A. Team form
    feats.update(team_form_features(match["team1"], prior_matches, "t1_"))
    feats.update(team_form_features(match["team2"], prior_matches, "t2_"))
    feats.update(head_to_head_features(match["team1"], match["team2"], prior_matches))

    # Current season form
    season = match.get("season", 0)
    feats.update(current_season_form(match["team1"], prior_matches, season, "t1_"))
    feats.update(current_season_form(match["team2"], prior_matches, season, "t2_"))

    # B. Phase features
    xi_data = all_xis.get(mid)
    if xi_data:
        feats.update(aggregate_phase_features(
            match["team1"], prior_matches, all_xis, "team1_xi", "t1_"))
        feats.update(aggregate_phase_features(
            match["team2"], prior_matches, all_xis, "team2_xi", "t2_"))

    # C-F. Player aggregate features
    if xi_data:
        feats.update(player_xi_features(
            xi_data["team1_xi"], match["team2"],
            match.get("city", ""), match["date"], "t1_"))
        feats.update(player_xi_features(
            xi_data["team2_xi"], match["team1"],
            match.get("city", ""), match["date"], "t2_"))

    # G. Venue features
    v_feats = venue_features(match.get("venue", ""), match.get("city", ""), prior_matches)
    feats.update(v_feats)

    # H. Toss
    feats.update(toss_features(match, v_feats))

    # I. Context
    feats.update(context_features(match, prior_matches))

    # Playoff flag
    mn = match.get("match_number", "")
    feats["is_playoff"] = 1 if any(
        k in mn for k in ("Qualifier", "Eliminator", "Final")
    ) else 0

    # J. Phase matchups (cross-team bat vs bowl)
    feats.update(phase_matchup_features(feats))

    # K. Diffs
    feats = add_diff_features(feats)

    return feats
