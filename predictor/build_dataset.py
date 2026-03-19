"""Build the training dataset with strict time ordering."""

import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from predictor.normalize import load_matches, normalize_team, ROOT
from predictor.playing_xi import extract_all_xis
from predictor.features import build_match_features, update_elo, _load_bbb

DATASET_PATH = ROOT / "dataset.csv"
WEATHER_CSV = ROOT / "match_weather.csv"
CAPTAINS_CSV = ROOT / "match_captains.csv"
FIELDING_CSV = ROOT / "player_fielding.csv"
FIELDING_INNINGS_DIR = ROOT / "player_fielding_innings"


def _load_weather() -> dict[str, dict]:
    """Load match_weather.csv into a dict keyed by match_id."""
    path = WEATHER_CSV
    if not path.exists():
        return {}
    lookup = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            lookup[row["match_id"]] = row
    return lookup

def _load_captains() -> dict[str, dict]:
    """Load match_captains.csv into a dict keyed by match_id.

    Returns {match_id: {team_name: captain_player_id}}.
    """
    path = CAPTAINS_CSV
    if not path.exists():
        return {}
    lookup = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mid = row["match_id"]
            lookup[mid] = {
                normalize_team(row["captain1_team"]): str(row["captain1_id"]),
                normalize_team(row["captain2_team"]): str(row["captain2_id"]),
            }
    return lookup


class CaptainTracker:
    """Track captain win rates and match counts."""

    def __init__(self):
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"wins": 0, "matches": 0}
        )

    def get_state(self, captain_id: str) -> dict:
        s = self._stats[captain_id]
        return {
            "captain_wins": s["wins"],
            "captain_matches": s["matches"],
            "captain_win_rate": s["wins"] / s["matches"] if s["matches"] > 0 else 0.5,
        }

    def update(self, captain_id: str, won: bool):
        self._stats[captain_id]["matches"] += 1
        if won:
            self._stats[captain_id]["wins"] += 1


class DeathBowlingTracker:
    """Track per-bowler death over (16-20) economy from BBB data."""

    def __init__(self):
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"runs": 0, "balls": 0}
        )

    def update_from_match(self, match_id: str, all_xis: dict):
        """Scan BBB for death overs and accumulate per-bowler stats."""
        balls = _load_bbb(str(match_id))
        if not balls:
            return
        for ball in balls:
            over_str = ball.get("over", "0")
            try:
                ov = int(str(over_str).split(".")[0])
            except ValueError:
                continue
            if ov < 16:
                continue
            bowl_id = ball.get("bowler_id", "")
            if not bowl_id:
                continue
            runs = int(ball.get("runs", 0))
            play_type = ball.get("play_type", "")
            self._stats[bowl_id]["runs"] += runs
            if play_type not in ("wide", "no ball"):
                self._stats[bowl_id]["balls"] += 1

    def get_xi_death_quality(self, xi: dict[str, str]) -> dict:
        """Aggregate death bowling economy for bowlers in an XI."""
        economies = []
        for pid in xi:
            s = self._stats[pid]
            if s["balls"] >= 12:
                econ = (s["runs"] / s["balls"]) * 6
                economies.append(econ)
        if not economies:
            return {"death_bowl_avg_econ": 10.0, "death_bowl_specialists": 0,
                    "death_bowl_best_econ": 10.0}
        return {
            "death_bowl_avg_econ": sum(economies) / len(economies),
            "death_bowl_specialists": len(economies),
            "death_bowl_best_econ": min(economies),
        }


def _load_fielding() -> dict[str, float]:
    """Load player_fielding.csv into {player_id: dismissals_per_innings}."""
    path = FIELDING_CSV
    if not path.exists():
        return {}
    lookup = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                lookup[row["player_id"]] = float(row["dismissals_per_innings"])
            except (ValueError, KeyError):
                pass
    return lookup


class FieldingTracker:
    """Time-ordered fielding stats from innings-level data."""

    _MONTH_MAP = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }

    def __init__(self):
        self._data: dict[str, list[tuple[str, int, int]]] = {}
        self._load()

    def _parse_date(self, s: str) -> str:
        parts = s.strip().split()
        if len(parts) < 3:
            return "0000-00-00"
        day = parts[0].rstrip(",")
        mon = self._MONTH_MAP.get(parts[1], 0)
        year = parts[2]
        return f"{year}-{mon:02d}-{int(day):02d}"

    def _load(self):
        if not FIELDING_INNINGS_DIR.exists():
            return
        for path in FIELDING_INNINGS_DIR.glob("*.csv"):
            with open(path) as f:
                for row in csv.DictReader(f):
                    pid = row["player_id"]
                    date = self._parse_date(row.get("start_date", ""))
                    try:
                        dis = int(row.get("dismissals", 0))
                        inns = int(row.get("innings", 1)) if row.get("innings") else 1
                    except ValueError:
                        continue
                    self._data.setdefault(pid, []).append((date, dis, inns))
        for pid in self._data:
            self._data[pid].sort(key=lambda x: x[0])

    def get_dpi(self, player_id: str, before_date: str) -> float:
        entries = self._data.get(player_id, [])
        total_dis = 0
        total_inns = 0
        for date, dis, inns in entries:
            if date >= before_date:
                break
            total_dis += dis
            total_inns += inns
        if total_inns < 5:
            return 0.3
        return total_dis / total_inns


class BatterBowlerTracker:
    """Track batter-vs-bowler matchup stats from BBB data."""

    _NON_BALL = {"wide", "no ball"}
    _NON_BAT_RUNS = {"leg bye", "bye"}
    MIN_BALLS = 6

    def __init__(self):
        self._stats: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"runs": 0, "balls": 0, "dismissals": 0}
        )

    def update_from_match(self, match_id: str):
        balls = _load_bbb(str(match_id))
        if not balls:
            return
        for b in balls:
            bat_id = b.get("batsman_id", "")
            bowl_id = b.get("bowler_id", "")
            if not bat_id or not bowl_id:
                continue
            pt = b.get("play_type", "")
            if pt in self._NON_BALL:
                continue
            key = (bat_id, bowl_id)
            self._stats[key]["balls"] += 1
            if pt not in self._NON_BAT_RUNS:
                self._stats[key]["runs"] += int(b.get("runs", 0))
            if b.get("is_wicket") == "True":
                self._stats[key]["dismissals"] += 1

    def get_matchup_features(self, team1_xi: dict, team2_xi: dict) -> dict:
        t1_sr, t1_dr = [], []
        for bat_id in team1_xi:
            for bowl_id in team2_xi:
                s = self._stats[(bat_id, bowl_id)]
                if s["balls"] >= self.MIN_BALLS:
                    w = s["balls"]
                    t1_sr.append(((s["runs"] / w) * 100, w))
                    t1_dr.append((s["dismissals"] / w, w))

        t2_sr, t2_dr = [], []
        for bat_id in team2_xi:
            for bowl_id in team1_xi:
                s = self._stats[(bat_id, bowl_id)]
                if s["balls"] >= self.MIN_BALLS:
                    w = s["balls"]
                    t2_sr.append(((s["runs"] / w) * 100, w))
                    t2_dr.append((s["dismissals"] / w, w))

        def _wavg(pairs, default):
            if not pairs:
                return default
            tw = sum(w for _, w in pairs)
            return sum(v * w for v, w in pairs) / tw

        t1_bat_sr = _wavg(t1_sr, 120.0)
        t1_bat_dr = _wavg(t1_dr, 0.05)
        t2_bat_sr = _wavg(t2_sr, 120.0)
        t2_bat_dr = _wavg(t2_dr, 0.05)

        return {
            "matchup_t1_bat_sr": round(t1_bat_sr, 2),
            "matchup_t1_dismiss_rate": round(t1_bat_dr, 4),
            "matchup_t2_bat_sr": round(t2_bat_sr, 2),
            "matchup_t2_dismiss_rate": round(t2_bat_dr, 4),
            "matchup_sr_diff": round(t1_bat_sr - t2_bat_sr, 2),
            "matchup_dismiss_diff": round(t2_bat_dr - t1_bat_dr, 4),
            "matchup_t1_pairs": len(t1_sr),
            "matchup_t2_pairs": len(t2_sr),
        }


class CollapseTracker:
    """Track batting collapse frequency per team from BBB wicket clusters."""

    def __init__(self):
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"collapses": 0, "innings": 0}
        )

    def update_from_match(self, match_id: str, team: str, xi_ids: set[str]):
        balls = _load_bbb(str(match_id))
        if not balls:
            return

        for innings_num in ("1", "2"):
            inn_balls = [b for b in balls if b.get("innings") == innings_num]
            if not inn_balls:
                continue
            bat_ids_in_inn = set(b.get("batsman_id", "") for b in inn_balls)
            if not bat_ids_in_inn & xi_ids:
                continue

            self._stats[team]["innings"] += 1

            wkt_overs = []
            for b in inn_balls:
                if b.get("is_wicket") == "True":
                    try:
                        ov = float(b.get("over", "0"))
                        wkt_overs.append(ov)
                    except ValueError:
                        pass

            for i in range(len(wkt_overs) - 2):
                span = wkt_overs[i + 2] - wkt_overs[i]
                if span <= 3.0:
                    self._stats[team]["collapses"] += 1
                    break

    def get_state(self, team: str) -> dict:
        s = self._stats[team]
        if s["innings"] == 0:
            return {"collapse_rate": 0.2}
        return {"collapse_rate": s["collapses"] / s["innings"]}


_CITY_COORDS = {
    "Mumbai": (19.08, 72.88), "Bengaluru": (12.97, 77.59),
    "Kolkata": (22.57, 88.36), "Delhi": (28.61, 77.23),
    "Chennai": (13.08, 80.27), "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79), "Mohali": (30.70, 76.72),
    "New Chandigarh": (30.70, 76.78), "Pune": (18.52, 73.86),
    "Ahmedabad": (23.02, 72.57), "Lucknow": (26.85, 80.95),
    "Visakhapatnam": (17.69, 83.22), "Rajkot": (22.30, 70.80),
    "Indore": (22.72, 75.86), "Guwahati": (26.14, 91.74),
    "Dharamsala": (32.22, 76.32), "Navi Mumbai": (19.03, 73.03),
    "Raipur": (21.25, 81.63), "Kanpur": (26.45, 80.35),
    "Ranchi": (23.34, 85.31), "Dubai": (25.20, 55.27),
    "Abu Dhabi": (24.45, 54.65), "Sharjah": (25.34, 55.41),
    "Cape Town": (-33.93, 18.42), "Centurion": (-25.86, 28.19),
    "Port Elizabeth": (-33.96, 25.60), "Johannesburg": (-26.20, 28.05),
    "Durban": (-29.86, 31.03),
}


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


class TravelTracker:
    """Track cumulative travel distance and schedule density per team."""

    def __init__(self):
        self._last_city: dict[str, str] = {}
        self._recent_travel: dict[str, list[float]] = defaultdict(list)

    def get_state(self, team: str) -> dict:
        recent = self._recent_travel.get(team, [])
        last5 = recent[-5:] if recent else []
        return {
            "travel_last5_km": sum(last5),
            "travel_avg_km": sum(last5) / len(last5) if last5 else 0,
        }

    def update(self, team: str, city: str):
        prev = self._last_city.get(team)
        if prev and prev != city:
            c1 = _CITY_COORDS.get(prev)
            c2 = _CITY_COORDS.get(city)
            if c1 and c2:
                dist = _haversine_km(c1[0], c1[1], c2[0], c2[1])
                self._recent_travel[team].append(dist)
            else:
                self._recent_travel[team].append(0)
        else:
            self._recent_travel[team].append(0)
        self._last_city[team] = city


class TopOrderTracker:
    """Track top-order batting concentration per team from BBB data."""

    def __init__(self):
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"top3_runs": 0, "total_runs": 0, "innings": 0}
        )

    def update_from_match(self, match_id: str, team: str, xi_ids: set[str]):
        """Compute runs by batting position for a team's innings."""
        balls = _load_bbb(str(match_id))
        if not balls:
            return

        for innings_num in ("1", "2"):
            inn_balls = [b for b in balls if b.get("innings") == innings_num]
            if not inn_balls:
                continue
            bat_ids_in_inn = set(b.get("batsman_id", "") for b in inn_balls)
            if not bat_ids_in_inn & xi_ids:
                continue

            order = []
            seen = set()
            for b in inn_balls:
                bid = b.get("batsman_id", "")
                if bid and bid not in seen:
                    seen.add(bid)
                    order.append(bid)

            top3_ids = set(order[:3])
            top3_runs = 0
            total_runs = 0
            for b in inn_balls:
                bid = b.get("batsman_id", "")
                if bid not in xi_ids:
                    continue
                runs = int(b.get("runs", 0))
                total_runs += runs
                if bid in top3_ids:
                    top3_runs += runs

            if total_runs > 0:
                self._stats[team]["top3_runs"] += top3_runs
                self._stats[team]["total_runs"] += total_runs
                self._stats[team]["innings"] += 1

    def get_state(self, team: str) -> dict:
        s = self._stats[team]
        if s["total_runs"] == 0:
            return {"top3_run_share": 0.5, "top3_innings": 0}
        return {
            "top3_run_share": s["top3_runs"] / s["total_runs"],
            "top3_innings": s["innings"],
        }


class PointsTable:
    """Track IPL points table state within a season."""

    def __init__(self):
        self._teams: dict[str, dict] = defaultdict(
            lambda: {"pts": 0, "played": 0, "won": 0,
                      "runs_for": 0, "overs_for": 0.0,
                      "runs_against": 0, "overs_against": 0.0}
        )
        self._season = None

    def _reset(self, season):
        self._teams.clear()
        self._season = season

    def _overs_from_bbb(self, match_id, innings_num):
        """Get runs scored and overs bowled for an innings from BBB data."""
        balls = _load_bbb(str(match_id))
        if not balls:
            return 0, 0.0
        inn_balls = [b for b in balls if b.get("innings") == str(innings_num)]
        if not inn_balls:
            return 0, 0.0
        last = inn_balls[-1]
        runs = int(last.get("total_runs", 0))
        wkts = int(last.get("total_wickets", 0))
        over_str = last.get("over", "0")
        try:
            parts = str(over_str).split(".")
            whole = int(parts[0])
            partial = int(parts[1]) if len(parts) > 1 else 0
            overs = whole + partial / 6.0
            if partial == 6:
                overs = whole + 1.0
        except (ValueError, IndexError):
            overs = 0.0
        if wkts >= 10:
            overs = min(overs, 20.0)
        return runs, overs

    def get_state(self, team, season):
        """Get pre-match table state for a team."""
        if season != self._season:
            return {"table_pts": 0, "table_pos": 5, "table_nrr": 0.0,
                    "table_played": 0, "table_win_pct": 0.5}
        s = self._teams[team]
        nrr = 0.0
        if s["overs_for"] > 0 and s["overs_against"] > 0:
            nrr = (s["runs_for"] / s["overs_for"]) - (s["runs_against"] / s["overs_against"])
        win_pct = s["won"] / s["played"] if s["played"] > 0 else 0.5

        sorted_teams = sorted(
            self._teams.keys(),
            key=lambda t: (-self._teams[t]["pts"],
                           -((self._teams[t]["runs_for"] / max(self._teams[t]["overs_for"], 1))
                             - (self._teams[t]["runs_against"] / max(self._teams[t]["overs_against"], 1))))
        )
        pos = sorted_teams.index(team) + 1 if team in sorted_teams else 5

        return {"table_pts": s["pts"], "table_pos": pos, "table_nrr": round(nrr, 3),
                "table_played": s["played"], "table_win_pct": round(win_pct, 3)}

    def update(self, match):
        """Update table after a match result."""
        season = match["season"]
        if season != self._season:
            self._reset(season)

        t1, t2 = match["team1"], match["team2"]
        mid = match["match_id"]
        winner = match.get("winner", "")
        toss_winner = match.get("toss_winner", "")
        toss_decision = match.get("toss_decision", "")

        if toss_decision == "bat":
            bat_first = toss_winner
        else:
            bat_first = t1 if toss_winner == t2 else t2
        bat_second = t2 if bat_first == t1 else t1

        inn1_runs, inn1_overs = self._overs_from_bbb(mid, 1)
        inn2_runs, inn2_overs = self._overs_from_bbb(mid, 2)

        if inn1_overs > 0 and inn2_overs > 0:
            self._teams[bat_first]["runs_for"] += inn1_runs
            self._teams[bat_first]["overs_for"] += inn1_overs
            self._teams[bat_first]["runs_against"] += inn2_runs
            self._teams[bat_first]["overs_against"] += inn2_overs

            self._teams[bat_second]["runs_for"] += inn2_runs
            self._teams[bat_second]["overs_for"] += inn2_overs
            self._teams[bat_second]["runs_against"] += inn1_runs
            self._teams[bat_second]["overs_against"] += inn1_overs

        self._teams[t1]["played"] += 1
        self._teams[t2]["played"] += 1
        if winner == t1:
            self._teams[t1]["pts"] += 2
            self._teams[t1]["won"] += 1
        elif winner == t2:
            self._teams[t2]["pts"] += 2
            self._teams[t2]["won"] += 1


_MARGIN_RUNS_RE = re.compile(r"won by (\d+) run")
_MARGIN_WKTS_RE = re.compile(r"won by (\d+) wicket")
_BALLS_REM_RE = re.compile(r"(\d+) balls? remaining")


def _parse_margin(result_text: str) -> float:
    """Parse a dominance score from result text (higher = more dominant win)."""
    m = _MARGIN_RUNS_RE.search(result_text)
    if m:
        return float(m.group(1))
    m = _MARGIN_WKTS_RE.search(result_text)
    if m:
        wkts = int(m.group(1))
        bm = _BALLS_REM_RE.search(result_text)
        balls_left = int(bm.group(1)) if bm else 0
        return wkts * 10.0 + balls_left
    return 0.0


def build():
    print("Loading matches...", flush=True)
    matches = load_matches()
    matches.sort(key=lambda m: m["date"])
    print(f"  {len(matches)} valid matches", flush=True)

    print("Extracting playing XIs...", flush=True)
    all_xis = extract_all_xis(matches)
    print(f"  XIs for {len(all_xis)} matches", flush=True)

    print("Loading weather data...", flush=True)
    weather_lookup = _load_weather()
    print(f"  Weather for {len(weather_lookup)} matches", flush=True)

    print("Loading captain data...", flush=True)
    captain_lookup = _load_captains()
    print(f"  Captains for {len(captain_lookup)} matches", flush=True)

    print("Loading fielding data...", flush=True)
    fielding_lookup = _load_fielding()
    fielding_tracker = FieldingTracker()
    print(f"  Fielding career for {len(fielding_lookup)} players, innings for {len(fielding_tracker._data)} players", flush=True)

    print("Building features (strict time ordering)...", flush=True)
    rows = []
    all_feature_keys = None
    elo_state: dict[str, float] = {}
    prev_season = None

    # State trackers for new features
    team_prev_xi: dict[str, set[str]] = {}
    team_margins: dict[tuple[str, int], list[float]] = defaultdict(list)
    points_table = PointsTable()
    captain_tracker = CaptainTracker()
    death_tracker = DeathBowlingTracker()
    toporder_tracker = TopOrderTracker()
    collapse_tracker = CollapseTracker()
    travel_tracker = TravelTracker()
    bvb_tracker = BatterBowlerTracker()

    for i, match in enumerate(matches):
        prior = matches[:i]

        feats = build_match_features(match, prior, all_xis, elo_state)

        mid = match["match_id"]
        t1, t2 = match["team1"], match["team2"]
        season = match["season"]

        # --- XI squad churn: players changed vs previous match ---
        xi_data = all_xis.get(mid)
        t1_changes, t2_changes = 0, 0
        if xi_data:
            t1_xi = set(xi_data["team1_xi"].keys())
            t2_xi = set(xi_data["team2_xi"].keys())
            t1_prev = team_prev_xi.get(t1, set())
            t2_prev = team_prev_xi.get(t2, set())
            if t1_prev:
                t1_changes = len(t1_xi - t1_prev)
            if t2_prev:
                t2_changes = len(t2_xi - t2_prev)
            team_prev_xi[t1] = t1_xi
            team_prev_xi[t2] = t2_xi

        feats["t1_xi_changes"] = t1_changes
        feats["t2_xi_changes"] = t2_changes

        # --- Win margin strength: avg margin in current season ---
        t1_season_margins = team_margins.get((t1, season), [])
        t2_season_margins = team_margins.get((t2, season), [])
        feats["t1_avg_win_margin"] = (
            sum(t1_season_margins[-10:]) / len(t1_season_margins[-10:])
            if t1_season_margins else 0.0
        )
        feats["t2_avg_win_margin"] = (
            sum(t2_season_margins[-10:]) / len(t2_season_margins[-10:])
            if t2_season_margins else 0.0
        )

        # Update margin state after recording pre-match features
        margin = _parse_margin(match.get("result", ""))
        winner = match.get("winner", "")
        if winner == t1:
            team_margins[(t1, season)].append(margin)
            team_margins[(t2, season)].append(-margin)
        elif winner == t2:
            team_margins[(t1, season)].append(-margin)
            team_margins[(t2, season)].append(margin)

        # --- Captain win rate / experience ---
        cap_data = captain_lookup.get(str(mid), {})
        t1_cap_id = cap_data.get(t1, "")
        t2_cap_id = cap_data.get(t2, "")
        if t1_cap_id:
            t1_cap = captain_tracker.get_state(t1_cap_id)
            feats["t1_captain_win_rate"] = t1_cap["captain_win_rate"]
            feats["t1_captain_matches"] = t1_cap["captain_matches"]
        else:
            feats["t1_captain_win_rate"] = 0.5
            feats["t1_captain_matches"] = 0
        if t2_cap_id:
            t2_cap = captain_tracker.get_state(t2_cap_id)
            feats["t2_captain_win_rate"] = t2_cap["captain_win_rate"]
            feats["t2_captain_matches"] = t2_cap["captain_matches"]
        else:
            feats["t2_captain_win_rate"] = 0.5
            feats["t2_captain_matches"] = 0

        # --- Death bowling quality ---
        if xi_data:
            t1_death = death_tracker.get_xi_death_quality(xi_data["team1_xi"])
            t2_death = death_tracker.get_xi_death_quality(xi_data["team2_xi"])
            for k, v in t1_death.items():
                feats[f"t1_{k}"] = v
            for k, v in t2_death.items():
                feats[f"t2_{k}"] = v
        else:
            for prefix in ("t1_", "t2_"):
                feats[f"{prefix}death_bowl_avg_econ"] = 10.0
                feats[f"{prefix}death_bowl_specialists"] = 0
                feats[f"{prefix}death_bowl_best_econ"] = 10.0

        # --- Top-order batting concentration ---
        t1_top = toporder_tracker.get_state(t1)
        t2_top = toporder_tracker.get_state(t2)
        feats["t1_top3_run_share"] = t1_top["top3_run_share"]
        feats["t2_top3_run_share"] = t2_top["top3_run_share"]

        # --- Collapse tendency ---
        t1_collapse = collapse_tracker.get_state(t1)
        t2_collapse = collapse_tracker.get_state(t2)
        feats["t1_collapse_rate"] = t1_collapse["collapse_rate"]
        feats["t2_collapse_rate"] = t2_collapse["collapse_rate"]

        # --- Fielding quality (time-ordered dismissals per innings) ---
        match_date = match.get("date", "")
        if xi_data and match_date:
            t1_dpi = [fielding_tracker.get_dpi(pid, match_date) for pid in xi_data["team1_xi"]]
            t2_dpi = [fielding_tracker.get_dpi(pid, match_date) for pid in xi_data["team2_xi"]]
            feats["t1_fielding_dpi"] = sum(t1_dpi) / len(t1_dpi) if t1_dpi else 0.3
            feats["t2_fielding_dpi"] = sum(t2_dpi) / len(t2_dpi) if t2_dpi else 0.3
            feats["t1_fielding_total"] = sum(t1_dpi)
            feats["t2_fielding_total"] = sum(t2_dpi)
        else:
            feats["t1_fielding_dpi"] = 0.3
            feats["t2_fielding_dpi"] = 0.3
            feats["t1_fielding_total"] = 3.3
            feats["t2_fielding_total"] = 3.3

        # --- Batter vs bowler matchups ---
        if xi_data:
            bvb = bvb_tracker.get_matchup_features(
                xi_data["team1_xi"], xi_data["team2_xi"]
            )
            feats.update(bvb)
        else:
            feats["matchup_t1_bat_sr"] = 120.0
            feats["matchup_t1_dismiss_rate"] = 0.05
            feats["matchup_t2_bat_sr"] = 120.0
            feats["matchup_t2_dismiss_rate"] = 0.05
            feats["matchup_sr_diff"] = 0.0
            feats["matchup_dismiss_diff"] = 0.0
            feats["matchup_t1_pairs"] = 0
            feats["matchup_t2_pairs"] = 0

        # --- Travel / schedule congestion ---
        t1_travel = travel_tracker.get_state(t1)
        t2_travel = travel_tracker.get_state(t2)
        feats["t1_travel_last5_km"] = t1_travel["travel_last5_km"]
        feats["t2_travel_last5_km"] = t2_travel["travel_last5_km"]
        feats["t1_travel_avg_km"] = t1_travel["travel_avg_km"]
        feats["t2_travel_avg_km"] = t2_travel["travel_avg_km"]

        # --- Points table position and NRR ---
        t1_table = points_table.get_state(t1, season)
        t2_table = points_table.get_state(t2, season)
        for k, v in t1_table.items():
            feats[f"t1_{k}"] = v
        for k, v in t2_table.items():
            feats[f"t2_{k}"] = v
        feats["diff_table_pts"] = t1_table["table_pts"] - t2_table["table_pts"]
        feats["diff_table_pos"] = t1_table["table_pos"] - t2_table["table_pos"]
        feats["diff_table_nrr"] = t1_table["table_nrr"] - t2_table["table_nrr"]
        feats["diff_table_win_pct"] = t1_table["table_win_pct"] - t2_table["table_win_pct"]

        # --- Weather and time features ---
        wdata = weather_lookup.get(str(mid), {})
        feats["is_night"] = 1 if wdata.get("floodlit") == "night" else 0
        feats["is_day_match"] = 1 if wdata.get("floodlit") in ("day", "daynight") else 0
        feats["temperature"] = float(wdata.get("temperature", 0))
        feats["humidity"] = float(wdata.get("humidity", 0))
        feats["dew_point"] = float(wdata.get("dew_point_approx", 0))
        feats["precipitation"] = float(wdata.get("precipitation", 0))
        feats["wind_speed"] = float(wdata.get("wind_speed", 0))
        feats["cloud_cover"] = float(wdata.get("cloud_cover", 0))

        # Update state trackers after recording pre-match features
        points_table.update(match)
        death_tracker.update_from_match(mid, all_xis)
        bvb_tracker.update_from_match(mid)
        if xi_data:
            toporder_tracker.update_from_match(mid, t1, set(xi_data["team1_xi"].keys()))
            toporder_tracker.update_from_match(mid, t2, set(xi_data["team2_xi"].keys()))
            collapse_tracker.update_from_match(mid, t1, set(xi_data["team1_xi"].keys()))
            collapse_tracker.update_from_match(mid, t2, set(xi_data["team2_xi"].keys()))
        city = match.get("city", "")
        travel_tracker.update(t1, city)
        travel_tracker.update(t2, city)
        if t1_cap_id:
            captain_tracker.update(t1_cap_id, winner == t1)
        if t2_cap_id:
            captain_tracker.update(t2_cap_id, winner == t2)

        elo_state = update_elo(elo_state, match, prev_season=prev_season)
        prev_season = match.get("season")
        feats["match_id"] = match["match_id"]
        feats["date"] = match["date"]
        feats["season"] = match["season"]
        feats["team1"] = match["team1"]
        feats["team2"] = match["team2"]
        feats["label"] = match["label"]

        rows.append(feats)

        if all_feature_keys is None:
            all_feature_keys = list(feats.keys())

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(matches)}", flush=True)

    print(f"  Done. {len(rows)} rows, {len(all_feature_keys)} columns", flush=True)

    # Fill missing keys with 0
    for row in rows:
        for k in all_feature_keys:
            row.setdefault(k, 0)

    print(f"Writing {DATASET_PATH}...", flush=True)
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_feature_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved: {len(rows)} rows x {len(all_feature_keys)} cols", flush=True)
    return rows, all_feature_keys


if __name__ == "__main__":
    t0 = time.time()
    build()
    print(f"Elapsed: {time.time() - t0:.0f}s")
