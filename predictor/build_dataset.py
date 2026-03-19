"""Build the training dataset with strict time ordering."""

import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from predictor.normalize import load_matches, ROOT
from predictor.playing_xi import extract_all_xis
from predictor.features import build_match_features, update_elo, _load_bbb

DATASET_PATH = ROOT / "dataset.csv"
WEATHER_CSV = ROOT / "match_weather.csv"


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

    print("Building features (strict time ordering)...", flush=True)
    rows = []
    all_feature_keys = None
    elo_state: dict[str, float] = {}
    prev_season = None

    # State trackers for new features
    team_prev_xi: dict[str, set[str]] = {}
    team_margins: dict[tuple[str, int], list[float]] = defaultdict(list)
    points_table = PointsTable()

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

        # Update points table after recording pre-match features
        points_table.update(match)

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
