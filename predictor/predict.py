"""Inference CLI: accept XIs + toss + venue, compute features, return probabilities."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from predictor.normalize import load_matches, normalize_team
from predictor.features import build_match_features, _player_batting_form, _player_bowling_form
from predictor.playing_xi import extract_all_xis

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_model():
    with open(MODELS_DIR / "bundle.json") as f:
        bundle = json.load(f)
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return model, bundle


def resolve_player_ids(names_or_ids: list[str], all_xis) -> dict[str, str]:
    name_to_id = {}
    id_to_name = {}
    for xi_data in all_xis.values():
        for xi in (xi_data["team1_xi"], xi_data["team2_xi"]):
            for pid, pname in xi.items():
                name_to_id[pname.lower()] = pid
                id_to_name[pid] = pname

    result = {}
    for entry in names_or_ids:
        entry = entry.strip()
        if entry in id_to_name:
            result[entry] = id_to_name[entry]
        elif entry.lower() in name_to_id:
            pid = name_to_id[entry.lower()]
            result[pid] = id_to_name[pid]
        else:
            for known_name, pid in name_to_id.items():
                if entry.lower() in known_name:
                    result[pid] = id_to_name[pid]
                    break
            else:
                print(f"  Warning: could not resolve player '{entry}'", file=sys.stderr)
    return result


def _pick_best_impact(candidates: dict[str, str], before_date: str) -> str | None:
    """Pick the strongest impact player candidate by composite score."""
    best_pid, best_score = None, -1.0
    for pid in candidates:
        bf = _player_batting_form(pid, before_date)
        wf = _player_bowling_form(pid, before_date)
        score = (
            bf["bat_runs_avg"]
            + (bf["bat_sr_avg"] / 100) * 10
            + wf["bowl_wkts_avg"] * 15
            + bf["bat_experience"]
            + wf["bowl_experience"]
        )
        if score > best_score:
            best_score = score
            best_pid = pid
    return best_pid


def predict(
    team1: str,
    team2: str,
    venue: str,
    city: str,
    toss_winner: str,
    toss_decision: str,
    team1_xi: list[str],
    team2_xi: list[str],
    team1_impact: list[str] | None = None,
    team2_impact: list[str] | None = None,
) -> dict:
    team1 = normalize_team(team1)
    team2 = normalize_team(team2)
    toss_winner = normalize_team(toss_winner)

    all_matches = load_matches()
    all_matches.sort(key=lambda m: m["date"])
    all_xis = extract_all_xis(all_matches)

    t1_xi = resolve_player_ids(team1_xi, all_xis)
    t2_xi = resolve_player_ids(team2_xi, all_xis)

    before_date = all_matches[-1]["date"]

    for xi, impact in ((t1_xi, team1_impact), (t2_xi, team2_impact)):
        if not impact:
            continue
        resolved = resolve_player_ids(impact, all_xis)
        already = set(xi.keys())
        candidates = {pid: name for pid, name in resolved.items() if pid not in already}
        if not candidates:
            continue
        best = _pick_best_impact(candidates, before_date)
        if best:
            xi[best] = candidates[best]

    match = {
        "match_id": "prediction",
        "date": before_date,
        "season": int(all_matches[-1]["season"]),
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": "",
        "venue": venue,
        "city": city,
    }

    all_xis["prediction"] = {"team1_xi": t1_xi, "team2_xi": t2_xi}
    feats = build_match_features(match, all_matches, all_xis)

    model, bundle = load_model()
    feature_cols = bundle["feature_cols"]

    X = np.array([[feats.get(c, 0) for c in feature_cols]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    prob = model.predict_proba(X)[0, 1]

    return {
        "team1": team1,
        "team2": team2,
        "team1_win_prob": round(float(prob), 4),
        "team2_win_prob": round(1 - float(prob), 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IPL Match Winner Prediction")
    parser.add_argument("--team1", required=True)
    parser.add_argument("--team2", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--toss-winner", required=True)
    parser.add_argument("--toss-decision", required=True, choices=["bat", "field"])
    parser.add_argument("--team1-xi", required=True, nargs="+",
                        help="Player names or IDs for team1")
    parser.add_argument("--team2-xi", required=True, nargs="+",
                        help="Player names or IDs for team2")
    args = parser.parse_args()

    result = predict(
        team1=args.team1,
        team2=args.team2,
        venue=args.venue,
        city=args.city,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
        team1_xi=args.team1_xi,
        team2_xi=args.team2_xi,
    )

    print(f"\n{'='*50}")
    print(f"  {result['team1']} vs {result['team2']}")
    print(f"{'='*50}")
    print(f"  {result['team1']}: {result['team1_win_prob']*100:.1f}%")
    print(f"  {result['team2']}: {result['team2_win_prob']*100:.1f}%")


if __name__ == "__main__":
    main()
