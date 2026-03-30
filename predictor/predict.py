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


def _ensure_player_data(pid: str, name: str):
    """Fetch T20 career innings for a player if not already cached."""
    import re
    player_dir = ROOT / "data" / "player_innings"
    pattern = f"*_{pid}.csv"
    if list(player_dir.glob(pattern)):
        return
    try:
        from cricdata import CricinfoClient
        ci = CricinfoClient()
        bat = ci.player_innings(pid, fmt="t20", stat_type="batting")
        bowl = ci.player_innings(pid, fmt="t20", stat_type="bowling")
    except Exception as e:
        print(f"  Warning: could not fetch data for {name} ({pid}): {e}", file=sys.stderr)
        return

    bat_inn = bat.get("innings", [])
    bowl_inn = bowl.get("innings", [])
    rows = []
    for i, bi in enumerate(bat_inn):
        bw = bowl_inn[i] if i < len(bowl_inn) else {}
        rows.append({
            "player_id": pid, "player_name": name,
            "start_date": bi.get("Start Date", ""),
            "opposition": bi.get("Opposition", ""),
            "ground": bi.get("Ground", ""),
            "bat_innings": bi.get("Inns", ""),
            "bat_position": bi.get("Pos", ""),
            "bat_runs": bi.get("Runs", ""),
            "bat_mins": bi.get("Mins", ""),
            "bat_bf": bi.get("BF", ""),
            "bat_4s": bi.get("4s", ""),
            "bat_6s": bi.get("6s", ""),
            "bat_sr": bi.get("SR", ""),
            "bat_dismissal": bi.get("Dismissal", ""),
            "bowl_innings": bw.get("Inns", ""),
            "bowl_overs": bw.get("Overs", ""),
            "bowl_maidens": bw.get("Mdns", ""),
            "bowl_runs": bw.get("Runs", ""),
            "bowl_wickets": bw.get("Wkts", ""),
            "bowl_economy": bw.get("Econ", ""),
        })
    if rows:
        import csv
        safe = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_") + f"_{pid}"
        outpath = player_dir / f"{safe}.csv"
        fields = list(rows[0].keys())
        with open(outpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  Fetched T20 data for {name} ({pid}): {len(rows)} innings", file=sys.stderr)


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
    squad_ids: dict | None = None,
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

    for team, xi, impact in (
        (team1, t1_xi, team1_impact),
        (team2, t2_xi, team2_impact),
    ):
        if not impact:
            continue
        resolved = resolve_player_ids(impact, all_xis)
        # For unresolved names, try squad_ids from the scorecard
        if squad_ids and len(resolved) < len(impact):
            team_squad = squad_ids.get(team, {})
            name_to_sid = {v.lower(): k for k, v in team_squad.items()}
            for name in impact:
                name_l = name.strip().lower()
                if name_l not in {v.lower() for v in resolved.values()}:
                    sid = name_to_sid.get(name_l)
                    if not sid:
                        for sq_name, sq_id in name_to_sid.items():
                            if name_l in sq_name:
                                sid = sq_id
                                break
                    if sid:
                        _ensure_player_data(sid, name.strip())
                        resolved[sid] = name.strip()
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
    base_models_dict = model["base_models"]
    meta_model = model["meta_model"]
    feat_sets = bundle["base_models"]

    base_preds = []
    for name, feats_list in feat_sets.items():
        X = np.array([[feats.get(c, 0) for c in feats_list]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        base_preds.append(base_models_dict[name].predict_proba(X)[0, 1])

    meta_X = np.array([base_preds])
    prob = meta_model.predict_proba(meta_X)[0, 1]

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
