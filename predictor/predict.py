"""Inference CLI: accept XIs + toss + venue, compute features, return probabilities."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from predictor.normalize import load_matches, normalize_team
from predictor.features import build_match_features
from predictor.playing_xi import extract_all_xis

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_model():
    with open(MODELS_DIR / "bundle.json") as f:
        bundle = json.load(f)
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    calibrator = None
    cal_cfg = bundle.get("calibrator", {})
    cal_path = cal_cfg.get("path")
    if cal_path:
        p = MODELS_DIR / cal_path
        if p.exists():
            with open(p, "rb") as f:
                calibrator = pickle.load(f)
    return model, bundle, calibrator


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


def _fill_xi_from_squad(
    xi: dict[str, str], raw_names: list[str], team: str, squad_ids: dict | None
):
    if not squad_ids:
        return
    team_squad = squad_ids.get(team, {})
    if not team_squad:
        return
    by_name = {v.lower(): k for k, v in team_squad.items()}
    have_names = {n.lower() for n in xi.values()}
    for name in raw_names:
        key = name.strip().lower()
        if not key or key in have_names:
            continue
        pid = by_name.get(key)
        if not pid:
            for squad_name, squad_pid in by_name.items():
                if key in squad_name:
                    pid = squad_pid
                    break
        if not pid or pid in xi:
            continue
        xi[pid] = name.strip()
        have_names.add(name.strip().lower())


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
    _fill_xi_from_squad(t1_xi, team1_xi, team1, squad_ids)
    _fill_xi_from_squad(t2_xi, team2_xi, team2, squad_ids)

    before_date = all_matches[-1]["date"]

    for pid, name in {**t1_xi, **t2_xi}.items():
        _ensure_player_data(pid, name)

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

    model, bundle, calibrator = load_model()
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
    if calibrator is not None:
        prob = float(calibrator.predict([prob])[0])

    return {
        "team1": team1,
        "team2": team2,
        "team1_win_prob": round(float(prob), 4),
        "team2_win_prob": round(1 - float(prob), 4),
    }


def last_xi_for_team(team: str, matches: list[dict], all_xis: dict) -> dict:
    """Return the most-recent non-empty {player_id: name} XI for ``team``.

    ``matches`` must already be sorted ascending by date. Skips matches whose
    ball-by-ball file has not been ingested yet (stub CSVs leave empty XIs).
    """
    for m in reversed(matches):
        entry = all_xis.get(m["match_id"])
        if not entry:
            continue
        if m["team1"] == team:
            xi = entry.get("team1_xi") or {}
        elif m["team2"] == team:
            xi = entry.get("team2_xi") or {}
        else:
            continue
        if xi:
            return xi
    return {}


def preview_toss_scenarios(
    team1: str,
    team2: str,
    venue: str,
    city: str,
) -> list[dict]:
    """Run ``predict()`` for all four toss-winner / decision combos using each
    team's last-known playing XI. Informational only, never used for betting.
    """
    team1 = normalize_team(team1)
    team2 = normalize_team(team2)

    matches = load_matches()
    matches.sort(key=lambda m: m["date"])
    all_xis = extract_all_xis(matches)

    t1_xi = last_xi_for_team(team1, matches, all_xis)
    t2_xi = last_xi_for_team(team2, matches, all_xis)
    if not t1_xi or not t2_xi:
        return []

    t1_ids = list(t1_xi.keys())
    t2_ids = list(t2_xi.keys())

    out = []
    for toss_winner in (team1, team2):
        for toss_decision in ("bat", "field"):
            r = predict(
                team1=team1, team2=team2,
                venue=venue, city=city,
                toss_winner=toss_winner, toss_decision=toss_decision,
                team1_xi=t1_ids, team2_xi=t2_ids,
            )
            out.append({
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "team1_win_prob": r["team1_win_prob"],
                "team2_win_prob": r["team2_win_prob"],
            })
    return out


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
