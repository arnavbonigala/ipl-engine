"""Sweep HALF_LIFE values on A11 walk-forward accuracy for 2026.

Reuses live_inference_sim building blocks. Only runs the 11-starter (A11)
variant since that's our live-inference baseline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predictor.features import build_match_features
from predictor.normalize import load_matches
from predictor.playing_xi import extract_all_xis
from predictor.train import (
    _all_feature_sets, COVID_SEASONS, MIN_TRAIN_SEASON,
)
from scripts.live_inference_sim import (
    DATASET, SEASON, META_COLS,
    build_announced_xi_for_match, build_name_to_bbb_id,
    load_announced_xis_by_match, _train_and_predict_one,
    IMPACT_CACHE,
)


def main():
    impact_cache = json.loads(IMPACT_CACHE.read_text()) if IMPACT_CACHE.exists() else {}

    print("Loading data…", flush=True)
    df = pd.read_csv(DATASET)
    df["season"] = df["season"].astype(int)
    for c in df.columns:
        if c in META_COLS:
            continue
        df[c] = df[c].fillna(0).replace([np.inf, -np.inf], 0)
    matches = load_matches()
    all_xis = extract_all_xis(matches)
    name_to_bbb = build_name_to_bbb_id(all_xis)
    announced = load_announced_xis_by_match()

    m26 = (df[df["season"] == SEASON]
           .sort_values("date")
           .reset_index(drop=True))
    match_meta = {m["match_id"]: m for m in matches}
    feat_sets = _all_feature_sets()
    pre = df[(df["season"] >= MIN_TRAIN_SEASON)
             & (df["season"] < SEASON)
             & ~df["season"].isin(COVID_SEASONS)]

    # Precompute ann11 XI + features for each match once.
    work = []
    for _, row in m26.iterrows():
        mid = str(row["match_id"])
        ann = announced.get(mid)
        meta = match_meta.get(mid)
        bbb_xi = all_xis.get(mid)
        if not ann or not meta or not bbb_xi:
            continue
        impact_used = impact_cache.get(mid, [])
        xi_ann11 = build_announced_xi_for_match(
            row, ann, name_to_bbb, bbb_xi, impact_names_used=impact_used)
        if not xi_ann11 or not xi_ann11["team1_xi"] or not xi_ann11["team2_xi"]:
            continue
        prior_m = [m for m in matches if m.get("date", "") < row["date"]
                   and m.get("season", 0) >= 2008]
        all_xis_11 = dict(all_xis); all_xis_11[mid] = xi_ann11
        feats11 = build_match_features(meta, prior_m, all_xis_11)
        bbb_feat_vec = {c: row[c] for c in df.columns if c not in META_COLS}
        earlier_26 = df[(df["season"] == SEASON) & (df["date"] < row["date"])]
        train_df = pd.concat([pre, earlier_26], ignore_index=True)
        work.append({
            "mid": mid,
            "date": row["date"],
            "label": int(row["label"]),
            "feats11": feats11,
            "bbb_feats": bbb_feat_vec,
            "train_df": train_df,
        })
    print(f"Graded matches: {len(work)}", flush=True)

    from predictor import train as train_mod
    base_boost = train_mod.IMPACT_ERA_BOOST
    for hl in [2.0, 2.5, 3.0]:
        for boost in [2.0, 3.0]:
            train_mod.IMPACT_ERA_BOOST = boost
            a11_ok = 0
            bbb_ok = 0
            for w in work:
                p11 = _train_and_predict_one(
                    w["train_df"], w["feats11"], feat_sets, half_life=hl)
                pbbb = _train_and_predict_one(
                    w["train_df"], w["bbb_feats"], feat_sets, half_life=hl)
                if (p11 > 0.5) == bool(w["label"]): a11_ok += 1
                if (pbbb > 0.5) == bool(w["label"]): bbb_ok += 1
            n = len(work)
            print(f"HL={hl:>4.1f} B={boost:>4.1f}  "
                  f"BBB 2026: {bbb_ok/n:.1%} ({bbb_ok}/{n})  "
                  f"A11 2026: {a11_ok/n:.1%} ({a11_ok}/{n})",
                  flush=True)
    train_mod.IMPACT_ERA_BOOST = base_boost


if __name__ == "__main__":
    main()
