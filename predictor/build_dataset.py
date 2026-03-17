"""Build the training dataset with strict time ordering."""

import csv
import sys
import time
from pathlib import Path

from predictor.normalize import load_matches, ROOT
from predictor.playing_xi import extract_all_xis
from predictor.features import build_match_features, update_elo

DATASET_PATH = ROOT / "dataset.csv"


def build():
    print("Loading matches...", flush=True)
    matches = load_matches()
    matches.sort(key=lambda m: m["date"])
    print(f"  {len(matches)} valid matches", flush=True)

    print("Extracting playing XIs...", flush=True)
    all_xis = extract_all_xis(matches)
    print(f"  XIs for {len(all_xis)} matches", flush=True)

    print("Building features (strict time ordering)...", flush=True)
    rows = []
    all_feature_keys = None
    elo_state: dict[str, float] = {}
    prev_season = None

    for i, match in enumerate(matches):
        prior = matches[:i]

        feats = build_match_features(match, prior, all_xis, elo_state)

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
