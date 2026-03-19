"""Train IPL match predictor: stacked ensemble of specialist LR models.

Key design decisions:
  - 4 base models: baseline (9 feats), composition (5), form (5), phase (4).
  - Each base model is a scaled LR (C=0.1) trained with exponential sample weights.
  - Meta-learner: LR (C=0.1) trained on out-of-fold base model probabilities.
  - Train on 2018+ excluding COVID neutral-venue seasons (2020, 2021).
  - Exponential sample weights (half-life 2y) with 3x boost for impact-player-era (2023+).
  - Validated 2023+2024+2025 holdout: avg 78.6% (56/73, 55/71, 58/71).
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "dataset.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

META_COLS = ["match_id", "date", "season", "team1", "team2", "label"]

SELECTED_FEATURES = [
    "diff_win_streak",
    "t2_spin_count",
    "t2_pace_count",
    "diff_season_matches",
    "h2h_t1_win_rate",
    "diff_left_bowl_count",
    "t2_low_bat_sr",
    "diff_middle_bowl_extras_per_match",
    "diff_table_nrr",
]

CLUSTER_FEATURES = {
    "composition": [
        "t2_specialist_bowler", "t2_bowling_ar", "t2_specialist_bat",
        "t2_allrounder", "t2_wk_bat",
    ],
    "form": [
        "t2_win_streak", "diff_season_matches", "diff_matches_played",
        "t1_win_rate_last10", "t2_loss_streak",
    ],
    "phase": [
        "t1_middle_bat_dot_pct", "t2_powerplay_bowl_rr",
        "diff_death_bowl_bound_pct", "diff_death_bowl_extras_per_match",
    ],
}

COVID_SEASONS = {2020, 2021}
MIN_TRAIN_SEASON = 2018
IMPACT_PLAYER_START = 2023
HALF_LIFE = 2.0
IMPACT_ERA_BOOST = 3.0
LR_C = 0.1
META_C = 0.1


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["season"] = df["season"].astype(int)
    return df


def compute_sample_weights(seasons: np.ndarray, holdout_year: int) -> np.ndarray:
    """Most recent year gets highest weight. Impact-player era gets 3x boost."""
    max_train_year = holdout_year - 1
    decay = np.log(2) / HALF_LIFE
    weights = np.exp(-decay * (max_train_year - seasons))
    weights[seasons >= IMPACT_PLAYER_START] *= IMPACT_ERA_BOOST
    return weights


def make_train_mask(df: pd.DataFrame, holdout_year: int) -> pd.Series:
    return (
        (df["season"] >= MIN_TRAIN_SEASON)
        & (df["season"] < holdout_year)
        & ~df["season"].isin(COVID_SEASONS)
    )


def train_base_model(X_train, y_train, w_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=LR_C, max_iter=5000)),
    ])
    pipe.fit(X_train, y_train, lr__sample_weight=w_train)
    return pipe


def _all_feature_sets():
    """Return ordered list of (name, feature_list) for all base models."""
    sets = [("baseline", SELECTED_FEATURES)]
    for name, feats in CLUSTER_FEATURES.items():
        sets.append((name, feats))
    return sets


def _generate_oof(df, train_df, feat_sets, holdout_year):
    """Generate out-of-fold predictions for each base model via leave-one-season-out."""
    train_seasons = sorted(train_df["season"].unique())
    y_train = train_df["label"].values
    n_models = len(feat_sets)
    oof = np.zeros((len(train_df), n_models))

    for mi, (_, feats) in enumerate(feat_sets):
        for val_season in train_seasons:
            tr_idx = train_df["season"] != val_season
            va_idx = train_df["season"] == val_season
            X_tr = train_df.loc[train_df.index[tr_idx], feats].fillna(0).values
            y_tr = y_train[tr_idx]
            X_va = train_df.loc[train_df.index[va_idx], feats].fillna(0).values
            seasons_tr = train_df.loc[train_df.index[tr_idx], "season"].values
            decay = np.log(2) / HALF_LIFE
            w_tr = np.exp(-decay * (val_season - 1 - seasons_tr))
            pipe = Pipeline([
                ("s", StandardScaler()),
                ("lr", LogisticRegression(C=LR_C, max_iter=5000)),
            ])
            pipe.fit(X_tr, y_tr, lr__sample_weight=w_tr)
            oof[va_idx, mi] = pipe.predict_proba(X_va)[:, 1]
    return oof


def evaluate_holdout(df, holdout_year, verbose=True):
    """Train stacked ensemble, evaluate on holdout."""
    mask = make_train_mask(df, holdout_year)
    train_df = df[mask]
    holdout = df[df["season"] == holdout_year]

    if len(holdout) == 0:
        return None

    feat_sets = _all_feature_sets()
    y_train = train_df["label"].values
    y = holdout["label"].values
    w = compute_sample_weights(train_df["season"].values, holdout_year)

    oof = _generate_oof(df, train_df, feat_sets, holdout_year)

    hold_preds = np.zeros((len(holdout), len(feat_sets)))
    for mi, (_, feats) in enumerate(feat_sets):
        pipe = train_base_model(
            train_df[feats].fillna(0).values, y_train, w
        )
        hold_preds[:, mi] = pipe.predict_proba(holdout[feats].fillna(0).values)[:, 1]

    meta = Pipeline([
        ("s", StandardScaler()),
        ("lr", LogisticRegression(C=META_C, max_iter=5000)),
    ])
    meta.fit(oof, y_train, lr__sample_weight=w)
    probs = meta.predict_proba(hold_preds)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    auc = roc_auc_score(y, probs)
    correct = int(acc * len(y))

    seasons_in_train = sorted(train_df["season"].unique())
    weight_summary = {
        int(s): f"{w[train_df['season'].values == s].mean():.2f}"
        for s in seasons_in_train
    }

    print(f"\n{'='*60}", flush=True)
    print(f"  {holdout_year} HOLDOUT", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Train: {len(train_df)} matches, seasons {seasons_in_train}", flush=True)
    print(f"  Weights per season: {weight_summary}", flush=True)
    print(f"  Holdout: {len(holdout)} matches", flush=True)
    print(f"  Accuracy:    {acc:.4f} ({correct}/{len(holdout)})", flush=True)
    print(f"  Log loss:    {ll:.4f}", flush=True)
    print(f"  Brier score: {brier_score_loss(y, probs):.4f}", flush=True)
    print(f"  ROC AUC:     {auc:.4f}", flush=True)

    meta_coefs = meta.named_steps["lr"].coef_[0]
    print(f"  Meta weights:", flush=True)
    for (name, _), coef in zip(feat_sets, meta_coefs):
        print(f"    {coef:+.4f}  {name}", flush=True)

    if verbose:
        for idx, (_, row) in enumerate(holdout.iterrows()):
            p = probs[idx]
            actual = int(row["label"])
            ok = "OK" if (p > 0.5) == bool(actual) else "XX"
            print(f"  {ok} {row['team1']:>25} v {row['team2']:<25} "
                  f"p={p:.3f} actual={actual}", flush=True)
        print(f"  Total: {correct}/{len(holdout)} correct", flush=True)

    return meta, acc


def train():
    print("Loading dataset...", flush=True)
    df = load_dataset()
    all_feats = set(SELECTED_FEATURES)
    for feats in CLUSTER_FEATURES.values():
        all_feats.update(feats)
    for f in all_feats:
        df[f] = df[f].fillna(0).replace([np.inf, -np.inf], 0)

    seasons = sorted(df["season"].unique())
    feat_sets = _all_feature_sets()
    total_feats = sum(len(fs) for _, fs in feat_sets)
    print(f"  {len(df)} rows, {len(feat_sets)} base models ({total_feats} total features), seasons: {seasons}", flush=True)

    for hy in [2023, 2024, 2025]:
        if hy not in seasons:
            continue
        evaluate_holdout(df, hy, verbose=(hy >= 2024))

    latest_season = max(seasons)
    print(f"\nTraining production model (target: {latest_season + 1})...", flush=True)
    prod_mask = (
        (df["season"] >= MIN_TRAIN_SEASON)
        & ~df["season"].isin(COVID_SEASONS)
    )
    prod_df = df[prod_mask]
    w_prod = compute_sample_weights(prod_df["season"].values, latest_season + 1)
    y_prod = prod_df["label"].values

    base_models = {}
    for name, feats in feat_sets:
        base_models[name] = train_base_model(
            prod_df[feats].fillna(0).values, y_prod, w_prod
        )

    oof_prod = _generate_oof(df, prod_df, feat_sets, latest_season + 1)
    meta_model = Pipeline([
        ("s", StandardScaler()),
        ("lr", LogisticRegression(C=META_C, max_iter=5000)),
    ])
    meta_model.fit(oof_prod, y_prod, lr__sample_weight=w_prod)

    meta_coefs = meta_model.named_steps["lr"].coef_[0]
    print("\nMeta-learner weights (production):", flush=True)
    for (name, _), coef in zip(feat_sets, meta_coefs):
        print(f"  {coef:+.4f}  {name}", flush=True)

    bundle = {
        "model_type": "stacked_lr",
        "base_models": {name: feats for name, feats in feat_sets},
        "lr_C": LR_C,
        "meta_C": META_C,
        "min_train_season": MIN_TRAIN_SEASON,
        "covid_seasons": list(COVID_SEASONS),
        "impact_era_boost": IMPACT_ERA_BOOST,
        "half_life": HALF_LIFE,
        "latest_train_season": int(latest_season),
    }
    with open(MODELS_DIR / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump({"base_models": base_models, "meta_model": meta_model}, f)

    print(f"\nModel saved to {MODELS_DIR}/", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"Elapsed: {time.time() - t0:.0f}s")
