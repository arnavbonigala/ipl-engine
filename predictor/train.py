"""Train IPL match predictor: impact-player-era-optimized scaled Logistic Regression.

Key design decisions:
  - 14 features selected by forward selection optimizing 2023+2024 holdout accuracy.
  - Train on 2018+ excluding COVID neutral-venue seasons (2020, 2021).
  - Exponential sample weights (half-life 2y) with 3x boost for impact-player-era data (2023+).
  - Most recent year always gets the highest weight.
  - Scaled LR (C=0.1) outperforms GBDT on this small dataset.
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
DATASET_PATH = ROOT / "dataset.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

META_COLS = ["match_id", "date", "season", "team1", "team2", "label"]

SELECTED_FEATURES = [
    "t2_frontline_bowlers",
    "t2_xi_not_out_pct",
    "t1_top_bat_avg",
    "t2_powerplay_bowl_dot_pct",
    "h2h_t1_win_rate",
    "t1_xi_bowl_wkts",
    "t1_set_win_rate",
    "diff_frontline_bowlers",
    "diff_xi_not_out_pct",
    "t2_season_chase_win_rate",
    "diff_win_rate_last10",
    "t2_elo",
    "diff_powerplay_bowl_bound_pct",
    "diff_days_rest",
]

COVID_SEASONS = {2020, 2021}
MIN_TRAIN_SEASON = 2018
IMPACT_PLAYER_START = 2023
HALF_LIFE = 2.0
IMPACT_ERA_BOOST = 3.0
LR_C = 0.1


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


def train_model(X_train, y_train, w_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=LR_C, max_iter=5000)),
    ])
    pipe.fit(X_train, y_train, lr__sample_weight=w_train)
    return pipe


def evaluate_holdout(df, feat_cols, holdout_year, verbose=True):
    """Train on all available data before holdout_year, evaluate on holdout."""
    mask = make_train_mask(df, holdout_year)
    train_df = df[mask]
    holdout = df[df["season"] == holdout_year]

    if len(holdout) == 0:
        return None

    w = compute_sample_weights(train_df["season"].values, holdout_year)
    model = train_model(train_df[feat_cols], train_df["label"].values, w)
    probs = model.predict_proba(holdout[feat_cols])[:, 1]
    preds = (probs > 0.5).astype(int)
    y = holdout["label"].values

    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    auc = roc_auc_score(y, probs)
    correct = int(acc * len(y))

    # Show weight distribution
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

    if verbose:
        for idx, (_, row) in enumerate(holdout.iterrows()):
            p = probs[idx]
            actual = int(row["label"])
            ok = "OK" if (p > 0.5) == bool(actual) else "XX"
            print(f"  {ok} {row['team1']:>25} v {row['team2']:<25} "
                  f"p={p:.3f} actual={actual}", flush=True)
        print(f"  Total: {correct}/{len(holdout)} correct", flush=True)

    return model, acc


def train():
    print("Loading dataset...", flush=True)
    df = load_dataset()
    feat_cols = SELECTED_FEATURES
    df[feat_cols] = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)

    seasons = sorted(df["season"].unique())
    print(f"  {len(df)} rows, {len(feat_cols)} features, seasons: {seasons}", flush=True)

    # Evaluate on each impact-player-era holdout season
    for hy in [2023, 2024, 2025]:
        if hy not in seasons:
            continue
        evaluate_holdout(df, feat_cols, hy, verbose=(hy >= 2024))

    # Production model: train on ALL available data, weighted toward most recent
    latest_season = max(seasons)
    print(f"\nTraining production model (target: {latest_season + 1})...", flush=True)
    prod_mask = (
        (df["season"] >= MIN_TRAIN_SEASON)
        & ~df["season"].isin(COVID_SEASONS)
    )
    prod_df = df[prod_mask]
    w_prod = compute_sample_weights(prod_df["season"].values, latest_season + 1)
    prod_model = train_model(prod_df[feat_cols], prod_df["label"].values, w_prod)

    # Feature coefficients from production model
    lr = prod_model.named_steps["lr"]
    print("\nFeature coefficients (production model):", flush=True)
    for name, coef in sorted(zip(feat_cols, lr.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {coef:+.4f}  {name}", flush=True)
    print(f"  Intercept: {lr.intercept_[0]:+.4f}", flush=True)

    # Save
    bundle = {
        "feature_cols": feat_cols,
        "model_type": "scaled_lr",
        "lr_C": LR_C,
        "min_train_season": MIN_TRAIN_SEASON,
        "covid_seasons": list(COVID_SEASONS),
        "impact_era_boost": IMPACT_ERA_BOOST,
        "half_life": HALF_LIFE,
        "latest_train_season": int(latest_season),
    }
    with open(MODELS_DIR / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(prod_model, f)

    print(f"\nModel saved to {MODELS_DIR}/", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"Elapsed: {time.time() - t0:.0f}s")
