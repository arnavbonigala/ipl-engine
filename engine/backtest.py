"""Replay a season backtest and populate engine_state.json for the dashboard.

Usage:
  python -m engine.backtest          # default: 2025
  python -m engine.backtest 2023     # run on 2023
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset.csv"
STATE_FILE = ROOT / "engine_state.json"
EVENTS_CACHE = ROOT / "engine" / f"pm_events_{SEASON}.json"
TRADEABLE_CACHE = ROOT / "engine" / f"tradeable_{SEASON}.json"

CLOB = "https://clob.polymarket.com"

KELLY_FRACTION = 0.5
MAX_BET_FRACTION = 0.25
MAX_ENTRY_PRICE = 0.55
USE_STOP_LOSS = False
STOP_LOSS = 0.15
MIN_CONFIDENCE = 0.60
MIN_EDGE = 0.03
STARTING_BANKROLL = 250.0

MIN_VOLUME = 100
MIN_PRICE_POINTS = 5
MAX_BET_VS_VOLUME = 0.10  # can't bet more than 10% of market volume
SLIPPAGE_BASE = 0.01      # 1 cent baseline slippage
SLIPPAGE_IMPACT = 0.05    # additional slippage proportional to bet/volume

SELECTED_FEATURES = [
    "diff_win_streak", "t2_spin_count", "t2_pace_count",
    "diff_season_matches", "h2h_t1_win_rate", "diff_left_bowl_count",
    "t2_low_bat_sr", "diff_middle_bowl_extras_per_match", "diff_table_nrr",
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
HALF_LIFE = 2.0
IMPACT_ERA_BOOST = 3.0
IMPACT_PLAYER_START = 2023
LR_C = 0.1
META_C = 0.1


def _feat_sets():
    sets = [("baseline", SELECTED_FEATURES)]
    for name, feats in CLUSTER_FEATURES.items():
        sets.append((name, feats))
    return sets


def _train_and_predict(train_df, predict_df, feat_sets):
    """Train stacked ensemble on train_df, return P(t1_wins) for predict_df rows."""
    y = train_df["label"].values
    max_yr = int(train_df["season"].max())
    decay = np.log(2) / HALF_LIFE
    w = np.exp(-decay * (max_yr - train_df["season"].values))
    w[train_df["season"].values >= IMPACT_PLAYER_START] *= IMPACT_ERA_BOOST

    seasons = sorted(train_df["season"].unique())
    oof = np.zeros((len(train_df), len(feat_sets)))
    for mi, (_, feats) in enumerate(feat_sets):
        for vs in seasons:
            tr = train_df["season"] != vs
            va = train_df["season"] == vs
            if tr.sum() == 0 or va.sum() == 0:
                continue
            s_tr = train_df.loc[train_df.index[tr], "season"].values
            w_tr = np.exp(-decay * (vs - 1 - s_tr))
            pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=LR_C, max_iter=5000))])
            pipe.fit(
                train_df.loc[train_df.index[tr], feats].fillna(0).values,
                y[tr], lr__sample_weight=w_tr,
            )
            oof[va, mi] = pipe.predict_proba(
                train_df.loc[train_df.index[va], feats].fillna(0).values
            )[:, 1]

    preds = np.zeros((len(predict_df), len(feat_sets)))
    for mi, (_, feats) in enumerate(feat_sets):
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=LR_C, max_iter=5000))])
        pipe.fit(train_df[feats].fillna(0).values, y, lr__sample_weight=w)
        preds[:, mi] = pipe.predict_proba(predict_df[feats].fillna(0).values)[:, 1]

    meta = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=META_C, max_iter=5000))])
    meta.fit(oof, y, lr__sample_weight=w)
    return meta.predict_proba(preds)[:, 1]


def _curl(url):
    try:
        r = subprocess.run(["curl", "-s", url], capture_output=True, text=True, timeout=15)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def _get_prices(token_id, match_date):
    """Get pre-match price and during-match low/high.

    Matches the original backtest methodology:
    - Fetch ±2 days at fidelity=5 (5-min candles) for maximum granularity
    - Pre-match price = last price before 08:00 UTC on match day
    - Match window = 10:00-22:00 UTC for low/high
    """
    dt = datetime.strptime(match_date, "%Y-%m-%d")
    from datetime import timedelta
    start_ts = int((dt - timedelta(days=2)).timestamp())
    end_ts = int((dt + timedelta(days=2)).timestamp())

    data = _curl(
        f"{CLOB}/prices-history?market={token_id}"
        f"&startTs={start_ts}&endTs={end_ts}&fidelity=5"
    )

    if not data or not data.get("history"):
        return None, None, None, 0

    history = data["history"]
    num_prices = len(history)

    midnight_utc = int(dt.replace(hour=0, minute=0, second=0).timestamp())
    target_pre = midnight_utc + 8 * 3600

    pre_prices = [h for h in history if int(h["t"]) <= target_pre]
    pre_price = float(pre_prices[-1]["p"]) if pre_prices else float(history[0]["p"])

    match_start = midnight_utc + 10 * 3600
    match_end = midnight_utc + 22 * 3600
    match_window = [h for h in history if match_start <= int(h["t"]) <= match_end]

    low = min(float(h["p"]) for h in match_window) if match_window else pre_price
    high = max(float(h["p"]) for h in match_window) if match_window else pre_price

    return pre_price, low, high, num_prices


def run_backtest():
    print("Loading dataset...", flush=True)
    df = pd.read_csv(DATASET)
    df["season"] = df["season"].astype(int)
    all_feats = set(SELECTED_FEATURES)
    for feats in CLUSTER_FEATURES.values():
        all_feats.update(feats)
    for f in all_feats:
        df[f] = df[f].fillna(0).replace([np.inf, -np.inf], 0)

    holdout = df[df["season"] == SEASON].copy().sort_values("date")
    print(f"  {len(holdout)} matches in {SEASON}\n", flush=True)

    with open(EVENTS_CACHE) as f:
        pm_events = json.load(f)
    print(f"Loaded {len(pm_events)} cached Polymarket events", flush=True)

    # === PHASE 1: Fetch all prices (no model needed) ===
    print("Fetching CLOB price history...\n", flush=True)
    price_info = {}
    skip_reasons = {}  # match_id -> reason (for matches without price data)
    for _, row in holdout.iterrows():
        mid = str(row["match_id"])
        date_str = str(row["date"])[:10]
        team1, team2 = row["team1"], row["team2"]

        if mid not in pm_events:
            skip_reasons[mid] = "no Polymarket market found"
            print(f"  {date_str} {team1:>28} v {team2:<28} -- no market", flush=True)
            continue

        pm = pm_events[mid]
        volume = pm.get("volume", 0) or 0

        print(f"  {date_str} {team1:>28} v {team2:<28}", end=" ", flush=True)

        if volume < MIN_VOLUME and SEASON >= 2025:
            skip_reasons[mid] = f"low volume (${volume:.0f})"
            print(f"-- low volume (${volume:.0f})", flush=True)
            continue

        pre_price, low, high, num_prices = _get_prices(pm["t1_token"], date_str)

        if num_prices <= MIN_PRICE_POINTS:
            skip_reasons[mid] = f"insufficient price data ({num_prices} points)"
            print(f"-- insufficient price data ({num_prices} points)", flush=True)
            time.sleep(0.2)
            continue

        if pre_price is None or pre_price < 0.05 or pre_price > 0.95:
            skip_reasons[mid] = f"bad pre-match price ({pre_price})"
            print(f"-- bad price ({pre_price})", flush=True)
            time.sleep(0.2)
            continue

        price_info[mid] = {
            "t1_pre": pre_price,
            "t1_low": low,
            "t1_high": high,
            "volume": volume,
            "t1_token": pm["t1_token"],
            "t2_token": pm["t2_token"],
            "slug": pm.get("slug", ""),
        }
        print(f"OK  price={pre_price:.2f}  low={low}  high={high}  ({num_prices} pts)", flush=True)
        time.sleep(0.15)

    print(f"\n{len(price_info)} matches with price data\n", flush=True)

    # === PHASE 2: Simulate with incremental retraining ===
    print("Simulating with per-day retraining...\n", flush=True)
    feat_sets = _feat_sets()
    train_mask = (
        (df["season"] >= MIN_TRAIN_SEASON)
        & (df["season"] < SEASON)
        & ~df["season"].isin(COVID_SEASONS)
    )
    current_train = df[train_mask].copy()

    dates = sorted(holdout["date"].str[:10].unique())
    bankroll = STARTING_BANKROLL
    history = []
    events = []
    all_tradeable = []
    last_retrain_size = 0

    events.append({
        "ts": f"{dates[0]}T07:00:00Z",
        "type": "discovery",
        "message": f"{SEASON} Season Backtest (incremental retrain) — ${bankroll:.2f}",
    })

    for date in dates:
        day_rows = holdout[holdout["date"].str[:10] == date]

        # Log non-tradeable matches
        for _, row in day_rows.iterrows():
            mid = str(row["match_id"])
            if mid not in price_info and mid in skip_reasons:
                winner = row["team1"] if int(row["label"]) == 1 else row["team2"]
                events.append({
                    "ts": f"{date}T08:00:00Z",
                    "type": "no_market",
                    "message": (
                        f"{row['team1']} vs {row['team2']}: {skip_reasons[mid]} "
                        f"(winner: {winner})"
                    ),
                })
                history.append({
                    "match_date": date,
                    "team1": row["team1"],
                    "team2": row["team2"],
                    "side": "-",
                    "team": "-",
                    "token_id": "",
                    "entry_price": 0,
                    "contracts": 0,
                    "bet_amount": 0,
                    "model_prob": 0,
                    "market_price": 0,
                    "edge": 0,
                    "status": "no_market",
                    "pnl": 0,
                    "current_price": 0,
                    "skip_reason": skip_reasons[mid],
                    "winner": winner,
                })

        tradeable = [r for _, r in day_rows.iterrows() if str(r["match_id"]) in price_info]
        if not tradeable:
            current_train = pd.concat([current_train, day_rows], ignore_index=True)
            continue

        # Retrain only if training data grew since last retrain
        if len(current_train) != last_retrain_size:
            n_season = int((current_train["season"] == SEASON).sum())
            if n_season > 0:
                events.append({
                    "ts": f"{date}T06:00:00Z",
                    "type": "retrain",
                    "message": f"Retrained model — {len(current_train)} rows ({n_season} from {SEASON})",
                })
                print(f"  [RETRAIN] {date}: {len(current_train)} training rows ({n_season} from {SEASON})", flush=True)
            last_retrain_size = len(current_train)

        predict_df = day_rows.loc[[r.name for r in tradeable]]
        probs = _train_and_predict(current_train, predict_df, feat_sets)

        for row, model_t1 in zip(tradeable, probs):
            mid = str(row["match_id"])
            date_str = str(row["date"])[:10]
            pi = price_info[mid]
            label = int(row["label"])
            model_t1 = float(model_t1)
            confidence = max(model_t1, 1 - model_t1)

            # Save all tradeable matches for variant replays
            picks_t1_cache = model_t1 > 0.5
            all_tradeable.append({
                "date": date_str,
                "team1": row["team1"],
                "team2": row["team2"],
                "match_id": mid,
                "label": label,
                "model_t1_prob": round(model_t1, 6),
                "t1_pre": pi["t1_pre"],
                "t1_low": pi["t1_low"],
                "t1_high": pi["t1_high"],
                "volume": pi["volume"],
                "our_entry": pi["t1_pre"] if picks_t1_cache else round(1.0 - pi["t1_pre"], 4),
                "our_prob": round(model_t1 if picks_t1_cache else 1 - model_t1, 6),
                "our_low": pi["t1_low"] if picks_t1_cache else round(1.0 - pi["t1_high"], 4),
                "side": "T1" if picks_t1_cache else "T2",
                "model_correct": (label == 1) if picks_t1_cache else (label == 0),
            })

            # Compute side/entry for all matches (needed for skip logging)
            picks_t1 = model_t1 > 0.5
            if picks_t1:
                entry = pi["t1_pre"]
                our_prob = model_t1
                team = row["team1"]
                side = "T1"
                won = label == 1
                low_during = pi["t1_low"]
            else:
                entry = 1.0 - pi["t1_pre"]
                our_prob = 1 - model_t1
                team = row["team2"]
                side = "T2"
                won = label == 0
                low_during = 1.0 - pi["t1_high"] if pi["t1_high"] else None

            edge = our_prob - entry
            winner = row["team1"] if label == 1 else row["team2"]

            def _skip(reason):
                events.append({
                    "ts": f"{date_str}T08:00:00Z",
                    "type": "skip",
                    "message": (
                        f"{row['team1']} vs {row['team2']}: {reason} | "
                        f"Model: {our_prob:.0%} {team}, Market: {entry:.2f}, "
                        f"Edge: {edge:+.2f} (winner: {winner})"
                    ),
                })
                history.append({
                    "match_date": date_str,
                    "team1": row["team1"],
                    "team2": row["team2"],
                    "side": side,
                    "team": team,
                    "token_id": pi["t1_token"] if picks_t1 else pi["t2_token"],
                    "entry_price": 0,
                    "contracts": 0,
                    "bet_amount": 0,
                    "model_prob": round(our_prob, 4),
                    "market_price": round(entry, 4),
                    "edge": round(edge, 4),
                    "status": "skipped",
                    "pnl": 0,
                    "current_price": 0,
                    "skip_reason": reason,
                    "winner": winner,
                    "model_correct": won,
                })

            if confidence < MIN_CONFIDENCE:
                _skip(f"confidence {confidence:.0%} < {MIN_CONFIDENCE:.0%}")
                continue

            if entry > MAX_ENTRY_PRICE:
                _skip(f"entry {entry:.2f} > {MAX_ENTRY_PRICE}")
                continue

            if edge < MIN_EDGE:
                _skip(f"edge {edge:+.2f} < {MIN_EDGE:.0%}")
                continue

            volume = pi["volume"]
            kelly = min((edge / (1 - entry)) * KELLY_FRACTION, MAX_BET_FRACTION)
            bet = round(bankroll * kelly, 2)
            max_bet = volume * MAX_BET_VS_VOLUME if volume > 0 else bet

            if volume > 0 and bet > max_bet and max_bet > 0:
                bet = round(max_bet, 2)

            impact = SLIPPAGE_BASE + SLIPPAGE_IMPACT * (bet / volume if volume > 0 else 0)
            fill_price = round(min(entry + impact, 0.99), 4)

            contracts = round(bet / fill_price, 2)
            stop_price = round(fill_price - STOP_LOSS, 4)

            stopped = False
            if USE_STOP_LOSS and low_during is not None and low_during <= stop_price:
                exit_price = max(stop_price - SLIPPAGE_BASE, 0.01)
                pnl = round((exit_price - fill_price) * contracts, 2)
                status = "stop_loss"
                stopped = True
            elif won:
                pnl = round((1.0 - fill_price) * contracts, 2)
                status = "won"
            else:
                pnl = round(-fill_price * contracts, 2)
                status = "lost"

            bankroll = round(bankroll + pnl, 2)

            events.append({
                "ts": f"{date_str}T08:30:00Z",
                "type": "signal",
                "message": (
                    f"Edge {edge:+.2f} on {team} "
                    f"(model {our_prob:.0%}, market {entry:.0%}). "
                    f"Kelly {kelly:.0%}, ${bet:.2f}"
                    f" (fill {fill_price:.2f}, slip {impact:.3f})"
                ),
            })
            events.append({
                "ts": f"{date_str}T08:30:05Z",
                "type": "bet",
                "message": (
                    f"Placed: {contracts:.1f} contracts of {team} "
                    f"@ {fill_price:.2f} (${bet:.2f}, vol=${volume:,.0f})"
                ),
            })

            if stopped:
                events.append({
                    "ts": f"{date_str}T12:00:00Z",
                    "type": "stop_loss",
                    "message": (
                        f"Stop-loss triggered for {team} @ {stop_price:.2f} "
                        f"(fill {fill_price:.2f}). P&L: ${pnl:.2f}"
                    ),
                })
            else:
                events.append({
                    "ts": f"{date_str}T16:00:00Z",
                    "type": "settle",
                    "message": f"{team} {'won' if won else 'lost'}. P&L: ${pnl:+.2f}. Bankroll: ${bankroll:.2f}",
                })

            history.append({
                "match_date": date_str,
                "team1": row["team1"],
                "team2": row["team2"],
                "side": side,
                "team": team,
                "token_id": pi["t1_token"] if picks_t1 else pi["t2_token"],
                "entry_price": fill_price,
                "contracts": contracts,
                "bet_amount": bet,
                "model_prob": round(our_prob, 4),
                "market_price": entry,
                "edge": round(edge, 4),
                "status": status,
                "pnl": pnl,
                "current_price": 1.0 if won and not stopped else (stop_price if stopped else 0.0),
            })

            print(
                f"  {date_str} {row['team1']:>28} v {row['team2']:<28} "
                f"{side} mkt={entry:.2f} fill={fill_price:.2f} edge={edge:+.2f} "
                f"bet=${bet:.2f} (cap=${max_bet:.0f}) -> {status:>9} "
                f"pnl=${pnl:+.2f}  bankroll=${bankroll:.2f}",
                flush=True,
            )

        # Add this day's results to training data for next day's model
        current_train = pd.concat([current_train, day_rows], ignore_index=True)

    events.append({
        "ts": f"{dates[-1]}T20:00:00Z",
        "type": "settle",
        "message": f"Season complete. Final bankroll: ${bankroll:.2f}",
    })

    actual_bets = [h for h in history if h["status"] not in ("skipped", "no_market")]
    wins = sum(1 for h in actual_bets if h["pnl"] > 0)
    losses = len(actual_bets) - wins
    total_pnl = sum(h["pnl"] for h in actual_bets)
    print(f"\n{'='*60}", flush=True)
    print(f"  Bets: {len(actual_bets)} ({wins}W / {losses}L)  |  Skipped: {len(history) - len(actual_bets)}", flush=True)
    print(f"  Final bankroll: ${bankroll:.2f}", flush=True)
    print(f"  Total P&L: ${total_pnl:+.2f}", flush=True)
    print(f"  Return: {total_pnl / STARTING_BANKROLL * 100:+.1f}%", flush=True)
    print(f"  Retrains: {int((current_train['season'] == SEASON).sum())} matches absorbed", flush=True)
    print(f"{'='*60}", flush=True)

    state = {
        "bankroll": bankroll,
        "starting_bankroll": STARTING_BANKROLL,
        "wallet_address": f"backtest-{SEASON}",
        "positions": [],
        "history": history,
        "events": events,
        "upcoming": [],
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\nWrote {STATE_FILE}", flush=True)

    with open(TRADEABLE_CACHE, "w") as f:
        json.dump(all_tradeable, f, indent=2)
    print(f"Wrote {TRADEABLE_CACHE} ({len(all_tradeable)} matches)", flush=True)


if __name__ == "__main__":
    run_backtest()
