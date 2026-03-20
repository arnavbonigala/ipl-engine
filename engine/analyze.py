"""Post-backtest diagnostics: decomposition, counterfactuals, and calibration."""

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "engine_state.json"
DATASET = ROOT / "data" / "dataset.csv"
TRADEABLE_CACHE = ROOT / "engine" / "tradeable_2025.json"

STOP_LOSS = 0.15
KELLY_FRACTION = 0.5
MAX_BET_FRACTION = 0.25
MAX_BET_VS_VOLUME = 0.10
SLIPPAGE_BASE = 0.01
SLIPPAGE_IMPACT = 0.05
MIN_EDGE = 0.03
STARTING_BANKROLL = 100.0


def _load():
    with open(STATE_FILE) as f:
        state = json.load(f)
    df = pd.read_csv(DATASET)
    df["season"] = df["season"].astype(int)
    history = state["history"]

    holdout = df[df["season"] == 2025].copy()
    label_map = {}
    for _, row in holdout.iterrows():
        key = (str(row["date"])[:10], row["team1"], row["team2"])
        label_map[key] = int(row["label"])

    for h in history:
        key = (h["match_date"], h["team1"], h["team2"])
        label = label_map.get(key)
        if h["side"] == "T1":
            h["model_correct"] = label == 1
        else:
            h["model_correct"] = label == 0
        hold_pnl = (
            (1.0 - h["entry_price"]) * h["contracts"]
            if h["model_correct"]
            else -h["entry_price"] * h["contracts"]
        )
        h["hold_pnl"] = round(hold_pnl, 2)

    return history


def _section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def decompose(history):
    """1. Four-bucket decomposition: prediction x trade outcome."""
    _section("1. PREDICTION vs TRADE MANAGEMENT DECOMPOSITION")

    buckets = {
        "Predicted RIGHT, Trade WON": [],
        "Predicted RIGHT, Trade LOST (stop fought model)": [],
        "Predicted WRONG, Trade WON (stop saved us)": [],
        "Predicted WRONG, Trade LOST": [],
    }

    for h in history:
        correct = h["model_correct"]
        trade_won = h["pnl"] > 0
        if correct and trade_won:
            buckets["Predicted RIGHT, Trade WON"].append(h)
        elif correct and not trade_won:
            buckets["Predicted RIGHT, Trade LOST (stop fought model)"].append(h)
        elif not correct and trade_won:
            buckets["Predicted WRONG, Trade WON (stop saved us)"].append(h)
        else:
            buckets["Predicted WRONG, Trade LOST"].append(h)

    total = len(history)
    for label, bets in buckets.items():
        n = len(bets)
        pnl = sum(b["pnl"] for b in bets)
        hold = sum(b["hold_pnl"] for b in bets)
        stopped = sum(1 for b in bets if b["status"] == "stop_loss")
        print(f"\n  {label}")
        print(f"    Count:       {n:>3} ({n / total * 100:.0f}%)")
        print(f"    Realized:   ${pnl:>+10.2f}")
        print(f"    If held:    ${hold:>+10.2f}")
        print(f"    Difference: ${hold - pnl:>+10.2f}  (value left on table / saved)")
        print(f"    Stopped:     {stopped}")

    right = sum(1 for h in history if h["model_correct"])
    print(f"\n  Model accuracy on betted matches: {right}/{total} ({right / total * 100:.0f}%)")
    print(f"  Trade win rate:                   {sum(1 for h in history if h['pnl'] > 0)}/{total} ({sum(1 for h in history if h['pnl'] > 0) / total * 100:.0f}%)")

    stopped_right = buckets["Predicted RIGHT, Trade LOST (stop fought model)"]
    if stopped_right:
        print(f"\n  Stop-loss cost on correct predictions:")
        print(f"    {len(stopped_right)} bets where model was right but stop triggered")
        lost = sum(b["pnl"] for b in stopped_right)
        wouldve = sum(b["hold_pnl"] for b in stopped_right)
        print(f"    Realized:  ${lost:>+10.2f}")
        print(f"    If held:   ${wouldve:>+10.2f}")
        print(f"    EV drag:   ${wouldve - lost:>+10.2f}")


def no_stop_counterfactual(history):
    """2. Replay with no stop-loss — hold everything to settlement."""
    _section("2. NO-STOP COUNTERFACTUAL")

    # Replay with stops
    bankroll_stop = STARTING_BANKROLL
    peak_stop = STARTING_BANKROLL
    max_dd_stop = 0.0
    returns_stop = []
    streak_stop = 0
    worst_streak_stop = 0

    # Replay without stops (same entry signals, different P&L)
    bankroll_no = STARTING_BANKROLL
    peak_no = STARTING_BANKROLL
    max_dd_no = 0.0
    returns_no = []
    streak_no = 0
    worst_streak_no = 0

    for h in history:
        entry = h["entry_price"]
        edge = h["edge"]
        mkt = h["market_price"]

        # --- WITH STOP (actual) ---
        pnl_s = h["pnl"]
        bankroll_stop += pnl_s
        ret_s = pnl_s / max(bankroll_stop - pnl_s, 1)
        returns_stop.append(ret_s)
        if bankroll_stop > peak_stop:
            peak_stop = bankroll_stop
        dd = (peak_stop - bankroll_stop) / peak_stop
        max_dd_stop = max(max_dd_stop, dd)
        if pnl_s < 0:
            streak_stop += 1
            worst_streak_stop = max(worst_streak_stop, streak_stop)
        else:
            streak_stop = 0

        # --- WITHOUT STOP (hold to settlement) ---
        # Recompute bet size from no-stop bankroll
        kelly = min((edge / (1 - mkt)) * KELLY_FRACTION, MAX_BET_FRACTION)
        bet_no = bankroll_no * kelly
        vol = h.get("bet_amount", 0) / (h.get("entry_price", 1))
        # Use same volume cap ratio
        if "bet_amount" in h and h["contracts"] > 0:
            orig_vol_cap = h["bet_amount"] / MAX_BET_VS_VOLUME if h["bet_amount"] < bankroll_no * kelly else float("inf")
        else:
            orig_vol_cap = float("inf")

        # Approximate volume from original data
        if h["bet_amount"] < bankroll_stop * kelly * 0.99:
            # Was volume-capped in original; use same cap
            volume_est = h["bet_amount"] / MAX_BET_VS_VOLUME
            max_bet_no = volume_est * MAX_BET_VS_VOLUME
            bet_no = min(bet_no, max_bet_no)

        impact = SLIPPAGE_BASE + SLIPPAGE_IMPACT * (bet_no / max(volume_est if 'volume_est' in dir() else 10000, 1))
        fill_no = min(mkt + impact, 0.99)
        contracts_no = bet_no / fill_no

        if h["model_correct"]:
            pnl_no = (1.0 - fill_no) * contracts_no
        else:
            pnl_no = -fill_no * contracts_no

        bankroll_no += pnl_no
        ret_no = pnl_no / max(bankroll_no - pnl_no, 1)
        returns_no.append(ret_no)
        if bankroll_no > peak_no:
            peak_no = bankroll_no
        dd = (peak_no - bankroll_no) / peak_no
        max_dd_no = max(max_dd_no, dd)
        if pnl_no < 0:
            streak_no += 1
            worst_streak_no = max(worst_streak_no, streak_no)
        else:
            streak_no = 0

    def _sharpe(rets):
        if len(rets) < 2:
            return 0
        m = sum(rets) / len(rets)
        v = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
        return m / (v ** 0.5) if v > 0 else 0

    print(f"\n  {'Metric':<30} {'With Stop':>15} {'No Stop':>15}")
    print(f"  {'─' * 60}")
    print(f"  {'Final bankroll':<30} ${bankroll_stop:>14.2f} ${bankroll_no:>14.2f}")
    print(f"  {'Total return':<30} {(bankroll_stop / STARTING_BANKROLL - 1) * 100:>14.1f}% {(bankroll_no / STARTING_BANKROLL - 1) * 100:>14.1f}%")
    print(f"  {'Max drawdown':<30} {max_dd_stop * 100:>14.1f}% {max_dd_no * 100:>14.1f}%")
    print(f"  {'Sharpe (per-bet)':<30} {_sharpe(returns_stop):>14.3f} {_sharpe(returns_no):>14.3f}")
    print(f"  {'Worst losing streak':<30} {worst_streak_stop:>15} {worst_streak_no:>15}")
    print(f"  {'Avg return per bet':<30} {sum(returns_stop) / len(returns_stop) * 100:>14.2f}% {sum(returns_no) / len(returns_no) * 100:>14.2f}%")
    print(f"  {'Return volatility':<30} {_std(returns_stop) * 100:>14.2f}% {_std(returns_no) * 100:>14.2f}%")


def _std(vals):
    if len(vals) < 2:
        return 0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def entry_price_buckets(history):
    """3. Split by entry price bucket."""
    _section("3. RESULTS BY ENTRY PRICE BUCKET")

    buckets = [
        ("Under 0.40", 0.0, 0.40),
        ("0.40 – 0.55", 0.40, 0.55),
        ("0.55 – 0.70", 0.55, 0.70),
        ("Over 0.70", 0.70, 1.01),
    ]

    print(f"\n  {'Bucket':<16} {'N':>4} {'W':>3} {'L':>3} {'Win%':>6} {'PnL':>10} {'Avg PnL':>9} {'Correct':>8} {'Avg Edge':>9}")
    print(f"  {'─' * 75}")

    for label, lo, hi in buckets:
        subset = [h for h in history if lo <= h["market_price"] < hi]
        if not subset:
            print(f"  {label:<16} {'—':>4}")
            continue
        wins = sum(1 for h in subset if h["pnl"] > 0)
        losses = len(subset) - wins
        pnl = sum(h["pnl"] for h in subset)
        correct = sum(1 for h in subset if h["model_correct"])
        avg_edge = sum(h["edge"] for h in subset) / len(subset)
        print(
            f"  {label:<16} {len(subset):>4} {wins:>3} {losses:>3} "
            f"{wins / len(subset) * 100:>5.0f}% ${pnl:>+9.2f} "
            f"${pnl / len(subset):>+8.2f} {correct}/{len(subset):>5} "
            f"{avg_edge:>+8.2f}"
        )


def edge_buckets(history):
    """4. Split by edge bucket."""
    _section("4. RESULTS BY EDGE BUCKET")

    buckets = [
        ("3–8%", 0.03, 0.08),
        ("8–15%", 0.08, 0.15),
        ("15–25%", 0.15, 0.25),
        ("25–35%", 0.25, 0.35),
        ("35%+", 0.35, 1.01),
    ]

    print(f"\n  {'Bucket':<12} {'N':>4} {'W':>3} {'L':>3} {'Win%':>6} {'PnL':>10} {'Avg PnL':>9} {'Correct':>8} {'Avg Entry':>10}")
    print(f"  {'─' * 75}")

    for label, lo, hi in buckets:
        subset = [h for h in history if lo <= h["edge"] < hi]
        if not subset:
            print(f"  {label:<12} {'—':>4}")
            continue
        wins = sum(1 for h in subset if h["pnl"] > 0)
        losses = len(subset) - wins
        pnl = sum(h["pnl"] for h in subset)
        correct = sum(1 for h in subset if h["model_correct"])
        avg_entry = sum(h["market_price"] for h in subset) / len(subset)
        print(
            f"  {label:<12} {len(subset):>4} {wins:>3} {losses:>3} "
            f"{wins / len(subset) * 100:>5.0f}% ${pnl:>+9.2f} "
            f"${pnl / len(subset):>+8.2f} {correct}/{len(subset):>5} "
            f"{avg_entry:>9.2f}"
        )


def contrarian_calibration(history):
    """5. Calibration on contrarian spots only."""
    _section("5. CALIBRATION ON CONTRARIAN SPOTS")

    print("\n  Only bets where model-market disagreement >= threshold (all bets qualify by definition)")
    print(f"  All bets have edge >= {MIN_EDGE:.0%}\n")

    # Bin by model probability
    bins = [
        ("60–65%", 0.60, 0.65),
        ("65–70%", 0.65, 0.70),
        ("70–80%", 0.70, 0.80),
        ("80–90%", 0.80, 0.90),
        ("90%+", 0.90, 1.01),
    ]

    print(f"  {'Model Prob':<12} {'N':>4} {'Correct':>8} {'Accuracy':>9} {'Avg Mkt':>8} {'Overconf':>10}")
    print(f"  {'─' * 55}")

    for label, lo, hi in bins:
        subset = [h for h in history if lo <= h["model_prob"] < hi]
        if not subset:
            print(f"  {label:<12} {'—':>4}")
            continue
        correct = sum(1 for h in subset if h["model_correct"])
        acc = correct / len(subset)
        avg_prob = sum(h["model_prob"] for h in subset) / len(subset)
        avg_mkt = sum(h["market_price"] for h in subset) / len(subset)
        overconf = avg_prob - acc
        print(
            f"  {label:<12} {len(subset):>4} {correct:>4}/{len(subset):<3} "
            f"{acc * 100:>8.0f}% {avg_mkt:>7.2f} "
            f"{overconf:>+9.2f}"
        )

    print(f"\n  Overconf > 0 means model is overconfident vs realized accuracy.")
    print(f"  Compare model prob to market price to see who was better calibrated.")

    # Also: market vs model — who was right more?
    print(f"\n  Market vs Model head-to-head:")
    model_closer = 0
    market_closer = 0
    for h in history:
        actual = 1.0 if h["model_correct"] else 0.0
        model_err = abs(h["model_prob"] - actual)
        market_err = abs(h["market_price"] - actual)
        if model_err < market_err:
            model_closer += 1
        elif market_err < model_err:
            market_closer += 1
    ties = len(history) - model_closer - market_closer
    print(f"    Model closer to truth: {model_closer}/{len(history)}")
    print(f"    Market closer to truth: {market_closer}/{len(history)}")
    print(f"    Ties: {ties}")

    # Brier scores
    model_brier = sum((h["model_prob"] - (1.0 if h["model_correct"] else 0.0)) ** 2 for h in history) / len(history)
    market_brier = sum((h["market_price"] - (1.0 if h["model_correct"] else 0.0)) ** 2 for h in history) / len(history)
    print(f"\n    Model Brier score (on bets):  {model_brier:.4f}")
    print(f"    Market Brier score (on bets): {market_brier:.4f}")
    print(f"    {'Model' if model_brier < market_brier else 'Market'} is better calibrated on these contrarian spots")


def run_variant(
    max_entry=1.0,
    use_stop=True,
    stop_cents=0.15,
    kelly_frac=0.5,
    max_bet_frac=0.25,
    min_confidence=0.60,
    min_edge=0.03,
    bankroll_start=100.0,
    label="Strategy Variant",
    date_range=None,
    _quiet=False,
):
    """Replay from tradeable cache with custom parameters."""
    with open(TRADEABLE_CACHE) as f:
        matches = json.load(f)

    if date_range:
        start, end = date_range
        if start:
            matches = [m for m in matches if m["date"] >= start]
        if end:
            matches = [m for m in matches if m["date"] < end]

    if not _quiet:
        _section(label)
        params = {
            "Max entry price": f"{max_entry:.2f}",
            "Stop-loss": f"{stop_cents * 100:.0f}c" if use_stop else "NONE",
            "Kelly fraction": f"{kelly_frac:.0%}",
            "Max bet fraction": f"{max_bet_frac:.0%}",
            "Min confidence": f"{min_confidence:.0%}",
            "Min edge": f"{min_edge:.0%}",
        }
        for k, v in params.items():
            print(f"  {k:<20} {v}")
        print()

    bankroll = bankroll_start
    peak = bankroll
    max_dd = 0.0
    history = []
    streak = 0
    worst_streak = 0
    returns = []

    for m in sorted(matches, key=lambda x: x["date"]):
        entry = m["our_entry"]
        prob = m["our_prob"]
        confidence = prob
        correct = m["model_correct"]

        if confidence < min_confidence:
            continue
        if entry > max_entry:
            continue

        edge = prob - entry
        if edge < min_edge:
            continue
        if entry <= 0.01 or entry >= 0.99:
            continue

        kelly = min((edge / (1 - entry)) * kelly_frac, max_bet_frac)
        if kelly <= 0.005:
            continue

        bet = bankroll * kelly
        volume = m["volume"]
        max_bet = volume * MAX_BET_VS_VOLUME
        if bet > max_bet and max_bet > 0:
            bet = max_bet

        impact = SLIPPAGE_BASE + SLIPPAGE_IMPACT * (bet / volume if volume > 0 else 0)
        fill = min(entry + impact, 0.99)
        contracts = bet / fill

        if use_stop:
            our_low = m["our_low"]
            stop_price = fill - stop_cents
            if our_low is not None and our_low <= stop_price:
                exit_p = max(stop_price - SLIPPAGE_BASE, 0.01)
                pnl = (exit_p - fill) * contracts
                status = "stop_loss"
            elif correct:
                pnl = (1.0 - fill) * contracts
                status = "won"
            else:
                pnl = -fill * contracts
                status = "lost"
        else:
            if correct:
                pnl = (1.0 - fill) * contracts
                status = "won"
            else:
                pnl = -fill * contracts
                status = "lost"

        pnl = round(pnl, 2)
        bankroll = round(bankroll + pnl, 2)
        ret = pnl / max(bankroll - pnl, 1)
        returns.append(ret)

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if pnl < 0:
            streak += 1
            worst_streak = max(worst_streak, streak)
        else:
            streak = 0

        history.append({
            "date": m["date"],
            "team1": m["team1"],
            "team2": m["team2"],
            "side": m["side"],
            "entry": entry,
            "fill": fill,
            "edge": edge,
            "bet": bet,
            "contracts": contracts,
            "pnl": pnl,
            "status": status,
            "correct": correct,
            "prob": prob,
            "bankroll": bankroll,
        })

        if not _quiet:
            team = m["team1"] if m["side"] == "T1" else m["team2"]
            print(
                f"  {m['date']} {m['team1']:>25} v {m['team2']:<25} "
                f"{m['side']} mkt={entry:.2f} fill={fill:.2f} edge={edge:+.2f} "
                f"bet=${bet:>7.1f} -> {status:>9} pnl=${pnl:>+8.2f}  bal=${bankroll:>8.2f}",
                flush=True,
            )

    if not _quiet and history:
        wins = sum(1 for h in history if h["pnl"] > 0)
        losses = len(history) - wins
        correct_count = sum(1 for h in history if h["correct"])

        def _sharpe(rets):
            if len(rets) < 2:
                return 0
            m = sum(rets) / len(rets)
            v = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
            return m / (v ** 0.5) if v > 0 else 0

        print(f"\n  {'─' * 50}")
        print(f"  Final:            ${bankroll:.2f}")
        print(f"  Return:           {(bankroll / bankroll_start - 1) * 100:+.1f}%")
        print(f"  Bets:             {len(history)} ({wins}W / {losses}L)")
        print(f"  Trade win rate:   {wins / len(history) * 100:.0f}%")
        print(f"  Model accuracy:   {correct_count}/{len(history)} ({correct_count / len(history) * 100:.0f}%)")
        print(f"  Peak:             ${peak:.2f}")
        print(f"  Max drawdown:     {max_dd * 100:.1f}%")
        print(f"  Worst lose streak:{worst_streak}")
        print(f"  Sharpe (per-bet): {_sharpe(returns):.3f}")
        if wins:
            avg_win = sum(h["pnl"] for h in history if h["pnl"] > 0) / wins
            avg_loss = sum(h["pnl"] for h in history if h["pnl"] <= 0) / losses if losses else 0
            print(f"  Avg win:          ${avg_win:.2f}")
            print(f"  Avg loss:         ${avg_loss:.2f}")

    return history


def stress_thresholds():
    """Sweep entry price thresholds."""
    _section("ENTRY PRICE THRESHOLD SWEEP")
    print(f"\n  {'Thresh':>6} {'Bets':>5} {'W':>3} {'L':>3} {'WR%':>5} {'Acc%':>5} {'Final':>10} {'Return':>8} {'MaxDD':>6} {'Sharpe':>7} {'Streak':>6}")
    print(f"  {'─' * 75}")

    for thresh in [0.45, 0.50, 0.55, 0.60]:
        h = run_variant(
            max_entry=thresh, use_stop=False, kelly_frac=0.5, max_bet_frac=0.25,
            label=f"  threshold={thresh:.2f}", _quiet=True,
        )
        _print_summary_row(f"{thresh:.2f}", h)


def stress_sizing():
    """Sweep Kelly fractions with the baseline entry filter."""
    _section("KELLY SIZING SWEEP (entry <= 0.55, no stop)")
    print(f"\n  {'Kelly':>6} {'Bets':>5} {'W':>3} {'L':>3} {'WR%':>5} {'Acc%':>5} {'Final':>10} {'Return':>8} {'MaxDD':>6} {'Sharpe':>7} {'Streak':>6}")
    print(f"  {'─' * 75}")

    for kf in [0.25, 0.50, 0.75]:
        label = {0.25: "quarter", 0.50: "half", 0.75: "3/4"}[kf]
        h = run_variant(
            max_entry=0.55, use_stop=False, kelly_frac=kf, max_bet_frac=0.25,
            label=f"  kelly={label}", _quiet=True,
        )
        _print_summary_row(f"{label}", h)


def season_splits():
    """Split by early/mid/late season."""
    _section("SEASON-SPLIT VALIDATION (entry <= 0.55, no stop, half-Kelly)")

    with open(TRADEABLE_CACHE) as f:
        matches = json.load(f)

    dates = sorted(set(m["date"] for m in matches))
    n = len(dates)
    early_end = dates[n // 3]
    mid_end = dates[2 * n // 3]

    splits = [
        ("Early (Mar 22 – ~Apr 8)", None, early_end),
        ("Mid (~Apr 8 – ~May 4)", early_end, mid_end),
        ("Late (~May 4 – Jun 3)", mid_end, None),
    ]

    print(f"\n  {'Period':<30} {'Bets':>5} {'W':>3} {'L':>3} {'WR%':>5} {'Acc%':>5} {'Return':>8} {'MaxDD':>6}")
    print(f"  {'─' * 70}")

    for label, start, end in splits:
        h = run_variant(
            max_entry=0.55, use_stop=False, kelly_frac=0.5, max_bet_frac=0.25,
            date_range=(start, end),
            label=f"  {label}", _quiet=True,
        )
        if not h:
            print(f"  {label:<30}   no bets")
            continue
        wins = sum(1 for b in h if b["pnl"] > 0)
        losses = len(h) - wins
        correct = sum(1 for b in h if b["correct"])
        bal = STARTING_BANKROLL
        peak = bal
        max_dd = 0
        for b in h:
            bal += b["pnl"]
            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        ret = (bal / STARTING_BANKROLL - 1) * 100
        wr = wins / len(h) * 100 if h else 0
        acc = correct / len(h) * 100 if h else 0
        print(f"  {label:<30} {len(h):>5} {wins:>3} {losses:>3} {wr:>4.0f}% {acc:>4.0f}% {ret:>+7.0f}% {max_dd * 100:>5.1f}%")


def _print_summary_row(label, h):
    if not h:
        print(f"  {label:>6}   no bets")
        return
    wins = sum(1 for b in h if b["pnl"] > 0)
    losses = len(h) - wins
    correct = sum(1 for b in h if b["correct"])
    bankroll = STARTING_BANKROLL + sum(b["pnl"] for b in h)
    # Recompute with compounding
    bal = STARTING_BANKROLL
    peak = bal
    max_dd = 0
    streak = 0
    worst = 0
    rets = []
    for b in h:
        bal += b["pnl"]
        r = b["pnl"] / max(bal - b["pnl"], 1)
        rets.append(r)
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        if b["pnl"] < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    wr = wins / len(h) * 100
    acc = correct / len(h) * 100
    sharpe = 0
    if len(rets) >= 2:
        m = sum(rets) / len(rets)
        v = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
        sharpe = m / (v ** 0.5) if v > 0 else 0
    print(f"  {label:>6} {len(h):>5} {wins:>3} {losses:>3} {wr:>4.0f}% {acc:>4.0f}% ${bal:>9.2f} {(bal / STARTING_BANKROLL - 1) * 100:>+7.0f}% {max_dd * 100:>5.1f}% {sharpe:>6.3f} {worst:>6}")


def threshold_sensitivity():
    """Fine-grained threshold sweep to check for plateau vs spike."""
    _section("1. ENTRY THRESHOLD SENSITIVITY (plateau check)")
    print(f"\n  {'Thresh':>6} {'Bets':>5} {'W':>3} {'L':>3} {'WR%':>5} {'Acc%':>5} {'Final':>10} {'Return':>8} {'MaxDD':>6} {'Sharpe':>7} {'Streak':>6}")
    print(f"  {'─' * 75}")

    for thresh in [0.53, 0.54, 0.55, 0.56, 0.57]:
        h = run_variant(
            max_entry=thresh, use_stop=False, kelly_frac=0.5, max_bet_frac=0.25,
            _quiet=True,
        )
        _print_summary_row(f"{thresh:.2f}", h)


def edge_threshold_sweep():
    """Edge floor sweep within the 0.55 universe."""
    _section("2. EDGE THRESHOLD SWEEP (entry <= 0.55, no stop, half-Kelly)")
    print(f"\n  {'Edge':>6} {'Bets':>5} {'W':>3} {'L':>3} {'WR%':>5} {'Acc%':>5} {'Final':>10} {'Return':>8} {'MaxDD':>6} {'Sharpe':>7} {'Streak':>6}")
    print(f"  {'─' * 75}")

    for edge_pct in [0.03, 0.05, 0.08, 0.10]:
        h = run_variant(
            max_entry=0.55, use_stop=False, kelly_frac=0.5, max_bet_frac=0.25,
            min_edge=edge_pct, _quiet=True,
        )
        _print_summary_row(f"{edge_pct:.0%}", h)


def exposure_concentration():
    """Check whether a few giant bets dominate the equity curve."""
    _section("3. EXPOSURE CONCENTRATION")

    h = run_variant(
        max_entry=0.55, use_stop=False, kelly_frac=0.5, max_bet_frac=0.25,
        _quiet=True,
    )
    if not h:
        print("  No bets")
        return

    # Bet size as % of bankroll at time of bet
    print(f"\n  Bet sizing distribution:")
    bet_pcts = []
    for b in h:
        pre_bal = b["bankroll"] - b["pnl"]
        pct = b["bet"] / pre_bal * 100 if pre_bal > 0 else 0
        bet_pcts.append(pct)

    bet_pcts_sorted = sorted(bet_pcts, reverse=True)
    print(f"    Max single bet:   {bet_pcts_sorted[0]:>5.1f}% of bankroll")
    print(f"    2nd largest:      {bet_pcts_sorted[1]:>5.1f}%")
    print(f"    Median bet:       {bet_pcts_sorted[len(bet_pcts_sorted)//2]:>5.1f}%")
    print(f"    Min bet:          {bet_pcts_sorted[-1]:>5.1f}%")

    # PnL concentration: top N bets as % of total profit
    total_pnl = sum(b["pnl"] for b in h)
    wins_sorted = sorted([b["pnl"] for b in h if b["pnl"] > 0], reverse=True)
    losses_sorted = sorted([b["pnl"] for b in h if b["pnl"] < 0])
    total_profit = sum(wins_sorted)
    total_loss = sum(losses_sorted)

    print(f"\n  PnL concentration:")
    print(f"    Total profit from wins:  ${total_profit:>+10.2f}")
    print(f"    Total cost of losses:    ${total_loss:>+10.2f}")
    print(f"    Net:                     ${total_pnl:>+10.2f}")
    for n in [1, 3, 5]:
        if n <= len(wins_sorted):
            top_n = sum(wins_sorted[:n])
            print(f"    Top {n} win(s):            ${top_n:>+10.2f}  ({top_n / total_profit * 100:.0f}% of all profit)")

    # Remove top N wins — is the strategy still profitable?
    print(f"\n  Robustness: remove largest wins")
    for n in [1, 2, 3]:
        if n <= len(wins_sorted):
            remaining = total_pnl - sum(wins_sorted[:n])
            print(f"    Without top {n} win(s):    ${remaining:>+10.2f}  ({'profitable' if remaining > 0 else 'NEGATIVE'})")

    # Per-team exposure
    from collections import Counter
    team_bets = Counter()
    team_pnl = {}
    for b in h:
        team = b["team1"] if b["side"] == "T1" else b["team2"]
        team_bets[team] += 1
        team_pnl[team] = team_pnl.get(team, 0) + b["pnl"]

    print(f"\n  Per-team concentration:")
    print(f"    {'Team':<30} {'Bets':>5} {'PnL':>10}")
    print(f"    {'─' * 48}")
    for team, count in team_bets.most_common():
        print(f"    {team:<30} {count:>5} ${team_pnl[team]:>+9.2f}")

    # Same-day exposure
    from collections import defaultdict
    day_bets = defaultdict(list)
    for b in h:
        day_bets[b["date"]].append(b)
    multi_days = {d: bs for d, bs in day_bets.items() if len(bs) > 1}

    if multi_days:
        print(f"\n  Same-day multi-bets: {len(multi_days)} days")
        for d, bs in sorted(multi_days.items()):
            total_bet = sum(b["bet"] for b in bs)
            pre_bal = bs[0]["bankroll"] - bs[0]["pnl"]
            pct = total_bet / pre_bal * 100 if pre_bal > 0 else 0
            print(f"    {d}: {len(bs)} bets, ${total_bet:.0f} total ({pct:.0f}% of bankroll)")
    else:
        print(f"\n  No same-day multi-bets")


def closing_market_comparison():
    """Check if entry prices consistently beat later market prices."""
    _section("4. CLOSING-MARKET COMPARISON")

    with open(TRADEABLE_CACHE) as f:
        matches = json.load(f)

    # Run baseline to get which matches were bet on
    h = run_variant(
        max_entry=0.55, use_stop=False, kelly_frac=0.5, max_bet_frac=0.25,
        _quiet=True,
    )
    bet_keys = set((b["date"], b["team1"], b["team2"]) for b in h)

    # Build lookup from tradeable cache
    match_lookup = {}
    for m in matches:
        match_lookup[(m["date"], m["team1"], m["team2"])] = m

    print(f"\n  For each bet: did the market move toward our position after entry?")
    print(f"  (our_entry vs match-window peak for our side)\n")

    print(f"  {'Date':>10} {'Matchup':>50} {'Entry':>6} {'Peak':>6} {'Move':>6} {'Correct':>8}")
    print(f"  {'─' * 90}")

    convergence = []
    for b in h:
        key = (b["date"], b["team1"], b["team2"])
        m = match_lookup.get(key)
        if not m:
            continue

        # Peak price for our side during the match
        if b["side"] == "T1":
            our_peak = m["t1_high"]
        else:
            our_peak = 1.0 - m["t1_low"]

        entry = b["entry"]
        move = our_peak - entry if our_peak else 0
        correct_str = "RIGHT" if b["correct"] else "WRONG"

        convergence.append({
            "entry": entry,
            "peak": our_peak,
            "move": move,
            "correct": b["correct"],
            "pnl": b["pnl"],
        })

        print(
            f"  {b['date']:>10} {b['team1']:>22} v {b['team2']:<22} "
            f"{entry:>5.2f} {our_peak:>5.2f} {move:>+5.2f} {correct_str:>8}"
        )

    if not convergence:
        return

    # Summary stats
    all_moves = [c["move"] for c in convergence]
    correct_moves = [c["move"] for c in convergence if c["correct"]]
    wrong_moves = [c["move"] for c in convergence if not c["correct"]]

    print(f"\n  Summary:")
    print(f"    Avg move toward position (all):     {sum(all_moves) / len(all_moves):>+.3f}")
    print(f"    Avg move toward position (correct): {sum(correct_moves) / len(correct_moves):>+.3f}" if correct_moves else "")
    print(f"    Avg move toward position (wrong):   {sum(wrong_moves) / len(wrong_moves):>+.3f}" if wrong_moves else "")

    # How often did market peak above our entry?
    above = sum(1 for c in convergence if c["move"] > 0)
    print(f"\n    Market peaked above entry: {above}/{len(convergence)} ({above / len(convergence) * 100:.0f}%)")

    # For correct predictions: how much higher did the market go?
    if correct_moves:
        print(f"    On correct calls:")
        print(f"      Avg peak above entry: +{sum(correct_moves) / len(correct_moves):.2f}")
        print(f"      Min:                  {min(correct_moves):+.2f}")
        print(f"      Max:                  {max(correct_moves):+.2f}")

    # Final value vs entry: settlement at 1.0 for correct, 0.0 for wrong
    avg_entry_correct = sum(c["entry"] for c in convergence if c["correct"]) / len(correct_moves) if correct_moves else 0
    avg_entry_wrong = sum(c["entry"] for c in convergence if not c["correct"]) / len(wrong_moves) if wrong_moves else 0
    print(f"\n    Avg entry on correct calls:  {avg_entry_correct:.3f}  (settles at 1.00, +{1.0 - avg_entry_correct:.3f} per contract)")
    if wrong_moves:
        print(f"    Avg entry on wrong calls:    {avg_entry_wrong:.3f}  (settles at 0.00, -{avg_entry_wrong:.3f} per contract)")
    print(f"    Payoff asymmetry:            {(1.0 - avg_entry_correct) / avg_entry_correct:.1f}:1 win/loss ratio per contract")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diagnostics"

    if mode == "variant":
        run_variant(
            max_entry=0.55,
            use_stop=False,
            kelly_frac=0.5,
            max_bet_frac=0.25,
            min_confidence=0.60,
            min_edge=0.03,
            label="BASELINE: Contrarian (<=0.55), no stop, half-Kelly",
        )
    elif mode == "stress":
        stress_thresholds()
        stress_sizing()
        season_splits()
    elif mode == "stress2":
        threshold_sensitivity()
        edge_threshold_sweep()
        exposure_concentration()
        closing_market_comparison()
    else:
        history = _load()
        decompose(history)
        no_stop_counterfactual(history)
        entry_price_buckets(history)
        edge_buckets(history)
        contrarian_calibration(history)
