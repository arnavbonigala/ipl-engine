"""Main orchestrator loop for the IPL betting engine (Kalshi).

Usage:
  python -m engine.run              # full live mode (requires KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH)
  python -m engine.run --paper      # paper trading: signals only, no real orders
"""

from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone, timedelta

from engine.config import DASHBOARD_PORT, TEAM_KEYWORDS
from engine.state import load_state, save_state, open_position, add_upcoming, clear_upcoming, get_bankroll
from engine.logger import log_event
from engine.market import find_ipl_markets
from engine.scraper import get_todays_ipl_matches, poll_until_toss
from engine.signal import generate_signal
from engine.monitor import monitor_all_positions
from engine.server import start_server

PAPER_MODE = "--paper" in sys.argv or not os.environ.get("KALSHI_API_KEY_ID")


def _sync_positions(state: dict):
    """Reconcile engine state with actual Kalshi positions on startup."""
    try:
        from engine.executor import get_positions
        kalshi_positions = get_positions()
    except Exception:
        return

    kalshi_tickers = {
        p["ticker"] for p in kalshi_positions
        if float(p.get("position_fp", 0)) > 0
    }
    engine_tickers = {p["ticker"] for p in state.get("positions", []) if p.get("status") == "open"}

    stale = engine_tickers - kalshi_tickers
    if stale:
        state["positions"] = [
            p for p in state["positions"]
            if not (p.get("status") == "open" and p["ticker"] in stale)
        ]
        log_event(state, "discovery", f"Removed {len(stale)} stale position(s) not on Kalshi")
        save_state(state)


def _start_dashboard():
    """Run the dashboard in a background thread."""
    t = threading.Thread(target=start_server, args=(DASHBOARD_PORT,), daemon=True)
    t.start()
    print(f"  Dashboard running at http://localhost:{DASHBOARD_PORT}")
    return t


def _discover_markets(state: dict, match_date: str):
    """Find today's IPL markets on Kalshi and add to upcoming."""
    log_event(state, "discovery", f"Searching Kalshi for IPL matches on {match_date}")
    markets = find_ipl_markets(match_date)

    if not markets:
        log_event(state, "discovery", "No IPL markets found on Kalshi for today")
        return []

    for mkt in markets:
        add_upcoming(state, {
            "match_date": mkt.get("match_date") or match_date,
            "team1": mkt["team1"],
            "team2": mkt["team2"],
            "event_ticker": mkt["event_ticker"],
            "t1_ticker": mkt["t1_ticker"],
            "t2_ticker": mkt["t2_ticker"],
            "model_prediction": None,
            "status": "awaiting_toss",
        })
        log_event(
            state, "discovery",
            f"Found market: {mkt['team1']} vs {mkt['team2']} ({mkt['event_ticker']})",
            data={"volume": mkt.get("volume", 0)},
        )

    save_state(state)
    return markets


def _finish_match(state: dict, market_info: dict, status: str, prediction: dict | None = None):
    """Move match from upcoming to history (idempotent per event_ticker)."""
    event_ticker = market_info.get("event_ticker") or ""
    existing = {h.get("event_ticker") for h in state.get("history", [])}
    if event_ticker and event_ticker in existing:
        clear_upcoming(state, event_ticker)
        return
    entry = {
        "match_date": date.today().isoformat(),
        "team1": market_info.get("team1", ""),
        "team2": market_info.get("team2", ""),
        "event_ticker": event_ticker,
        "status": status,
    }
    if prediction:
        entry["model_t1_prob"] = prediction.get("model_t1_prob")
        entry["model_prob"] = prediction.get("model_prob")
        entry["chosen_team"] = prediction.get("chosen_team")
        entry["market_price"] = prediction.get("market_price")
        entry["edge"] = prediction.get("edge")
    state.setdefault("history", []).append(entry)
    clear_upcoming(state, event_ticker)


def _already_acted(state: dict, event_ticker: str) -> bool:
    """True if we already have a position or history entry for this event."""
    for p in state.get("positions", []):
        if p.get("event_ticker") == event_ticker:
            return True
    for h in state.get("history", []):
        if h.get("event_ticker") == event_ticker:
            return True
    return False


def _process_match(state: dict, match: dict, market_info: dict):
    """Process a single match: scrape, signal, bet, monitor."""
    event_ticker = market_info.get("event_ticker", "")
    if _already_acted(state, event_ticker):
        return

    team1 = match.get("team1") or market_info.get("team1")
    team2 = match.get("team2") or market_info.get("team2")

    series_slug = match.get("series_slug", "")
    match_slug = match.get("match_slug", "")

    # Wait until 2 hours before match start before polling
    start_time_str = match.get("start_time", "")
    if start_time_str:
        try:
            match_start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            poll_start = match_start - timedelta(hours=2)
            now = datetime.now(timezone.utc)
            wait_secs = (poll_start - now).total_seconds()
            if wait_secs > 60:
                log_event(
                    state, "scrape",
                    f"Sleeping until {poll_start.strftime('%H:%M UTC')} "
                    f"({wait_secs/3600:.1f}h) before polling for toss: {team1} vs {team2}",
                )
                save_state(state)
                time.sleep(max(wait_secs, 0))
        except (ValueError, TypeError):
            pass

    log_event(state, "scrape", f"Polling for toss/XIs: {team1} vs {team2}")
    save_state(state)

    if series_slug and match_slug:
        details = poll_until_toss(series_slug, match_slug, timeout=14400)
    else:
        log_event(state, "error", f"No cricinfo match slug for {team1} vs {team2} — skipping")
        _finish_match(state, market_info, "skipped_no_slug")
        return

    if not details:
        log_event(state, "error", f"Toss/XIs not available for {team1} vs {team2} — timed out")
        _finish_match(state, market_info, "skipped_timeout")
        return

    log_event(
        state, "scrape",
        f"Toss: {details['toss_winner']} won, chose to {details['toss_decision']}",
        data={"team1_xi": details["team1_xi"][:3], "team2_xi": details["team2_xi"][:3]},
    )

    try:
        from engine.executor import get_balance
        bankroll = get_balance()
    except Exception:
        bankroll = get_bankroll(state)
    sig, prediction = generate_signal(details, market_info, bankroll)

    if not sig:
        mkt_price = prediction.get("market_price")
        mkt_str = f", market {mkt_price:.0%}" if mkt_price else ""
        edge = prediction.get("edge")
        edge_str = f", edge {edge:+.2f}" if edge is not None else ""
        log_event(
            state, "signal",
            f"No edge for {team1} vs {team2} — model picks "
            f"{prediction['chosen_team']} @ {prediction['model_prob']:.0%}"
            f"{mkt_str}{edge_str}",
            data=prediction,
        )
        _finish_match(state, market_info, "skipped_no_edge", prediction)
        return

    log_event(
        state, "signal",
        f"Edge {sig['edge']:+.2f} on {sig['team']} "
        f"(model {sig['model_prob']:.0%}, market {sig['market_price']:.0%}). "
        f"Kelly {sig['kelly_fraction']:.0%}, betting ${sig['bet_amount']:.2f}",
        data=sig,
    )

    if not PAPER_MODE:
        try:
            from engine.executor import get_positions
            for kp in get_positions():
                if kp["ticker"] == sig["ticker"] and float(kp.get("position_fp", 0)) > 0:
                    log_event(state, "bet", f"Already have Kalshi position on {sig['ticker']} — skipping")
                    _finish_match(state, market_info, "skipped_already_positioned", prediction)
                    return
        except Exception:
            pass

    if PAPER_MODE:
        order_id = "paper-" + date.today().isoformat()
        log_event(
            state, "bet",
            f"[PAPER] Would place: {sig['contracts']:.1f} contracts of {sig['team']} @ "
            f"{sig['market_price']:.2f} (${sig['bet_amount']:.2f})",
            data=sig,
        )
    else:
        from engine.executor import place_bet
        order_id = place_bet(sig)
        if not order_id:
            log_event(state, "error", f"Order placement failed for {sig['team']}")
            return
        log_event(
            state, "bet",
            f"Placed: {sig['contracts']:.1f} contracts of {sig['team']} @ "
            f"{sig['market_price']:.2f} (${sig['bet_amount']:.2f})",
            data={"order_id": order_id},
        )

    position = {
        "match_date": date.today().isoformat(),
        "team1": sig["team1"],
        "team2": sig["team2"],
        "side": sig["side"],
        "team": sig["team"],
        "ticker": sig["ticker"],
        "event_ticker": market_info.get("event_ticker", ""),
        "entry_price": sig["market_price"],
        "contracts": sig["contracts"],
        "bet_amount": sig["bet_amount"],
        "model_prob": sig["model_prob"],
        "market_price": sig["market_price"],
        "edge": sig["edge"],
        "order_id": order_id,
    }
    state["bankroll"] = round(state["bankroll"] - sig["bet_amount"], 4)
    open_position(state, position)


def _pair_matches_to_markets(cricket_matches: list, kalshi_markets: list) -> list:
    """Match cricinfo matches to Kalshi markets by team names."""
    pairs = []
    used = set()
    for cm in cricket_matches:
        for i, km in enumerate(kalshi_markets):
            if i in used:
                continue
            cm_teams = {cm.get("team1"), cm.get("team2")}
            km_teams = {km.get("team1"), km.get("team2")}
            if cm_teams == km_teams:
                pairs.append((cm, km))
                used.add(i)
                break
    return pairs


def _retrain(state: dict):
    """Fetch new data, rebuild dataset, and retrain the model."""
    log_event(state, "retrain", "Starting post-match retrain pipeline...")
    save_state(state)

    steps = [
        ("fetch_data", [sys.executable, "scripts/fetch_data.py"]),
        ("build_dataset", [sys.executable, "-m", "predictor.build_dataset"]),
        ("train", [sys.executable, "-m", "predictor.train"]),
    ]

    for name, cmd in steps:
        log_event(state, "retrain", f"Running {name}...")
        save_state(state)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log_event(
                state, "error",
                f"Retrain step '{name}' failed (exit {result.returncode}): "
                f"{result.stderr[:300]}",
            )
            save_state(state)
            return

    log_event(state, "retrain", "Retrain complete — model updated for next match day")
    save_state(state)


def _run_day(state: dict):
    """Run the engine for a single match day."""
    today = date.today().isoformat()
    log_event(state, "discovery", f"Starting match day: {today}")
    save_state(state)

    # Step 1: Find Kalshi markets
    kalshi_markets = _discover_markets(state, today)

    # Step 2: Find cricket matches
    cricket_matches = get_todays_ipl_matches()
    if cricket_matches:
        log_event(state, "scrape", f"Found {len(cricket_matches)} IPL match(es) on ESPNCricinfo")
    else:
        log_event(state, "scrape", "No live IPL matches found on ESPNCricinfo")
    save_state(state)

    # Step 3: Pair them up
    pairs = _pair_matches_to_markets(cricket_matches, kalshi_markets)

    if not pairs:
        log_event(state, "discovery", "No actionable match/market pairs for today")
        save_state(state)
        return

    # Step 4: Process each match (sleeps until ~2h before start, then polls for toss)
    threads = []
    for cm, km in pairs:
        market_info = {
            "t1_ticker": km["t1_ticker"],
            "t2_ticker": km["t2_ticker"],
            "team1": km["team1"],
            "team2": km["team2"],
            "event_ticker": km["event_ticker"],
        }
        t = threading.Thread(target=_process_match, args=(state, cm, market_info))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Step 5: Monitor all open positions
    log_event(state, "discovery", "All matches processed. Monitoring open positions...")
    save_state(state)
    monitor_all_positions(state)

    try:
        from engine.executor import get_balance
        final_bal = get_balance()
    except Exception:
        final_bal = get_bankroll(state)
    log_event(state, "settle", f"Day complete. Final balance: ${final_bal:.2f}")
    save_state(state)

    # Step 6: Retrain model with today's new match data
    _retrain(state)


def _seconds_until_next_morning() -> float:
    """Seconds until 08:00 UTC the next day (~1:30 PM IST, well before any match)."""
    now = datetime.now(timezone.utc)
    tomorrow_8am = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    return (tomorrow_8am - now).total_seconds()


def run():
    """Main entry point — runs continuously, one match day per iteration."""
    print("=" * 50)
    print(f"  IPL Betting Engine [Kalshi] {'[PAPER MODE]' if PAPER_MODE else '[LIVE]'}")
    print("=" * 50)

    state = load_state()
    try:
        from engine.executor import get_balance
        bal = get_balance()
    except Exception:
        bal = get_bankroll(state)
    log_event(state, "discovery", f"Engine started. Balance: ${bal:.2f}")
    save_state(state)

    _sync_positions(state)
    _start_dashboard()

    try:
        while True:
            _run_day(state)

            sleep_secs = _seconds_until_next_morning()
            log_event(
                state, "discovery",
                f"Day done. Sleeping {sleep_secs/3600:.1f}h until next morning scan.",
            )
            save_state(state)
            time.sleep(sleep_secs)

            state = load_state()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
