"""Main orchestrator loop for the IPL betting engine."""

import subprocess
import sys
import threading
import time
from datetime import date

from engine.config import DASHBOARD_PORT
from engine.state import load_state, save_state, open_position, add_upcoming, get_bankroll
from engine.logger import log_event
from engine.market import find_ipl_markets
from engine.scraper import get_todays_ipl_matches, poll_until_toss
from engine.signal import generate_signal
from engine.executor import place_bet
from engine.monitor import monitor_all_positions
from engine.server import start_server


def _start_dashboard():
    """Run the dashboard in a background thread."""
    t = threading.Thread(target=start_server, args=(DASHBOARD_PORT,), daemon=True)
    t.start()
    print(f"  Dashboard running at http://localhost:{DASHBOARD_PORT}")
    return t


def _discover_markets(state: dict, match_date: str):
    """Find today's IPL markets on Polymarket and add to upcoming."""
    log_event(state, "discovery", f"Searching Polymarket for IPL matches on {match_date}")
    markets = find_ipl_markets(match_date)

    if not markets:
        log_event(state, "discovery", "No IPL markets found on Polymarket for today")
        return []

    for mkt in markets:
        add_upcoming(state, {
            "match_date": match_date,
            "team1": mkt["team1"],
            "team2": mkt["team2"],
            "polymarket_slug": mkt["slug"],
            "t1_token_id": mkt["t1_token_id"],
            "t2_token_id": mkt["t2_token_id"],
            "model_prediction": None,
            "status": "awaiting_toss",
        })
        log_event(
            state, "discovery",
            f"Found market: {mkt['team1']} vs {mkt['team2']} ({mkt['slug']})",
            data={"event_id": mkt["event_id"], "volume": mkt.get("volume", 0)},
        )

    save_state(state)
    return markets


def _process_match(state: dict, match: dict, market_info: dict):
    """Process a single match: scrape, signal, bet, monitor."""
    team1 = match.get("team1") or market_info.get("team1")
    team2 = match.get("team2") or market_info.get("team2")

    log_event(state, "scrape", f"Waiting for toss/XIs: {team1} vs {team2}")

    series_slug = match.get("series_slug", "")
    match_slug = match.get("match_slug", "")

    if series_slug and match_slug:
        details = poll_until_toss(series_slug, match_slug, timeout=7200)
    else:
        log_event(state, "error", f"No cricinfo match slug for {team1} vs {team2} — skipping")
        return

    if not details:
        log_event(state, "error", f"Toss/XIs not available for {team1} vs {team2} — timed out")
        return

    log_event(
        state, "scrape",
        f"Toss: {details['toss_winner']} won, chose to {details['toss_decision']}",
        data={"team1_xi": details["team1_xi"][:3], "team2_xi": details["team2_xi"][:3]},
    )

    bankroll = get_bankroll(state)
    sig = generate_signal(details, market_info, bankroll)

    if not sig:
        log_event(state, "signal", f"No edge for {team1} vs {team2} — skipping")
        return

    log_event(
        state, "signal",
        f"Edge {sig['edge']:+.2f} on {sig['team']} "
        f"(model {sig['model_prob']:.0%}, market {sig['market_price']:.0%}). "
        f"Kelly {sig['kelly_fraction']:.0%}, betting ${sig['bet_amount']:.2f}",
        data=sig,
    )

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
        "token_id": sig["token_id"],
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


def _pair_matches_to_markets(cricket_matches: list, polymarket_markets: list) -> list:
    """Match cricinfo matches to Polymarket markets by team names."""
    pairs = []
    used = set()
    for cm in cricket_matches:
        for i, pm in enumerate(polymarket_markets):
            if i in used:
                continue
            if (cm.get("team1") == pm.get("team1") and cm.get("team2") == pm.get("team2")) or \
               (cm.get("team1") == pm.get("team2") and cm.get("team2") == pm.get("team1")):
                pairs.append((cm, pm))
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


def run():
    """Main entry point."""
    print("=" * 50)
    print("  IPL Betting Engine")
    print("=" * 50)

    state = load_state()
    log_event(state, "discovery", f"Engine started. Bankroll: ${get_bankroll(state):.2f}")
    save_state(state)

    _start_dashboard()

    today = date.today().isoformat()

    # Step 1: Find Polymarket markets
    pm_markets = _discover_markets(state, today)

    # Step 2: Find cricket matches
    cricket_matches = get_todays_ipl_matches()
    if cricket_matches:
        log_event(state, "scrape", f"Found {len(cricket_matches)} IPL match(es) on ESPNCricinfo")
    else:
        log_event(state, "scrape", "No live IPL matches found on ESPNCricinfo")

    # Step 3: Pair them up
    pairs = _pair_matches_to_markets(cricket_matches, pm_markets)

    if not pairs:
        log_event(state, "discovery", "No actionable match/market pairs for today")
        save_state(state)
        print("\n  No matches to trade. Dashboard will stay live.")
        print("  Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        return

    # Step 4: Process each match (scrape toss, generate signal, bet)
    threads = []
    for cm, pm in pairs:
        market_info = {
            "t1_token_id": pm["t1_token_id"],
            "t2_token_id": pm["t2_token_id"],
            "team1": pm["team1"],
            "team2": pm["team2"],
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

    log_event(
        state, "settle",
        f"Day complete. Final bankroll: ${get_bankroll(state):.2f}",
    )
    save_state(state)

    # Step 6: Retrain model with today's new match data
    _retrain(state)

    print(f"\n  Day complete. Bankroll: ${get_bankroll(state):.2f}")
    print("  Dashboard will stay live. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
