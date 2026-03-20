"""Live price polling and stop-loss enforcement."""

import time
import threading

from engine.config import POLL_INTERVAL_SECS, STOP_LOSS_CENTS, USE_STOP_LOSS
from engine.market import get_market_price
from engine.executor import sell_position
from engine.state import close_position, save_state
from engine.logger import log_event


def monitor_position(state: dict, position: dict):
    """Poll a single open position until settled or stopped out."""
    token_id = position["token_id"]
    entry = position["entry_price"]
    contracts = position["contracts"]
    stop_price = entry - STOP_LOSS_CENTS

    while position.get("status") == "open":
        price = get_market_price(token_id)
        if price is None:
            time.sleep(POLL_INTERVAL_SECS)
            continue

        position["current_price"] = round(price, 4)
        save_state(state)

        # stop-loss trigger
        if USE_STOP_LOSS and price <= stop_price:
            log_event(
                state, "stop_loss",
                f"Stop-loss triggered for {position['team']} @ {price:.2f} "
                f"(entry {entry:.2f}, stop {stop_price:.2f})",
            )
            sell_position(token_id, contracts, price)
            pnl = (price - entry) * contracts
            close_position(state, token_id, "stop_loss", pnl)
            return

        # check if market is settled (price at 0 or 1)
        if price >= 0.99:
            pnl = (1.0 - entry) * contracts
            log_event(state, "settle", f"{position['team']} won! P&L: +${pnl:.2f}")
            close_position(state, token_id, "won", pnl)
            return
        elif price <= 0.01:
            pnl = -entry * contracts
            log_event(state, "settle", f"{position['team']} lost. P&L: -${abs(pnl):.2f}")
            close_position(state, token_id, "lost", pnl)
            return

        time.sleep(POLL_INTERVAL_SECS)


def monitor_all_positions(state: dict):
    """Monitor all open positions concurrently."""
    open_positions = [p for p in state.get("positions", []) if p.get("status") == "open"]
    if not open_positions:
        return

    threads = []
    for pos in open_positions:
        t = threading.Thread(target=monitor_position, args=(state, pos), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
