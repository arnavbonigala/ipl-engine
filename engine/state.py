"""JSON persistence for positions, bankroll, and event log."""

import json
from pathlib import Path

from engine.config import STATE_FILE

_STATE_PATH = Path(STATE_FILE)

_DEFAULT = {
    "bankroll": 100.0,
    "wallet_address": "",
    "positions": [],
    "history": [],
    "events": [],
    "upcoming": [],
}


def load_state() -> dict:
    if _STATE_PATH.exists():
        with open(_STATE_PATH) as f:
            return json.load(f)
    return dict(_DEFAULT)


def save_state(state: dict):
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_bankroll(state: dict) -> float:
    return state.get("bankroll", 0.0)


def open_position(state: dict, position: dict):
    position.setdefault("status", "open")
    position.setdefault("pnl", None)
    position.setdefault("current_price", position.get("entry_price", 0))
    state["positions"].append(position)
    save_state(state)


def close_position(state: dict, ticker: str, status: str, pnl: float):
    cost = 0.0
    for pos in state["positions"]:
        if pos["ticker"] == ticker and pos["status"] == "open":
            pos["status"] = status
            pos["pnl"] = round(pnl, 4)
            cost = pos.get("bet_amount", 0.0)
            state["history"].append(pos)
            break
    state["positions"] = [
        p for p in state["positions"]
        if not (p["ticker"] == ticker and p["status"] != "open")
    ]
    # cost was already subtracted at placement; add it back along with pnl
    state["bankroll"] = round(state["bankroll"] + cost + pnl, 4)
    save_state(state)


def add_upcoming(state: dict, match: dict):
    existing = {m.get("event_ticker") for m in state["upcoming"]}
    if match.get("event_ticker") not in existing:
        state["upcoming"].append(match)
        save_state(state)


def clear_upcoming(state: dict, event_ticker: str):
    state["upcoming"] = [m for m in state["upcoming"] if m.get("event_ticker") != event_ticker]
    save_state(state)
