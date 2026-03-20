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


def close_position(state: dict, token_id: str, status: str, pnl: float):
    for pos in state["positions"]:
        if pos["token_id"] == token_id and pos["status"] == "open":
            pos["status"] = status
            pos["pnl"] = round(pnl, 4)
            state["history"].append(pos)
            break
    state["positions"] = [
        p for p in state["positions"]
        if not (p["token_id"] == token_id and p["status"] != "open")
    ]
    state["bankroll"] = round(state["bankroll"] + pnl, 4)
    save_state(state)


def add_upcoming(state: dict, match: dict):
    existing_slugs = {m.get("polymarket_slug") for m in state["upcoming"]}
    if match.get("polymarket_slug") not in existing_slugs:
        state["upcoming"].append(match)
        save_state(state)


def clear_upcoming(state: dict, slug: str):
    state["upcoming"] = [m for m in state["upcoming"] if m.get("polymarket_slug") != slug]
    save_state(state)
