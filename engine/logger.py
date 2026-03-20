"""Structured event logging for the betting engine."""

from datetime import datetime, timezone


def log_event(state: dict, event_type: str, message: str, data: dict | None = None):
    """Append a structured event to state and print it."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "message": message,
    }
    if data:
        entry["data"] = data
    state.setdefault("events", []).append(entry)
    print(f"  [{event_type.upper():>10}] {message}")
    return entry
