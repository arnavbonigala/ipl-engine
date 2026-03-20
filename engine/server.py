"""FastAPI app serving REST API + static dashboard."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from engine.state import load_state

app = FastAPI(title="IPL Betting Engine")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (_STATIC_DIR / "index.html").read_text()


@app.get("/api/status")
def status():
    s = load_state()
    open_count = sum(1 for p in s.get("positions", []) if p.get("status") == "open")
    return {
        "bankroll": s.get("bankroll", 0),
        "wallet_address": s.get("wallet_address", ""),
        "engine_state": "running" if open_count > 0 else "idle",
        "open_positions": open_count,
    }


@app.get("/api/positions")
def positions():
    s = load_state()
    return [p for p in s.get("positions", []) if p.get("status") == "open"]


@app.get("/api/history")
def history():
    s = load_state()
    return list(reversed(s.get("history", [])))


@app.get("/api/upcoming")
def upcoming():
    s = load_state()
    return s.get("upcoming", [])


@app.get("/api/events")
def events():
    s = load_state()
    return list(reversed(s.get("events", [])))[:200]


@app.get("/api/stats")
def stats():
    s = load_state()
    hist = s.get("history", [])
    bets = [h for h in hist if h.get("status") not in ("skipped", "no_market")]
    skipped = len(hist) - len(bets)
    if not bets:
        return {
            "total_return_pct": 0, "win_rate": 0, "total_bets": 0,
            "total_pnl": 0, "max_drawdown": 0,
            "total_matches": len(hist), "skipped_matches": skipped,
        }

    wins = sum(1 for h in bets if (h.get("pnl") or 0) > 0)
    total_pnl = sum(h.get("pnl", 0) for h in bets)

    bankroll_trace = [250.0]
    for h in bets:
        bankroll_trace.append(bankroll_trace[-1] + (h.get("pnl") or 0))
    peak = bankroll_trace[0]
    max_dd = 0
    for val in bankroll_trace:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "total_return_pct": round(total_pnl / 250 * 100, 2),
        "win_rate": round(wins / len(bets) * 100, 1),
        "total_bets": len(bets),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd * 100, 1),
        "total_matches": len(hist),
        "skipped_matches": skipped,
    }


def start_server(port: int = 8050):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
