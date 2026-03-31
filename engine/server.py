"""FastAPI app serving REST API + static dashboard."""

from dotenv import load_dotenv
load_dotenv()

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from engine.state import load_state

app = FastAPI(title="IPL Betting Engine")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Background caches -- polled by dedicated threads so /api/live returns instantly
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_live_scores = []          # from cricdata
_live_scores_ts = 0.0
_kalshi_odds = {}           # ticker -> {bid, ask, last, vol_24h, status, result}
_kalshi_odds_ts = 0.0
_kalshi_markets = []        # from find_ipl_markets
_kalshi_markets_ts = 0.0

SCORE_POLL_SECS = 3
ODDS_POLL_SECS = 3
MARKET_DISCOVERY_SECS = 60


def _poll_scores():
    global _live_scores, _live_scores_ts
    while True:
        try:
            from engine.scraper import get_live_scores
            data = get_live_scores()
            with _cache_lock:
                _live_scores = data
                _live_scores_ts = time.time()
        except Exception:
            pass
        time.sleep(SCORE_POLL_SECS)


def _poll_odds():
    global _kalshi_odds, _kalshi_odds_ts
    while True:
        try:
            from engine.market import get_market_data
            with _cache_lock:
                tickers = set()
                for m in _kalshi_markets:
                    tickers.add(m["t1_ticker"])
                    tickers.add(m["t2_ticker"])

            new_odds = {}
            for ticker in tickers:
                mkt = get_market_data(ticker)
                if mkt:
                    new_odds[ticker] = {
                        "bid": float(mkt.get("yes_bid_dollars", 0)),
                        "ask": float(mkt.get("yes_ask_dollars", 0)),
                        "last": float(mkt.get("last_price_dollars", 0)),
                        "vol_24h": mkt.get("volume_24h_fp", "0"),
                        "status": mkt.get("status", ""),
                        "result": mkt.get("result", ""),
                    }

            with _cache_lock:
                _kalshi_odds = new_odds
                _kalshi_odds_ts = time.time()
        except Exception:
            pass
        time.sleep(ODDS_POLL_SECS)


def _poll_markets():
    global _kalshi_markets, _kalshi_markets_ts
    while True:
        try:
            from engine.market import find_ipl_markets
            data = find_ipl_markets()
            with _cache_lock:
                _kalshi_markets = data
                _kalshi_markets_ts = time.time()
        except Exception:
            pass
        time.sleep(MARKET_DISCOVERY_SECS)


@app.on_event("startup")
def _start_pollers():
    for fn in (_poll_scores, _poll_odds, _poll_markets):
        t = threading.Thread(target=fn, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (_STATIC_DIR / "index.html").read_text()


@app.get("/api/status")
def status():
    s = load_state()
    open_count = sum(1 for p in s.get("positions", []) if p.get("status") == "open")

    kalshi_balance = None
    try:
        from engine.executor import get_balance
        kalshi_balance = get_balance()
    except Exception:
        pass

    key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    masked = f"{key_id[:8]}...{key_id[-4:]}" if len(key_id) > 12 else key_id

    return {
        "balance": kalshi_balance,
        "exchange": "kalshi",
        "api_key": masked,
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
    stored = {m.get("event_ticker"): m for m in s.get("upcoming", []) if m.get("event_ticker")}
    history_tickers = {h.get("event_ticker") for h in s.get("history", []) if h.get("event_ticker")}

    try:
        from engine.scraper import get_ipl_fixtures
        fixtures = get_ipl_fixtures("ipl-2026-1510719")
    except Exception:
        fixtures = []

    results = []
    seen_teams = set()
    for f in fixtures:
        if f["status"] not in ("PRE", "LIVE"):
            continue
        key = frozenset((f["team1"], f["team2"], f["date"]))
        if key in seen_teams:
            continue
        seen_teams.add(key)

        entry = {
            "match_date": f["date"],
            "team1": f["team1"],
            "team2": f["team2"],
            "status": "awaiting_toss",
            "model_prediction": None,
        }
        for st in stored.values():
            if ({st.get("team1"), st.get("team2")} == {f["team1"], f["team2"]}
                    and st.get("match_date") == f["date"]):
                if st.get("event_ticker") in history_tickers:
                    entry = None
                    break
                entry.update({k: v for k, v in st.items() if v is not None})
                break
        if entry:
            results.append(entry)

    return sorted(results, key=lambda m: m.get("match_date", ""))[:15]


@app.get("/api/events")
def events():
    s = load_state()
    return list(reversed(s.get("events", [])))[:200]


@app.get("/api/stats")
def stats():
    s = load_state()
    hist = s.get("history", [])
    bets = [h for h in hist if not (h.get("status", "").startswith("skipped") or h.get("status") == "no_market")]
    skipped = len(hist) - len(bets)
    if not bets:
        return {
            "total_return_pct": 0, "win_rate": 0, "total_bets": 0,
            "total_pnl": 0, "max_drawdown": 0,
            "total_matches": len(hist), "skipped_matches": skipped,
        }

    wins = sum(1 for h in bets if (h.get("pnl") or 0) > 0)
    total_pnl = sum(h.get("pnl", 0) for h in bets)

    start = s.get("starting_bankroll")
    if not start:
        try:
            from engine.executor import get_balance
            start = get_balance() - total_pnl
        except Exception:
            start = 25.0
    bankroll_trace = [start]
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
        "total_return_pct": round(total_pnl / start * 100, 2),
        "win_rate": round(wins / len(bets) * 100, 1),
        "total_bets": len(bets),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd * 100, 1),
        "total_matches": len(hist),
        "skipped_matches": skipped,
    }


@app.get("/api/trades/{ticker}")
def trades(ticker: str, limit: int = 500):
    """Return recent Kalshi trades for a market ticker (for the chance chart)."""
    import requests as _req

    from engine.config import KALSHI_API_BASE

    url = f"{KALSHI_API_BASE}/markets/trades"
    all_trades = []
    cursor = None
    remaining = limit
    while remaining > 0:
        params = {"ticker": ticker, "limit": min(remaining, 200)}
        if cursor:
            params["cursor"] = cursor
        try:
            r = _req.get(url, params=params, timeout=5)
            if r.status_code != 200:
                break
            body = r.json()
        except Exception:
            break
        batch = body.get("trades", [])
        if not batch:
            break
        all_trades.extend(batch)
        remaining -= len(batch)
        cursor = body.get("cursor")
        if not cursor:
            break

    return [
        {
            "ts": t["created_time"],
            "price": float(t["yes_price_dollars"]),
        }
        for t in reversed(all_trades)
    ]


@app.get("/api/live")
def live():
    """Live match data from background caches -- returns instantly."""
    with _cache_lock:
        scores = list(_live_scores)
        odds = dict(_kalshi_odds)
        markets = list(_kalshi_markets)
        scores_age = time.time() - _live_scores_ts if _live_scores_ts else None
        odds_age = time.time() - _kalshi_odds_ts if _kalshi_odds_ts else None

    s = load_state()
    open_positions = [p for p in s.get("positions", []) if p.get("status") == "open"]

    results = []
    for sc in scores:
        if sc.get("match_state") in ("POST", "COMPLETE", "RESULT"):
            continue
        t1, t2 = sc["team1"], sc["team2"]

        km = None
        for m in markets:
            if {m["team1"], m["team2"]} == {t1, t2}:
                km = m
                break

        t1_odds = t2_odds = t1_vol = t2_vol = None
        t1_ticker = t2_ticker = None
        if km:
            team_to_ticker = {km["team1"]: km["t1_ticker"], km["team2"]: km["t2_ticker"]}
            t1_ticker = team_to_ticker.get(t1)
            t2_ticker = team_to_ticker.get(t2)

            t1_data = odds.get(t1_ticker)
            t2_data = odds.get(t2_ticker)

            if t1_data and t1_data["bid"] > 0:
                t1_odds = round((t1_data["bid"] + t1_data["ask"]) / 2, 4)
                t1_vol = t1_data["vol_24h"]
            if t2_data and t2_data["bid"] > 0:
                t2_odds = round((t2_data["bid"] + t2_data["ask"]) / 2, 4)
                t2_vol = t2_data["vol_24h"]

        active_bet = None
        for p in open_positions:
            if {p.get("team1"), p.get("team2")} == {t1, t2}:
                cur_price = None
                ticker = p.get("ticker", "")
                od = odds.get(ticker)
                if od and od["bid"] > 0:
                    cur_price = round((od["bid"] + od["ask"]) / 2, 4)
                unrealized = 0
                if cur_price:
                    unrealized = round((cur_price - p["entry_price"]) * p["contracts"], 2)
                active_bet = {
                    "team": p["team"],
                    "side": p["side"],
                    "entry_price": p["entry_price"],
                    "current_price": cur_price,
                    "contracts": p["contracts"],
                    "bet_amount": p["bet_amount"],
                    "unrealized_pnl": unrealized,
                    "model_prob": p.get("model_prob"),
                    "edge": p.get("edge"),
                }
                break

        results.append({
            **sc,
            "t1_kalshi_odds": t1_odds,
            "t2_kalshi_odds": t2_odds,
            "t1_volume_24h": t1_vol,
            "t2_volume_24h": t2_vol,
            "t1_ticker": t1_ticker,
            "t2_ticker": t2_ticker,
            "event_ticker": km["event_ticker"] if km else None,
            "active_bet": active_bet,
            "scores_age_secs": round(scores_age, 1) if scores_age else None,
            "odds_age_secs": round(odds_age, 1) if odds_age else None,
        })

    return results


def start_server(port: int = 8050):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
