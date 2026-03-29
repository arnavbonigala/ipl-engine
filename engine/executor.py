"""Kalshi API authentication and order placement."""

import base64
import datetime
import os
import uuid

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from engine.config import KALSHI_API_BASE

_private_key = None
_api_key_id = None


def _load_credentials():
    global _private_key, _api_key_id
    if _private_key is not None:
        return

    _api_key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    if not _api_key_id or not key_path:
        raise RuntimeError(
            "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars required"
        )

    with open(key_path, "rb") as f:
        _private_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )


def _sign(timestamp: str, method: str, path: str) -> str:
    path_clean = path.split("?")[0]
    message = f"{timestamp}{method}{path_clean}".encode("utf-8")
    sig = _private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def _headers(method: str, path: str) -> dict:
    _load_credentials()
    ts = str(int(datetime.datetime.now().timestamp() * 1000))
    return {
        "KALSHI-ACCESS-KEY": _api_key_id,
        "KALSHI-ACCESS-SIGNATURE": _sign(ts, method, path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


def _auth_get(path: str):
    full_path = f"/trade-api/v2{path}"
    resp = requests.get(
        f"{KALSHI_API_BASE}{path}",
        headers=_headers("GET", full_path),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _auth_post(path: str, data: dict):
    full_path = f"/trade-api/v2{path}"
    resp = requests.post(
        f"{KALSHI_API_BASE}{path}",
        headers=_headers("POST", full_path),
        json=data,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _auth_delete(path: str):
    full_path = f"/trade-api/v2{path}"
    resp = requests.delete(
        f"{KALSHI_API_BASE}{path}",
        headers=_headers("DELETE", full_path),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_balance() -> float:
    """Return account balance in dollars."""
    data = _auth_get("/portfolio/balance")
    return data.get("balance", 0) / 100.0


def place_bet(signal: dict) -> str | None:
    """Place a limit buy YES order on Kalshi. Returns the order_id or None."""
    ticker = signal["ticker"]
    price = signal["market_price"]
    count = max(int(signal["contracts"]), 1)

    order_data = {
        "ticker": ticker,
        "action": "buy",
        "side": "yes",
        "count": count,
        "type": "limit",
        "yes_price_dollars": f"{price:.4f}",
        "client_order_id": str(uuid.uuid4()),
    }

    try:
        resp = _auth_post("/portfolio/orders", order_data)
        return resp.get("order", {}).get("order_id")
    except Exception as e:
        print(f"  [ORDER ERROR] place_bet failed: {e}")
        return None


def sell_position(ticker: str, count: int, price: float) -> str | None:
    """Sell a YES position at given price."""
    order_data = {
        "ticker": ticker,
        "action": "sell",
        "side": "yes",
        "count": count,
        "type": "limit",
        "yes_price_dollars": f"{price:.4f}",
        "client_order_id": str(uuid.uuid4()),
    }

    try:
        resp = _auth_post("/portfolio/orders", order_data)
        return resp.get("order", {}).get("order_id")
    except Exception as e:
        print(f"  [ORDER ERROR] sell_position failed: {e}")
        return None


def cancel_order(order_id: str) -> bool:
    try:
        _auth_delete(f"/portfolio/orders/{order_id}")
        return True
    except Exception as e:
        print(f"  [ORDER ERROR] cancel_order failed: {e}")
        return False
