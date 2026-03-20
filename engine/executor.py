"""Authenticated py-clob-client order placement and cancellation."""

import os

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from engine.config import POLYMARKET_HOST, CHAIN_ID


_client: ClobClient | None = None


def init_client() -> ClobClient:
    """Create and return an authenticated ClobClient."""
    global _client
    key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if not key:
        raise RuntimeError("POLYMARKET_PRIVATE_KEY env var not set")

    _client = ClobClient(
        POLYMARKET_HOST,
        key=key,
        chain_id=CHAIN_ID,
    )
    _client.set_api_creds(_client.create_or_derive_api_creds())
    return _client


def _get_client() -> ClobClient:
    if _client is None:
        return init_client()
    return _client


def place_bet(signal: dict) -> str | None:
    """Place a limit buy order for the signal's token.

    Returns the order ID on success, None on failure.
    """
    client = _get_client()
    token_id = signal["token_id"]
    price = signal["market_price"]
    size = signal["contracts"]

    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=BUY,
    )

    try:
        signed = client.create_order(order_args)
        resp = client.post_order(signed, OrderType.GTC)
        return resp.get("orderID") or resp.get("id")
    except Exception as e:
        print(f"  [ORDER ERROR] place_bet failed: {e}")
        return None


def sell_position(token_id: str, size: float, price: float) -> str | None:
    """Sell a position at a given price (used for stop-loss and manual exit)."""
    client = _get_client()

    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side=SELL,
    )

    try:
        signed = client.create_order(order_args)
        resp = client.post_order(signed, OrderType.GTC)
        return resp.get("orderID") or resp.get("id")
    except Exception as e:
        print(f"  [ORDER ERROR] sell_position failed: {e}")
        return None


def cancel_order(order_id: str) -> bool:
    """Cancel an order by ID."""
    client = _get_client()
    try:
        client.cancel(order_id)
        return True
    except Exception as e:
        print(f"  [ORDER ERROR] cancel_order failed: {e}")
        return False
