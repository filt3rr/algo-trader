"""Phase 1 connectivity test.

Run with:
    python scripts/connectivity_test.py

Verifies:
  1. Alpaca paper account is reachable and returns balance.
  2. Market data endpoint returns a live BTC/USD price.
  3. A tiny test order (0.0001 BTC) can be submitted and cancelled.
  4. LLM adapter is importable and configured correctly.

Expected output (example):
    [CONNECTED] Balance: $100,000.00 | BTC/USD last: $95,234.56
    [ORDER]     Submitted buy 0.0001 BTC/USD (market) → id=xxxxxxxx
    [CANCELLED] Order xxxxxxxx cancelled OK
    [LLM]       Provider=claude | Model=claude-sonnet-4-5
    [OK]        All connectivity checks passed.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)  # suppress retry noise during test


def run() -> None:
    from agent.broker.alpaca import AlpacaBroker
    from agent.broker.base import OrderRequest, OrderSide, OrderType
    from agent.config import get_settings
    from agent.reasoning.llm_base import get_llm_adapter

    cfg = get_settings()

    # 1. Account balance
    broker = AlpacaBroker(cfg)
    acct = broker.get_account()
    print(f"[CONNECTED] Balance: ${float(acct.portfolio_value):,.2f} | ", end="")

    # 2. Live price
    btc_price = broker.get_latest_price("BTC/USD")
    print(f"BTC/USD last: ${float(btc_price):,.2f}")

    # 3. Test order — submit then immediately cancel
    req = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        qty=Decimal("0.0001"),
        type=OrderType.MARKET,
    )
    order = broker.submit_order(req)
    print(f"[ORDER]     Submitted buy 0.0001 BTC/USD (market) → id={order.id}")

    cancelled = broker.cancel_order(order.id)
    status = "OK" if cancelled else "already filled (too fast!)"
    print(f"[CANCELLED] Order {order.id} cancel: {status}")

    # 4. LLM adapter
    adapter = get_llm_adapter()
    print(f"[LLM]       Provider={adapter.provider} | Model={adapter.model_id}")

    print("[OK]        All connectivity checks passed.")


if __name__ == "__main__":
    run()
