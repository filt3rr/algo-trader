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

    # 3. Test order — limit buy at 50% of market price (will never fill; safe to cancel)
    #    Alpaca requires cost_basis >= $10, so 0.001 BTC at ~50% price is well over that.
    limit_price = (btc_price * Decimal("0.5")).quantize(Decimal("0.01"))
    req = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        qty=Decimal("0.001"),
        type=OrderType.LIMIT,
        limit_price=limit_price,
    )
    order = broker.submit_order(req)
    print(f"[ORDER]     Submitted limit buy 0.001 BTC/USD @ ${float(limit_price):,.2f} -> id={order.id}")

    cancelled = broker.cancel_order(order.id)
    status = "OK" if cancelled else "already filled (unexpected!)"
    print(f"[CANCELLED] Order {order.id} cancel: {status}")

    # 4. LLM adapter
    adapter = get_llm_adapter()
    print(f"[LLM]       Provider={adapter.provider} | Model={adapter.model_id}")

    print("[OK]        All connectivity checks passed.")


if __name__ == "__main__":
    run()
