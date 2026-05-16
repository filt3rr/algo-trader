"""Alpaca paper-trading broker adapter.

Uses alpaca-py (v0.38+). Crypto markets are always open, so we never gate on
market hours. All orders go to the paper endpoint.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide as AlpacaSide,
    OrderType as AlpacaOrderType,
    TimeInForce as AlpacaTIF,
)
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.broker.base import (
    AccountInfo,
    BrokerAdapter,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from agent.config import Settings, get_settings

log = logging.getLogger(__name__)


def _to_order_status(raw: str) -> OrderStatus:
    mapping = {
        "new": OrderStatus.PENDING,
        "accepted": OrderStatus.PENDING,
        "pending_new": OrderStatus.PENDING,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "cancelled": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
        "expired": OrderStatus.EXPIRED,
        "done_for_day": OrderStatus.EXPIRED,
    }
    return mapping.get(raw.lower(), OrderStatus.PENDING)


def _alpaca_side(side: OrderSide) -> AlpacaSide:
    return AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL


def _alpaca_tif(tif: TimeInForce) -> AlpacaTIF:
    return {
        TimeInForce.GTC: AlpacaTIF.GTC,
        TimeInForce.IOC: AlpacaTIF.IOC,
        TimeInForce.FOK: AlpacaTIF.FOK,
        TimeInForce.DAY: AlpacaTIF.DAY,
    }[tif]


class AlpacaBroker(BrokerAdapter):
    """Paper-trading adapter backed by Alpaca's paper endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()  # runs assert_paper_only()
        cfg = settings or get_settings()

        self._trading = TradingClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            paper=True,
            url_override=cfg.alpaca_base_url,
        )
        self._data = CryptoHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
        )
        log.info("AlpacaBroker initialised (paper=True, url=%s)", cfg.alpaca_base_url)

    # ── Account ──────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_account(self) -> AccountInfo:
        acct = self._trading.get_account()
        return AccountInfo(
            id=str(acct.id),
            cash=Decimal(str(acct.cash)),
            portfolio_value=Decimal(str(acct.portfolio_value)),
            equity=Decimal(str(acct.equity)),
            buying_power=Decimal(str(acct.buying_power)),
            currency=str(acct.currency),
            is_paper=True,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_positions(self) -> list[Position]:
        raw_positions = self._trading.get_all_positions()
        return [self._parse_position(p) for p in raw_positions]

    def get_position(self, symbol: str) -> Position | None:
        try:
            raw = self._trading.get_open_position(symbol.replace("/", ""))
            return self._parse_position(raw)
        except Exception:
            return None

    def _parse_position(self, raw: Any) -> Position:
        qty = Decimal(str(raw.qty))
        avg_entry = Decimal(str(raw.avg_entry_price))
        current = Decimal(str(raw.current_price))
        market_val = Decimal(str(raw.market_value))
        unrealized = Decimal(str(raw.unrealized_pl))
        unrealized_pct = float(raw.unrealized_plpc) if raw.unrealized_plpc else 0.0

        symbol = str(raw.symbol)
        if "/" not in symbol and len(symbol) > 3:
            symbol = symbol[:-3] + "/" + symbol[-3:]

        return Position(
            symbol=symbol,
            qty=qty,
            avg_entry_price=avg_entry,
            current_price=current,
            market_value=market_val,
            unrealized_pnl=unrealized,
            unrealized_pnl_pct=unrealized_pct,
        )

    # ── Orders ───────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def submit_order(self, request: OrderRequest) -> Order:
        # Alpaca crypto symbols use no slash: BTC/USD → BTCUSD
        symbol = request.symbol.replace("/", "")
        side = _alpaca_side(request.side)
        tif = _alpaca_tif(request.time_in_force)
        qty = float(request.qty)

        if request.type == OrderType.MARKET:
            alpaca_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
            )
        else:
            if request.limit_price is None:
                raise ValueError("limit_price required for LIMIT orders")
            alpaca_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
                limit_price=float(request.limit_price),
            )

        raw = self._trading.submit_order(alpaca_req)
        log.info(
            "Order submitted id=%s symbol=%s side=%s qty=%s",
            raw.id, request.symbol, request.side, request.qty,
        )
        return self._parse_order(raw, request.symbol)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            log.info("Order cancelled id=%s", order_id)
            return True
        except Exception as exc:
            log.warning("Could not cancel order id=%s: %s", order_id, exc)
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_order(self, order_id: str) -> Order:
        raw = self._trading.get_order_by_id(order_id)
        symbol = str(raw.symbol)
        if "/" not in symbol and len(symbol) > 3:
            symbol = symbol[:-3] + "/" + symbol[-3:]
        return self._parse_order(raw, symbol)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_open_orders(self) -> list[Order]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        raws = self._trading.get_orders(filter=req)
        orders = []
        for raw in raws:
            symbol = str(raw.symbol)
            if "/" not in symbol and len(symbol) > 3:
                symbol = symbol[:-3] + "/" + symbol[-3:]
            orders.append(self._parse_order(raw, symbol))
        return orders

    def _parse_order(self, raw: Any, symbol: str) -> Order:
        return Order(
            id=str(raw.id),
            symbol=symbol,
            side=OrderSide(str(raw.side.value).lower()),
            type=OrderType(str(raw.type.value).lower()),
            qty=Decimal(str(raw.qty or 0)),
            status=_to_order_status(str(raw.status.value)),
            submitted_at=raw.submitted_at,
            filled_qty=Decimal(str(raw.filled_qty or 0)),
            filled_avg_price=(
                Decimal(str(raw.filled_avg_price)) if raw.filled_avg_price else None
            ),
            limit_price=(
                Decimal(str(raw.limit_price)) if raw.limit_price else None
            ),
            client_order_id=str(raw.client_order_id or ""),
            raw=raw.model_dump() if hasattr(raw, "model_dump") else {},
        )

    # ── Market data ───────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_latest_price(self, symbol: str) -> Decimal:
        req = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = self._data.get_crypto_latest_quote(req)
        quote = quotes[symbol]
        mid = (Decimal(str(quote.ask_price)) + Decimal(str(quote.bid_price))) / 2
        return mid

    def is_market_open(self) -> bool:
        return True

    @property
    def is_paper(self) -> bool:
        return True
