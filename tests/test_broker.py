"""Tests for agent/broker — base guard and Alpaca adapter (mocked)."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from agent.broker.base import (
    AccountInfo,
    BrokerAdapter,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    assert_paper_only,
)


# ── assert_paper_only ─────────────────────────────────────────────────────────

def test_paper_only_passes_when_live_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "False")
    assert_paper_only()  # must not raise


def test_paper_only_passes_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    assert_paper_only()


def test_paper_only_raises_when_enabled_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.delenv("LIVE_TRADING_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="LIVE_TRADING_ENABLED=True"):
        assert_paper_only()


def test_paper_only_raises_when_enabled_with_wrong_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("LIVE_TRADING_SECRET", "wrong_value")
    with pytest.raises(RuntimeError, match="LIVE_TRADING_ENABLED=True"):
        assert_paper_only()


def test_paper_only_warns_when_fully_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("LIVE_TRADING_SECRET", "I_ACCEPT_ALL_FINANCIAL_RISK")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_paper_only()
        assert len(w) == 1
        assert "LIVE TRADING IS ENABLED" in str(w[0].message)


# ── AlpacaBroker (mocked) ─────────────────────────────────────────────────────

@pytest.fixture
def mock_trading_client():
    with patch("agent.broker.alpaca.TradingClient") as mock:
        yield mock


@pytest.fixture
def mock_data_client():
    with patch("agent.broker.alpaca.CryptoHistoricalDataClient") as mock:
        yield mock


@pytest.fixture
def broker(mock_trading_client, mock_data_client, monkeypatch: pytest.MonkeyPatch):
    from agent.broker.alpaca import AlpacaBroker
    from agent.config import Settings

    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    cfg = Settings()
    return AlpacaBroker(settings=cfg)


def test_broker_is_paper(broker) -> None:
    assert broker.is_paper is True


def test_broker_market_always_open(broker) -> None:
    assert broker.is_market_open() is True


def test_get_account_parses_response(broker, mock_trading_client) -> None:
    raw_acct = MagicMock()
    raw_acct.id = "test-account-id"
    raw_acct.cash = "95000.00"
    raw_acct.portfolio_value = "100000.00"
    raw_acct.equity = "100000.00"
    raw_acct.buying_power = "190000.00"
    raw_acct.currency = "USD"

    mock_trading_client.return_value.get_account.return_value = raw_acct

    acct = broker.get_account()

    assert isinstance(acct, AccountInfo)
    assert acct.cash == Decimal("95000.00")
    assert acct.portfolio_value == Decimal("100000.00")
    assert acct.is_paper is True
    assert acct.currency == "USD"


def test_get_positions_returns_list(broker, mock_trading_client) -> None:
    raw_pos = MagicMock()
    raw_pos.symbol = "BTCUSD"
    raw_pos.qty = "0.5"
    raw_pos.avg_entry_price = "45000.00"
    raw_pos.current_price = "47000.00"
    raw_pos.market_value = "23500.00"
    raw_pos.unrealized_pl = "1000.00"
    raw_pos.unrealized_plpc = "0.04255"

    mock_trading_client.return_value.get_all_positions.return_value = [raw_pos]

    positions = broker.get_positions()

    assert len(positions) == 1
    p = positions[0]
    assert isinstance(p, Position)
    assert p.symbol == "BTC/USD"
    assert p.qty == Decimal("0.5")
    assert p.avg_entry_price == Decimal("45000.00")


def test_submit_market_order(broker, mock_trading_client) -> None:
    from datetime import datetime, timezone

    raw_order = MagicMock()
    raw_order.id = "order-123"
    raw_order.symbol = "BTCUSD"
    raw_order.side.value = "buy"
    raw_order.type.value = "market"
    raw_order.qty = "0.01"
    raw_order.status.value = "new"
    raw_order.submitted_at = datetime.now(tz=timezone.utc)
    raw_order.filled_qty = "0"
    raw_order.filled_avg_price = None
    raw_order.limit_price = None
    raw_order.client_order_id = ""
    raw_order.model_dump.return_value = {}

    mock_trading_client.return_value.submit_order.return_value = raw_order

    req = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        type=OrderType.MARKET,
    )
    order = broker.submit_order(req)

    assert isinstance(order, Order)
    assert order.id == "order-123"
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.PENDING


def test_cancel_order_returns_true_on_success(broker, mock_trading_client) -> None:
    mock_trading_client.return_value.cancel_order_by_id.return_value = None
    result = broker.cancel_order("order-123")
    assert result is True


def test_cancel_order_returns_false_on_exception(broker, mock_trading_client) -> None:
    mock_trading_client.return_value.cancel_order_by_id.side_effect = Exception("not found")
    result = broker.cancel_order("nonexistent")
    assert result is False


def test_get_latest_price(broker, mock_data_client) -> None:
    mock_quote = MagicMock()
    mock_quote.ask_price = "47100.00"
    mock_quote.bid_price = "46900.00"

    mock_data_client.return_value.get_crypto_latest_quote.return_value = {
        "BTC/USD": mock_quote
    }

    price = broker.get_latest_price("BTC/USD")
    assert price == Decimal("47000.00")
