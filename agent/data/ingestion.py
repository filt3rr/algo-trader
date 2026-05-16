"""OHLCV data ingestion from Alpaca.

Fetches historical bars and latest quotes for the configured trading universe.
Results are returned as pandas DataFrames with a standardised schema.

DataFrame column schema (OHLCV):
    timestamp (index, UTC)  — bar open time
    open, high, low, close  — float64, USD
    volume                  — float64
    trade_count             — int64 (number of trades in bar)
    vwap                    — float64
    symbol                  — str (category)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.config import Settings, get_settings

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

OHLCV_COLS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

_TF_MAP: dict[str, TimeFrame] = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class MarketDataClient:
    """Wraps Alpaca's CryptoHistoricalDataClient with retry logic and DataFrame output."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._client = CryptoHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
        )
        self._universe = cfg.universe
        log.info("MarketDataClient initialised, universe=%s", self._universe)

    # ── Bars ─────────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def fetch_bars(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for one or more symbols.

        Returns a DataFrame indexed by (symbol, timestamp) with OHLCV_COLS.
        """
        syms = symbols or self._universe
        tf = _TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe {timeframe!r}. Valid: {list(_TF_MAP)}")

        now = datetime.now(tz=timezone.utc)
        if end is None:
            end = now
        if start is None:
            start = end - timedelta(days=30)

        req = CryptoBarsRequest(
            symbol_or_symbols=syms,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )
        raw = self._client.get_crypto_bars(req)
        df = raw.df

        if df is None or df.empty:
            log.warning("No bars returned for %s [%s, %s]", syms, start, end)
            return pd.DataFrame(columns=["symbol"] + OHLCV_COLS)

        df = df.reset_index()
        df = df.rename(columns={"trade_count": "trade_count"})

        # Normalise symbol format: BTCUSD → BTC/USD
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].apply(_normalise_symbol)

        # Ensure UTC index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index(["symbol", "timestamp"]).sort_index()
        elif df.index.names and "timestamp" in (df.index.names or []):
            df.index = df.index.set_levels(
                pd.to_datetime(df.index.get_level_values("timestamp"), utc=True),
                level="timestamp",
            )

        # Cast dtypes
        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        if "trade_count" in df.columns:
            df["trade_count"] = df["trade_count"].fillna(0).astype("int64")

        log.info(
            "Fetched %d bars | symbols=%s | tf=%s | range=[%s, %s]",
            len(df), syms, timeframe, start.date(), end.date(),
        )
        return df

    def fetch_bars_single(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Convenience wrapper for a single symbol — returns flat (timestamp-indexed) df."""
        df = self.fetch_bars([symbol], timeframe, start, end, limit)
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        return df

    # ── Latest quotes ─────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def fetch_latest_prices(
        self, symbols: list[str] | None = None
    ) -> dict[str, Decimal]:
        """Return the latest mid-price for each symbol."""
        syms = symbols or self._universe
        req = CryptoLatestQuoteRequest(symbol_or_symbols=syms)
        quotes = self._client.get_crypto_latest_quote(req)

        prices: dict[str, Decimal] = {}
        for sym, quote in quotes.items():
            mid = (Decimal(str(quote.ask_price)) + Decimal(str(quote.bid_price))) / 2
            prices[_normalise_symbol(sym)] = mid

        log.debug("Latest prices: %s", {k: float(v) for k, v in prices.items()})
        return prices

    @property
    def universe(self) -> list[str]:
        return self._universe


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_symbol(raw: str) -> str:
    """Convert BTCUSD → BTC/USD. No-op if already has slash."""
    raw = raw.strip()
    if "/" in raw:
        return raw
    # Assume USD quote currency (3 chars)
    if len(raw) > 3 and raw.endswith("USD"):
        return raw[:-3] + "/USD"
    if len(raw) > 4 and raw.endswith("USDT"):
        return raw[:-4] + "/USDT"
    return raw
