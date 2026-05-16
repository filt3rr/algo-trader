"""Parquet-based OHLCV cache with staleness detection.

Layout on disk:
    data/parquet/{symbol_safe}/{timeframe}.parquet

Where symbol_safe replaces "/" with "_": BTC/USD → BTC_USD.

Strategy:
- On read: load existing Parquet (if any), check freshness.
- On write: append new rows, deduplicate by timestamp, sort, write back.
- Staleness: a cache hit is "fresh" if the newest bar is within 2× the
  timeframe duration of now (e.g., for 1h bars, fresh if newest bar < 2h ago).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agent.config import get_settings

log = logging.getLogger(__name__)

_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _symbol_dir(base: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return base / safe


def _cache_path(base: Path, symbol: str, timeframe: str) -> Path:
    return _symbol_dir(base, symbol) / f"{timeframe}.parquet"


class OHLCVCache:
    """Read/write OHLCV Parquet files with staleness checking."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        cfg = get_settings()
        self.base = Path(base_dir or cfg.parquet_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def is_fresh(self, symbol: str, timeframe: str) -> bool:
        """Return True if cached data exists and is within the staleness window."""
        path = _cache_path(self.base, symbol, timeframe)
        if not path.exists():
            return False

        df = self.load(symbol, timeframe)
        if df is None or df.empty:
            return False

        newest = df.index.max()
        if newest is pd.NaT or newest is None:
            return False

        newest_dt: datetime = newest.to_pydatetime()
        if newest_dt.tzinfo is None:
            newest_dt = newest_dt.replace(tzinfo=timezone.utc)

        tf_secs = _TIMEFRAME_SECONDS.get(timeframe, 3600)
        stale_after = timedelta(seconds=tf_secs * 2)
        age = datetime.now(tz=timezone.utc) - newest_dt

        fresh = age < stale_after
        if not fresh:
            log.debug("Cache stale for %s/%s (age=%s, limit=%s)", symbol, timeframe, age, stale_after)
        return fresh

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Load cached bars; returns None if not found."""
        path = _cache_path(self.base, symbol, timeframe)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            elif df.index.name == "timestamp":
                df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            log.debug("Cache hit %s/%s: %d rows", symbol, timeframe, len(df))
            return df
        except Exception as exc:
            log.warning("Failed to read cache %s: %s", path, exc)
            return None

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Append new bars to cache, deduplicating on timestamp index."""
        if df is None or df.empty:
            return

        path = _cache_path(self.base, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = self.load(symbol, timeframe)

        # Ensure incoming df has a proper DatetimeIndex named "timestamp"
        incoming = df.copy()
        if incoming.index.name != "timestamp" or not isinstance(
            incoming.index, pd.DatetimeIndex
        ):
            if "timestamp" in incoming.columns:
                incoming = incoming.set_index("timestamp")
            incoming.index = pd.to_datetime(incoming.index, utc=True)
            incoming.index.name = "timestamp"

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, incoming])
        else:
            combined = incoming

        # Ensure unified DatetimeIndex before dedup + sort
        combined.index = pd.to_datetime(combined.index, utc=True)
        combined.index.name = "timestamp"
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        # Reset index so timestamp is a column (more portable)
        out = combined.reset_index()
        if "timestamp" not in out.columns and combined.index.name == "timestamp":
            out = combined.reset_index()

        table = pa.Table.from_pandas(out)
        pq.write_table(table, path, compression="snappy")
        log.debug("Cache saved %s/%s: %d total rows", symbol, timeframe, len(combined))

    def load_range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        """Load a date-filtered slice from cache."""
        df = self.load(symbol, timeframe)
        if df is None or df.empty:
            return None
        start_ts = pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end)
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        sliced = df.loc[mask]
        return sliced if not sliced.empty else None

    def missing_range(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> tuple[datetime, datetime] | None:
        """Return the (start, end) gap that is not covered by the cache.

        Returns None if the cache fully covers the requested range.
        """
        df = self.load(symbol, timeframe)
        if df is None or df.empty:
            return start, end

        start_ts = pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end)
        cached_start = df.index.min()
        cached_end = df.index.max()

        if cached_start <= start_ts and cached_end >= end_ts:
            return None  # fully covered

        fetch_start = min(start_ts, cached_start).to_pydatetime()
        fetch_end = max(end_ts, cached_end).to_pydatetime()
        return fetch_start, fetch_end
