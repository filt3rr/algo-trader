"""Dynamic universe screener — CoinGecko ranking + Alpaca tradeable validation.

Flow each refresh:
    1. Fetch Alpaca's live tradeable crypto assets (cached per instance)
    2. Fetch CoinGecko top-250 coins ordered by market cap
    3. Map CoinGecko tickers → Alpaca format (e.g. BTC → BTC/USD)
    4. Filter to the Alpaca tradeable set
    5. Return top UNIVERSE_SIZE symbols, BTC/USD always pinned first

Refresh cadence: startup + once daily (scheduled in main.py).
CoinGecko free-tier rate limit: ~30 calls/min, ~10k calls/month.
Daily refresh consumes 2 calls (assets + coins/markets) — well within quota.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent.config import get_settings

log = logging.getLogger(__name__)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_ALPACA_ASSETS_PATH = "/v2/assets"

# BTC is always first — the main loop feeds its composite signal as macro
# context to every subsequent coin decision in the same cycle.
_PINNED = "BTC/USD"

# CoinGecko IDs of stablecoins and non-tradeable pegged assets.
# These always generate flat signals and would waste every LLM call.
_STABLECOIN_IDS: frozenset[str] = frozenset({
    # USD stablecoins
    "tether", "usd-coin", "binance-usd", "dai", "frax", "true-usd",
    "pax-dollar", "gemini-dollar", "liquity-usd", "neutrino", "usdd",
    "usdg", "paypal-usd", "first-digital-usd", "ethena-usde",
    "usual-usd", "mountain-protocol-usdm", "angle-protocol",
    "crvusd", "rai", "fei-usd", "mim", "bean", "terra-usd",
    # EUR / other fiat stablecoins
    "stasis-eurs", "tether-eurt", "ageur",
    # Wrapped BTC — tracks BTC, redundant signal
    "wrapped-bitcoin",
})

# CoinGecko coin IDs that don't map cleanly to symbol + /USD on Alpaca.
# Key: CoinGecko coin id (lowercase)  Value: Alpaca ticker prefix
_ID_OVERRIDES: dict[str, str] = {
    "matic-network": "MATIC",
    "shiba-inu": "SHIB",
    "avalanche-2": "AVAX",
    "binancecoin": "BNB",
    "uniswap": "UNI",
    "chainlink": "LINK",
    "aave": "AAVE",
    "the-graph": "GRT",
    "decentraland": "MANA",
    "the-sandbox": "SAND",
    "basic-attention-token": "BAT",
    "yearn-finance": "YFI",
    "compound-governance-token": "COMP",
    "maker": "MKR",
    "sushi": "SUSHI",
    "curve-dao-token": "CRV",
    "0x": "ZRX",
    "loopring": "LRC",
    "enjincoin": "ENJ",
    "chiliz": "CHZ",
    "ankr": "ANKR",
    "gala": "GALA",
}


class UniverseScreener:
    """Screens and ranks the tradeable crypto universe using CoinGecko data.

    Args:
        api_key: CoinGecko API key. Falls back to settings if omitted.
    """

    def __init__(self, api_key: str | None = None) -> None:
        cfg = get_settings()
        self._api_key = api_key or cfg.coingecko_api_key
        self._universe_size = cfg.universe_size
        self._alpaca_key = cfg.alpaca_api_key
        self._alpaca_secret = cfg.alpaca_secret_key
        self._alpaca_base = cfg.alpaca_base_url.rstrip("/")
        # Cache Alpaca tradeable set — changes infrequently (new listings/delistings)
        self._tradeable_cache: set[str] | None = None

    async def screen(self, hold_symbols: list[str] | None = None) -> list[str]:
        """Return ranked tradeable symbols, up to universe_size.

        BTC/USD is always first. Any symbol in hold_symbols (open position)
        is retained even if it falls outside the top-N, so open positions
        are never orphaned by a universe rotation.

        Args:
            hold_symbols: Alpaca symbols currently held — preserved in output.

        Returns:
            Ordered list of Alpaca symbol strings, e.g. ['BTC/USD', 'ETH/USD', ...].
            Returns [] on network failure (caller keeps the previous universe).
        """
        try:
            tradeable = await self._get_alpaca_tradeable()
            candidates = await self._get_coingecko_top(n=250)
        except httpx.HTTPError as exc:
            log.error("Universe screener HTTP error: %s", exc)
            return []
        except Exception as exc:
            log.error("Universe screener unexpected error: %s", exc)
            return []

        ranked: list[str] = []
        for coin in candidates:
            cg_id = (coin.get("id") or "").lower()
            if cg_id in _STABLECOIN_IDS:
                continue
            sym = self._resolve_symbol(coin)
            if sym and sym in tradeable and sym not in ranked:
                ranked.append(sym)
            if len(ranked) >= self._universe_size:
                break

        # Pin BTC at position 0
        if _PINNED in ranked:
            ranked.remove(_PINNED)
        if _PINNED in tradeable:
            ranked.insert(0, _PINNED)

        # Preserve open positions even if they fell out of top-N
        if hold_symbols:
            for sym in hold_symbols:
                if sym in tradeable and sym not in ranked:
                    ranked.append(sym)
                    log.info("Retained held position in universe: %s", sym)

        log.info(
            "Universe screened: %d symbols selected (Alpaca pool: %d)",
            len(ranked), len(tradeable),
        )
        return ranked

    async def _get_alpaca_tradeable(self) -> set[str]:
        """Return Alpaca-tradeable active crypto symbols (e.g. 'BTC/USD').

        Result is cached — Alpaca's asset list changes only on new listings or
        delistings, so re-fetching every call is wasteful.
        """
        if self._tradeable_cache is not None:
            return self._tradeable_cache

        url = f"{self._alpaca_base}{_ALPACA_ASSETS_PATH}"
        headers = {
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret,
        }
        params = {"asset_class": "crypto", "status": "active"}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            assets: list[dict[str, Any]] = resp.json()

        symbols: set[str] = {
            a["symbol"]
            for a in assets
            if a.get("symbol") and a.get("tradable", False)
        }
        self._tradeable_cache = symbols
        log.info("Alpaca tradeable crypto pool: %d symbols", len(symbols))
        return symbols

    async def _get_coingecko_top(self, n: int = 250) -> list[dict[str, Any]]:
        """Fetch top N coins from CoinGecko ordered by market cap (descending)."""
        url = f"{_COINGECKO_BASE}/coins/markets"
        params: dict[str, Any] = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(n, 250),
            "page": 1,
            "sparkline": "false",
        }
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _resolve_symbol(coin: dict[str, Any]) -> str | None:
        """Map a CoinGecko coin record to an Alpaca symbol string.

        Resolution order:
            1. Known CoinGecko coin ID override (handles MATIC, SHIB, AVAX etc.)
            2. Uppercase CoinGecko ticker symbol + '/USD' (covers ~95% of majors)
        """
        cg_id = (coin.get("id") or "").lower()
        cg_sym = (coin.get("symbol") or "").upper()
        if not cg_sym:
            return None
        prefix = _ID_OVERRIDES.get(cg_id, cg_sym)
        return f"{prefix}/USD"
