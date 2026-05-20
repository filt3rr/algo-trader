"""Centralised settings loaded from .env.

All other modules import `get_settings()` — never read os.environ directly.

Note on pydantic-settings 2.7+:
  List fields are expected in JSON format in env files. To avoid forcing
  users to write JSON in .env, we store list-like settings as plain strings
  and expose them via properties. This is intentional — do not change
  `trading_universe` back to `list[str]`.
"""

from __future__ import annotations

import random
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    CLAUDE = "claude"
    LOCAL = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Alpaca ────────────────────────────────────────────────────────────────
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"

    # ── LLM ──────────────────────────────────────────────────────────────────
    # LLM_PROVIDER controls trading decisions. REFLECTION_LLM_PROVIDER controls
    # the nightly reflection call — defaults to the same value as LLM_PROVIDER
    # but can be set to "claude" to keep the complex reflection on the stronger
    # model while routing cheap per-symbol decisions to local inference.
    llm_provider: LLMProvider = LLMProvider.CLAUDE
    reflection_llm_provider: LLMProvider | None = None  # None = same as llm_provider
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    local_llm_base_url: str = "http://localhost:11434"
    local_llm_model: str = "qwen2.5:7b"
    # Maximum seconds to wait for Ollama to finish generating a response.
    # Increase if your GPU is slow or you switch to a larger model.
    local_llm_timeout: int = 120

    # ── Pre-filter ────────────────────────────────────────────────────────────
    # Skip the LLM entirely for symbols with no open position and a composite
    # signal weaker than this threshold. The LLM was going to hold anyway.
    # Set to 0.0 to disable (always call LLM).
    signal_prefilter_threshold: float = 0.2

    # ── Universe ──────────────────────────────────────────────────────────────
    # Kept as a plain str so pydantic-settings 2.7 does not attempt JSON-decode.
    # Use the `.universe` property for the parsed list.
    # When COINGECKO_API_KEY is set, trading_universe is used only as the
    # startup fallback — the live universe is managed by UniverseScreener.
    trading_universe: str = Field(
        default="BTC/USD,ETH/USD,SOL/USD,AVAX/USD,MATIC/USD,LINK/USD,DOT/USD,ADA/USD",
    )

    # ── CoinGecko ─────────────────────────────────────────────────────────────
    coingecko_api_key: str = ""
    # Max symbols in the dynamic universe (Alpaca crypto pool is ~30–40 total)
    universe_size: int = 20

    # ── Risk ─────────────────────────────────────────────────────────────────
    max_risk_per_trade_pct: float = 0.02
    max_position_pct: float = 0.10
    max_open_positions: int = 5
    daily_loss_limit_pct: float = 0.05

    # ── Scheduler ─────────────────────────────────────────────────────────────
    trading_loop_seconds: int = 3600
    reflection_hour: int = 23

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str = "data/trader.db"
    parquet_dir: str = "data/parquet"
    log_level: str = "INFO"

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000

    # ── Feature flags ─────────────────────────────────────────────────────────
    live_trading_enabled: bool = False
    dry_run: bool = True   # True = log decisions, no real orders submitted

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("alpaca_base_url", mode="before")
    @classmethod
    def normalise_base_url(cls, v: str) -> str:
        """alpaca-py appends /v2 internally — strip it if the user included it."""
        return v.rstrip("/").removesuffix("/v2")

    @model_validator(mode="after")
    def validate_live_trading(self) -> "Settings":
        """Live trading requires an explicit secret env var — not just the flag."""
        if self.live_trading_enabled:
            raise ValueError(
                "LIVE_TRADING_ENABLED=True is blocked at the Settings level. "
                "See agent/broker/base.py for the multi-step unlock process."
            )
        return self

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def universe(self) -> list[str]:
        """Parsed trading universe as a list of symbol strings."""
        return [s.strip() for s in self.trading_universe.split(",") if s.strip()]

    def seed_everything(self) -> None:
        """Seed Python random and numpy for reproducible runs."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    @property
    def db_url(self) -> str:
        path = Path(self.db_path)
        if not path.is_absolute():
            # Resolve relative paths from the project root (parent of agent/)
            path = Path(__file__).parent.parent / path
        return f"sqlite+aiosqlite:///{path}"

    @property
    def parquet_path(self) -> Path:
        return Path(self.parquet_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance (cached after first call)."""
    s = Settings()
    s.seed_everything()
    return s
