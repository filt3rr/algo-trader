"""Pydantic schemas for the reasoning layer — inputs to and outputs from the LLM.

DecisionInput  — assembled by ReasoningAgent from live market + account state.
TradingDecision — validated JSON response produced by the LLM.

The LLM is instructed to return a single JSON object matching TradingDecision.
The schema is embedded verbatim in the prompt so the model knows the contract.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Sub-models used in DecisionInput ─────────────────────────────────────────

class SignalSnapshot(BaseModel):
    """Condensed signal state for a single symbol, passed to the LLM."""
    symbol: str
    direction: Literal["buy", "sell", "hold"]
    composite: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    ema_score: float = Field(ge=-1.0, le=1.0)
    rsi_score: float = Field(ge=-1.0, le=1.0)
    roc_score: float = Field(ge=-1.0, le=1.0)
    regime: str
    close_price: float
    signal_weights: dict[str, float]
    volume_mult: float = Field(default=1.0, ge=0.0, le=1.0)   # volume confirmation [0.4, 1.0]
    signal_agreement: float = Field(default=1.0, ge=0.0, le=1.0)  # sub-signal alignment [0, 1]


class PositionSnapshot(BaseModel):
    """Current open position summary."""
    symbol: str
    qty: str              # Decimal serialised to string
    market_value: str     # Decimal serialised to string
    unrealized_pnl: str   # Decimal serialised to string
    unrealized_pnl_pct: float
    concentration_pct: float  # position_value / portfolio_value


class AccountSnapshot(BaseModel):
    """Portfolio-level account state."""
    portfolio_value: str   # Decimal → string
    cash: str              # Decimal → string
    cash_pct: float        # cash / portfolio_value
    daily_pnl_pct: float   # today's return so far
    open_position_count: int
    circuit_breaker_active: bool


class TimeframeSignal(BaseModel):
    """Signal state for one supplementary timeframe (15m or 4h), for MTF context."""
    timeframe: str                              # "15m" or "4h"
    direction: Literal["buy", "sell", "hold"]
    composite: float = Field(ge=-1.0, le=1.0)
    regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    volume_mult: float = Field(ge=0.0, le=1.0)
    signal_agreement: float = Field(ge=0.0, le=1.0)


class RecentLesson(BaseModel):
    """A past lesson retrieved from the reflection store."""
    lesson: str
    source: str           # 'reflection' or 'manual'
    created_at: datetime


class RecentDecision(BaseModel):
    """A summary of a recent LLM decision (for short-term context)."""
    symbol: str
    action: Literal["buy", "sell", "hold"]
    rationale_summary: str  # first 200 chars of rationale
    confidence: float
    outcome_pnl_pct: float | None  # None until the trade closes
    decided_at: datetime


# ── DecisionInput ─────────────────────────────────────────────────────────────

class DecisionInput(BaseModel):
    """Full context assembled by ReasoningAgent and fed to the LLM."""

    timestamp: datetime
    symbol: str
    signal: SignalSnapshot
    account: AccountSnapshot
    open_positions: list[PositionSnapshot] = Field(default_factory=list)
    recent_lessons: list[RecentLesson] = Field(default_factory=list)
    recent_decisions: list[RecentDecision] = Field(default_factory=list)
    dry_run: bool = True
    btc_composite: float | None = None   # BTC macro signal; None when symbol IS BTC/USD
    mtf_signals: list[TimeframeSignal] = Field(default_factory=list)  # 4h then 15m

    @model_validator(mode="after")
    def signal_symbol_matches(self) -> "DecisionInput":
        if self.signal.symbol != self.symbol:
            raise ValueError(
                f"signal.symbol {self.signal.symbol!r} != symbol {self.symbol!r}"
            )
        return self


# ── TradingDecision (LLM output) ──────────────────────────────────────────────

class TradingDecision(BaseModel):
    """Structured decision returned by the LLM as JSON.

    The prompt instructs the model to output ONLY this JSON object (no prose
    outside the JSON block). ReasoningAgent parses and validates it here.
    """

    action: Literal["buy", "sell", "hold"] = Field(
        description="Trade direction: buy, sell, or hold."
    )
    symbol: str = Field(
        description="Ticker symbol this decision applies to (e.g. BTC/USD)."
    )
    rationale: str = Field(
        min_length=80,
        description=(
            "Specific analysis referencing actual signal values. "
            "Min 80 chars — generic statements like 'signal is weak' are rejected."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Conviction in this decision [0.0–1.0]. Used for position sizing.",
    )
    edge_estimate: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated edge: probability the trade beats the risk-free rate "
            "[0.0–1.0]. Feeds the Kelly sizing formula."
        ),
    )
    risk_notes: str = Field(
        default="",
        description=(
            "Key risks or what market condition would change this decision. "
            "Required for hold decisions (min 1 sentence)."
        ),
    )
    dry_run: bool = Field(
        default=True,
        description="Mirror of the input dry_run flag — must match the input value.",
    )

    @field_validator("confidence", "edge_estimate", mode="before")
    @classmethod
    def clamp_to_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def hold_has_no_edge(self) -> "TradingDecision":
        """Hold always has zero edge. Also enforces non-empty risk_notes for holds."""
        if self.action == "hold":
            object.__setattr__(self, "edge_estimate", 0.0)
            if not self.risk_notes.strip():
                raise ValueError(
                    "risk_notes must not be empty for hold decisions — "
                    "state at least one risk factor or what would change this view."
                )
        return self

    @classmethod
    def json_schema_for_prompt(cls) -> str:
        """Return a compact JSON schema string to embed in the LLM prompt."""
        return """{
  "action": "<buy|sell|hold>",
  "symbol": "<SYMBOL/USD>",
  "rationale": "<REQUIRED: min 3 sentences. Cite actual signal values (composite=X, regime=Y, volume=Z%, MTF alignment). Explain what the data shows and why that leads to this specific decision. Generic phrases like 'signal is weak' will fail validation.>",
  "confidence": <float 0.0-1.0, calibrated from signal data — derive from composite strength and MTF alignment, not a default>,
  "edge_estimate": <float 0.0-1.0, 0.0 for hold>,
  "risk_notes": "<REQUIRED for hold: at least 1 sentence. State a specific risk factor or what signal change would alter this decision.>",
  "dry_run": <true|false>
}"""


# ── Reflection schemas ────────────────────────────────────────────────────────

class AttributedDecision(BaseModel):
    """A past decision with its simulated outcome — fed to the reflection LLM."""
    symbol: str
    action: Literal["buy", "sell", "hold"]
    composite_signal: float
    signal_regime: str
    confidence: float
    edge_estimate: float
    ema_contribution: float   # composite * ema_weight — directional vote
    rsi_contribution: float
    roc_contribution: float
    outcome_pnl_pct: float    # simulated P&L measured N hours after decision
    decided_at: datetime


class ReflectionInput(BaseModel):
    """Full context for the nightly reflection LLM call."""
    reflection_date: datetime
    decisions: list[AttributedDecision]
    current_weights: dict[str, float]
    portfolio_value: str
    cumulative_return_pct: float | None
    recent_lessons: list[RecentLesson] = Field(default_factory=list)


class ReflectionOutput(BaseModel):
    """Structured output from the reflection LLM call."""
    lessons: list[str] = Field(
        min_length=1,
        description="1-5 specific, actionable lessons learned from today's decisions.",
    )
    weight_adjustments: dict[str, float] = Field(
        description="New signal weights (ema, rsi, roc) that sum to approximately 1.0.",
    )
    weight_rationale: str = Field(
        min_length=20,
        description="Explanation for the weight changes.",
    )

    @field_validator("weight_adjustments", mode="after")
    @classmethod
    def normalise_weights(cls, v: dict[str, float]) -> dict[str, float]:
        required = {"ema", "rsi", "roc"}
        if not required.issubset(v.keys()):
            raise ValueError(f"weight_adjustments must contain keys: {required}")
        clamped = {k: max(0.0, v[k]) for k in required}
        total = sum(clamped.values())
        if total <= 0:
            return {"ema": 1/3, "rsi": 1/3, "roc": 1/3}
        return {k: clamped[k] / total for k in required}

    @classmethod
    def json_schema_for_prompt(cls) -> str:
        return """{
  "lessons": ["<lesson 1>", "<lesson 2>"],
  "weight_adjustments": {"ema": <float>, "rsi": <float>, "roc": <float>},
  "weight_rationale": "<explanation of weight changes, min 20 chars>"
}"""


# ── Decision record stored in DB ──────────────────────────────────────────────

class DecisionRecord(BaseModel):
    """Row written to the decisions table after each reasoning cycle."""

    id: int | None = None
    symbol: str
    action: Literal["buy", "sell", "hold"]
    rationale: str
    confidence: float
    edge_estimate: float
    risk_notes: str
    composite_signal: float
    signal_regime: str
    portfolio_value_at_decision: str    # Decimal → string
    decided_at: datetime
    dry_run: bool
    llm_model: str
    input_tokens: int = 0
    output_tokens: int = 0
    order_id: str | None = None          # filled after broker execution
    outcome_pnl_pct: float | None = None # filled by learning module

    # Phase 5+ fields — enable accurate attribution and reflection
    close_price_at_decision: float | None = None   # signal close price; entry price for attribution
    signal_weights_json: str | None = None          # JSON of {ema,rsi,roc} weights at decision time
