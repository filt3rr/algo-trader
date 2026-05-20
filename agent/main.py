"""Main trading loop — Phase 5+.

Full pipeline:
    MarketDataClient -> SignalEngine -> RiskChecker -> ReasoningAgent -> AlpacaBroker
    + PriceBasedAttributor (every cycle)
    + PerformanceTracker (every cycle)
    + ReflectionEngine (nightly at REFLECTION_HOUR UTC)
    + Order reconciliation (every 5 minutes)

DRY_RUN=True (default): decisions are computed and logged to SQLite but no
orders are submitted to the broker.

Usage:
    python -m agent.main
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import AsyncSession

from agent.broker.base import OrderSide, OrderStatus, OrderType, OrderRequest
from agent.config import get_settings
from agent.learning.attribution import PriceBasedAttributor
from agent.learning.metrics import PerformanceTracker
from agent.learning.reflection import ReflectionEngine
from agent.memory.db import init_db
from agent.memory.queries import (
    count_all_decisions,
    count_all_trades,
    get_agent_state,
    get_current_weights,
    get_decisions_since,
    get_pending_trades,
    get_recent_decisions,
    get_recent_lessons,
    link_order_to_decision,
    save_agent_state,
    save_decision,
    save_trade,
    update_trade_fill,
)
from agent.reasoning.agent import ReasoningAgent
from agent.reasoning.llm_base import get_llm_adapter
from agent.reasoning.schemas import (
    AccountSnapshot,
    DecisionInput,
    PositionSnapshot,
    SignalSnapshot,
    TimeframeSignal,
    TradingDecision,
)
from agent.risk.checks import RiskChecker
from agent.risk.sizing import compute_order_size
from agent.strategy.signals import SignalEngine, TIMEFRAME_PRESETS, TIMEFRAME_FETCH_DAYS
from agent.data.universe import UniverseScreener

log = structlog.get_logger(__name__)

_STALE_ORDER_HOURS = 2  # cancel orders pending longer than this


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


# ── Snapshot builders ─────────────────────────────────────────────────────────

def _build_account_snapshot(
    account,
    positions,
    circuit_breaker_active: bool,
    daily_pnl_pct: float = 0.0,
) -> AccountSnapshot:
    pv = float(account.portfolio_value)
    cash = float(account.cash)
    return AccountSnapshot(
        portfolio_value=str(account.portfolio_value),
        cash=str(account.cash),
        cash_pct=cash / pv if pv > 0 else 0.0,
        daily_pnl_pct=daily_pnl_pct,
        open_position_count=len(positions),
        circuit_breaker_active=circuit_breaker_active,
    )


def _build_position_snapshots(positions, portfolio_value: Decimal) -> list[PositionSnapshot]:
    pv = float(portfolio_value)
    return [
        PositionSnapshot(
            symbol=p.symbol,
            qty=str(p.qty),
            market_value=str(p.market_value),
            unrealized_pnl=str(p.unrealized_pnl),
            unrealized_pnl_pct=p.unrealized_pnl_pct,
            concentration_pct=float(p.market_value) / pv if pv > 0 else 0.0,
        )
        for p in positions
    ]


def _build_signal_snapshot(signal_result) -> SignalSnapshot:
    return SignalSnapshot(
        symbol=signal_result.symbol,
        direction=signal_result.direction,
        composite=signal_result.composite,
        confidence=signal_result.confidence,
        ema_score=signal_result.components.ema,
        rsi_score=signal_result.components.rsi,
        roc_score=signal_result.components.roc,
        regime=signal_result.regime,
        close_price=signal_result.close_price,
        signal_weights=signal_result.weights,
        volume_mult=signal_result.components.volume_mult,
        signal_agreement=signal_result.signal_agreement,
    )


# ── Core pipeline ─────────────────────────────────────────────────────────────

class TradingPipeline:
    """Orchestrates the full agent pipeline across all symbols."""

    def __init__(self) -> None:
        self.cfg = get_settings()
        self.signal_engine = SignalEngine(**TIMEFRAME_PRESETS["1h"])  # primary (adaptive weights)
        self._mtf_engines: dict[str, SignalEngine] = {
            "15m": SignalEngine(**TIMEFRAME_PRESETS["15m"]),
            "4h":  SignalEngine(**TIMEFRAME_PRESETS["4h"]),
        }
        self.risk_checker = RiskChecker()
        self.reasoning_agent: ReasoningAgent | None = None
        self.reflection_engine: ReflectionEngine | None = None
        self.performance_tracker: PerformanceTracker | None = None
        self.attributor = PriceBasedAttributor(attribution_window_hours=24)
        self.session_factory = None
        self._broker = None
        self._data_client = None
        self._decision_count = 0
        self._trade_count = 0
        self._universe_screener: UniverseScreener | None = None
        # Active symbol list; updated at startup and daily by refresh_universe()
        self._active_universe: list[str] = list(self.cfg.universe)

    async def setup(self) -> None:
        """Initialise async resources — call once at startup."""
        _, self.session_factory = await init_db(self.cfg.db_url)

        # Decision adapter: uses LLM_PROVIDER (local by default when configured)
        decision_adapter = get_llm_adapter()
        # Reflection adapter: uses REFLECTION_LLM_PROVIDER if set, else same as decisions.
        # Keeps the heavier nightly task on Claude while decisions run locally.
        reflection_provider = self.cfg.reflection_llm_provider
        reflection_adapter = (
            get_llm_adapter(provider=str(reflection_provider))
            if reflection_provider and reflection_provider != self.cfg.llm_provider
            else decision_adapter
        )
        log.info(
            "LLM adapters",
            decisions=decision_adapter.model_id,
            reflection=reflection_adapter.model_id,
        )
        self.reasoning_agent = ReasoningAgent(adapter=decision_adapter)
        self.reflection_engine = ReflectionEngine(
            adapter=reflection_adapter,
            signal_engine=self.signal_engine,
        )

        async with self.session_factory() as session:
            weights = await get_current_weights(session)
            self._decision_count = await count_all_decisions(session)
            self._trade_count = await count_all_trades(session)
            cb_json = await get_agent_state(session, "circuit_breaker")

        if weights:
            # Propagate restored weights to ALL signal engines (primary + MTF)
            self.signal_engine.update_weights(weights)
            for mtf_engine in self._mtf_engines.values():
                mtf_engine.update_weights(weights)
            log.info("Restored signal weights from DB", weights=weights)

        if cb_json:
            try:
                self.risk_checker.import_state(json.loads(cb_json))
            except Exception as exc:
                log.warning("Could not restore circuit breaker state", error=str(exc))

        # Dynamic universe via CoinGecko (optional — falls back to static .env list)
        if self.cfg.coingecko_api_key:
            self._universe_screener = UniverseScreener()
            screened = await self._universe_screener.screen()
            if screened:
                self._active_universe = screened
                log.info("Dynamic universe loaded", count=len(screened), symbols=screened)
            else:
                log.warning("Universe screen returned empty — keeping static universe")
        else:
            log.info("No COINGECKO_API_KEY set — using static universe")

        log.info(
            "Pipeline ready",
            model=self.reasoning_agent.model_id,
            dry_run=self.cfg.dry_run,
            universe=self._active_universe,
            decisions_restored=self._decision_count,
            trades_restored=self._trade_count,
        )

    def _broker_instance(self):
        if self._broker is None:
            from agent.broker.alpaca import AlpacaBroker
            self._broker = AlpacaBroker()
        return self._broker

    def _data_client_instance(self):
        if self._data_client is None:
            from agent.data.ingestion import MarketDataClient
            self._data_client = MarketDataClient()
        return self._data_client

    async def run_cycle(self) -> None:
        """Execute one full trading cycle."""
        log.info("Trading cycle start", ts=datetime.now(tz=timezone.utc).isoformat())
        broker = self._broker_instance()

        try:
            account = broker.get_account()
            positions = broker.get_positions()
        except Exception as exc:
            log.error("Cannot fetch account/positions, skipping cycle", error=str(exc))
            return

        self.risk_checker.update_daily_pnl(account)
        daily_pnl_pct = self.risk_checker.get_daily_pnl_pct(account.portfolio_value)

        account_snap = _build_account_snapshot(
            account, positions, self.risk_checker.circuit_breaker_active,
            daily_pnl_pct=daily_pnl_pct,
        )
        position_snaps = _build_position_snapshots(positions, account.portfolio_value)

        if self.performance_tracker is None:
            self.performance_tracker = PerformanceTracker(account.portfolio_value)

        current_prices: dict[str, float] = {}
        btc_composite: float | None = None

        for symbol in self._active_universe:
            try:
                close_price, sig_composite = await self._process_symbol(
                    symbol=symbol,
                    account=account,
                    account_snap=account_snap,
                    position_snaps=position_snaps,
                    positions=positions,
                    btc_composite=None if symbol == "BTC/USD" else btc_composite,
                )
                if close_price is not None:
                    current_prices[symbol] = close_price
                if symbol == "BTC/USD" and sig_composite is not None:
                    btc_composite = sig_composite
            except Exception as exc:
                log.exception("Unhandled error", symbol=symbol, error=str(exc))

        # Attribution: entry price comes from DB column; current prices = exit proxy
        if current_prices:
            async with self.session_factory() as session:
                n = await self.attributor.attribute_with_prices(
                    session,
                    entry_prices={},       # entry resolved from close_price_at_decision in DB
                    current_prices=current_prices,
                )
                if n:
                    log.info("Attributed %d decisions this cycle", n)

        async with self.session_factory() as session:
            attributed_outcomes = await self._get_recent_outcomes(session)
            await self.performance_tracker.snapshot(
                session,
                portfolio_value=account.portfolio_value,
                total_decisions=self._decision_count,
                total_trades=self._trade_count,
                attributed_outcomes=attributed_outcomes,
            )

        # Persist circuit breaker state across restarts
        async with self.session_factory() as session:
            await save_agent_state(
                session,
                key="circuit_breaker",
                value=json.dumps(self.risk_checker.export_state()),
            )

        log.info("Trading cycle complete", daily_pnl_pct=f"{daily_pnl_pct:+.3%}")

    async def _get_recent_outcomes(self, session: AsyncSession) -> list[float]:
        since = datetime.now(tz=timezone.utc) - timedelta(days=30)
        rows = await get_decisions_since(session, since=since, with_outcome_only=True)
        return [r.outcome_pnl_pct for r in rows if r.outcome_pnl_pct is not None]

    async def _process_symbol(
        self,
        symbol: str,
        account,
        account_snap: AccountSnapshot,
        position_snaps: list[PositionSnapshot],
        positions,
        btc_composite: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Process one symbol. Returns (close_price, composite_signal) or (None, None)."""
        data_client = self._data_client_instance()
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=60)

        try:
            df = data_client.fetch_bars(symbols=[symbol], timeframe="1h", start=start, end=end)
        except Exception as exc:
            log.warning("Failed to fetch bars", symbol=symbol, error=str(exc))
            return None, None

        if df.empty:
            return None, None

        # Slice MultiIndex (symbol, timestamp) → flat timestamp index before signal computation
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level="symbol")
            except KeyError:
                log.warning("Symbol %s not found in bars MultiIndex", symbol)
                return None, None

        signal_result = self.signal_engine.compute_latest(df, symbol)
        if signal_result is None:
            log.debug("Insufficient data for signal", symbol=symbol)
            return None, None

        signal_snap = _build_signal_snapshot(signal_result)

        # Pre-filter: skip the LLM for obviously-weak signals with no open position.
        # If |composite| is below the threshold AND we hold nothing in this symbol,
        # the LLM would hold regardless — no need to spend inference time.
        threshold = self.cfg.signal_prefilter_threshold
        has_position = any(p.symbol == symbol for p in positions)
        if threshold > 0 and not has_position and abs(signal_result.composite) < threshold:
            log.debug(
                "Pre-filter auto-hold",
                symbol=symbol,
                composite=f"{signal_result.composite:.4f}",
                threshold=threshold,
            )
            return signal_result.close_price, signal_result.composite

        mtf_signals, (recent_lessons, recent_decisions) = await asyncio.gather(
            self._compute_mtf_signals(symbol, end),
            self._fetch_context(symbol),
        )

        ctx = DecisionInput(
            timestamp=datetime.now(tz=timezone.utc),
            symbol=symbol,
            signal=signal_snap,
            account=account_snap,
            open_positions=position_snaps,
            recent_lessons=recent_lessons,
            recent_decisions=recent_decisions,
            dry_run=self.cfg.dry_run,
            btc_composite=btc_composite,
            mtf_signals=mtf_signals,
        )

        decision = await self.reasoning_agent.decide(ctx)
        self._decision_count += 1

        log.info(
            "Decision",
            symbol=symbol,
            action=decision.action,
            confidence=f"{decision.confidence:.2f}",
            edge=f"{decision.edge_estimate:.2f}",
            regime=signal_result.regime,
            dry_run=self.cfg.dry_run,
        )

        # Save decision FIRST to get a DB id for trade linking
        record = self.reasoning_agent.build_record(ctx=ctx, decision=decision)
        async with self.session_factory() as session:
            decision_id = await save_decision(session, record)

        # Execute trade and link back to the decision row
        order_id: str | None = None
        if decision.action != "hold" and not self.cfg.dry_run:
            order_id = await self._execute_trade(
                symbol, decision, account, positions, decision_id=decision_id
            )
            if order_id:
                self._trade_count += 1
                async with self.session_factory() as session:
                    await link_order_to_decision(session, decision_id, order_id)

        return signal_result.close_price, signal_result.composite

    async def _fetch_context(self, symbol: str):
        """Fetch lessons and recent decisions from DB (extracted for gather parallelism)."""
        async with self.session_factory() as session:
            lessons = await get_recent_lessons(session, symbol=symbol, limit=5)
            decisions = await get_recent_decisions(session, symbol=symbol, limit=5)
        return lessons, decisions

    async def _compute_mtf_signals(
        self, symbol: str, end: datetime
    ) -> list[TimeframeSignal]:
        """Fetch 4h and 15m bars and return supplementary timeframe signals.

        Order: 4h first (macro trend), then 15m (entry timing).
        Any timeframe that fails to fetch or compute is silently skipped.
        """
        data_client = self._data_client_instance()
        configs = [
            ("4h",  timedelta(days=TIMEFRAME_FETCH_DAYS["4h"])),
            ("15m", timedelta(days=TIMEFRAME_FETCH_DAYS["15m"])),
        ]
        results: list[TimeframeSignal] = []
        for tf, lookback in configs:
            try:
                df = data_client.fetch_bars(
                    symbols=[symbol], timeframe=tf,
                    start=end - lookback, end=end,
                )
                if df.empty:
                    continue
                # Slice MultiIndex to flat timestamp index
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level="symbol")
                sr = self._mtf_engines[tf].compute_latest(df, symbol)
                if sr is None:
                    continue
                results.append(TimeframeSignal(
                    timeframe=tf,
                    direction=sr.direction,
                    composite=sr.composite,
                    regime=sr.regime,
                    confidence=sr.confidence,
                    volume_mult=sr.components.volume_mult,
                    signal_agreement=sr.signal_agreement,
                ))
            except Exception as exc:
                log.warning("MTF signal failed", symbol=symbol, tf=tf, error=str(exc))
        return results

    async def _execute_trade(
        self,
        symbol: str,
        decision,
        account,
        positions,
        decision_id: int | None = None,
    ) -> str | None:
        """Submit an order to the broker and persist to the trades table.

        SELL: closes the full open position quantity. If no position exists,
              the sell is skipped — there's nothing to close.
        BUY:  uses Half-Kelly sizing capped at the concentration limit.
              Passes current position value so the cap is correctly computed.
        """
        broker = self._broker_instance()

        if decision.action == "sell":
            # Close the actual open position, not a Kelly-computed quantity
            position = next((p for p in positions if p.symbol == symbol), None)
            if position is None or position.qty <= 0:
                log.info("No open position to sell", symbol=symbol)
                return None
            qty = position.qty
            # Use current market price for the risk check (position.current_price)
            entry_price = position.current_price
        else:
            # BUY: fetch live price and compute Kelly size
            try:
                entry_price = broker.get_latest_price(symbol)
            except Exception as exc:
                log.error("Cannot fetch price for sizing", symbol=symbol, error=str(exc))
                return None

            # Pass current position value so concentration cap is accurate
            current_pos = next((p for p in positions if p.symbol == symbol), None)
            current_position_value = current_pos.market_value if current_pos else Decimal("0")

            qty = compute_order_size(
                portfolio_value=account.portfolio_value,
                entry_price=entry_price,
                confidence=decision.confidence,
                edge_estimate=decision.edge_estimate,
                current_position_value=current_position_value,
            )
            if qty == Decimal("0"):
                log.info("Kelly sizing returned 0, skipping", symbol=symbol)
                return None

        side = OrderSide.BUY if decision.action == "buy" else OrderSide.SELL
        order_req = OrderRequest(
            symbol=symbol, side=side, qty=qty,
            type=OrderType.LIMIT, limit_price=entry_price,
        )
        risk_result = self.risk_checker.check(order_req, account, positions, entry_price=entry_price)
        if not risk_result.approved:
            log.warning("Order blocked by risk", symbol=symbol, reason=risk_result.rejection_reason)
            return None

        try:
            order = broker.submit_order(order_req)
            log.info(
                "Order submitted",
                symbol=symbol, side=str(side), qty=str(qty), order_id=order.id,
            )
            async with self.session_factory() as session:
                await save_trade(
                    session,
                    decision_id=decision_id,
                    order_id=order.id,
                    symbol=symbol,
                    side=str(side),
                    qty=str(qty),
                    status=str(order.status),
                    submitted_at=order.submitted_at,
                    dry_run=False,
                )
            return order.id
        except Exception as exc:
            log.error("Order submission failed", symbol=symbol, error=str(exc))
            return None

    async def run_order_reconciliation(self) -> None:
        """Poll the broker for pending order status and reconcile DB records.

        Runs every 5 minutes. Skipped entirely in DRY_RUN mode.
        """
        if self.cfg.dry_run:
            return

        broker = self._broker_instance()
        stale_cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_STALE_ORDER_HOURS)

        async with self.session_factory() as session:
            pending = await get_pending_trades(session)

        if not pending:
            return

        log.debug("Order reconciliation: checking %d pending orders", len(pending))

        for trade in pending:
            try:
                order = broker.get_order(trade.order_id)
            except Exception as exc:
                log.warning("Cannot fetch order status", order_id=trade.order_id, error=str(exc))
                continue

            status_str = str(order.status)

            if order.status == OrderStatus.FILLED:
                fill_price = str(order.filled_avg_price) if order.filled_avg_price else None
                order_value = (
                    float(order.filled_avg_price * order.filled_qty)
                    if order.filled_avg_price and order.filled_qty
                    else None
                )
                async with self.session_factory() as session:
                    await update_trade_fill(
                        session,
                        order_id=trade.order_id,
                        status="filled",
                        fill_price=fill_price,
                        filled_at=datetime.now(tz=timezone.utc),
                        order_value=order_value,
                    )
                log.info("Order filled", order_id=trade.order_id, fill_price=fill_price)

            elif order.status in (
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            ):
                async with self.session_factory() as session:
                    await update_trade_fill(
                        session,
                        order_id=trade.order_id,
                        status=status_str,
                        fill_price=None,
                        filled_at=None,
                        order_value=None,
                    )
                log.info("Order terminal", order_id=trade.order_id, status=status_str)

            elif order.status == OrderStatus.PARTIALLY_FILLED:
                fill_price = str(order.filled_avg_price) if order.filled_avg_price else None
                order_value = (
                    float(order.filled_avg_price * order.filled_qty)
                    if order.filled_avg_price and order.filled_qty
                    else None
                )
                async with self.session_factory() as session:
                    await update_trade_fill(
                        session,
                        order_id=trade.order_id,
                        status="partially_filled",
                        fill_price=fill_price,
                        filled_at=None,
                        order_value=order_value,
                    )

            else:
                # Still pending — cancel if stale
                try:
                    submitted = datetime.fromisoformat(trade.submitted_at)
                    if submitted.tzinfo is None:
                        submitted = submitted.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError, TypeError):
                    submitted = None

                if submitted and submitted < stale_cutoff:
                    try:
                        broker.cancel_order(trade.order_id)
                        async with self.session_factory() as session:
                            await update_trade_fill(
                                session,
                                order_id=trade.order_id,
                                status="stale",
                                fill_price=None,
                                filled_at=None,
                                order_value=None,
                            )
                        log.warning(
                            "Cancelled stale order",
                            order_id=trade.order_id,
                            submitted_at=trade.submitted_at,
                        )
                    except Exception as exc:
                        log.error(
                            "Cannot cancel stale order",
                            order_id=trade.order_id,
                            error=str(exc),
                        )

    async def refresh_universe(self) -> None:
        """Daily universe refresh — re-screens CoinGecko top coins.

        Held positions are always preserved in the universe so open trades are
        never orphaned by a rotation. Falls back silently on network errors.
        """
        if self._universe_screener is None:
            return

        # Gather currently held symbols so they are preserved
        hold_symbols: list[str] = []
        try:
            broker = self._broker_instance()
            positions = broker.get_positions()
            hold_symbols = [p.symbol for p in positions]
        except Exception as exc:
            log.warning("Could not fetch positions for universe refresh", error=str(exc))

        screened = await self._universe_screener.screen(hold_symbols=hold_symbols)
        if screened:
            self._active_universe = screened
            log.info(
                "Universe refreshed",
                count=len(screened),
                held=hold_symbols,
                symbols=screened,
            )
        else:
            log.warning("Universe refresh returned empty — keeping previous universe")

    async def run_reflection(self) -> None:
        """Nightly reflection job — extracts lessons, updates signal weights."""
        log.info("Nightly reflection start")
        async with self.session_factory() as session:
            result = await self.reflection_engine.run(session)

        if result:
            # Propagate updated weights to MTF engines (primary already updated inside engine.run)
            w = result.weight_adjustments
            for mtf_engine in self._mtf_engines.values():
                mtf_engine.update_weights(w)
            log.info(
                "Reflection complete",
                lessons=len(result.lessons),
                weights_ema=f"{w['ema']:.3f}",
                weights_rsi=f"{w['rsi']:.3f}",
                weights_roc=f"{w['roc']:.3f}",
            )
        else:
            log.info("Reflection complete (no attributed decisions)")


# ── Scheduler + entrypoint ────────────────────────────────────────────────────

async def main() -> None:
    cfg = get_settings()
    _configure_logging(cfg.log_level)

    pipeline = TradingPipeline()
    await pipeline.setup()

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        pipeline.run_cycle,
        trigger="interval",
        seconds=cfg.trading_loop_seconds,
        id="trading_loop",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        pipeline.run_reflection,
        trigger="cron",
        hour=cfg.reflection_hour,
        minute=0,
        id="nightly_reflection",
    )
    scheduler.add_job(
        pipeline.run_order_reconciliation,
        trigger="interval",
        minutes=5,
        start_date=datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        + timedelta(seconds=90),
        id="order_reconciliation",
        max_instances=1,
        coalesce=True,
    )
    # Universe refresh runs daily at 01:00 UTC — after midnight attribution,
    # before the next trading cycle opens. Only active when CoinGecko key is set.
    if cfg.coingecko_api_key:
        scheduler.add_job(
            pipeline.refresh_universe,
            trigger="cron",
            hour=1,
            minute=0,
            id="universe_refresh",
        )
    scheduler.start()

    log.info(
        "Scheduler started",
        loop_seconds=cfg.trading_loop_seconds,
        reflection_hour=cfg.reflection_hour,
        dry_run=cfg.dry_run,
    )

    stop_event = asyncio.Event()

    def _handle_signal(*_):
        log.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            signal.signal(sig, _handle_signal)

    await pipeline.run_cycle()
    await stop_event.wait()
    scheduler.shutdown(wait=False)
    log.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
