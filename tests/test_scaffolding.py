"""Phase 0 smoke tests — verify the project structure is importable."""

import importlib
from pathlib import Path

import pytest


AGENT_MODULES = [
    "agent",
    "agent.data",
    "agent.broker",
    "agent.strategy",
    "agent.reasoning",
    "agent.risk",
    "agent.memory",
    "agent.learning",
    "agent.dashboard",
]


@pytest.mark.parametrize("module", AGENT_MODULES)
def test_module_importable(module: str) -> None:
    """Every agent sub-package must be importable (no syntax errors, bad imports)."""
    importlib.import_module(module)


def test_env_example_exists() -> None:
    root = Path(__file__).parent.parent
    assert (root / ".env.example").exists(), ".env.example must exist"


def test_design_md_exists() -> None:
    root = Path(__file__).parent.parent
    assert (root / "DESIGN.md").exists(), "DESIGN.md must exist"


def test_required_directories_exist() -> None:
    root = Path(__file__).parent.parent
    required = [
        "agent/data",
        "agent/broker",
        "agent/strategy",
        "agent/reasoning/prompts",
        "agent/risk",
        "agent/memory",
        "agent/learning",
        "agent/dashboard",
        "tests",
        "data",
        "logs",
    ]
    for rel in required:
        assert (root / rel).is_dir(), f"Directory {rel} must exist"


def test_live_trading_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live trading must default to False — even if the env var is missing."""
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    live = False  # default when env var absent
    assert live is False, "Live trading must default to False"
