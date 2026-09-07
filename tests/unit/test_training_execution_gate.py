from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.interactive.config_loader import ConfigurationError


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "train_agentgraph_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location("training_execution_gate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_closed_execution_gate_allows_only_prepare_only() -> None:
    config = {
        "execution_gate": {
            "execution_enabled": False,
            "allowed_mode": "prepare_only",
        }
    }

    _MODULE._require_execution_gate(config, prepare_only=True)
    with pytest.raises(ConfigurationError, match="only --prepare-only"):
        _MODULE._require_execution_gate(config, prepare_only=False)


def test_legacy_config_without_gate_is_unchanged() -> None:
    _MODULE._require_execution_gate({}, prepare_only=False)


@pytest.mark.parametrize("value", (None, 1, "false"))
def test_declared_gate_requires_boolean_execution_state(value: object) -> None:
    config = {
        "execution_gate": {
            "execution_enabled": value,
            "allowed_mode": "prepare_only",
        }
    }
    with pytest.raises(ConfigurationError, match="must be bool"):
        _MODULE._require_execution_gate(config, prepare_only=True)
