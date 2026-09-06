"""Tempered Trajectory Balance objective for structured AgentGraph actions.

Source map (thin adaptation, objective math only):

* ``training/trajectory.py::split_think_and_action`` in the referenced
  SkillFlow implementation keeps reasoning in the model context and exposes
  only the structured action as the prediction target.
* ``training/flow_metrics.py::edge_logprob_tilde`` and
  ``effective_paper_steps`` average each edge over its structured-action
  tokens and normalize a trajectory by its number of action-bearing edges.
* ``training/gflownet_trainer.py::compute_ttb_loss`` and
  ``_compute_action_logprob_forward`` implement the Tempered Trajectory
  Balance residual used here.

The authoritative files audited for this branch are in the released SkillFlow
repository at revision ``74be52bb6bd9f0e9e68dacb72636b75649197983``. This
module is a local mathematical adaptation rather than a direct import. It does
not tokenize model output: callers must score structured action tokens only,
with any reasoning tokens supplied solely as context. Likewise,
``r_tilde`` must already contain the caller's explicit epsilon shift or clip;
this objective rejects non-positive rewards and applies no hidden clipping.

This module intentionally contains no GRPO advantages, KL regularization,
MACE, or Skill-evolution logic.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Sequence


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_token_count(value: Any, *, edge_index: int) -> int:
    if isinstance(value, bool):
        raise TypeError(
            f"action_token_counts[{edge_index}] must be an integer, not bool"
        )
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"action_token_counts[{edge_index}] must be an integer"
        ) from exc
    if count <= 0:
        raise ValueError(
            f"action_token_counts[{edge_index}] must be positive"
        )
    return int(count)


def _validated_python_edges(
    forward_action_logprob_sums: Sequence[float],
    backward_action_logprob_sums: Sequence[float],
    action_token_counts: Sequence[int],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    if isinstance(forward_action_logprob_sums, (str, bytes)):
        raise TypeError("forward_action_logprob_sums must be a sequence of scalars")
    if isinstance(backward_action_logprob_sums, (str, bytes)):
        raise TypeError("backward_action_logprob_sums must be a sequence of scalars")
    if isinstance(action_token_counts, (str, bytes)):
        raise TypeError("action_token_counts must be a sequence of integers")

    try:
        forward_values = tuple(forward_action_logprob_sums)
        backward_values = tuple(backward_action_logprob_sums)
        count_values = tuple(action_token_counts)
    except TypeError as exc:
        raise TypeError("edge inputs must be finite sequences") from exc

    edge_count = len(forward_values)
    if edge_count == 0:
        raise ValueError("at least one structured-action edge is required")
    if len(backward_values) != edge_count or len(count_values) != edge_count:
        raise ValueError(
            "forward sums, backward sums, and action token counts must have "
            "equal lengths"
        )

    forward = tuple(
        _finite_float(f"forward_action_logprob_sums[{index}]", value)
        for index, value in enumerate(forward_values)
    )
    backward = tuple(
        _finite_float(f"backward_action_logprob_sums[{index}]", value)
        for index, value in enumerate(backward_values)
    )
    counts = tuple(
        _positive_token_count(value, edge_index=index)
        for index, value in enumerate(count_values)
    )
    return forward, backward, counts


def edge_action_logprob_mean(logprob_sum: float, action_token_count: int) -> float:
    """Return one edge's mean log-probability over structured-action tokens.

    ``logprob_sum`` must not include reasoning-token log-probabilities.
    """

    value = _finite_float("logprob_sum", logprob_sum)
    count = _positive_token_count(action_token_count, edge_index=0)
    return value / count


def tempered_trajectory_balance_residual(
    *,
    log_z: float,
    forward_action_logprob_sums: Sequence[float],
    backward_action_logprob_sums: Sequence[float],
    action_token_counts: Sequence[int],
    r_tilde: float,
    beta: float = 1.0,
) -> float:
    """Compute ``Delta = logZ + sum(fwd) - beta*log(r_tilde) - sum(bwd)``.

    Each forward/backward edge term is its structured-action log-probability
    sum divided by the corresponding structured-action token count.
    ``r_tilde`` must already be shifted/clipped by the caller and be positive.
    """

    z_value = _finite_float("log_z", log_z)
    reward = _finite_float("r_tilde", r_tilde)
    beta_value = _finite_float("beta", beta)
    if reward <= 0.0:
        raise ValueError(
            "r_tilde must be positive; apply the configured epsilon shift or "
            "clip before calling the TTB objective"
        )
    if beta_value <= 0.0:
        raise ValueError("beta must be positive")

    forward, backward, counts = _validated_python_edges(
        forward_action_logprob_sums,
        backward_action_logprob_sums,
        action_token_counts,
    )
    forward_sum = sum(value / count for value, count in zip(forward, counts))
    backward_sum = sum(value / count for value, count in zip(backward, counts))
    return z_value + forward_sum - beta_value * math.log(reward) - backward_sum


def tempered_trajectory_balance_loss(
    *,
    log_z: float,
    forward_action_logprob_sums: Sequence[float],
    backward_action_logprob_sums: Sequence[float],
    action_token_counts: Sequence[int],
    r_tilde: float,
    beta: float = 1.0,
) -> float:
    """Return the single-trajectory SkillFlow loss ``(Delta / T) ** 2``."""

    edge_count = len(tuple(action_token_counts))
    residual = tempered_trajectory_balance_residual(
        log_z=log_z,
        forward_action_logprob_sums=forward_action_logprob_sums,
        backward_action_logprob_sums=backward_action_logprob_sums,
        action_token_counts=action_token_counts,
        r_tilde=r_tilde,
        beta=beta,
    )
    return (residual / edge_count) ** 2


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError("PyTorch is required for the differentiable TTB objective") from exc
    return torch


def _validated_torch_edges(
    forward_action_logprob_sums: Any,
    backward_action_logprob_sums: Any,
    action_token_counts: Any,
) -> tuple[Any, Any, Any]:
    torch = _require_torch()
    tensors = {
        "forward_action_logprob_sums": forward_action_logprob_sums,
        "backward_action_logprob_sums": backward_action_logprob_sums,
        "action_token_counts": action_token_counts,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if forward_action_logprob_sums.numel() == 0:
        raise ValueError("at least one structured-action edge is required")
    if not (
        forward_action_logprob_sums.shape
        == backward_action_logprob_sums.shape
        == action_token_counts.shape
    ):
        raise ValueError(
            "forward sums, backward sums, and action token counts must have "
            "equal shapes"
        )
    if not forward_action_logprob_sums.is_floating_point():
        raise TypeError("forward_action_logprob_sums must have a floating dtype")
    if not backward_action_logprob_sums.is_floating_point():
        raise TypeError("backward_action_logprob_sums must have a floating dtype")
    if action_token_counts.dtype == torch.bool or action_token_counts.is_complex():
        raise TypeError("action_token_counts must have a real, non-boolean dtype")
    if forward_action_logprob_sums.device != backward_action_logprob_sums.device:
        raise ValueError("forward and backward sums must be on the same device")
    if forward_action_logprob_sums.dtype != backward_action_logprob_sums.dtype:
        raise ValueError("forward and backward sums must have the same dtype")
    if action_token_counts.device != forward_action_logprob_sums.device:
        raise ValueError("action token counts must be on the same device as log-probs")
    if not torch.isfinite(forward_action_logprob_sums).all().detach().item():
        raise ValueError("forward_action_logprob_sums must be finite")
    if not torch.isfinite(backward_action_logprob_sums).all().detach().item():
        raise ValueError("backward_action_logprob_sums must be finite")
    if not torch.isfinite(action_token_counts).all().detach().item():
        raise ValueError("action_token_counts must be finite")
    if not torch.equal(action_token_counts, action_token_counts.floor()):
        raise ValueError("action_token_counts must contain integers")
    if not torch.all(action_token_counts > 0).detach().item():
        raise ValueError("every action token count must be positive")
    return (
        forward_action_logprob_sums,
        backward_action_logprob_sums,
        action_token_counts.to(
            dtype=forward_action_logprob_sums.dtype,
            device=forward_action_logprob_sums.device,
        ),
    )


def _torch_scalar(name: str, value: Any, *, reference: Any) -> Any:
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on the same device as log-probs")
        result = value.reshape(()).to(dtype=reference.dtype)
    else:
        result = torch.as_tensor(
            _finite_float(name, value),
            dtype=reference.dtype,
            device=reference.device,
        )
    if not torch.isfinite(result).detach().item():
        raise ValueError(f"{name} must be finite")
    return result


def torch_tempered_trajectory_balance_residual(
    *,
    log_z: Any,
    forward_action_logprob_sums: Any,
    backward_action_logprob_sums: Any,
    action_token_counts: Any,
    r_tilde: Any,
    beta: float = 1.0,
) -> Any:
    """Differentiable single-trajectory TTB residual with a lazy torch import."""

    torch = _require_torch()
    forward, backward, counts = _validated_torch_edges(
        forward_action_logprob_sums,
        backward_action_logprob_sums,
        action_token_counts,
    )
    z_value = _torch_scalar("log_z", log_z, reference=forward)
    reward = _torch_scalar("r_tilde", r_tilde, reference=forward)
    beta_value = _finite_float("beta", beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive")
    if not (reward > 0).detach().item():
        raise ValueError(
            "r_tilde must be positive; apply the configured epsilon shift or "
            "clip before calling the TTB objective"
        )
    return (
        z_value
        + (forward / counts).sum()
        - beta_value * torch.log(reward)
        - (backward / counts).sum()
    )


def torch_tempered_trajectory_balance_loss(
    *,
    log_z: Any,
    forward_action_logprob_sums: Any,
    backward_action_logprob_sums: Any,
    action_token_counts: Any,
    r_tilde: Any,
    beta: float = 1.0,
) -> Any:
    """Return differentiable ``(Delta / T) ** 2`` for one trajectory."""

    residual = torch_tempered_trajectory_balance_residual(
        log_z=log_z,
        forward_action_logprob_sums=forward_action_logprob_sums,
        backward_action_logprob_sums=backward_action_logprob_sums,
        action_token_counts=action_token_counts,
        r_tilde=r_tilde,
        beta=beta,
    )
    edge_count = forward_action_logprob_sums.numel()
    return (residual / edge_count) ** 2


__all__ = [
    "edge_action_logprob_mean",
    "tempered_trajectory_balance_loss",
    "tempered_trajectory_balance_residual",
    "torch_tempered_trajectory_balance_loss",
    "torch_tempered_trajectory_balance_residual",
]
