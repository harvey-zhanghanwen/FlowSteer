"""Terminal-only, action-masked one-pass group policy objective.

This is intentionally separate from the legacy structural-reward trainer.  It
rejects reconstructed/off-policy receipts and groups only by exact
``(task, condition, policy_version)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


GroupKey = Tuple[str, str, str]


@dataclass(frozen=True)
class GRPOTrajectory:
    trajectory_id: str
    task_id: str
    condition_id: str
    policy_version: str
    terminal_reward: float
    token_log_probs: Sequence[float]
    action_mask: Sequence[int]
    evaluator_valid: bool = True
    explicit_finish: bool = True
    forced_probe: bool = False
    fallback_or_manual_repair: bool = False
    reconstructed_context: bool = False
    exact_receipt_verified: bool = True

    @property
    def group_key(self) -> GroupKey:
        return (self.task_id, self.condition_id, self.policy_version)

    @property
    def eligible(self) -> bool:
        binary_mask = all(type(value) is int and value in (0, 1) for value in self.action_mask)
        finite_log_probs = all(math.isfinite(float(value)) for value in self.token_log_probs)
        return bool(
            self.evaluator_valid
            and self.explicit_finish
            and not self.forced_probe
            and not self.fallback_or_manual_repair
            and not self.reconstructed_context
            and self.exact_receipt_verified
            and len(self.token_log_probs) == len(self.action_mask)
            and binary_mask
            and finite_log_probs
            and sum(int(value) for value in self.action_mask) > 0
            and math.isfinite(self.terminal_reward)
        )


@dataclass(frozen=True)
class OnePassLossResult:
    loss: float
    eligible_trajectories: int
    groups: int
    zero_information_groups: int
    advantages: Mapping[str, float]


def group_eligible_trajectories(
    trajectories: Iterable[GRPOTrajectory],
) -> Dict[GroupKey, List[GRPOTrajectory]]:
    groups: Dict[GroupKey, List[GRPOTrajectory]] = {}
    for trajectory in trajectories:
        if trajectory.eligible:
            groups.setdefault(trajectory.group_key, []).append(trajectory)
    return groups


def same_condition_advantages(
    trajectories: Sequence[GRPOTrajectory],
    epsilon: float = 1e-8,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not trajectories:
        return np.zeros(0, dtype=np.float64)
    key = trajectories[0].group_key
    if any(item.group_key != key for item in trajectories):
        raise ValueError("advantages may only be computed within one exact group key")
    rewards = np.asarray([item.terminal_reward for item in trajectories], dtype=np.float64)
    if rewards.size < 2:
        return np.zeros_like(rewards)
    std = float(rewards.std(ddof=0))
    if not math.isfinite(std) or std <= epsilon:
        return np.zeros_like(rewards)
    return (rewards - float(rewards.mean())) / std


def action_masked_one_pass_loss(
    trajectories: Iterable[GRPOTrajectory],
    *,
    advantage_epsilon: float = 1e-8,
    equal_weight_per_group: bool = True,
) -> OnePassLossResult:
    """Compute the scalar objective from supplied differentiable log-prob values.

    This NumPy form is an audit/reference implementation.  The training code can
    use :func:`torch_action_masked_one_pass_loss` for backpropagation.
    """

    groups = group_eligible_trajectories(trajectories)
    all_advantages: Dict[str, float] = {}
    group_losses: list[float] = []
    flat_losses: list[float] = []
    zero_groups = 0
    eligible_count = 0

    for group in groups.values():
        advantages = same_condition_advantages(group, advantage_epsilon)
        if not np.any(advantages):
            zero_groups += 1
        current: list[float] = []
        for trajectory, advantage in zip(group, advantages):
            mask = np.asarray(trajectory.action_mask, dtype=np.float64)
            log_probs = np.asarray(trajectory.token_log_probs, dtype=np.float64)
            token_count = float(mask.sum())
            normalized_log_prob = float((log_probs * mask).sum() / token_count)
            contribution = -float(advantage) * normalized_log_prob
            current.append(contribution)
            flat_losses.append(contribution)
            all_advantages[trajectory.trajectory_id] = float(advantage)
            eligible_count += 1
        group_losses.append(float(np.mean(current)))

    if equal_weight_per_group:
        loss = float(np.mean(group_losses)) if group_losses else 0.0
    else:
        loss = float(np.mean(flat_losses)) if flat_losses else 0.0
    return OnePassLossResult(
        loss=loss,
        eligible_trajectories=eligible_count,
        groups=len(groups),
        zero_information_groups=zero_groups,
        advantages=all_advantages,
    )


def torch_action_masked_one_pass_loss(
    token_log_probs: Sequence["object"],
    action_masks: Sequence["object"],
    advantages: Sequence[float],
) -> "object":
    """Differentiable per-trajectory-normalized loss with a lazy torch import."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError("PyTorch is required for the differentiable objective") from exc
    if not (len(token_log_probs) == len(action_masks) == len(advantages)):
        raise ValueError("log-probs, masks, and advantages must have equal lengths")
    if not token_log_probs:
        raise ValueError("at least one trajectory is required")
    terms = []
    for log_probs, mask, advantage in zip(token_log_probs, action_masks, advantages):
        if log_probs.shape != mask.shape:
            raise ValueError("each action mask must match its token log-probs")
        if not torch.all((mask == 0) | (mask == 1)).detach().item():
            raise ValueError("action masks must be binary")
        if not torch.isfinite(log_probs).all().detach().item():
            raise ValueError("token log-probs must be finite")
        float_mask = mask.to(dtype=log_probs.dtype, device=log_probs.device)
        count = float_mask.sum()
        if count.detach().item() <= 0:
            raise ValueError("each trajectory needs at least one consumed action token")
        terms.append(-float(advantage) * (log_probs * float_mask).sum() / count)
    return torch.stack(terms).mean()
