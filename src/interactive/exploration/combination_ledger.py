"""Dynamic combination posterior from the in-loop ledger specification.

The ledger is an exploration-plane model, not a policy-training loss.  Its
Gaussian comparison posterior is fitted only from same-snapshot paired probes.
Natural and probe trajectories may update the task-family baseline and surface
signal sensors, but they never update the comparison posterior directly.

The implementation deliberately keeps the sufficient statistics in precision
form and uses Cholesky solves for posterior reads and joint samples.  Columns
are added dynamically as model/role/relation levels are encountered.  A
second-order column becomes active only after the corresponding combination
has been touched by ``n_act`` paired probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


LEDGER_STATE_VERSION = "combination-posterior-v1"
ROLE_CLUSTERS = frozenset(
    {
        "solve",
        "verify",
        "plan",
        "summarize",
        "arbitrate",
        "retrieve",
        "code",
        "test",
    }
)
EDGE_TYPES = frozenset({"independent", "unidirectional", "bidirectional"})
STAGES = frozenset({"before_output", "other"})
SURFACE_SIGNAL_KINDS = frozenset(
    {"verifier_pass", "bidir_consensus", "exec_error", "none"}
)
FIRST_ORDER_FACTORS = ("role", "model", "rel")
NINETY_PERCENT_NORMAL_QUANTILE = 1.6448536269514722


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _readonly_array(
    value: Iterable[float] | np.ndarray,
    *,
    ndim: int,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got {result.ndim}")
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("ledger arrays must be finite")
    result.setflags(write=False)
    return result


def _solve_cholesky(precision: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    if precision.shape == (0, 0):
        return np.zeros_like(right_hand_side, dtype=np.float64)
    factor = np.linalg.cholesky(precision)
    intermediate = np.linalg.solve(factor, right_hand_side)
    return np.linalg.solve(factor.T, intermediate)


@dataclass(frozen=True)
class DecisionKey:
    """One role/model/relation decision represented in the ledger."""

    task_family: str
    role_cluster: str
    model_id: str
    edge_type: str
    same_model_as_upstream: bool
    stage: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_family", _non_empty(self.task_family, "task_family"))
        object.__setattr__(self, "model_id", _non_empty(self.model_id, "model_id"))
        if self.role_cluster not in ROLE_CLUSTERS:
            raise ValueError(
                f"role_cluster must be one of {sorted(ROLE_CLUSTERS)}, "
                f"got {self.role_cluster!r}"
            )
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(
                f"edge_type must be one of {sorted(EDGE_TYPES)}, got {self.edge_type!r}"
            )
        if type(self.same_model_as_upstream) is not bool:
            raise TypeError("same_model_as_upstream must be bool")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}, got {self.stage!r}")

    @property
    def relation_level(self) -> str:
        model_relation = "same" if self.same_model_as_upstream else "different"
        return f"{self.edge_type}|{model_relation}"

    def factor_levels(self) -> dict[str, str]:
        """Return the three factors included in the comparison design."""

        return {
            "role": self.role_cluster,
            "model": self.model_id,
            "rel": self.relation_level,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "task_family": self.task_family,
            "role_cluster": self.role_cluster,
            "model_id": self.model_id,
            "edge_type": self.edge_type,
            "same_model_as_upstream": self.same_model_as_upstream,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "DecisionKey":
        return cls(
            task_family=str(state["task_family"]),
            role_cluster=str(state["role_cluster"]),
            model_id=str(state["model_id"]),
            edge_type=str(state["edge_type"]),
            same_model_as_upstream=state["same_model_as_upstream"],
            stage=str(state["stage"]),
        )


@dataclass(frozen=True)
class SurfaceSignal:
    """Observable post-execution signal and its versioned sensor class."""

    kind: str
    value: bool
    sensor_class: str

    def __post_init__(self) -> None:
        if self.kind not in SURFACE_SIGNAL_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(SURFACE_SIGNAL_KINDS)}, got {self.kind!r}"
            )
        if type(self.value) is not bool:
            raise TypeError("surface signal value must be bool")
        object.__setattr__(
            self, "sensor_class", _non_empty(self.sensor_class, "sensor_class")
        )

    @property
    def is_normal(self) -> bool:
        """Whether latent-risk inference is permitted after this signal."""

        if self.kind == "none":
            return True
        if self.kind in {"verifier_pass", "bidir_consensus"}:
            return self.value
        return not self.value  # ``exec_error=False`` means execution was normal.

    @property
    def is_positive_evidence(self) -> bool:
        return self.kind in {"verifier_pass", "bidir_consensus"} and self.value

    @classmethod
    def for_key(cls, kind: str, value: bool, key: DecisionKey) -> "SurfaceSignal":
        suffix = "same" if key.same_model_as_upstream else "different"
        return cls(kind=kind, value=value, sensor_class=f"{kind}|{suffix}")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "value": self.value,
            "sensor_class": self.sensor_class,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "SurfaceSignal":
        return cls(
            kind=str(state["kind"]),
            value=state["value"],
            sensor_class=str(state["sensor_class"]),
        )


@dataclass(frozen=True)
class ContrastEstimate:
    """Gaussian posterior readout for one same-snapshot contrast."""

    mean: float
    variance: float
    lower: float
    upper: float

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class ProbeUpdate:
    """Sufficient statistics computed from one paired Bernoulli probe."""

    delta_hat: float
    noise_var: float
    keep_rate: float
    switch_rate: float
    repeats: int
    activated_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class LedgerState:
    """Serializable snapshot of all dynamic ledger sufficient statistics."""

    epoch: int
    policy_version: str
    columns: Mapping[str, int]
    Lambda: np.ndarray
    eta: np.ndarray
    sigma_f2: Mapping[str, float]
    sigma_eta2: float
    task_baseline: Mapping[str, tuple[float, int]]
    sensor_counts: Mapping[str, np.ndarray]
    calibration_quantile: float
    interaction_touches: Mapping[str, int] = field(default_factory=dict)
    baseline_history: Mapping[str, tuple[tuple[int, float, int], ...]] = field(
        default_factory=dict
    )
    probe_count: int = 0
    low_coverage_epochs: int = 0
    version: str = LEDGER_STATE_VERSION

    def __post_init__(self) -> None:
        if self.version != LEDGER_STATE_VERSION:
            raise ValueError(f"unsupported ledger state version: {self.version}")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        object.__setattr__(
            self, "policy_version", _non_empty(self.policy_version, "policy_version")
        )
        columns = {str(name): int(index) for name, index in self.columns.items()}
        if sorted(columns.values()) != list(range(len(columns))):
            raise ValueError("column indices must be unique and contiguous from zero")
        dimension = len(columns)
        precision = _readonly_array(
            self.Lambda, ndim=2, shape=(dimension, dimension)
        )
        if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-12):
            raise ValueError("Lambda must be symmetric")
        if dimension:
            np.linalg.cholesky(precision)
        information = _readonly_array(self.eta, ndim=1, shape=(dimension,))
        variances = {
            str(name): _finite_positive(value, f"sigma_f2[{name!r}]")
            for name, value in self.sigma_f2.items()
        }
        if set(variances) != set(FIRST_ORDER_FACTORS):
            raise ValueError(f"sigma_f2 must contain exactly {FIRST_ORDER_FACTORS}")
        sigma_eta2 = _finite_positive(self.sigma_eta2, "sigma_eta2")
        task_baseline: dict[str, tuple[float, int]] = {}
        for family, pair in self.task_baseline.items():
            if len(pair) != 2:
                raise ValueError("task baseline values must be (mean, count)")
            mean, count = float(pair[0]), int(pair[1])
            if not math.isfinite(mean) or count < 0:
                raise ValueError("task baseline statistics must be finite and non-negative")
            task_baseline[str(family)] = (mean, count)
        sensor_counts: dict[str, np.ndarray] = {}
        for sensor_class, counts in self.sensor_counts.items():
            array = _readonly_array(counts, ndim=2, shape=(2, 2))
            if np.any(array <= 0.0):
                raise ValueError("sensor beta counts must stay positive")
            sensor_counts[str(sensor_class)] = array
        calibration_quantile = float(self.calibration_quantile)
        if not math.isfinite(calibration_quantile) or calibration_quantile <= 0.0:
            raise ValueError("calibration_quantile must be finite and positive")
        touches = {str(name): int(count) for name, count in self.interaction_touches.items()}
        if any(count < 0 for count in touches.values()):
            raise ValueError("interaction touch counts must be non-negative")
        history: dict[str, tuple[tuple[int, float, int], ...]] = {}
        for family, entries in self.baseline_history.items():
            checked_entries: list[tuple[int, float, int]] = []
            for epoch, total, count in entries:
                checked = (int(epoch), float(total), int(count))
                if checked[0] < 0 or not math.isfinite(checked[1]) or checked[2] < 0:
                    raise ValueError("invalid task baseline history entry")
                checked_entries.append(checked)
            history[str(family)] = tuple(checked_entries)
        if self.probe_count < 0 or self.low_coverage_epochs < 0:
            raise ValueError("ledger counters must be non-negative")

        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "Lambda", precision)
        object.__setattr__(self, "eta", information)
        object.__setattr__(self, "sigma_f2", variances)
        object.__setattr__(self, "sigma_eta2", sigma_eta2)
        object.__setattr__(self, "task_baseline", task_baseline)
        object.__setattr__(self, "sensor_counts", sensor_counts)
        object.__setattr__(self, "calibration_quantile", calibration_quantile)
        object.__setattr__(self, "interaction_touches", touches)
        object.__setattr__(self, "baseline_history", history)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "epoch": self.epoch,
            "policy_version": self.policy_version,
            "columns": dict(self.columns),
            "Lambda": self.Lambda.tolist(),
            "eta": self.eta.tolist(),
            "sigma_f2": dict(self.sigma_f2),
            "sigma_eta2": self.sigma_eta2,
            "task_baseline": {
                family: [mean, count]
                for family, (mean, count) in self.task_baseline.items()
            },
            "sensor_counts": {
                sensor_class: counts.tolist()
                for sensor_class, counts in self.sensor_counts.items()
            },
            "calibration_quantile": self.calibration_quantile,
            "interaction_touches": dict(self.interaction_touches),
            "baseline_history": {
                family: [list(entry) for entry in entries]
                for family, entries in self.baseline_history.items()
            },
            "probe_count": self.probe_count,
            "low_coverage_epochs": self.low_coverage_epochs,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "LedgerState":
        return cls(
            version=str(state.get("version", "")),
            epoch=int(state["epoch"]),
            policy_version=str(state["policy_version"]),
            columns=dict(state["columns"]),
            Lambda=np.asarray(state["Lambda"], dtype=np.float64),
            eta=np.asarray(state["eta"], dtype=np.float64),
            sigma_f2=dict(state["sigma_f2"]),
            sigma_eta2=float(state["sigma_eta2"]),
            task_baseline={
                str(family): (float(pair[0]), int(pair[1]))
                for family, pair in dict(state.get("task_baseline", {})).items()
            },
            sensor_counts={
                str(sensor_class): np.asarray(counts, dtype=np.float64)
                for sensor_class, counts in dict(state.get("sensor_counts", {})).items()
            },
            calibration_quantile=float(state["calibration_quantile"]),
            interaction_touches={
                str(name): int(count)
                for name, count in dict(state.get("interaction_touches", {})).items()
            },
            baseline_history={
                str(family): tuple(
                    (int(entry[0]), float(entry[1]), int(entry[2]))
                    for entry in entries
                )
                for family, entries in dict(state.get("baseline_history", {})).items()
            },
            probe_count=int(state.get("probe_count", 0)),
            low_coverage_epochs=int(state.get("low_coverage_epochs", 0)),
        )


class CombinationPosterior:
    """Dynamic Gaussian ledger fitted from paired Bernoulli differences."""

    def __init__(
        self,
        policy_version: str,
        *,
        epoch: int = 0,
        n_act: int = 5,
        variance_floor: float = 0.01,
        initial_factor_variance: float = 0.05,
        initial_interaction_variance: float = 0.05,
        baseline_prior_mean: float = 0.5,
        baseline_prior_count: int = 4,
        calibration_quantile: float = NINETY_PERCENT_NORMAL_QUANTILE,
        calibration_alpha: float = 0.05,
    ) -> None:
        self.policy_version = _non_empty(policy_version, "policy_version")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if n_act <= 0:
            raise ValueError("n_act must be positive")
        if baseline_prior_count <= 0:
            raise ValueError("baseline_prior_count must be positive")
        if not 0.0 <= baseline_prior_mean <= 1.0:
            raise ValueError("baseline_prior_mean must lie in [0, 1]")
        if not 0.0 < calibration_alpha < 1.0:
            raise ValueError("calibration_alpha must lie in (0, 1)")
        self.epoch = int(epoch)
        self.n_act = int(n_act)
        self.variance_floor = _finite_positive(variance_floor, "variance_floor")
        factor_variance = _finite_positive(
            initial_factor_variance, "initial_factor_variance"
        )
        self._sigma_f2 = {factor: factor_variance for factor in FIRST_ORDER_FACTORS}
        self._sigma_eta2 = _finite_positive(
            initial_interaction_variance, "initial_interaction_variance"
        )
        self.baseline_prior_mean = float(baseline_prior_mean)
        self.baseline_prior_count = int(baseline_prior_count)
        self.calibration_quantile = _finite_positive(
            calibration_quantile, "calibration_quantile"
        )
        self.calibration_alpha = float(calibration_alpha)

        self._columns: dict[str, int] = {}
        self._Lambda = np.zeros((0, 0), dtype=np.float64)
        self._eta = np.zeros(0, dtype=np.float64)
        self._prior_precision_diagonal = np.zeros(0, dtype=np.float64)
        self._interaction_touches: dict[str, int] = {}
        self._baseline_history: dict[str, list[tuple[int, float, int]]] = {}
        self._sensor_counts: dict[str, np.ndarray] = {}
        self._probe_count = 0
        self._low_coverage_epochs = 0

    @property
    def dimension(self) -> int:
        return len(self._columns)

    @property
    def columns(self) -> dict[str, int]:
        return dict(self._columns)

    @property
    def Lambda(self) -> np.ndarray:
        result = self._Lambda.copy()
        result.setflags(write=False)
        return result

    @property
    def precision(self) -> np.ndarray:
        return self.Lambda

    @property
    def eta(self) -> np.ndarray:
        result = self._eta.copy()
        result.setflags(write=False)
        return result

    @property
    def information(self) -> np.ndarray:
        return self.eta

    @property
    def mean(self) -> np.ndarray:
        result = _solve_cholesky(self._Lambda, self._eta)
        result.setflags(write=False)
        return result

    @property
    def covariance(self) -> np.ndarray:
        identity = np.eye(self.dimension, dtype=np.float64)
        result = _solve_cholesky(self._Lambda, identity)
        result = 0.5 * (result + result.T)
        result.setflags(write=False)
        return result

    @property
    def sigma_f2(self) -> dict[str, float]:
        return dict(self._sigma_f2)

    @property
    def sigma_eta2(self) -> float:
        return self._sigma_eta2

    @property
    def probe_count(self) -> int:
        return self._probe_count

    @property
    def interaction_touches(self) -> dict[str, int]:
        return dict(self._interaction_touches)

    @property
    def sensor_counts(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for sensor_class, counts in self._sensor_counts.items():
            copied = counts.copy()
            copied.setflags(write=False)
            result[sensor_class] = copied
        return result

    @property
    def task_baseline(self) -> dict[str, tuple[float, int]]:
        return {
            family: self._aggregate_baseline(entries)
            for family, entries in self._baseline_history.items()
        }

    @staticmethod
    def _first_order_name(factor: str, level: str) -> str:
        return f"{factor}={level}"

    @staticmethod
    def _column_factor(column: str) -> str:
        if "&" in column:
            return "interaction"
        return column.split("=", 1)[0]

    @staticmethod
    def _interaction_name(left: str, right: str) -> str:
        left_factor = left.split("=", 1)[0]
        right_factor = right.split("=", 1)[0]
        left_order = FIRST_ORDER_FACTORS.index(left_factor)
        right_order = FIRST_ORDER_FACTORS.index(right_factor)
        return f"{left}&{right}" if left_order < right_order else f"{right}&{left}"

    def _key_first_order_columns(self, key: DecisionKey) -> tuple[str, ...]:
        levels = key.factor_levels()
        return tuple(
            self._first_order_name(factor, levels[factor])
            for factor in FIRST_ORDER_FACTORS
        )

    def _key_interaction_columns(self, key: DecisionKey) -> tuple[str, ...]:
        first_order = self._key_first_order_columns(key)
        return tuple(
            self._interaction_name(first_order[left], first_order[right])
            for left in range(len(first_order))
            for right in range(left + 1, len(first_order))
        )

    def _prior_variance_for_column(self, column: str) -> float:
        factor = self._column_factor(column)
        return self._sigma_eta2 if factor == "interaction" else self._sigma_f2[factor]

    def _ensure_column(self, name: str) -> bool:
        if name in self._columns:
            return False
        old_dimension = self.dimension
        self._columns[name] = old_dimension
        expanded = np.zeros((old_dimension + 1, old_dimension + 1), dtype=np.float64)
        if old_dimension:
            expanded[:old_dimension, :old_dimension] = self._Lambda
        prior_precision = 1.0 / self._prior_variance_for_column(name)
        expanded[old_dimension, old_dimension] = prior_precision
        self._Lambda = expanded
        self._eta = np.pad(self._eta, (0, 1))
        self._prior_precision_diagonal = np.append(
            self._prior_precision_diagonal, prior_precision
        )
        return True

    def register_key(self, key: DecisionKey) -> None:
        for column in self._key_first_order_columns(key):
            self._ensure_column(column)

    def design_vector(self, key: DecisionKey) -> np.ndarray:
        """Construct ``phi(key)`` and register newly observed first-order levels."""

        if not isinstance(key, DecisionKey):
            raise TypeError("key must be DecisionKey")
        self.register_key(key)
        result = np.zeros(self.dimension, dtype=np.float64)
        for column in self._key_first_order_columns(key):
            result[self._columns[column]] = 1.0
        for column in self._key_interaction_columns(key):
            index = self._columns.get(column)
            if index is not None:
                result[index] = 1.0
        result.setflags(write=False)
        return result

    def prefix_vector(self, keys: Sequence[DecisionKey]) -> np.ndarray:
        keys = tuple(keys)
        for key in keys:
            self.register_key(key)
        result = np.zeros(self.dimension, dtype=np.float64)
        for key in keys:
            # All first-order levels have already been registered, so vector
            # dimensions remain stable throughout the sum.
            result += self.design_vector(key)
        result.setflags(write=False)
        return result

    @staticmethod
    def _changed_factor(key_switch: DecisionKey, key_keep: DecisionKey) -> str:
        if key_switch.task_family != key_keep.task_family:
            raise ValueError("paired decisions must have the same task_family")
        if key_switch.stage != key_keep.stage:
            raise ValueError("paired decisions must have the same stage")
        switch_levels = key_switch.factor_levels()
        keep_levels = key_keep.factor_levels()
        changed = [
            factor
            for factor in FIRST_ORDER_FACTORS
            if switch_levels[factor] != keep_levels[factor]
        ]
        if len(changed) != 1:
            raise ValueError("paired decisions must differ in exactly one design factor")
        return changed[0]

    def contrast_vector(
        self, key_switch: DecisionKey, key_keep: DecisionKey
    ) -> np.ndarray:
        self._changed_factor(key_switch, key_keep)
        self.register_key(key_switch)
        self.register_key(key_keep)
        switch = self.design_vector(key_switch)
        keep = self.design_vector(key_keep)
        result = np.array(switch - keep, copy=True)
        result.setflags(write=False)
        return result

    def predict_contrast(
        self,
        key_switch: DecisionKey,
        key_keep: DecisionKey,
        *,
        interval_quantile: float | None = None,
    ) -> ContrastEstimate:
        vector = self.contrast_vector(key_switch, key_keep)
        mean = float(vector @ self.mean)
        solved = _solve_cholesky(self._Lambda, vector)
        variance = max(0.0, float(vector @ solved))
        quantile = (
            self.calibration_quantile
            if interval_quantile is None
            else _finite_positive(interval_quantile, "interval_quantile")
        )
        radius = quantile * math.sqrt(variance)
        return ContrastEstimate(
            mean=mean,
            variance=variance,
            lower=mean - radius,
            upper=mean + radius,
        )

    def contrast(
        self,
        key_switch: DecisionKey,
        key_keep: DecisionKey,
        *,
        interval_quantile: float | None = None,
    ) -> ContrastEstimate:
        return self.predict_contrast(
            key_switch, key_keep, interval_quantile=interval_quantile
        )

    def sample_parameters(
        self, rng: np.random.Generator, *, size: int = 1
    ) -> np.ndarray:
        """Draw joint parameter samples with one shared draw per alternative set."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if size <= 0:
            raise ValueError("size must be positive")
        if self.dimension == 0:
            return np.zeros((size, 0), dtype=np.float64)
        factor = np.linalg.cholesky(self._Lambda)
        noise = rng.standard_normal((size, self.dimension))
        transformed = np.linalg.solve(factor.T, noise.T).T
        return self.mean[None, :] + transformed

    def _touch_and_activate(
        self, key_keep: DecisionKey, key_switch: DecisionKey
    ) -> tuple[str, ...]:
        touched = set(self._key_interaction_columns(key_keep))
        touched.update(self._key_interaction_columns(key_switch))
        activated: list[str] = []
        for column in sorted(touched):
            count = self._interaction_touches.get(column, 0) + 1
            self._interaction_touches[column] = count
            if count >= self.n_act and self._ensure_column(column):
                activated.append(column)
        return tuple(activated)

    @staticmethod
    def _binary_returns(values: Iterable[int], name: str) -> np.ndarray:
        array = np.asarray(tuple(values), dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
        if not np.all(np.isin(array, (0.0, 1.0))):
            raise ValueError(f"{name} must contain only binary terminal outcomes")
        return array

    def update_probe(
        self,
        key_keep: DecisionKey,
        key_switch: DecisionKey,
        returns_keep: Iterable[int],
        returns_switch: Iterable[int],
    ) -> ProbeUpdate:
        """Update the comparison posterior from one same-snapshot paired probe."""

        self._changed_factor(key_switch, key_keep)
        keep = self._binary_returns(returns_keep, "returns_keep")
        switch = self._binary_returns(returns_switch, "returns_switch")
        if keep.size != switch.size:
            raise ValueError("paired probe branches must use the same repeat count")
        self.register_key(key_keep)
        self.register_key(key_switch)
        activated = self._touch_and_activate(key_keep, key_switch)
        vector = self.contrast_vector(key_switch, key_keep)
        keep_rate = float(np.mean(keep))
        switch_rate = float(np.mean(switch))
        delta_hat = switch_rate - keep_rate
        repeats = int(keep.size)
        noise_var = (
            switch_rate * (1.0 - switch_rate)
            + keep_rate * (1.0 - keep_rate)
        ) / repeats + self.variance_floor
        self._Lambda += np.outer(vector, vector) / noise_var
        self._eta += vector * delta_hat / noise_var
        self._probe_count += 1
        return ProbeUpdate(
            delta_hat=delta_hat,
            noise_var=noise_var,
            keep_rate=keep_rate,
            switch_rate=switch_rate,
            repeats=repeats,
            activated_columns=activated,
        )

    def _aggregate_baseline(
        self, entries: Sequence[tuple[int, float, int]]
    ) -> tuple[float, int]:
        observed_sum = sum(total for _, total, _ in entries)
        observed_count = sum(count for _, _, count in entries)
        total_count = self.baseline_prior_count + observed_count
        total_sum = self.baseline_prior_mean * self.baseline_prior_count + observed_sum
        return total_sum / total_count, total_count

    def _baseline_for(self, task_family: str) -> tuple[float, int]:
        family = _non_empty(task_family, "task_family")
        return self._aggregate_baseline(self._baseline_history.get(family, ()))

    def expected_terminal(
        self,
        task_family: str,
        prefix_keys: Sequence[DecisionKey],
        *,
        parameter: Iterable[float] | np.ndarray | None = None,
    ) -> float:
        """Return clipped ``Q_hat(prefix)`` from baseline plus configuration effect."""

        baseline, _ = self._baseline_for(task_family)
        vector = self.prefix_vector(prefix_keys)
        theta = self.mean if parameter is None else np.asarray(parameter, dtype=np.float64)
        if theta.shape != (self.dimension,) or not np.all(np.isfinite(theta)):
            raise ValueError(f"parameter must be finite with shape {(self.dimension,)}")
        return float(np.clip(baseline + vector @ theta, 0.02, 0.98))

    def update_task_baseline(
        self,
        task_family: str,
        prefix_keys: Sequence[DecisionKey],
        terminal_reward: int,
    ) -> float:
        """Update the family running mean after removing current configuration effect."""

        if terminal_reward not in (0, 1):
            raise ValueError("terminal_reward must be binary")
        family = _non_empty(task_family, "task_family")
        vector = self.prefix_vector(prefix_keys)
        residual = float(terminal_reward - vector @ self.mean)
        entries = self._baseline_history.setdefault(family, [])
        for index, (epoch, total, count) in enumerate(entries):
            if epoch == self.epoch:
                entries[index] = (epoch, total + residual, count + 1)
                break
        else:
            entries.append((self.epoch, residual, 1))
        return residual

    def _ensure_sensor(self, sensor_class: str) -> np.ndarray:
        name = _non_empty(sensor_class, "sensor_class")
        if name not in self._sensor_counts:
            self._sensor_counts[name] = np.ones((2, 2), dtype=np.float64)
        return self._sensor_counts[name]

    def update_sensor(self, surface: SurfaceSignal, terminal_reward: int) -> None:
        if not isinstance(surface, SurfaceSignal):
            raise TypeError("surface must be SurfaceSignal")
        if terminal_reward not in (0, 1):
            raise ValueError("terminal_reward must be binary")
        counts = self._ensure_sensor(surface.sensor_class)
        row = 0 if surface.value else 1
        column = 0 if terminal_reward == 1 else 1
        counts[row, column] += 1.0

    def update_trajectory(
        self,
        task_family: str,
        prefix_keys: Sequence[DecisionKey],
        surfaces: Iterable[SurfaceSignal],
        terminal_reward: int,
    ) -> float:
        """Update only the absolute baseline and sensors from a valid trajectory."""

        residual = self.update_task_baseline(
            task_family, prefix_keys, terminal_reward
        )
        for surface in surfaces:
            self.update_sensor(surface, terminal_reward)
        return residual

    def sensor_rates(self, sensor_class: str) -> tuple[float, float]:
        counts = self._ensure_sensor(sensor_class)
        n11, n10 = counts[0]
        n01, n00 = counts[1]
        sensitivity = float(n11 / (n11 + n01))
        false_positive_rate = float(n10 / (n10 + n00))
        return sensitivity, false_positive_rate

    def likelihood_ratio(self, surface: SurfaceSignal) -> float:
        if not isinstance(surface, SurfaceSignal):
            raise TypeError("surface must be SurfaceSignal")
        sensitivity, false_positive_rate = self.sensor_rates(surface.sensor_class)
        if surface.value:
            return sensitivity / false_positive_rate
        return (1.0 - sensitivity) / (1.0 - false_positive_rate)

    def sample_sensor_rates(
        self,
        sensor_class: str,
        rng: np.random.Generator,
        *,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if size <= 0:
            raise ValueError("size must be positive")
        counts = self._ensure_sensor(sensor_class)
        sensitivity = rng.beta(counts[0, 0], counts[1, 0], size=size)
        false_positive_rate = rng.beta(counts[0, 1], counts[1, 1], size=size)
        return sensitivity, false_positive_rate

    def sample_sensor(
        self,
        surface: SurfaceSignal,
        rng: np.random.Generator,
        *,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(surface, SurfaceSignal):
            raise TypeError("surface must be SurfaceSignal")
        return self.sample_sensor_rates(surface.sensor_class, rng, size=size)

    def empirical_bayes_update(self) -> dict[str, float]:
        """Refresh factor prior variances while retaining accumulated data precision."""

        if self.dimension == 0:
            return {**self._sigma_f2, "interaction": self._sigma_eta2}
        mean = self.mean
        covariance_diagonal = np.diag(self.covariance)
        updated_factors: dict[str, float] = {}
        for factor in FIRST_ORDER_FACTORS:
            indices = [
                index
                for column, index in self._columns.items()
                if self._column_factor(column) == factor
            ]
            if indices:
                value = max(
                    float(np.var(mean[indices]))
                    - float(np.mean(covariance_diagonal[indices])),
                    1e-3,
                )
                updated_factors[factor] = value
            else:
                updated_factors[factor] = self._sigma_f2[factor]
        interaction_indices = [
            index
            for column, index in self._columns.items()
            if self._column_factor(column) == "interaction"
        ]
        if interaction_indices:
            updated_interaction = max(
                float(np.var(mean[interaction_indices]))
                - float(np.mean(covariance_diagonal[interaction_indices])),
                1e-3,
            )
        else:
            updated_interaction = self._sigma_eta2

        data_precision = self._Lambda - np.diag(self._prior_precision_diagonal)
        self._sigma_f2 = updated_factors
        self._sigma_eta2 = updated_interaction
        new_prior = np.array(
            [
                1.0 / self._prior_variance_for_column(column)
                for column, _ in sorted(self._columns.items(), key=lambda item: item[1])
            ],
            dtype=np.float64,
        )
        self._prior_precision_diagonal = new_prior
        self._Lambda = data_precision + np.diag(new_prior)
        self._Lambda = 0.5 * (self._Lambda + self._Lambda.T)
        np.linalg.cholesky(self._Lambda)
        return {**self._sigma_f2, "interaction": self._sigma_eta2}

    def _trim_baseline_history(self) -> None:
        minimum_epoch = max(0, self.epoch - 2)
        for family in tuple(self._baseline_history):
            retained = [
                entry
                for entry in self._baseline_history[family]
                if entry[0] >= minimum_epoch
            ]
            if retained:
                self._baseline_history[family] = retained
            else:
                del self._baseline_history[family]

    def _reset_statistics_to_prior(self) -> None:
        self._Lambda = np.diag(self._prior_precision_diagonal)
        self._eta = np.zeros(self.dimension, dtype=np.float64)
        self._baseline_history.clear()
        self._sensor_counts = {
            sensor_class: np.ones((2, 2), dtype=np.float64)
            for sensor_class in self._sensor_counts
        }

    def refresh_policy(
        self,
        new_policy_version: str,
        *,
        heldout_coverage: float | None = None,
        decay: float = 0.1,
    ) -> float:
        """Apply epoch-boundary decay after the frozen policy triple changes."""

        policy_version = _non_empty(new_policy_version, "new_policy_version")
        standard_decay = float(decay)
        if not 0.0 <= standard_decay <= 1.0:
            raise ValueError("decay must lie in [0, 1]")
        if heldout_coverage is not None:
            coverage = float(heldout_coverage)
            if not 0.0 <= coverage <= 1.0:
                raise ValueError("heldout_coverage must lie in [0, 1]")
            coverage_floor = 1.0 - self.calibration_alpha - 0.1
            if coverage < coverage_floor:
                self._low_coverage_epochs += 1
                standard_decay = 0.5
            else:
                self._low_coverage_epochs = 0

        self.epoch += 1
        self.policy_version = policy_version
        if self._low_coverage_epochs >= 2:
            self._reset_statistics_to_prior()
        else:
            prior = np.diag(self._prior_precision_diagonal)
            self._Lambda = (1.0 - standard_decay) * self._Lambda + standard_decay * prior
            self._eta *= 1.0 - standard_decay
            for counts in self._sensor_counts.values():
                counts *= 0.9
        self._trim_baseline_history()
        return standard_decay

    def snapshot(self) -> LedgerState:
        return LedgerState(
            epoch=self.epoch,
            policy_version=self.policy_version,
            columns=self._columns,
            Lambda=self._Lambda,
            eta=self._eta,
            sigma_f2=self._sigma_f2,
            sigma_eta2=self._sigma_eta2,
            task_baseline=self.task_baseline,
            sensor_counts=self._sensor_counts,
            calibration_quantile=self.calibration_quantile,
            interaction_touches=self._interaction_touches,
            baseline_history={
                family: tuple(entries)
                for family, entries in self._baseline_history.items()
            },
            probe_count=self._probe_count,
            low_coverage_epochs=self._low_coverage_epochs,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: LedgerState,
        *,
        n_act: int = 5,
        variance_floor: float = 0.01,
        baseline_prior_mean: float = 0.5,
        baseline_prior_count: int = 4,
        calibration_alpha: float = 0.05,
    ) -> "CombinationPosterior":
        ledger = cls(
            snapshot.policy_version,
            epoch=snapshot.epoch,
            n_act=n_act,
            variance_floor=variance_floor,
            initial_factor_variance=next(iter(snapshot.sigma_f2.values())),
            initial_interaction_variance=snapshot.sigma_eta2,
            baseline_prior_mean=baseline_prior_mean,
            baseline_prior_count=baseline_prior_count,
            calibration_quantile=snapshot.calibration_quantile,
            calibration_alpha=calibration_alpha,
        )
        ledger._columns = dict(snapshot.columns)
        ledger._Lambda = np.array(snapshot.Lambda, copy=True)
        ledger._eta = np.array(snapshot.eta, copy=True)
        ledger._sigma_f2 = dict(snapshot.sigma_f2)
        ledger._sigma_eta2 = snapshot.sigma_eta2
        ledger._prior_precision_diagonal = np.array(
            [
                1.0 / ledger._prior_variance_for_column(column)
                for column, _ in sorted(ledger._columns.items(), key=lambda item: item[1])
            ],
            dtype=np.float64,
        )
        ledger._interaction_touches = dict(snapshot.interaction_touches)
        ledger._baseline_history = {
            family: list(entries) for family, entries in snapshot.baseline_history.items()
        }
        if not ledger._baseline_history and snapshot.task_baseline:
            # Compatibility for snapshots that contain only the aggregate.
            for family, (mean, count) in snapshot.task_baseline.items():
                observed_count = max(0, count - ledger.baseline_prior_count)
                if observed_count:
                    observed_sum = (
                        mean * count
                        - ledger.baseline_prior_mean * ledger.baseline_prior_count
                    )
                    ledger._baseline_history[family] = [
                        (ledger.epoch, observed_sum, observed_count)
                    ]
        ledger._sensor_counts = {
            sensor_class: np.array(counts, copy=True)
            for sensor_class, counts in snapshot.sensor_counts.items()
        }
        ledger._probe_count = snapshot.probe_count
        ledger._low_coverage_epochs = snapshot.low_coverage_epochs
        return ledger

    def state_dict(self) -> dict[str, object]:
        result = self.snapshot().to_dict()
        result["hyperparameters"] = {
            "n_act": self.n_act,
            "variance_floor": self.variance_floor,
            "baseline_prior_mean": self.baseline_prior_mean,
            "baseline_prior_count": self.baseline_prior_count,
            "calibration_alpha": self.calibration_alpha,
        }
        return result

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "CombinationPosterior":
        snapshot = LedgerState.from_dict(state)
        hyperparameters = dict(state.get("hyperparameters", {}))
        return cls.from_snapshot(
            snapshot,
            n_act=int(hyperparameters.get("n_act", 5)),
            variance_floor=float(hyperparameters.get("variance_floor", 0.01)),
            baseline_prior_mean=float(hyperparameters.get("baseline_prior_mean", 0.5)),
            baseline_prior_count=int(hyperparameters.get("baseline_prior_count", 4)),
            calibration_alpha=float(hyperparameters.get("calibration_alpha", 0.05)),
        )

    def dumps(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def loads(cls, payload: str) -> "CombinationPosterior":
        state = json.loads(payload)
        if not isinstance(state, dict):
            raise ValueError("serialized ledger must be a JSON object")
        return cls.from_state_dict(state)
