"""Phase-B tests for ledger readouts and posterior latent risk."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.interactive.exploration.combination_ledger import DecisionKey, SurfaceSignal
from src.interactive.exploration.latent_loss import (
    CandidateEstimate,
    LatentLossConfig,
    changed_field,
    joint_maximum_estimate,
    post_execution_latent_loss,
    pre_execution_readout,
    single_field_candidates,
    straddle_score,
)


def _key(
    *,
    role: str = "solve",
    model: str = "A",
    edge: str = "independent",
    same: bool = False,
) -> DecisionKey:
    return DecisionKey(
        task_family="hotpotqa",
        role_cluster=role,
        model_id=model,
        edge_type=edge,
        same_model_as_upstream=same,
        stage="other",
    )


class _PosteriorStub:
    """Small frozen Gaussian/sensor posterior implementing the public ledger API."""

    def __init__(
        self,
        *,
        mean: tuple[float, ...] = (0.30, 0.0, 0.0, 0.0),
        covariance_scale: float = 1e-8,
        probe_count: int = 10,
        q0: float = 0.5,
        sensor_rates: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.covariance = covariance_scale * np.eye(self.mean.size)
        self.probe_count = probe_count
        self._q0 = q0
        self._sensor_rates = sensor_rates or {}

    def design_vector(self, key: DecisionKey) -> np.ndarray:
        return np.asarray(
            [
                float(key.model_id == "B"),
                float(key.edge_type == "bidirectional"),
                float(key.role_cluster == "verify"),
                float(key.same_model_as_upstream),
            ],
            dtype=np.float64,
        )

    def contrast_vector(self, switch: DecisionKey, keep: DecisionKey) -> np.ndarray:
        return self.design_vector(switch) - self.design_vector(keep)

    def predict_contrast(self, switch: DecisionKey, keep: DecisionKey) -> object:
        vector = self.contrast_vector(switch, keep)
        mean = float(vector @ self.mean)
        variance = float(vector @ self.covariance @ vector)
        radius = 1.6448536269514722 * np.sqrt(variance)
        return SimpleNamespace(
            mean=mean,
            variance=variance,
            lower=mean - radius,
            upper=mean + radius,
        )

    def sample_parameters(
        self, rng: np.random.Generator, *, size: int
    ) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.covariance, size=size)

    def expected_terminal(
        self,
        task_family: str,
        prefix_keys: tuple[DecisionKey, ...],
        parameter: np.ndarray | None = None,
    ) -> float:
        assert task_family == "hotpotqa"
        assert prefix_keys
        return self._q0

    def sample_sensor_rates(
        self,
        sensor_class: str,
        rng: np.random.Generator,
        *,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del rng
        sensitivity, false_positive_rate = self._sensor_rates.get(
            sensor_class, (0.5, 0.5)
        )
        return (
            np.full(size, sensitivity, dtype=np.float64),
            np.full(size, false_positive_rate, dtype=np.float64),
        )

    def likelihood_ratio(self, surface: SurfaceSignal) -> float:
        sensitivity, false_positive_rate = self._sensor_rates.get(
            surface.sensor_class, (0.5, 0.5)
        )
        if surface.value:
            return sensitivity / false_positive_rate
        return (1.0 - sensitivity) / (1.0 - false_positive_rate)


def test_candidate_set_changes_exactly_one_field_and_deduplicates() -> None:
    current = _key()
    candidates = single_field_candidates(
        current,
        model_ids=["A", "B", "B"],
        edge_types=["unidirectional", "bidirectional"],
        role_clusters=["solve", "verify"],
        same_model_options=[False, True],
    )

    assert len(candidates) == 5
    assert {changed_field(current, candidate) for candidate in candidates} == {
        "model_id",
        "edge_type",
        "role_cluster",
        "same_model_as_upstream",
    }
    with pytest.raises(ValueError, match="exactly one"):
        changed_field(
            current,
            _key(role="verify", model="B"),
        )


def test_pre_readout_ranks_top_k_without_triggering_a_revision() -> None:
    current = _key()
    candidates = single_field_candidates(
        current,
        model_ids=["B"],
        edge_types=["bidirectional"],
        role_clusters=["verify"],
    )
    posterior = _PosteriorStub(mean=(0.30, 0.10, 0.20, 0.0))
    readout = pre_execution_readout(
        posterior,
        step_id=2,
        current=current,
        candidates=candidates,
        config=LatentLossConfig(top_k_candidates=2),
    )

    assert [item.changed_field for item in readout.candidates] == [
        "model_id",
        "role_cluster",
    ]
    assert "[Ledger] step 2 candidates" in readout.text
    assert "model=B" in readout.text
    assert "risk=" not in readout.text


def test_same_model_self_check_pass_has_high_latent_risk() -> None:
    current = _key(same=True)
    candidate = _key(model="B", same=True)
    posterior = _PosteriorStub(
        sensor_rates={"verifier_pass|same": (0.75, 0.60)}
    )
    result = post_execution_latent_loss(
        posterior,
        step_id=3,
        prefix_before=(),
        current=current,
        candidates=[candidate],
        surface=SurfaceSignal(
            kind="verifier_pass",
            value=True,
            sensor_class="verifier_pass|same",
        ),
        remaining_rounds=3,
        rng=np.random.default_rng(7),
    )

    assert result.computed
    assert result.risk_probability is not None
    assert result.risk_probability >= 0.8
    assert result.warning
    assert "You may revise the last decision or continue." in result.text


def test_independent_check_pass_has_low_latent_risk() -> None:
    current = _key(same=False)
    candidate = _key(model="B", same=False)
    posterior = _PosteriorStub(
        sensor_rates={"verifier_pass|different": (0.90, 0.10)}
    )
    result = post_execution_latent_loss(
        posterior,
        step_id=1,
        prefix_before=(),
        current=current,
        candidates=[candidate],
        surface=SurfaceSignal(
            kind="verifier_pass",
            value=True,
            sensor_class="verifier_pass|different",
        ),
        remaining_rounds=3,
        rng=np.random.default_rng(8),
    )

    assert result.computed
    assert result.risk_probability is not None
    assert result.risk_probability <= 0.2
    assert not result.warning
    assert result.text == ""


def test_empty_ledger_stays_in_intermediate_region() -> None:
    current = _key()
    candidate = _key(model="B")
    posterior = _PosteriorStub(
        mean=(0.0, 0.0, 0.0, 0.0),
        covariance_scale=0.25,
        probe_count=0,
        sensor_rates={"none|different": (0.5, 0.5)},
    )
    result = post_execution_latent_loss(
        posterior,
        step_id=0,
        prefix_before=(),
        current=current,
        candidates=[candidate],
        surface=SurfaceSignal(
            kind="none",
            value=False,
            sensor_class="none|different",
        ),
        remaining_rounds=4,
        rng=np.random.default_rng(19),
        config=LatentLossConfig(posterior_samples=4_000),
    )

    assert result.risk_probability is not None
    assert 0.2 < result.risk_probability < 0.8
    assert not result.warning
    assert result.candidates[0].unknown


def test_joint_sampling_maximum_has_mathematically_correct_direction() -> None:
    current = _key()
    candidates = [_key(model="B"), _key(edge="bidirectional")]
    posterior = _PosteriorStub(
        mean=(0.0, 0.0, 0.0, 0.0), covariance_scale=1.0, probe_count=0
    )
    estimate = joint_maximum_estimate(
        posterior,
        current=current,
        candidates=candidates,
        rng=np.random.default_rng(23),
        samples=20_000,
    )

    # Jensen's inequality gives E[max X] >= max E[X].  The reversed sentence
    # in Phase-B item 4 of the supplied specification is therefore treated as
    # an ambiguity, while the required joint posterior sampling is preserved.
    assert estimate.posterior_expected_maximum > estimate.plug_in_maximum + 0.4


def test_abnormal_surface_defers_to_canvas_recovery() -> None:
    current = _key()
    result = post_execution_latent_loss(
        _PosteriorStub(),
        step_id=1,
        prefix_before=(),
        current=current,
        candidates=[_key(model="B")],
        surface=SurfaceSignal(
            kind="exec_error",
            value=True,
            sensor_class="exec_error|different",
        ),
        remaining_rounds=3,
        rng=np.random.default_rng(1),
    )

    assert not result.computed
    assert result.reason == "abnormal_surface"
    assert result.risk_probability is None
    assert not result.warning


def test_high_risk_warning_is_suppressed_when_revision_is_impossible() -> None:
    current = _key(same=True)
    posterior = _PosteriorStub(
        sensor_rates={"verifier_pass|same": (0.75, 0.60)}
    )
    result = post_execution_latent_loss(
        posterior,
        step_id=4,
        prefix_before=(),
        current=current,
        candidates=[_key(model="B", same=True)],
        surface=SurfaceSignal(
            kind="verifier_pass",
            value=True,
            sensor_class="verifier_pass|same",
        ),
        remaining_rounds=1,
        rng=np.random.default_rng(2),
    )

    assert result.risk_probability is not None
    assert result.risk_probability >= 0.8
    assert result.warning_suppressed_by_round_budget
    assert not result.warning
    assert result.reason == "round_budget"


def test_straddle_uses_only_intervals_crossing_delta_min() -> None:
    base = _key()
    crossing = CandidateEstimate(
        key=_key(model="B"),
        changed_field="model_id",
        changed_value="B",
        mean=0.06,
        variance=0.01,
        lower=-0.10,
        upper=0.20,
    )
    non_crossing = CandidateEstimate(
        key=_key(edge="bidirectional"),
        changed_field="edge_type",
        changed_value="bidirectional",
        mean=0.25,
        variance=0.01,
        lower=0.10,
        upper=0.40,
    )
    assert base != crossing.key
    assert straddle_score([crossing, non_crossing], delta_min=0.05) == pytest.approx(
        0.30
    )

