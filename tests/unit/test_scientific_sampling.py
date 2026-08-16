from __future__ import annotations

import random

from src.interactive.scientific_sampling import (
    GenerationPhase,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
)


def coordinate(
    task_id: str = "hotpotqa:one",
    *,
    condition: str = "architecture-dev",
    rollout_ordinal: int = 0,
    anchor: int = 0,
) -> ScientificSamplingCoordinate:
    return ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=17),
        schedule_purpose=condition,
        ordered_sequence_hash=stable_hash([task_id]),
        sequence_position=rollout_ordinal,
        task_id=task_id,
        optimizer_step_or_anchor_ordinal=anchor,
    )


def test_skillflow_seed_is_stable_distinct_and_does_not_touch_global_rng() -> None:
    random.seed(314159)
    state = random.getstate()
    value = coordinate()

    first = derive_generation_seed(
        base_seed=17,
        coordinate=value,
        step_index=1,
        phase=GenerationPhase.ACTION,
    )
    repeated = derive_generation_seed(
        base_seed=17,
        coordinate=value,
        step_index=1,
        phase=GenerationPhase.ACTION,
    )
    next_turn = derive_generation_seed(
        base_seed=17,
        coordinate=value,
        step_index=2,
        phase=GenerationPhase.ACTION,
    )
    reasoning = derive_generation_seed(
        base_seed=17,
        coordinate=value,
        step_index=1,
        phase=GenerationPhase.REASONING,
    )

    assert first == repeated
    assert len({first, next_turn, reasoning}) == 3
    assert random.getstate() == state


def test_task_condition_rollout_anchor_and_turn_are_result_coordinates() -> None:
    variants = (
        coordinate(),
        coordinate("hotpotqa:two"),
        coordinate(condition="confirmation"),
        coordinate(rollout_ordinal=1),
        coordinate(anchor=1),
    )
    seeds = {
        derive_generation_seed(
            base_seed=17,
            coordinate=value,
            step_index=1,
            phase=GenerationPhase.ACTION,
        )
        for value in variants
    }
    assert len(seeds) == len(variants)


def test_same_task_seed_is_independent_of_selected_task_order_or_subset() -> None:
    task_id = "hotpotqa:one"

    def task_seeds(selected: list[str]) -> tuple[int, ...]:
        assert task_id in selected
        # Evaluation assigns rollout ordinal zero per task.  The selected-list
        # position is deliberately absent from the coordinate.
        value = coordinate(task_id, rollout_ordinal=0)
        return tuple(
            derive_generation_seed(
                base_seed=17,
                coordinate=value,
                step_index=step,
                phase=GenerationPhase.ACTION,
            )
            for step in (1, 2, 3)
        )

    assert task_seeds([task_id, "hotpotqa:two"]) == task_seeds(
        ["hotpotqa:two", task_id]
    )
    assert task_seeds([task_id]) == task_seeds([task_id, "hotpotqa:two"])


def test_coordinate_round_trip_preserves_skillflow_wire_fields() -> None:
    value = coordinate(rollout_ordinal=3, anchor=2)
    assert ScientificSamplingCoordinate.from_value(value.to_value()) == value
