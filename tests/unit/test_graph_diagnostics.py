from __future__ import annotations

import unittest

from src.interactive.graph_diagnostics import (
    _runtime_delivery_evidence,
    aggregate_trajectory_diagnostics,
    diagnose_trajectory,
    graph_from_receipt,
)


def trajectory() -> dict[str, object]:
    graph = {
        "nodes": [
            {
                "id": "a",
                "model_id": "m",
                "contract": "produce evidence",
                "role_family": "evidence",
            },
            {
                "id": "b",
                "model_id": "m",
                "contract": "consume a evidence",
                "role_family": "format",
            },
        ],
        "relations": [
            {
                "source_id": "a",
                "target_id": "b",
                "source_to_target": True,
                "target_to_source": False,
            }
        ],
        "output_agent_id": "b",
        "revision": 4,
    }
    actions = [
        {"action": "add_agent"},
        {"action": "add_agent"},
        {"action": "set_relation"},
        {"action": "set_output"},
        {"action": "finish"},
    ]
    turns: list[dict[str, object]] = []
    for index, action in enumerate(actions):
        executions: list[dict[str, object]] = []
        if index == 3:
            executions = [
                {
                    "execution_id": "execution-a",
                    "agent_id": "a",
                    "metadata": {
                        "request": {
                            "graph_revision": 4,
                            "upstream": [],
                        }
                    },
                },
                {
                    "execution_id": "execution-b",
                    "agent_id": "b",
                    "metadata": {
                        "request": {
                            "graph_revision": 4,
                            "upstream": [
                                {
                                    "source_agent_id": "a",
                                    "target_agent_id": "b",
                                    "graph_revision": 4,
                                    "content": "evidence artifact",
                                }
                            ],
                        }
                    },
                },
            ]
        turns.append(
            {
                "action": action,
                "canvas_feedback": (
                    "workflow finished"
                    if index == 4
                    else f"accepted {action['action']} at revision {min(index + 1, 4)}"
                ),
                "executions": executions,
                "graph_snapshot": graph,
                "prompt": (
                    "Choose one\n\n"
                    '{"max_rounds":20,"recent_canvas_history":[]}'
                ),
            }
        )
    return {
        "task": {"task_id": "hotpotqa:test"},
        "turns": turns,
        "explicit_finish": True,
        "termination_reason": "finish",
    }


class GraphDiagnosticTests(unittest.TestCase):
    def test_peer_draft_delivery_uses_runtime_evidence_constraints(self) -> None:
        graph = graph_from_receipt(
            {
                "nodes": [
                    {"id": "a", "model_id": "m", "contract": ""},
                    {"id": "b", "model_id": "m", "contract": ""},
                ],
                "relations": [
                    {
                        "source_id": "a",
                        "target_id": "b",
                        "source_to_target": True,
                        "target_to_source": True,
                    }
                ],
                "output_agent_id": "b",
                "revision": 4,
            }
        )
        def request(
            peer_draft: dict[str, object], *, include_upstream: bool = True
        ) -> dict[str, object]:
            return {
                "graph_revision": 4,
                "upstream": (
                    [
                        {
                            "source_agent_id": "a",
                            "target_agent_id": "b",
                            "graph_revision": 4,
                            "content": "upstream artifact",
                        },
                        {
                            "source_agent_id": "b",
                            "target_agent_id": "a",
                            "graph_revision": 4,
                            "content": "duplicate peer artifact",
                        },
                    ]
                    if include_upstream
                    else []
                ),
                "peer_draft": peer_draft,
            }

        valid_peer = {
            "source_agent_id": "b",
            "target_agent_id": "a",
            "graph_revision": 4,
            "content": "peer artifact",
        }
        evidence = _runtime_delivery_evidence(
            [
                {
                    "executions": [
                        {
                            "execution_id": "valid",
                            "metadata": {"request": request(valid_peer)},
                        },
                        {
                            "execution_id": "wrong-revision",
                            "metadata": {
                                "request": request(
                                    {**valid_peer, "graph_revision": 3},
                                    include_upstream=False,
                                )
                            },
                        },
                        {
                            "execution_id": "non-edge",
                            "metadata": {
                                "request": request(
                                    {**valid_peer, "target_agent_id": "missing"},
                                    include_upstream=False,
                                )
                            },
                        },
                    ]
                }
            ],
            graph,
        )

        self.assertEqual(
            [("a", "b", "weak", "valid"), ("b", "a", "weak", "valid")],
            [
                (item.source_id, item.target_id, item.status, item.evidence_id)
                for item in evidence
            ],
        )

    def test_runtime_delivery_is_weak_not_verified_dependency_evidence(self) -> None:
        item = diagnose_trajectory(trajectory())

        self.assertEqual(2, item.structural_depth)
        self.assertEqual(("evidence", "format"), item.role_families)
        self.assertEqual(2, item.effective_dependency_depth)
        self.assertEqual("weak", item.effective_dependency_status)
        self.assertEqual("weak", item.full_structural_depth_evidence_status)
        self.assertEqual(1, item.execution_turn_count)
        self.assertEqual(2, item.executor_call_count)
        self.assertEqual(5, item.minimum_final_construction_actions)
        self.assertEqual(0, item.turn_overhead)

    def test_aggregate_reports_atomic_cost_and_distributions(self) -> None:
        report = aggregate_trajectory_diagnostics([trajectory()])

        self.assertEqual(1, report["task_count"])
        self.assertEqual({"2": 1}, report["agent_count_distribution"])
        self.assertEqual({"2": 1}, report["structural_depth_distribution"])
        self.assertEqual({"serial_2": 1}, report["topology_family_distribution"])
        self.assertEqual(
            {"evidence": 1, "format": 1},
            report["role_family_distribution"],
        )
        self.assertEqual(0, report["rejected_turn_count"])
        self.assertEqual(0, report["three_plus_agent_count"])
        self.assertEqual(
            5.0,
            report["atomic_cost_by_agent_count"]["2"][
                "mean_minimum_final_construction_actions"
            ],
        )


if __name__ == "__main__":
    unittest.main()
