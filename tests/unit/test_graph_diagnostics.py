from __future__ import annotations

import unittest

from src.interactive.graph_diagnostics import (
    aggregate_trajectory_diagnostics,
    diagnose_trajectory,
)


def trajectory() -> dict[str, object]:
    graph = {
        "nodes": [
            {"id": "a", "model_id": "m", "contract": "produce evidence"},
            {"id": "b", "model_id": "m", "contract": "consume a evidence"},
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
    def test_runtime_delivery_is_weak_not_verified_dependency_evidence(self) -> None:
        item = diagnose_trajectory(trajectory())

        self.assertEqual(2, item.structural_depth)
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
