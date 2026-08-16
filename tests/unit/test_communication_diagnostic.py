from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnose_hotpotqa_communication.py"
SPEC = importlib.util.spec_from_file_location("communication_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trajectory(task_id: str, *, multi_agent: bool) -> dict:
    nodes = [
        {"id": "out", "model_id": "m1", "contract": "answer"},
    ]
    relations = []
    if multi_agent:
        nodes.insert(0, {"id": "evidence", "model_id": "m1", "contract": "facts"})
        relations.append(
            {
                "source_id": "evidence",
                "target_id": "out",
                "source_to_target": True,
                "target_to_source": False,
            }
        )
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "explicit_finish": True,
        "final_answer": "<answer>a</answer>",
        "turns": [
            {
                "graph_snapshot": {
                    "nodes": nodes,
                    "relations": relations,
                    "output_agent_id": "out",
                    "revision": 1,
                }
            }
        ],
    }


def arm(diagnostic_id: str, answer: str, em: float, f1: float) -> dict:
    return {
        "diagnostic_id": diagnostic_id,
        "final_answer": f"<answer>{answer}</answer>",
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": em, "token_f1": f1},
            "details": {"scored_prediction": answer},
        },
    }


class CommunicationDiagnosticTests(unittest.TestCase):
    def test_selection_is_structural_and_preserves_source_order(self) -> None:
        values = [
            trajectory("single", multi_agent=False),
            trajectory("first", multi_agent=True),
            trajectory("second", multi_agent=True),
        ]
        selected = MODULE.select_diagnostic_trajectories(values, 1)
        self.assertEqual(["trajectory-first"], [item["trajectory_id"] for item in selected])

    def test_pair_record_is_diagnostic_only_and_uses_existing_evaluator_metrics(self) -> None:
        result = MODULE.paired_result(
            pair_id="pair-1",
            source_trajectory_id="source-1",
            task_id="task-1",
            normal=arm("normal-1", "The Alpha", 1.0, 1.0),
            masked=arm("masked-1", "Beta", 0.0, 0.0),
        )
        self.assertTrue(result["normal_correct_to_masked_wrong"])
        self.assertTrue(result["normalized_answer_changed"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["grpo_eligible"])
        self.assertEqual(-1.0, result["delta_masked_minus_normal"]["exact_match"])

    def test_diagnostic_outputs_cannot_overlap_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(MODULE.CommunicationDiagnosticError):
                MODULE._validate_isolated_paths(
                    [root / "trajectory.jsonl"],
                    [root / "trajectory.jsonl"],
                )


if __name__ == "__main__":
    unittest.main()
