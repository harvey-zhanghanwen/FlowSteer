from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "interactive"
    / "healthbench_professional_grader.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "healthbench_professional_grader_under_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_GRADER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GRADER
_SPEC.loader.exec_module(_GRADER)

HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION = (
    _GRADER.HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION
)
HEALTHBENCH_PROFESSIONAL_GRADER_MODEL = (
    _GRADER.HEALTHBENCH_PROFESSIONAL_GRADER_MODEL
)
HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER = (
    _GRADER.HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER
)
HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS = (
    _GRADER.HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS
)
HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT = (
    _GRADER.HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT
)
HealthBenchProfessionalGrader = _GRADER.HealthBenchProfessionalGrader
HealthBenchProfessionalGraderError = _GRADER.HealthBenchProfessionalGraderError
InMemoryJSONTransport = _GRADER.InMemoryJSONTransport

_WORKER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "healthbench_professional_grader_worker.py"
)
_WORKER_SPEC = importlib.util.spec_from_file_location(
    "healthbench_professional_grader_worker_under_test", _WORKER_PATH
)
assert _WORKER_SPEC is not None and _WORKER_SPEC.loader is not None
_WORKER = importlib.util.module_from_spec(_WORKER_SPEC)
sys.modules[_WORKER_SPEC.name] = _WORKER
_WORKER_SPEC.loader.exec_module(_WORKER)


def _receipt(
    request: dict,
    *,
    termination: str = "graded",
    grader_model: str = HEALTHBENCH_PROFESSIONAL_GRADER_MODEL,
) -> bytes:
    answer = request["candidate_answer"]
    raw_score = 0.75 if termination == "graded" else None
    adjusted_score = (
        raw_score
        - HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS
        * ((len(answer) - HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER) / 500.0)
        if raw_score is not None
        else None
    )
    rubric_receipts = (
        [
            {
                **item,
                "criteria_met": item["points"] > 0,
                "explanation": "synthetic grader explanation",
            }
            for item in request["private_case"]["rubrics"]
        ]
        if termination == "graded"
        else []
    )
    value = {
        "api_call_receipts": [
            {
                "api_call_index": 1,
                "attempt": 1,
                "latency_ms": 12.0,
                "status": "success",
                "token_usage": {
                    "input_tokens": 40,
                    "input_cached_tokens": 0,
                    "output_tokens": 10,
                    "output_reasoning_tokens": 2,
                    "total_tokens": 50,
                },
            }
        ],
        "evaluator_version": HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION,
        "grader_api_calls": 1,
        "grader_error": (
            None
            if termination == "graded"
            else {"error_type": "SyntheticProviderError", "message": "failed"}
        ),
        "grader_latency_ms": 12.0,
        "grader_model": grader_model,
        "grader_reasoning_effort": HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT,
        "grader_token_usage": {
            "input_tokens": 40,
            "input_cached_tokens": 0,
            "output_tokens": 10,
            "output_reasoning_tokens": 2,
            "total_tokens": 50,
        },
        "length_adjustment_center": HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER,
        "length_adjustment_penalty_per_500_chars": (
            HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS
        ),
        "overall_score": raw_score,
        "overall_score_length_adjusted": adjusted_score,
        "provider_errors": (
            []
            if termination == "graded"
            else [
                {
                    "api_call_index": 1,
                    "error_type": "SyntheticProviderError",
                    "message": "failed",
                    "request_id": None,
                    "status_code": 503,
                }
            ]
        ),
        "response_characters": len(answer),
        "rubric_level_receipts": rubric_receipts,
        "task_id": request["task_id"],
        "termination": termination,
        "triggered_negative_rubric_count": 0,
    }
    return json.dumps(value, separators=(",", ":")).encode()


class HealthBenchProfessionalGraderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.official_source = self.root / "simple-evals"
        self.official_source.mkdir()
        (self.official_source / "healthbench_eval.py").write_text(
            "# pinned test source\n", encoding="utf-8"
        )
        self.private_cases = self.root / "private_cases.jsonl"
        self.private_cases.write_text(
            json.dumps(
                {
                    "id": "healthbench-professional:case-1",
                    "conversation": {
                        "messages": [
                            {"role": "user", "content": "Private conversation"}
                        ]
                    },
                    "rubric_items": [
                        {"criterion_text": "Addresses the request", "points": 5},
                        {"criterion_text": "Contains unsafe advice", "points": -2},
                    ],
                    "physician_response": "Evaluator-only reference",
                    "specialty": "Evaluator-only metadata",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _grader(self, transport: InMemoryJSONTransport) -> HealthBenchProfessionalGrader:
        return HealthBenchProfessionalGrader.from_private_cases_jsonl(
            private_cases_path=self.private_cases,
            official_source_root=self.official_source,
            interpreter_path=Path(sys.executable).resolve(),
            transport=transport,
        )

    async def test_private_join_projects_only_official_grader_inputs(self) -> None:
        transport = InMemoryJSONTransport(
            lambda encoded: _receipt(json.loads(encoded))
        )
        grader = self._grader(transport)

        result = await grader.grade(
            "healthbench-professional:case-1", "Candidate assistant response"
        )

        self.assertEqual("graded", result["termination"])
        self.assertEqual(0.75, result["overall_score"])
        self.assertEqual(2, len(result["rubric_level_receipts"]))
        request = json.loads(transport.requests[0])
        self.assertEqual("grade", request["operation"])
        self.assertEqual(
            {"prompt", "rubrics"}, set(request["private_case"])
        )
        serialized = json.dumps(request, ensure_ascii=False)
        self.assertNotIn("Evaluator-only reference", serialized)
        self.assertNotIn("Evaluator-only metadata", serialized)
        self.assertEqual(
            "Addresses the request",
            request["private_case"]["rubrics"][0]["criterion"],
        )

    async def test_preflight_uses_synthetic_case_not_public_test_store(self) -> None:
        transport = InMemoryJSONTransport(
            lambda encoded: _receipt(json.loads(encoded))
        )
        grader = self._grader(transport)

        result = await grader.preflight()

        request = json.loads(transport.requests[0])
        self.assertEqual("preflight", request["operation"])
        self.assertEqual(
            "healthbench-professional:synthetic-preflight", request["task_id"]
        )
        self.assertNotIn("Private conversation", json.dumps(request))
        self.assertEqual("graded", result["termination"])
        self.assertEqual(
            HEALTHBENCH_PROFESSIONAL_GRADER_MODEL, result["grader_model"]
        )
        self.assertEqual(
            HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT,
            result["grader_reasoning_effort"],
        )

    async def test_versioned_non_reference_grader_is_explicitly_bound(self) -> None:
        grader_model = "gpt-5.4"
        transport = InMemoryJSONTransport(
            lambda encoded: _receipt(
                json.loads(encoded), grader_model=grader_model
            )
        )
        grader = HealthBenchProfessionalGrader.from_private_cases_jsonl(
            private_cases_path=self.private_cases,
            official_source_root=self.official_source,
            interpreter_path=Path(sys.executable).resolve(),
            grader_model=grader_model,
            transport=transport,
        )

        result = await grader.preflight()

        self.assertEqual(grader_model, result["grader_model"])
        self.assertEqual(grader_model, grader.grader_model)
        self.assertEqual(
            ["--grader-model", grader_model],
            list(grader.worker.command[-2:]),
        )

    async def test_provider_failure_is_an_invalid_receipt_not_score_zero(self) -> None:
        transport = InMemoryJSONTransport(
            lambda encoded: _receipt(json.loads(encoded), termination="grader_error")
        )
        grader = self._grader(transport)

        result = await grader.grade(
            "healthbench-professional:case-1", "Candidate assistant response"
        )

        self.assertEqual("grader_error", result["termination"])
        self.assertIsNone(result["overall_score"])
        self.assertIsNone(result["overall_score_length_adjusted"])
        self.assertEqual("SyntheticProviderError", result["grader_error"]["error_type"])
        self.assertEqual(503, result["provider_errors"][0]["status_code"])

    async def test_unknown_task_never_falls_back_to_another_private_case(self) -> None:
        transport = InMemoryJSONTransport(
            lambda encoded: _receipt(json.loads(encoded))
        )
        grader = self._grader(transport)

        with self.assertRaisesRegex(
            HealthBenchProfessionalGraderError, "not privately routed"
        ):
            await grader.grade("healthbench-professional:missing", "Candidate")
        self.assertEqual([], transport.requests)

    async def test_response_contract_rejects_wrong_model_or_field_set(self) -> None:
        def wrong_model(encoded: bytes) -> bytes:
            request = json.loads(encoded)
            value = json.loads(_receipt(request))
            value["grader_model"] = "unapproved-grader"
            return json.dumps(value).encode()

        grader = self._grader(InMemoryJSONTransport(wrong_model))
        with self.assertRaisesRegex(
            HealthBenchProfessionalGraderError, "grader_model differs"
        ):
            await grader.grade(
                "healthbench-professional:case-1", "Candidate assistant response"
            )

    def test_duplicate_private_task_id_is_rejected(self) -> None:
        duplicated = self.private_cases.read_text(encoding="utf-8") * 2
        self.private_cases.write_text(duplicated, encoding="utf-8")
        transport = InMemoryJSONTransport(
            lambda encoded: _receipt(json.loads(encoded))
        )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self._grader(transport)


class BoundedResponsesSamplerTests(unittest.TestCase):
    def test_exact_model_low_reasoning_and_bounded_json_repair(self) -> None:
        calls: list[dict] = []
        outputs = iter(
            [
                SimpleNamespace(output_text="not json", usage=None),
                SimpleNamespace(
                    output_text='{"explanation":"fixed","criteria_met":true}',
                    usage=None,
                ),
            ]
        )

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return next(outputs)

        sampler = _WORKER._BoundedResponsesSampler(
            client=SimpleNamespace(responses=Responses()),
            sampler_response_type=SimpleNamespace,
            max_parse_attempts=2,
            max_calls=2,
        )
        messages = [{"role": "user", "content": "Synthetic rubric prompt"}]

        first = sampler(messages)
        second = sampler(messages)

        self.assertEqual("not json", first.response_text)
        self.assertIn("criteria_met", second.response_text)
        self.assertEqual(HEALTHBENCH_PROFESSIONAL_GRADER_MODEL, calls[0]["model"])
        self.assertEqual(
            {"effort": HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT},
            calls[0]["reasoning"],
        )
        self.assertEqual("assistant", calls[1]["input"][-2]["role"])
        self.assertIn("Repair", calls[1]["input"][-1]["content"])
        with self.assertRaisesRegex(RuntimeError, "bounded JSON parse attempts"):
            sampler(messages)

    def test_transient_provider_error_is_retried_with_receipts(self) -> None:
        calls = 0

        class Responses:
            def create(self, **kwargs):
                nonlocal calls
                del kwargs
                calls += 1
                if calls == 1:
                    raise RuntimeError("temporary provider error")
                return SimpleNamespace(
                    output_text=(
                        '{"explanation":"recovered","criteria_met":true}'
                    ),
                    usage=None,
                )

        sampler = _WORKER._BoundedResponsesSampler(
            client=SimpleNamespace(responses=Responses()),
            sampler_response_type=SimpleNamespace,
            max_parse_attempts=2,
            max_calls=6,
            max_provider_attempts=3,
        )
        response = sampler([{"role": "user", "content": "rubric"}])

        self.assertIn("criteria_met", response.response_text)
        self.assertEqual(2, sampler.calls)
        self.assertEqual(1, len(sampler.provider_errors))
        self.assertEqual("provider_error", sampler.api_call_receipts[0]["status"])
        self.assertEqual("success", sampler.api_call_receipts[1]["status"])
        self.assertEqual(2, sampler.api_call_receipts[1]["provider_attempt"])


if __name__ == "__main__":
    unittest.main()
