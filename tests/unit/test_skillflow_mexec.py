from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

# Load the isolated adapter without importing ``src.interactive.__init__``.
# The base test image carries jsonschema 3.x, whereas that package initializer
# correctly requires Draft 2020-12 from the project's declared runtime.  This
# unit targets only the dependency-free MExec/parser boundary added here.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "interactive"
    / "skillflow_mexec.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_flowsteer_skillflow_mexec_test_target",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SkillFlowEditParseError = _MODULE.SkillFlowEditParseError
SkillFlowExactEdit = _MODULE.SkillFlowExactEdit
SkillFlowExactEditGenerator = _MODULE.SkillFlowExactEditGenerator
SkillFlowMExec = _MODULE.SkillFlowMExec
SkillFlowMExecError = _MODULE.SkillFlowMExecError
build_skillflow_edit_prompt = _MODULE.build_skillflow_edit_prompt
parse_skillflow_exact_edit = _MODULE.parse_skillflow_exact_edit


class _StubMExec:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def execute(self, **arguments: object) -> str:
        self.calls.append(dict(arguments))
        return self.response


class _FakeCompletions:
    def __init__(self, response: object = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **arguments: object) -> object:
        self.calls.append(dict(arguments))
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class SkillFlowExactEditParserTests(unittest.TestCase):
    def test_parses_fenced_balanced_and_python_literal_objects(self) -> None:
        fenced = parse_skillflow_exact_edit(
            'preface\n```json\n{"old_content":"left - right",'
            '"new_content":"left + right"}\n```\n'
        )
        self.assertEqual(
            SkillFlowExactEdit("left - right", "left + right"),
            fenced,
        )

        literal = parse_skillflow_exact_edit(
            "{'old_content': 'return False', 'new_content': 'return True'}"
        )
        self.assertEqual("return False", literal.old_content)
        self.assertEqual("return True", literal.new_content)

    def test_repairs_only_the_upstream_missing_final_string_quote_case(self) -> None:
        repaired = parse_skillflow_exact_edit(
            '{"old_content":"return left - right",'
            '"new_content":"return left + right}'
        )
        self.assertEqual("return left - right", repaired.old_content)
        self.assertEqual("return left + right", repaired.new_content)

    def test_rejects_execution_errors_non_strings_and_extra_keys(self) -> None:
        with self.assertRaises(SkillFlowMExecError):
            parse_skillflow_exact_edit("[EXECUTION_ERROR] request_failed")
        with self.assertRaises(SkillFlowEditParseError):
            parse_skillflow_exact_edit(
                '{"old_content":1,"new_content":"replacement"}'
            )
        with self.assertRaises(SkillFlowEditParseError):
            parse_skillflow_exact_edit(
                '{"old_content":"old","new_content":"new","path":"x.py"}'
            )
        with self.assertRaises(SkillFlowEditParseError):
            parse_skillflow_exact_edit("no object here")


class SkillFlowEditPromptTests(unittest.TestCase):
    def test_prompt_preserves_upstream_bounds_and_source_separation(self) -> None:
        issue = "I" * 2700
        excerpt = "A" * 6500 + "M" * 2000 + "Z" * 6500
        recent = "R" * 2200
        prompt = build_skillflow_edit_prompt(
            issue=issue,
            path="pkg/query.py",
            target_excerpt=excerpt,
            instruction="Change QuerySet.ordered using the viewed local condition.",
            recent_context=recent,
        )

        self.assertIn("[ISSUE TRUNCATED]", prompt)
        self.assertNotIn("I" * 2501, prompt)
        self.assertIn("[TRUNCATED 3000 chars of middle]", prompt)
        self.assertIn("A" * 6000, prompt)
        self.assertIn("Z" * 6000, prompt)
        self.assertNotIn("M" * 100, prompt)
        self.assertIn("R" * 2000, prompt)
        self.assertNotIn("R" * 2001, prompt)
        self.assertIn("read-only; NOT the edit target", prompt)
        self.assertIn("Target file: pkg/query.py", prompt)
        self.assertIn("Return ONLY valid JSON with exactly these keys", prompt)

    def test_public_callable_uses_code_generation_mexec_contract(self) -> None:
        stub = _StubMExec(
            '{"old_content":"return left - right",'
            '"new_content":"return left + right"}'
        )
        generate = SkillFlowExactEditGenerator(stub)

        edit = generate(
            issue="Addition subtracts the operands.",
            path="pkg/maths.py",
            target_excerpt="def add(left, right):\n    return left - right\n",
            instruction="In add, return the sum instead of the difference.",
            recent_context="search_code found add at line 1",
        )

        self.assertEqual("return left - right", edit.old_content)
        self.assertEqual("return left + right", edit.new_content)
        self.assertEqual(1, len(stub.calls))
        call = stub.calls[0]
        self.assertEqual("", call["context"])
        self.assertEqual("code_generation", call["task_type"])
        self.assertEqual(4096, call["max_tokens"])
        self.assertIn("Addition subtracts the operands.", call["instruction"])


class SkillFlowMExecTransportTests(unittest.TestCase):
    def test_local_openai_transport_matches_upstream_request(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "<think>private scratchpad</think>\n"
                            '{"old_content":"old","new_content":"new"}'
                        ),
                        reasoning_content=None,
                    )
                )
            ]
        )
        completions = _FakeCompletions(response=response)
        factory_calls: list[dict[str, object]] = []

        def factory(**arguments: object) -> _FakeClient:
            factory_calls.append(dict(arguments))
            return _FakeClient(completions)

        executor = SkillFlowMExec(
            "http://127.0.0.1:8015/v1/",
            "supervisor_theta",
            api_key="EMPTY",
            client_factory=factory,
        )
        raw = executor.execute(
            "produce an edit",
            context="bounded context",
            task_type="code_generation",
            max_tokens=4096,
        )

        self.assertEqual('{"old_content":"old","new_content":"new"}', raw)
        self.assertEqual("http://127.0.0.1:8015/v1", factory_calls[0]["base_url"])
        self.assertEqual(0, factory_calls[0]["max_retries"])
        request = completions.calls[0]
        self.assertEqual("supervisor_theta", request["model"])
        self.assertEqual(0.1, request["temperature"])
        self.assertEqual(4096, request["max_tokens"])
        self.assertEqual(
            {"chat_template_kwargs": {"enable_thinking": False}},
            request["extra_body"],
        )
        self.assertEqual(
            ["system", "user", "assistant", "user"],
            [message["role"] for message in request["messages"]],
        )
        self.assertEqual(1, executor.stats["total_calls"])
        self.assertEqual(0, executor.stats["total_errors"])

    def test_transport_failure_is_fail_closed_without_an_api_call_in_test(self) -> None:
        completions = _FakeCompletions(error=RuntimeError("offline fake"))
        executor = SkillFlowMExec(
            "http://127.0.0.1:8015/v1",
            "supervisor_theta",
            max_retries=1,
            retry_delay=0,
            client_factory=lambda **_: _FakeClient(completions),
        )

        with self.assertRaises(SkillFlowMExecError):
            executor.execute("produce an edit", task_type="code_generation")
        self.assertEqual(1, executor.stats["total_errors"])


if __name__ == "__main__":
    unittest.main()
