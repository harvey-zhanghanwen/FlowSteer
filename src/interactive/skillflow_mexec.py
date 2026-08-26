"""SkillFlow MExec transport and exact-edit response adapter.

Source map (DIRECT_REUSE):

* ``SkillFlow/src/executor/m_exec.py``: ``MExec`` message construction,
  OpenAI-compatible local transport, disabled thinking, retries, and response
  text extraction.
* ``SkillFlow/training/environment.py::_m_exec_generate_edit``: exact-edit
  prompt, bounded issue/source/recent context, and tolerant JSON extraction.

This module deliberately stops at ``old_content``/``new_content`` generation.
Repository mutation, syntax checking, diff materialization, and evaluator
receipts remain responsibilities of the task-scoped repository backend.  It
does not add an Agent role or a Director action: MExec is the private executor
behind SkillFlow's Agent-visible ``edit_file(path, instruction)`` Tool.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
import re
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol

try:  # Runtime dependency; kept lazy-testable for repository-only unit tests.
    from openai import APIError as _OpenAIAPIError
except ImportError:  # pragma: no cover - exercised only without runtime extras.
    class _OpenAIAPIError(Exception):
        pass


def _openai_client_factory(**arguments: object) -> Any:
    """Construct the same upstream OpenAI client only when runtime needs it."""

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - configuration failure.
        raise SkillFlowMExecError(
            "the openai runtime dependency is required for local MExec"
        ) from exc
    return OpenAI(**arguments)


_SYSTEM_PROMPT = (
    "You are a capable AI assistant. "
    "Execute the given instruction accurately and completely. "
    "Be concise but thorough. Do not add unnecessary caveats or disclaimers."
)

_EXECUTION_ERROR_PREFIX = "[EXECUTION_ERROR]"


class SkillFlowMExecError(RuntimeError):
    """Raised when the local executor fails and evaluation must fail closed."""


class SkillFlowEditParseError(ValueError):
    """Raised when MExec does not return one strict exact-edit object."""


class MExecTransport(Protocol):
    """Small synchronous boundary implemented by :class:`SkillFlowMExec`."""

    def execute(
        self,
        instruction: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "",
    ) -> str:
        ...


@dataclass(frozen=True, slots=True)
class SkillFlowExactEdit:
    """Strict parsed result from SkillFlow's semantic ``edit_file`` Tool."""

    old_content: str
    new_content: str

    def __post_init__(self) -> None:
        if not isinstance(self.old_content, str):
            raise TypeError("old_content must be text")
        if not isinstance(self.new_content, str):
            raise TypeError("new_content must be text")

    def to_dict(self) -> dict[str, str]:
        return {
            "old_content": self.old_content,
            "new_content": self.new_content,
        }


class SkillFlowMExec:
    """Synchronous SkillFlow MExec client for a local OpenAI-compatible server.

    ``client_factory`` is an injection seam for deterministic tests.  Runtime
    callers normally leave it unset so the upstream ``openai.OpenAI`` client is
    used against the configured local SGLang ``/v1`` endpoint.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "",
        default_temperature: float = 0.1,
        default_max_tokens: int = 8192,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_timeout: float = 60.0,
        fail_closed: bool = True,
        *,
        client_factory: Callable[..., Any] = _openai_client_factory,
    ) -> None:
        if not isinstance(api_base, str) or not api_base.strip():
            raise ValueError("api_base must be non-empty text")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty text")
        if default_temperature < 0:
            raise ValueError("default_temperature must be non-negative")
        if default_max_tokens < 1:
            raise ValueError("default_max_tokens must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if type(fail_closed) is not bool:
            raise TypeError("fail_closed must be boolean")

        self._api_base = api_base.rstrip("/")
        self._api_key = (
            api_key
            or os.environ.get("MEXEC_API_KEY")
            or os.environ.get("SGLANG_API_KEY", "EMPTY")
        )
        self._thread_local = threading.local()
        self._client_factory = client_factory
        self.model_name = model_name
        self.default_temperature = float(default_temperature)
        self.default_max_tokens = int(default_max_tokens)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.request_timeout = float(request_timeout)
        self.fail_closed = fail_closed
        self._total_calls = 0
        self._total_errors = 0

    @property
    def client(self) -> Any:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = self._client_factory(
                base_url=self._api_base,
                api_key=self._api_key,
                timeout=self.request_timeout,
                max_retries=0,
            )
            self._thread_local.client = client
        return client

    def execute(
        self,
        instruction: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "",
    ) -> str:
        """Execute one upstream-compatible MExec completion synchronously."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty text")
        if not isinstance(context, str):
            raise TypeError("context must be text")
        requested_max_tokens = (
            int(max_tokens) if max_tokens is not None else self.default_max_tokens
        )
        if requested_max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        requested_temperature = (
            float(temperature)
            if temperature is not None
            else self.default_temperature
        )
        if requested_temperature < 0:
            raise ValueError("temperature must be non-negative")

        messages = self._build_messages(instruction, context)
        self._total_calls += 1
        last_error: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=requested_temperature,
                    max_tokens=requested_max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                )
                return self._response_text(response)
            except _OpenAIAPIError as exc:
                self._total_errors += 1
                last_error = exc
                if attempt < self.max_retries - 1 and self.retry_delay:
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as exc:
                self._total_errors += 1
                if self.fail_closed:
                    raise SkillFlowMExecError(
                        f"MExec {task_type or 'request'} failed: "
                        f"{type(exc).__name__}"
                    ) from exc
                return f"{_EXECUTION_ERROR_PREFIX} {type(exc).__name__}"

        if self.fail_closed:
            raise SkillFlowMExecError(
                f"MExec {task_type or 'request'} exhausted retries"
            ) from last_error
        return f"{_EXECUTION_ERROR_PREFIX} Max retries exceeded"

    @property
    def stats(self) -> Mapping[str, float | int]:
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(1, self._total_calls),
        }

    @staticmethod
    def _build_messages(instruction: str, context: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Here is some relevant context:\n\n{context}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Understood. I have read the context and will use it "
                        "to answer."
                    ),
                }
            )
        messages.append({"role": "user", "content": instruction})
        return messages

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @classmethod
    def _response_text(cls, response: Any) -> str:
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise SkillFlowMExecError("MExec response has no completion message") from exc
        result = cls._strip_thinking(getattr(message, "content", None) or "")
        if not result:
            result = getattr(message, "reasoning_content", None) or ""
        if not isinstance(result, str) or not result.strip():
            raise SkillFlowMExecError("MExec response has no text content")
        return result.strip()


def _looks_like_edit_object(text: str) -> bool:
    return (
        ('"old_content"' in text or "'old_content'" in text)
        and ('"new_content"' in text or "'new_content'" in text)
    )


def _balanced_json_candidates(text: str) -> list[str]:
    """DIRECT_REUSE of SkillFlow's quote-aware balanced-object scan."""

    output: list[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                quote = ""
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                in_string = False
            continue
        if character in ("'", '"'):
            in_string = True
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                if _looks_like_edit_object(candidate):
                    output.append(candidate)
                start = None
    return output


def _repair_missing_final_string_quote(text: str) -> Optional[dict[str, str]]:
    """DIRECT_REUSE of SkillFlow's bounded final-string quote repair."""

    value = (text or "").strip()
    brace = value.find("{")
    if brace > 0:
        value = value[brace:].strip()
    if not (
        value.startswith("{")
        and "old_content" in value
        and "new_content" in value
    ):
        return None
    decoder = json.JSONDecoder()
    old_match = re.search(r'"old_content"\s*:\s*', value)
    if not old_match:
        return None
    try:
        old_value, old_end = decoder.raw_decode(value, old_match.end())
    except Exception:
        return None
    new_match = re.search(r'"new_content"\s*:\s*', value[old_end:])
    if not new_match:
        return None
    new_start = old_end + new_match.end()
    try:
        new_value, _ = decoder.raw_decode(value, new_start)
        if isinstance(old_value, str) and isinstance(new_value, str):
            return {"old_content": old_value, "new_content": new_value}
    except Exception:
        pass

    if new_start >= len(value) or value[new_start] != '"':
        return None
    tail = value[new_start + 1 :].rstrip()
    if not tail.endswith("}"):
        return None
    tail = tail[:-1].rstrip()
    if not tail:
        return None
    try:
        new_value = json.loads('"' + tail + '"')
    except Exception:
        try:
            new_value = json.loads('"' + tail.replace("\n", "\\n") + '"')
        except Exception:
            return None
    if isinstance(old_value, str) and isinstance(new_value, str):
        return {"old_content": old_value, "new_content": new_value}
    return None


def parse_skillflow_exact_edit(raw: str) -> SkillFlowExactEdit:
    """Parse MExec output, then require exactly two string-valued fields.

    Candidate extraction and the narrow quote repair are direct SkillFlow
    reuse.  Exact-key/type validation is the fail-closed adapter boundary that
    the upstream prompt requires but its episode method leaves to later code.
    """

    if not isinstance(raw, str):
        raise SkillFlowEditParseError("MExec output must be text")
    if raw.strip().startswith(_EXECUTION_ERROR_PREFIX):
        raise SkillFlowMExecError(raw.strip())

    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", raw):
        fenced = match.group(1).strip()
        if fenced:
            candidates.extend(_balanced_json_candidates(fenced))
            if fenced.startswith("{") and _looks_like_edit_object(fenced):
                candidates.append(fenced)
    candidates.extend(_balanced_json_candidates(raw))
    broad = re.search(
        r'\{[\s\S]*(?:"old_content"|\'old_content\')[\s\S]*'
        r'(?:"new_content"|\'new_content\')[\s\S]*\}',
        raw,
    )
    if broad:
        candidates.append(broad.group(0))
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        repaired = _repair_missing_final_string_quote(raw)
        if repaired is not None:
            return _strict_edit_object(repaired)
        raw_text = raw.strip()
        if (
            raw_text.startswith("{")
            and "old_content" in raw_text
            and "new_content" in raw_text
        ):
            raise SkillFlowEditParseError(
                "partial JSON object in MExec output"
            )
        raise SkillFlowEditParseError("no exact-edit JSON in MExec output")

    parsed: object | None = None
    last_error: BaseException | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
        try:
            parsed = json.loads(candidate.replace("\r", ""))
            break
        except json.JSONDecodeError as exc:
            last_error = exc
        try:
            literal = ast.literal_eval(candidate)
            if isinstance(literal, dict):
                parsed = literal
                break
        except Exception as exc:
            last_error = exc

    if parsed is None:
        for candidate in [*candidates, raw]:
            parsed = _repair_missing_final_string_quote(candidate)
            if parsed is not None:
                break
    if parsed is None:
        raise SkillFlowEditParseError(
            f"invalid exact-edit JSON: {type(last_error).__name__}"
        ) from last_error
    return _strict_edit_object(parsed)


def _strict_edit_object(value: object) -> SkillFlowExactEdit:
    if not isinstance(value, Mapping):
        raise SkillFlowEditParseError("exact-edit JSON must be an object")
    if set(value) != {"old_content", "new_content"}:
        raise SkillFlowEditParseError(
            "exact-edit JSON must contain only old_content and new_content"
        )
    old_content = value["old_content"]
    new_content = value["new_content"]
    if not isinstance(old_content, str) or not isinstance(new_content, str):
        raise SkillFlowEditParseError(
            "old_content and new_content must both be strings"
        )
    return SkillFlowExactEdit(old_content, new_content)


def build_skillflow_edit_prompt(
    *,
    issue: str,
    path: str,
    target_excerpt: str,
    instruction: str,
    recent_context: str = "",
) -> str:
    """Build the bounded prompt used by SkillFlow's ``edit_file`` backend."""

    for field_name, value in (
        ("issue", issue),
        ("path", path),
        ("target_excerpt", target_excerpt),
        ("instruction", instruction),
        ("recent_context", recent_context),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be text")
    if not path.strip():
        raise ValueError("path must be non-empty text")
    if not instruction.strip():
        raise ValueError("instruction must be non-empty text")

    max_chars = 12_000
    if len(target_excerpt) > max_chars:
        half = max_chars // 2
        content_for_prompt = (
            target_excerpt[:half]
            + "\n\n... [TRUNCATED "
            + str(len(target_excerpt) - max_chars)
            + " chars of middle] ...\n\n"
            + target_excerpt[-half:]
        )
    else:
        content_for_prompt = target_excerpt
    issue_text = issue
    if len(issue_text) > 2500:
        issue_text = issue_text[:2500] + "\n[ISSUE TRUNCATED]"
    extra_context_block = ""
    if recent_context:
        extra_context_block = (
            "Additional recent source/search context (read-only; NOT the edit "
            "target; do not copy old_content from here):\n"
            f"```\n{recent_context[:2000]}\n```\n\n"
        )

    return (
        "You are a precise source-code editor. Produce one minimal exact "
        "string replacement.\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"old_content": "...", "new_content": "..."}\n\n'
        "Source of truth: the issue text and the target file/excerpt below. "
        "Do not use tests, gold patches, or hidden feedback.\n\n"
        f"Issue:\n{issue_text}\n\n"
        f"{extra_context_block}"
        f"Target file: {path}\n"
        f"Target file content/excerpt:\n```\n{content_for_prompt}\n```\n\n"
        f"Requested change:\n{instruction}\n\n"
        "Rules:\n"
        "1. old_content must be copied verbatim from the target file/excerpt; "
        "do not include displayed line numbers.\n"
        "2. Keep old_content short and unique, usually 1-12 lines. For an "
        "insertion, replace a small anchor block with anchor+inserted code.\n"
        "3. new_content is replacement text, not a diff. Preserve indentation, "
        "imports, aliases, and local API style.\n"
        "4. Make the smallest source-code change that fixes the issue; do not "
        "edit tests/docs/examples or unrelated behavior.\n"
        '5. If a safe exact edit is not possible in this target file, return '
        '{"old_content":"","new_content":""}.\n'
        "6. The JSON must be complete and parseable; escape newlines as \\n "
        'and quotes as \\".\n'
    )


class SkillFlowExactEditGenerator:
    """Public synchronous callable for semantic request -> exact replacement."""

    def __init__(self, m_exec: MExecTransport) -> None:
        if not callable(getattr(m_exec, "execute", None)):
            raise TypeError("m_exec must provide a synchronous execute method")
        self._m_exec = m_exec

    def __call__(
        self,
        *,
        issue: str,
        path: str,
        target_excerpt: str,
        instruction: str,
        recent_context: str = "",
    ) -> SkillFlowExactEdit:
        prompt = build_skillflow_edit_prompt(
            issue=issue,
            path=path,
            target_excerpt=target_excerpt,
            instruction=instruction,
            recent_context=recent_context,
        )
        raw = self._m_exec.execute(
            instruction=prompt,
            context="",
            task_type="code_generation",
            max_tokens=4096,
        )
        return parse_skillflow_exact_edit(raw)


__all__ = [
    "MExecTransport",
    "SkillFlowEditParseError",
    "SkillFlowExactEdit",
    "SkillFlowExactEditGenerator",
    "SkillFlowMExec",
    "SkillFlowMExecError",
    "build_skillflow_edit_prompt",
    "parse_skillflow_exact_edit",
]
