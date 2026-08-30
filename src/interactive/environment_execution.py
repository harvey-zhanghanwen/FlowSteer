"""Bounded ReAct execution for RAGEN-compatible benchmark environments.

The control flow is a thin adaptation of SkillFlow's ``BoundedAgent`` and
``RolloutEnvironmentSession`` contracts: one model turn selects one public
environment action, the action is executed, and only the resulting public
observation is returned to the next model turn.  The concrete ALFWorld and
WebShop bridge remains the deployed ``RAGENAdapter``; evaluator labels and
terminal success predicates are deliberately outside this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
import inspect
import json
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Any, Optional, Protocol, Union

from .agent_runtime import AgentGateway, AgentRequest, AgentResponse, GatewayResponse
from .tool_runtime import (
    ActionKind,
    StructuredAction,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


class EnvironmentExecutionError(RuntimeError):
    """A public environment session could not complete its execution contract."""

    def __init__(
        self,
        message: str,
        *,
        environment_reset_receipt: Optional[Mapping[str, object]] = None,
        environment_receipts: Sequence[Mapping[str, object]] = (),
        environment_current_state: Optional[Mapping[str, object]] = None,
        evaluator_environment_trace: Sequence[Mapping[str, object]] = (),
        tool_receipts: Sequence[Mapping[str, object]] = (),
        model_calls: Sequence[Mapping[str, object]] = (),
        environment_revision: int = 0,
        environment_terminal: bool = False,
        cause_error_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        # DIRECT_REUSE: SkillFlow persists every completed Action--Observation
        # edge even when a bounded episode later fails.  Carry the same public
        # prefix through AgentRuntime; it is diagnostic state, never a terminal
        # artifact and never evaluator input.
        self.environment_reset_receipt = (
            None
            if environment_reset_receipt is None
            else dict(environment_reset_receipt)
        )
        self.environment_receipts = tuple(
            dict(item) for item in environment_receipts
        )
        self.environment_current_state = (
            None
            if environment_current_state is None
            else dict(environment_current_state)
        )
        self.evaluator_environment_trace = tuple(
            dict(item) for item in evaluator_environment_trace
        )
        self.tool_receipts = tuple(dict(item) for item in tool_receipts)
        self.model_calls = tuple(dict(item) for item in model_calls)
        self.environment_revision = environment_revision
        self.environment_terminal = environment_terminal
        self.cause_error_type = cause_error_type


class EnvironmentSession(Protocol):
    """Public RAGEN-compatible session; evaluator state is intentionally absent."""

    @property
    def environment_id(self) -> str: ...

    @property
    def task_family(self) -> str: ...

    @property
    def available_actions(self) -> object: ...

    def reset(self) -> Union[str, Awaitable[str]]: ...

    def step(
        self, action: str
    ) -> Union[
        tuple[str, object, bool, Mapping[str, object]],
        Awaitable[tuple[str, object, bool, Mapping[str, object]]],
    ]: ...


EnvironmentSessionFactory = Callable[[AgentRequest], EnvironmentSession]


@dataclass(slots=True)
class _EnvironmentTransition:
    # ``observation`` is retained verbatim for official evaluator replay.
    # ``public_observation`` is the projection returned to the Agent/Canvas.
    observation: str
    public_observation: str
    reward: object
    terminal: bool
    info: Mapping[str, object]


@dataclass(slots=True)
class _EnvironmentEpisode:
    session: EnvironmentSession
    observation: str
    raw_observation: str
    revision: int = 0
    pending_transition: Optional[_EnvironmentTransition] = None


@dataclass(slots=True)
class _EnvironmentRolloutState:
    """Public/evaluator state accumulated across one bounded episode.

    ``EnvironmentExecutionAdapter`` historically kept this state on the Python
    stack while one Agent invocation ran every environment turn.  The
    stepwise Director condition retains the same SkillFlow Action--Observation
    state across successive Canvas turns and commits exactly one turn per
    invocation.
    """

    episode: _EnvironmentEpisode
    episode_id: str
    original_task_instruction: str
    reset_receipt: dict[str, object]
    receipts: list[dict[str, object]] = field(default_factory=list)
    evaluator_trace: list[dict[str, object]] = field(default_factory=list)
    tool_receipts: list[dict[str, object]] = field(default_factory=list)
    model_calls: list[dict[str, object]] = field(default_factory=list)
    turns_used: int = 0
    terminal: bool = False


_EXECUTION_INTERFACE_SEPARATOR = "\n\nExecution interface:"


def _original_task_instruction(problem: str) -> str:
    """Return the benchmark instruction before the runtime interface suffix."""

    return str(problem).partition(_EXECUTION_INTERFACE_SEPARATOR)[0].strip()


_WEBSHOP_SCORE_MARKER = re.compile(
    r"Your\s+score\s*\(\s*min\s+0\.0\s*,\s*max\s+1\.0\s*\)",
    re.IGNORECASE,
)


def _public_environment_observation(
    task_family: str,
    observation: str,
    *,
    terminal: bool,
) -> str:
    """Remove WebShop's evaluator-only score block from public feedback.

    WebShop's upstream ``done_page.html`` renders purchased/target details in a
    hidden ``div`` and renders ``Your score`` in the visible terminal page.
    ``WebAgentTextEnv`` flattens both blocks into text.  The raw page is required
    for deterministic evaluator replay, while the public policy boundary may
    receive only the upstream visible acknowledgement without hidden target or
    reward fields.
    """

    if task_family.strip().lower() != "webshop" or not terminal:
        return observation
    acknowledgement = "Thank you for shopping with us!"
    if acknowledgement.casefold() in observation.casefold():
        return acknowledgement
    marker = _WEBSHOP_SCORE_MARKER.search(observation)
    if marker is None:
        return observation
    prefix = observation[: marker.start()]
    # ``observation_mode=text`` separates the HTML labels with ``[SEP]``;
    # ``text_rich`` uses whitespace/newlines.  Both formats originate from the
    # same upstream ``Reward`` label immediately before the score marker.
    prefix = re.sub(
        r"(?:\s*\[SEP\]\s*|\s+)Reward\s*(?:\[SEP\]\s*)?$",
        "",
        prefix,
        flags=re.IGNORECASE,
    )
    return prefix.rstrip()


class EnvironmentToolBackend:
    """Request-scoped backend for one real RAGEN environment resource.

    The backend is registered in the same :class:`ToolRegistry` exposed to the
    Director and AgentRuntime.  An episode is bound only while its execution
    adapter is active, so concurrent requests cannot address each other's
    simulator state.  Public ``ToolResult`` values intentionally omit reward
    and evaluator ``info``; the adapter consumes those fields through the
    private episode boundary solely to build deterministic evaluator replay.
    """

    def __init__(
        self,
        *,
        session_factory: EnvironmentSessionFactory,
        task_family: str,
        tool_id: Optional[str] = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not isinstance(task_family, str) or not task_family.strip():
            raise ValueError("task_family must be non-empty")
        family = task_family.strip().lower()
        resolved_tool_id = tool_id or f"{family}.environment"
        if not isinstance(resolved_tool_id, str) or not resolved_tool_id.strip():
            raise ValueError("tool_id must be non-empty")
        self.session_factory = session_factory
        self.task_family = family
        self.tool_id = resolved_tool_id.strip()
        self._episode: ContextVar[Optional[_EnvironmentEpisode]] = ContextVar(
            f"environment_episode_{id(self)}", default=None
        )

    async def begin(
        self, request: AgentRequest
    ) -> tuple[_EnvironmentEpisode, Token[Optional[_EnvironmentEpisode]]]:
        if self._episode.get() is not None:
            raise EnvironmentExecutionError(
                "environment episode is already active in this execution context"
            )
        session = self.session_factory(request)
        if str(session.task_family).strip().lower() != self.task_family:
            raise EnvironmentExecutionError(
                "environment session task family does not match its tool capability"
            )
        raw_observation = await _resolve(session.reset())
        if not isinstance(raw_observation, str):
            raise EnvironmentExecutionError("environment reset must return text")
        if raw_observation.startswith("[ENV_UNAVAILABLE]"):
            raise EnvironmentExecutionError(raw_observation)
        observation = _public_environment_observation(
            self.task_family,
            raw_observation,
            terminal=False,
        )
        episode = _EnvironmentEpisode(
            session=session,
            observation=observation,
            raw_observation=raw_observation,
        )
        return episode, self._episode.set(episode)

    def end(self, token: Token[Optional[_EnvironmentEpisode]]) -> None:
        self._episode.reset(token)

    def bind(
        self,
        episode: _EnvironmentEpisode,
    ) -> Token[Optional[_EnvironmentEpisode]]:
        """Bind an existing task-scoped episode for one serialized step."""

        if not isinstance(episode, _EnvironmentEpisode):
            raise TypeError("environment episode has an incompatible type")
        if self._episode.get() is not None:
            raise EnvironmentExecutionError(
                "environment episode is already active in this execution context"
            )
        return self._episode.set(episode)

    async def invoke(self, request: ToolRequest) -> ToolResult:
        episode = self._episode.get()
        if episode is None:
            raise EnvironmentExecutionError(
                "environment tool requires an active request-scoped episode"
            )
        if dict(request.arguments):
            raise EnvironmentExecutionError(
                "environment actions do not accept an arguments object"
            )
        admissible_actions, has_search_bar = _admissible_actions(
            self.task_family, episode.session.available_actions
        )
        if (
            _parse_action(
                request.action,
                task_family=self.task_family,
                admissible_actions=admissible_actions,
                webshop_has_search_bar=has_search_bar,
            )
            != request.action
        ):
            raise EnvironmentExecutionError(
                "environment tool action is not currently admissible"
            )
        transition = await _resolve(episode.session.step(request.action))
        if not isinstance(transition, tuple) or len(transition) != 4:
            raise EnvironmentExecutionError(
                "environment step must return observation, reward, terminal, and info"
            )
        raw_observation, reward, terminal, info = transition
        if not isinstance(raw_observation, str):
            raise EnvironmentExecutionError("environment observation must be text")
        if type(terminal) is not bool:
            raise EnvironmentExecutionError("environment terminal flag must be boolean")
        if not isinstance(info, Mapping):
            raise EnvironmentExecutionError("environment info must be a mapping")
        if raw_observation.startswith(("[ENV_UNAVAILABLE]", "[ERROR]")):
            raise EnvironmentExecutionError(raw_observation)
        public_observation = _public_environment_observation(
            self.task_family,
            raw_observation,
            terminal=terminal,
        )
        episode.revision += 1
        episode.observation = public_observation
        episode.raw_observation = raw_observation
        episode.pending_transition = _EnvironmentTransition(
            observation=raw_observation,
            public_observation=public_observation,
            reward=reward,
            terminal=terminal,
            info=MappingProxyType(dict(info)),
        )
        return ToolResult(
            {
                "environment_revision": episode.revision,
                "observation": public_observation,
                "terminal": terminal,
            }
        )

    def take_transition(self) -> _EnvironmentTransition:
        episode = self._episode.get()
        if episode is None or episode.pending_transition is None:
            raise EnvironmentExecutionError(
                "registered environment backend produced no transition"
            )
        transition = episode.pending_transition
        episode.pending_transition = None
        return transition


@dataclass(slots=True)
class RAGENEnvironmentSession:
    """Bind SkillFlow's deployed ``RAGENAdapter`` to one benchmark episode."""

    adapter: object
    env_type: str
    env_config: Mapping[str, object]
    question: str
    extra: Mapping[str, object]
    reset_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.env_type, str) or not self.env_type.strip():
            raise ValueError("env_type must be non-empty")
        if not hasattr(self.adapter, "reset") or not hasattr(self.adapter, "step"):
            raise TypeError("adapter must provide reset and step")
        self.env_type = self.env_type.strip()
        self.env_config = MappingProxyType(dict(self.env_config))
        self.extra = MappingProxyType(dict(self.extra))
        if self.reset_seed is not None and (
            isinstance(self.reset_seed, bool) or not isinstance(self.reset_seed, int)
        ):
            raise TypeError("reset_seed must be an integer when supplied")

    @property
    def environment_id(self) -> str:
        return f"ragen:{self.env_type}"

    @property
    def task_family(self) -> str:
        return self.env_type

    @property
    def available_actions(self) -> object:
        return getattr(self.adapter, "available_actions", ())

    def reset(self) -> str:
        # Match task_evaluator's WebShop boundary: imports and the dependency
        # check may mutate global RNG, so seed immediately before SimServer is
        # constructed by RAGENAdapter.reset.
        if self.reset_seed is not None:
            random.seed(self.reset_seed)
        return str(
            self.adapter.reset(  # type: ignore[attr-defined]
                self.env_type,
                dict(self.env_config),
                question=self.question,
                extra=dict(self.extra),
            )
        )

    def step(self, action: str) -> tuple[str, object, bool, Mapping[str, object]]:
        transition = self.adapter.step(action)  # type: ignore[attr-defined]
        if not isinstance(transition, tuple) or len(transition) != 4:
            raise EnvironmentExecutionError(
                "RAGEN step must return observation, reward, terminal, and info"
            )
        observation, reward, terminal, info = transition
        if type(terminal) is not bool:
            raise EnvironmentExecutionError("environment terminal flag must be boolean")
        if not isinstance(info, Mapping):
            raise EnvironmentExecutionError("environment info must be a mapping")
        return str(observation), reward, terminal, info


@dataclass(frozen=True, slots=True)
class RAGENEnvironmentSessionFactory:
    """Create one RAGEN episode for each AgentRuntime execution request."""

    adapter_factory: Callable[[], object]
    env_type: str
    env_config: Mapping[str, object]
    extra: Mapping[str, object] = field(default_factory=dict)
    question: Optional[str] = None
    reset_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not callable(self.adapter_factory):
            raise TypeError("adapter_factory must be callable")
        if not isinstance(self.env_type, str) or not self.env_type.strip():
            raise ValueError("env_type must be non-empty")
        object.__setattr__(self, "env_type", self.env_type.strip())
        object.__setattr__(self, "env_config", MappingProxyType(dict(self.env_config)))
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError("question must be text when supplied")
        if self.reset_seed is not None and (
            isinstance(self.reset_seed, bool) or not isinstance(self.reset_seed, int)
        ):
            raise TypeError("reset_seed must be an integer when supplied")

    def __call__(self, request: AgentRequest) -> RAGENEnvironmentSession:
        return RAGENEnvironmentSession(
            adapter=self.adapter_factory(),
            env_type=self.env_type,
            env_config=self.env_config,
            question=request.problem if self.question is None else self.question,
            extra=self.extra,
            reset_seed=self.reset_seed,
        )


def evaluator_locked_ragen_session_factory(
    *,
    record: object,
    dataset: str,
    ragen_adapter_path: Union[str, Path],
    max_environment_steps: Optional[int] = None,
) -> RAGENEnvironmentSessionFactory:
    """Build the live episode from task_evaluator's authoritative task lock.

    This deliberately reuses the evaluator's loader, aligned-record parser,
    ALFWorld inventory lock, and WebShop dependency boundary.  The resulting
    live trace can therefore be replayed by ``evaluate_task`` against a fresh
    identically locked environment instead of maintaining a second task-
    selection implementation here.
    """

    if not isinstance(dataset, str) or dataset.strip().lower() not in {
        "alfworld",
        "webshop",
    }:
        raise ValueError("dataset must be alfworld or webshop")
    if max_environment_steps is not None and (
        isinstance(max_environment_steps, bool)
        or not isinstance(max_environment_steps, int)
        or max_environment_steps < 1
    ):
        raise ValueError("max_environment_steps must be positive when supplied")

    # Local import keeps the generic environment execution module independent
    # of evaluator loading until a formal RAGEN session is explicitly built.
    from . import task_evaluator as evaluator

    normalized_dataset = dataset.strip().lower()
    module = evaluator._load_ragen_module(Path(ragen_adapter_path))
    env_type, config = evaluator._environment_config(record, normalized_dataset)
    if normalized_dataset == "alfworld":
        config, _ = evaluator._lock_alfworld_task(module, config)
        if (
            max_environment_steps is not None
            and "max_steps" in config
            and config["max_steps"] != max_environment_steps
        ):
            raise ValueError("ALFWorld record step limit does not match runtime budget")
        reset_seed = None
    else:
        check_webshop = getattr(module, "_check_webshop", None)
        if callable(check_webshop) and not bool(check_webshop()):
            raise RuntimeError("formal WebShop dependencies are unavailable")
        reset_seed = int(config.get("env_seed", 1000))

    return RAGENEnvironmentSessionFactory(
        adapter_factory=module.RAGENAdapter,
        env_type=env_type,
        env_config=config,
        extra=dict(evaluator._metadata(record)),
        question=str(evaluator._record_field(record, "question", "")),
        reset_seed=reset_seed,
    )


def _admissible_actions(
    task_family: str, available_actions: object
) -> tuple[tuple[str, ...], bool]:
    """Use the public action projection exposed by the deployed RAGEN bridge."""

    if task_family.lower() == "webshop" and isinstance(available_actions, Mapping):
        has_search_bar = bool(available_actions.get("has_search_bar"))
        actions: list[str] = ["search[<your query>]"] if has_search_bar else []
        clickables = available_actions.get("clickables", ())
        if isinstance(clickables, Sequence) and not isinstance(
            clickables, (str, bytes)
        ):
            actions.extend(f"click[{value}]" for value in clickables)
        return tuple(actions), has_search_bar
    if isinstance(available_actions, Sequence) and not isinstance(
        available_actions, (str, bytes)
    ):
        return tuple(str(action) for action in available_actions), False
    return (), False


def _parse_action(
    output: object,
    *,
    task_family: str,
    admissible_actions: Sequence[str],
    webshop_has_search_bar: bool,
) -> Optional[str]:
    """Accept one complete admissible action or one explicit action tag."""

    if not isinstance(output, str) or not output.strip():
        return None
    raw = output.strip()
    tagged = re.findall(
        r"<action>\s*(.*?)\s*</action>", raw, re.IGNORECASE | re.DOTALL
    )
    if len(tagged) > 1:
        return None
    candidate = tagged[0].strip() if tagged else raw
    if not candidate:
        return None
    if candidate in admissible_actions and candidate != "search[<your query>]":
        return candidate
    if task_family.lower() == "webshop" and webshop_has_search_bar:
        match = re.fullmatch(r"search\[([^\[\]\n]+)\]", candidate)
        if match and match.group(1).strip() not in {"", "<your query>"}:
            return candidate
    return None


def _webshop_structured_action_schema(
    admissible_actions: Sequence[str],
    *,
    has_search_bar: bool,
) -> dict[str, object]:
    """Project the live WebShop domain to SkillFlow's StructuredAction wire."""

    branches: list[dict[str, object]] = []
    if has_search_bar:
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "arguments",
                    "kind",
                    "name",
                    "resource_id",
                    "skill_id",
                ],
                "properties": {
                    "arguments": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string", "minLength": 1}
                        },
                    },
                    "kind": {"const": "tool"},
                    "name": {"const": "search"},
                    "resource_id": {"const": "webshop"},
                    "skill_id": {"type": "null"},
                },
            }
        )
    click_targets = [
        action[len("click[") : -1]
        for action in admissible_actions
        if action.startswith("click[") and action.endswith("]")
    ]
    if click_targets:
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "arguments",
                    "kind",
                    "name",
                    "resource_id",
                    "skill_id",
                ],
                "properties": {
                    "arguments": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["target"],
                        "properties": {"target": {"enum": click_targets}},
                    },
                    "kind": {"const": "tool"},
                    "name": {"const": "click"},
                    "resource_id": {"const": "webshop"},
                    "skill_id": {"type": "null"},
                },
            }
        )
    if not branches:
        raise EnvironmentExecutionError(
            "WebShop exposed no StructuredAction-compatible action"
        )
    return branches[0] if len(branches) == 1 else {"oneOf": branches}


def _parse_webshop_structured_action(
    output: object,
    *,
    admissible_actions: Sequence[str],
    webshop_has_search_bar: bool,
) -> tuple[Optional[str], str]:
    """Parse SkillFlow's five-field action and serialize one native command.

    The public SkillFlow resource is ``webshop``.  The local ToolRegistry uses
    ``webshop.environment`` as its task-scoped dispatch identifier, so this
    boundary validates the former and dispatches the resulting native command
    through the latter without rewriting sampled semantic fields.
    """

    if not isinstance(output, str) or not output.strip():
        return None, "parse_error"
    try:
        value = json.loads(output)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "parse_error"
    format_normalized = False
    if isinstance(value, Mapping):
        # Some OpenAI-compatible providers omit JSON-Schema ``const`` fields
        # from an otherwise complete object.  SkillFlow's WebShop action space
        # fixes these two values for every environment action, so filling only
        # those constants is a wire-format normalization and never chooses or
        # changes the semantic action (``name`` + ``arguments``).  Extra
        # fields, missing semantic fields, or conflicting constants still fail
        # closed through ``StructuredAction.from_value`` below.
        expected_fields = {
            "arguments",
            "kind",
            "name",
            "resource_id",
            "skill_id",
        }
        semantic_fields = {"arguments", "name", "resource_id"}
        present_fields = set(value)
        if semantic_fields.issubset(present_fields) and present_fields.issubset(
            expected_fields
        ):
            normalized_value = dict(value)
            if "kind" not in normalized_value:
                normalized_value["kind"] = "tool"
                format_normalized = True
            if "skill_id" not in normalized_value:
                normalized_value["skill_id"] = None
                format_normalized = True
            value = normalized_value
    try:
        action = StructuredAction.from_value(value)
    except (KeyError, TypeError, ValueError):
        return None, "schema_invalid"
    if (
        action.kind is not ActionKind.TOOL
        or action.resource_id != "webshop"
        or action.skill_id is not None
        or not isinstance(action.arguments, dict)
    ):
        return None, "schema_invalid"
    native: Optional[str] = None
    if action.name == "search" and set(action.arguments) == {"query"}:
        query = action.arguments.get("query")
        if isinstance(query, str) and query.strip():
            native = f"search[{query.strip()}]"
    elif action.name == "click" and set(action.arguments) == {"target"}:
        target = action.arguments.get("target")
        if isinstance(target, str) and target.strip():
            native = f"click[{target.strip()}]"
    elif action.name == "purchase" and not action.arguments:
        native = "click[buy now]"
    if native is None:
        return None, "schema_invalid"
    admitted = _parse_action(
        native,
        task_family="webshop",
        admissible_actions=admissible_actions,
        webshop_has_search_bar=webshop_has_search_bar,
    )
    if admitted != native:
        return None, "schema_invalid"
    return native, "format_normalized" if format_normalized else "success"


def _history_text(receipts: Sequence[Mapping[str, object]]) -> str:
    if not receipts:
        return "(none)"
    lines = []
    for receipt in receipts[-4:]:
        line = (
            "[Turn {turn}: action={action!r}, status={observation_status}, "
            "state_advanced={state_advanced}, terminal={terminal}]".format(
                **receipt
            )
        )
        raw_output = receipt.get("raw_model_output")
        if (
            receipt.get("observation_status") == "parse_error"
            and isinstance(raw_output, str)
            and raw_output.strip()
        ):
            compact = " ".join(raw_output.split())[:240]
            line += (
                " Previous response was not an admissible native action and the "
                f"environment state did not change: {compact!r}."
            )
        lines.append(line)
    return "\n".join(lines)


def _alfworld_task_facts(task: str) -> dict[str, object]:
    """Parse only task-visible ALFWorld fields used by SkillFlow state feedback.

    This is a compact adaptation of SkillFlow
    ``training.environment._parse_alfworld_task``.  It deliberately returns
    no action recommendation and consumes neither simulator internals nor
    evaluator state.
    """

    facts: dict[str, object] = {
        "target_class": None,
        "destination_class": None,
        "source_hint": None,
        "required_transform": None,
        "count": 1,
        "examine_with_desklamp": False,
    }
    text = " ".join(str(task or "").lower().rstrip(".").split())
    if not text:
        return facts
    match = re.search(
        r"pick up (?:the |a |some )?(\S+?)(?: \d)? from (\S+?)(?: \d)? "
        r"and put it (?:in/on|in|on) (\S+?)(?: \d)?$",
        text,
    )
    if match:
        facts.update(
            target_class=match.group(1),
            source_hint=match.group(2),
            destination_class=match.group(3),
        )
        return facts
    match = re.search(
        r"(?:examine|look at) (?:the |a |some )?(\S+?)(?: \d)? "
        r"(?:with|under|by) (?:the |a )?(?:desklamp|lamp)",
        text,
    )
    if match:
        facts.update(
            target_class=match.group(1),
            required_transform="examine_with_desklamp",
            examine_with_desklamp=True,
        )
        return facts
    match = re.search(
        r"(heat|cool|clean) (?:some |a |the )?(\S+?)(?: \d)? and put it "
        r"(?:in/on|in|on) (\S+?)(?: \d)?$",
        text,
    )
    if match:
        facts.update(
            required_transform=match.group(1),
            target_class=match.group(2),
            destination_class=match.group(3),
        )
        return facts
    match = re.search(
        r"(?:find|put) two (\S+?)(?: \d)? (?:and put them )?"
        r"(?:in/on|in|on) (\S+?)(?: \d)?$",
        text,
    )
    if match:
        facts.update(
            target_class=match.group(1),
            destination_class=match.group(2),
            count=2,
        )
        return facts
    match = re.search(
        r"put (?:a |some |the )?(?:(clean|washed|cool|cold|hot|heated|cooked) )?"
        r"(\S+?)(?: \d)? (?:in/on|in|on) (\S+?)(?: \d)?$",
        text,
    )
    if match:
        adjective, target, destination = match.groups()
        transform = None
        if adjective in {"clean", "washed"}:
            transform = "clean"
        elif adjective in {"cool", "cold"}:
            transform = "cool"
        elif adjective in {"hot", "heated", "cooked"}:
            transform = "heat"
        facts.update(
            target_class=target,
            destination_class=destination,
            required_transform=transform,
        )
    return facts


def _alfworld_object_class(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:a|an|the|some)\s+", "", text)
    return re.sub(r"\s+\d+\s*$", "", text).strip()


def _alfworld_action_object(action: object) -> str:
    text = str(action or "").strip()
    for pattern in (
        r"^take\s+(.+?)\s+from\s+",
        r"^move\s+(.+?)\s+to\s+",
        r"^(?:clean|heat|cool)\s+(.+?)\s+with\s+",
        r"^use\s+(.+?)\s+",
    ):
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


_WEBSHOP_NAV_CLICKABLES = frozenset(
    {
        "back to search",
        "< prev",
        "prev",
        "next >",
        "description",
        "features",
        "reviews",
        "buy now",
    }
)
_WEBSHOP_OPTION_LABELS = frozenset(
    {
        "color",
        "size",
        "scent",
        "flavor name",
        "flavor",
        "flavour",
        "style",
        "pattern",
        "quantity",
        "pack",
        "count",
        "dimension",
        "dimensions",
        "material",
        "fit",
        "fit type",
        "item shape",
        "shape",
    }
)


def _webshop_norm(text: object) -> str:
    """Reuse SkillFlow's normalization for visible WebShop option values."""

    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9.%]+", " ", str(text).lower()),
    ).strip()


def _webshop_price_limit(task_instruction: str) -> Optional[float]:
    """Thin-adapt SkillFlow's public WebShop price-limit parser."""

    match = re.search(
        r"price\s+lower\s+than\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        str(task_instruction).lower(),
    )
    return float(match.group(1)) if match is not None else None


def _webshop_product_price(observation: object) -> Optional[float]:
    """Thin-adapt SkillFlow's public product-page price parser."""

    for token in _webshop_tokens(observation):
        match = re.search(
            r"price:\s*\$\s*([0-9]+(?:\.[0-9]+)?)",
            token,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return float(match.group(1))
    return None


def _webshop_tokens(observation: object) -> list[str]:
    """Split the two public WebShop observation formats used by SkillFlow."""

    text = str(observation or "")
    parts = (
        re.split(r"\s*\[SEP\]\s*", text)
        if "[SEP]" in text
        else re.split(r"\s*\|\s*", text)
    )
    return [token.strip() for token in parts if token and token.strip()]


def _webshop_parse_product_options(
    observation: object,
) -> tuple[dict[str, list[str]], str]:
    """Thin-adapt SkillFlow's product-page option parser."""

    tokens = _webshop_tokens(observation)
    lowered = [token.lower() for token in tokens]
    if "buy now" not in lowered or "< prev" not in lowered:
        return {}, ""
    try:
        start = lowered.index("< prev") + 1
    except ValueError:
        start = 0
    title_index: Optional[int] = None
    for index, token in enumerate(lowered):
        if token.startswith("price:"):
            title_index = max(index - 1, start)
            break
    if title_index is None:
        return {}, ""

    groups: dict[str, list[str]] = {}
    current_group: Optional[str] = None
    for token in tokens[start:title_index]:
        key = token.strip().lower()
        if key in _WEBSHOP_OPTION_LABELS:
            current_group = key
            groups.setdefault(current_group, [])
        elif current_group and token.strip():
            groups[current_group].append(token.strip())

    for group, values in tuple(groups.items()):
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            normalized = _webshop_norm(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(value)
        groups[group] = unique
    title = tokens[title_index] if 0 <= title_index < len(tokens) else ""
    return groups, title


def _webshop_clicked_option_assignments(
    receipts: Sequence[Mapping[str, object]],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Return the latest visible option assignment for each product group.

    SkillFlow bounds option history to the current product.  The original
    WebShop environment assigns ``session[\"options\"][group] = value``;
    therefore a later click replaces the earlier value in that group rather
    than accumulating mutually exclusive selections.
    """

    value_groups: dict[str, list[tuple[str, str]]] = {}
    for group, values in groups.items():
        for value in values:
            normalized = _webshop_norm(value)
            if normalized:
                value_groups.setdefault(normalized, []).append((group, value))

    selected: dict[str, str] = {}
    for receipt in reversed(receipts):
        if receipt.get("state_advanced") is not True:
            continue
        raw_action = receipt.get("action")
        if not isinstance(raw_action, str):
            continue
        action = raw_action.strip()
        lowered = action.lower()
        if lowered.startswith("search[") or lowered == "click[back to search]":
            break
        match = re.fullmatch(r"click\[(.*)\]", action, flags=re.IGNORECASE)
        if match is None:
            continue
        value = match.group(1).strip()
        value_lower = value.lower()
        if re.fullmatch(r"b[0-9a-z]{9}", value_lower):
            break
        if value_lower in _WEBSHOP_NAV_CLICKABLES:
            continue
        candidates = value_groups.get(_webshop_norm(value), ())
        # When the same display value belongs to multiple option groups, the
        # flattened observation does not expose the HTML option name.  Retain
        # the native action instead of guessing an entity--attribute binding.
        if len(candidates) != 1:
            continue
        group, canonical_value = candidates[0]
        selected.setdefault(group, canonical_value)
    return {group: selected[group] for group in groups if group in selected}


def _webshop_model_visible_actions(
    *,
    task_instruction: str = "",
    observation: object,
    receipts: Sequence[Mapping[str, object]],
    native_actions: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, list[str]], dict[str, str]]:
    """Project state-dependent WebShop actions without changing native legality.

    The returned action list is a model-input projection.  Native legal actions
    remain unchanged in environment and evaluator receipts.
    """

    groups, _ = _webshop_parse_product_options(observation)
    selected = _webshop_clicked_option_assignments(receipts, groups)
    selected_values = {_webshop_norm(value) for value in selected.values()}
    purchase = _webshop_purchase_preconditions(
        task_instruction=task_instruction,
        observation=observation,
        receipts=receipts,
    )
    visible: list[str] = []
    for action in native_actions:
        match = re.fullmatch(r"click\[(.*)\]", action, flags=re.IGNORECASE)
        if match is not None and _webshop_norm(match.group(1)) in selected_values:
            continue
        if (
            match is not None
            and match.group(1).strip().casefold() == "buy now"
            and purchase["admissible"] is not True
        ):
            continue
        if _webshop_repeats_unchanged_click(
            action=action,
            observation=observation,
            receipts=receipts,
        ):
            continue
        visible.append(action)
    return tuple(visible), groups, selected


def _webshop_option_targets(
    task_instruction: str,
    groups: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Match public task text to option values visible on the current page.

    WebShop exposes option labels and values in the public observation.  This
    projection performs only deterministic normalization over those public
    strings; it does not read the evaluator goal object or reward.
    """

    task_normalized = _webshop_norm(task_instruction)
    targets: dict[str, list[str]] = {}
    for group, values in groups.items():
        group_normalized = _webshop_norm(group)
        group_aliases = {group_normalized}
        if group_normalized in {"flavor", "flavour", "flavor name"}:
            group_aliases.update({"flavor", "flavour", "flavor name"})
        matches: list[tuple[int, str]] = []
        for value in values:
            normalized = _webshop_norm(value)
            if not normalized:
                continue
            compact = normalized.replace(" ", "")
            explicit_binding = False
            for label in group_aliases:
                label_pattern = re.escape(label)
                value_patterns = {re.escape(normalized)}
                if len(compact) >= 5:
                    value_patterns.add(re.escape(compact))
                for value_pattern in value_patterns:
                    if re.search(
                        rf"(?:^| ){label_pattern}(?: is| of| equals|:)? "
                        rf"{value_pattern}(?: |$)",
                        task_normalized,
                    ) is not None or re.search(
                        rf"(?:^| ){value_pattern} {label_pattern}(?: |$)",
                        task_normalized,
                    ) is not None:
                        explicit_binding = True
                        break
                if explicit_binding:
                    break
            if group_normalized == "color":
                color_patterns = {re.escape(normalized)}
                if len(compact) >= 5:
                    color_patterns.add(re.escape(compact))
                explicit_binding = explicit_binding or any(
                    re.search(
                        rf"(?:^| ){pattern} (?:color|colored)(?: |$)",
                        task_normalized,
                    ) is not None
                    or re.search(
                        rf"(?:^| )with {pattern} color(?: |$)",
                        task_normalized,
                    ) is not None
                    for pattern in color_patterns
                )
            if explicit_binding:
                matches.append((len(compact), str(value)))
        if not matches:
            continue
        longest = max(length for length, _ in matches)
        targets[group] = [
            value for length, value in matches if length == longest
        ]
    return targets


def _webshop_option_values_equivalent(left: object, right: object) -> bool:
    left_normalized = _webshop_norm(left)
    right_normalized = _webshop_norm(right)
    if left_normalized == right_normalized:
        return True
    return (
        len(left_normalized.replace(" ", "")) >= 5
        and left_normalized.replace(" ", "")
        == right_normalized.replace(" ", "")
    )


def _webshop_purchase_preconditions(
    *,
    task_instruction: str,
    observation: object,
    receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return public option-binding preconditions for a purchase action."""

    groups, _ = _webshop_parse_product_options(observation)
    selected = _webshop_clicked_option_assignments(receipts, groups)
    targets = _webshop_option_targets(task_instruction, groups)
    missing = [group for group in targets if group not in selected]
    mismatched = [
        group
        for group, values in targets.items()
        if group in selected
        and not any(
            _webshop_option_values_equivalent(selected[group], value)
            for value in values
        )
    ]
    price_limit = _webshop_price_limit(task_instruction)
    product_price = _webshop_product_price(observation)
    price_evidence_missing = price_limit is not None and product_price is None
    price_exceeds_limit = bool(
        price_limit is not None
        and product_price is not None
        and product_price > price_limit
    )
    return {
        "admissible": (
            not missing
            and not mismatched
            and not price_evidence_missing
            and not price_exceeds_limit
        ),
        "required_option_targets": targets,
        "selected_options": selected,
        "missing_option_groups": missing,
        "mismatched_option_groups": mismatched,
        "price_limit": price_limit,
        "product_price": product_price,
        "price_evidence_missing": price_evidence_missing,
        "price_exceeds_limit": price_exceeds_limit,
        "source": "public_task_and_observation",
    }


def _webshop_zero_result_queries(
    receipts: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return normalized exact queries with an observed zero-result page."""

    queries: list[str] = []
    seen: set[str] = set()
    for receipt in receipts:
        if receipt.get("state_advanced") is not True:
            continue
        action = receipt.get("action")
        next_observation = receipt.get("next_observation")
        if not isinstance(action, str) or not isinstance(next_observation, str):
            continue
        match = re.fullmatch(r"search\[(.*)\]", action, flags=re.IGNORECASE)
        total = re.search(
            r"\bTotal\s+results\s*:\s*([0-9]+)\b",
            next_observation,
            flags=re.IGNORECASE,
        )
        if match is None or total is None or int(total.group(1)) != 0:
            continue
        query = _webshop_norm(match.group(1))
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    return tuple(queries)


def _webshop_repeats_unchanged_click(
    *,
    action: str,
    observation: object,
    receipts: Sequence[Mapping[str, object]],
) -> bool:
    """Detect a click already observed to leave this exact public state unchanged."""

    if re.fullmatch(r"click\[(.*)\]", action, flags=re.IGNORECASE) is None:
        return False
    return any(
        receipt.get("state_advanced") is True
        and receipt.get("action") == action
        and receipt.get("observation") == observation
        and receipt.get("next_observation") == observation
        for receipt in receipts
    )


def _webshop_action_precondition_failure(
    *,
    action: str,
    task_instruction: str,
    observation: object,
    receipts: Sequence[Mapping[str, object]],
) -> Optional[str]:
    """Validate one sampled action against public state-dependent constraints."""

    search = re.fullmatch(r"search\[(.*)\]", action, flags=re.IGNORECASE)
    if search is not None:
        query = _webshop_norm(search.group(1))
        if query in _webshop_zero_result_queries(receipts):
            return "known_zero_result_query"
    if _webshop_repeats_unchanged_click(
        action=action,
        observation=observation,
        receipts=receipts,
    ):
        return "repeated_unchanged_click"
    if action.casefold() == "click[buy now]":
        purchase = _webshop_purchase_preconditions(
            task_instruction=task_instruction,
            observation=observation,
            receipts=receipts,
        )
        if purchase["admissible"] is not True:
            return "purchase_option_precondition"
    return None


def _webshop_task_constraints(task: str) -> tuple[str, ...]:
    """Project SkillFlow's visible WebShop price/attribute fields only."""

    text = str(task or "")
    constraints: list[str] = []
    price = re.search(
        r"price\s+lower\s+than\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if price:
        constraints.append(f"price_lower_than={price.group(1)}")
    labels = (
        "color",
        "size",
        "fit type",
        "flavor name",
        "flavor",
        "scent",
        "style",
        "pattern",
        "count",
        "number",
        "dimension",
        "dimensions",
        "width",
        "height",
        "item shape",
        "shape",
    )
    labels_pattern = "|".join(re.escape(label) for label in labels)
    for label in labels:
        match = re.search(
            rf"\b{re.escape(label)}\s*:\s*(.*?)"
            rf"(?=,\s*(?:and\s+)?(?:{labels_pattern})\b\s*:?|"
            r",?\s*and\s+price\s+lower\s+than\b|$)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            value = re.sub(
                r"^\s*and\s+", "", match.group(1), flags=re.IGNORECASE
            ).strip(" ,.;")
            if value:
                constraints.append(f"{label.replace(' ', '_')}={value}")
    return tuple(dict.fromkeys(constraints))


def _public_transition_summary(
    *,
    task_family: str,
    task_instruction: str = "",
    observation: str,
    receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Project public Action--Observation progress for Agent and Director.

    The projection is a thin visible-state adaptation of SkillFlow's bounded
    environment history.  It consumes no reward, evaluator ``info`` or hidden
    target and never ranks candidates or recommends the next action.
    """

    completed = [
        item
        for item in receipts
        if item.get("state_advanced") is True
        and isinstance(item.get("action"), str)
    ]
    attempted = [
        item
        for item in receipts
        if isinstance(item.get("action"), str)
    ]
    recent_actions = [str(item["action"]) for item in completed[-8:]]
    latest = receipts[-1] if receipts else None
    observation_changed: Optional[bool] = None
    latest_transition: dict[str, object] = {
        "action": None,
        "observation_status": "reset",
        "state_advanced": None,
        "observation_changed": None,
        "terminal": False,
        "result_observation_field": "current_observation",
        "result_is_current_state": None,
    }
    if isinstance(latest, Mapping):
        previous_observation = latest.get("observation")
        next_observation = latest.get("next_observation")
        if isinstance(previous_observation, str) and isinstance(
            next_observation, str
        ):
            observation_changed = previous_observation != next_observation
        latest_transition.update(
            {
                "action": latest.get("action"),
                "observation_status": latest.get("observation_status"),
                "precondition_failure_reason": latest.get(
                    "precondition_failure_reason"
                ),
                "state_advanced": latest.get("state_advanced"),
                "observation_changed": observation_changed,
                "terminal": latest.get("terminal") is True,
                "result_is_current_state": next_observation == observation,
            }
        )

    repeated_state_action_count = 0
    if isinstance(latest, Mapping) and isinstance(latest.get("action"), str):
        latest_action = latest["action"]
        latest_pre_observation = latest.get("observation")
        latest_next_observation = latest.get("next_observation")
        for item in attempted:
            if (
                item.get("action") == latest_action
                and item.get("observation") == latest_pre_observation
                and item.get("next_observation") == latest_next_observation
            ):
                repeated_state_action_count += 1
    recent_transitions = [
        (
            str(item["action"]),
            item.get("observation"),
            item.get("next_observation"),
        )
        for item in completed[-4:]
    ]
    action_cycle = bool(
        len(recent_transitions) == 4
        and recent_transitions[-4] == recent_transitions[-2]
        and recent_transitions[-3] == recent_transitions[-1]
    )
    no_progress_reasons: list[str] = []
    if (
        isinstance(latest, Mapping)
        and latest.get("observation_status") == "precondition_failed"
    ):
        reason = latest.get("precondition_failure_reason")
        no_progress_reasons.append(
            str(reason) if isinstance(reason, str) and reason else "precondition_failed"
        )
    # A repeated exact public state--action--state transition is measured
    # no-progress even when the native WebShop action remains legal.  The
    # environment/evaluator domain stays lossless; only the next model/Canvas
    # decision consumes this public diagnosis.
    if repeated_state_action_count >= 2:
        no_progress_reasons.append("repeated_state_action")
    if action_cycle:
        no_progress_reasons.append("action_cycle")

    # SkillFlow conditions every next Action on public observed history.  Keep
    # a bounded state--action visit projection so revisiting a state exposes
    # which Actions have already been tried from the same public precondition,
    # before the Agent blindly repeats one.  This is advisory public history;
    # it neither masks a legal native action nor recommends an alternative.
    previous_actions_reversed: list[str] = []
    seen_actions: set[str] = set()
    for item in reversed(attempted):
        action = str(item["action"])
        if item.get("observation") != observation or action in seen_actions:
            continue
        previous_actions_reversed.append(action)
        seen_actions.add(action)
        if len(previous_actions_reversed) == 8:
            break
    previous_actions_from_current_state = list(
        reversed(previous_actions_reversed)
    )

    summary: dict[str, object] = {
        "latest_transition": latest_transition,
        "recent_actions": recent_actions,
        "previous_actions_from_current_state": previous_actions_from_current_state,
        "no_progress": {
            "detected": bool(no_progress_reasons),
            "reasons": no_progress_reasons,
            "repeated_state_action_count": repeated_state_action_count,
            "action_cycle": action_cycle,
        },
    }
    if task_family.casefold() == "webshop":
        searches: list[str] = []
        opened_asins: list[str] = []
        click_targets: list[str] = []
        for action in recent_actions:
            search = re.fullmatch(r"search\[(.*)\]", action, flags=re.IGNORECASE)
            click = re.fullmatch(r"click\[(.*)\]", action, flags=re.IGNORECASE)
            if search and search.group(1).strip():
                searches.append(search.group(1).strip())
            if click and click.group(1).strip():
                target = click.group(1).strip()
                click_targets.append(target)
                if re.fullmatch(r"b[0-9a-z]{9}", target, flags=re.IGNORECASE):
                    opened_asins.append(target.lower())
        latest_search_outcome: Optional[dict[str, str]] = None
        if isinstance(latest, Mapping) and isinstance(latest.get("action"), str):
            search = re.fullmatch(
                r"search\[(.*)\]",
                str(latest["action"]),
                flags=re.IGNORECASE,
            )
            if search and search.group(1).strip():
                outcome = "unknown"
                next_observation = latest.get("next_observation")
                if isinstance(next_observation, str):
                    total = re.search(
                        r"\bTotal\s+results\s*:\s*([0-9]+)\b",
                        next_observation,
                        flags=re.IGNORECASE,
                    )
                    if total is not None:
                        outcome = (
                            "zero_results"
                            if int(total.group(1)) == 0
                            else "results_observed"
                        )
                latest_search_outcome = {
                    "query": search.group(1).strip(),
                    "outcome": outcome,
                }
        groups, title = _webshop_parse_product_options(observation)
        selected_options = _webshop_clicked_option_assignments(receipts, groups)
        purchase_preconditions = _webshop_purchase_preconditions(
            task_instruction=task_instruction,
            observation=observation,
            receipts=receipts,
        )
        summary.update(
            {
                "recent_search_queries": searches[-3:],
                "zero_result_queries": list(
                    _webshop_zero_result_queries(receipts)
                ),
                "opened_asins": opened_asins[-8:],
                "recent_click_targets": click_targets[-8:],
                "latest_search_outcome": latest_search_outcome,
                "purchase_preconditions": purchase_preconditions,
                "current_product": (
                    {
                        "title": title,
                        "visible_option_groups": {
                            group: list(values) for group, values in groups.items()
                        },
                        "selected_options": selected_options,
                    }
                    if groups
                    else None
                ),
            }
        )
    return summary


def _public_state_feedback(
    request: AgentRequest,
    *,
    task_family: str,
    observation: str,
    admissible_actions: Sequence[str],
    receipts: Sequence[Mapping[str, object]],
) -> str:
    """Summarize public task/environment state without evaluator leakage.

    SkillFlow's ALFWorld/WebShop visible-state feedback is intentionally
    reduced to neutral facts.  The block may expose a repeated-state warning,
    but it never ranks actions, blocks a legal action, or reads reward/``won``.
    """

    lines = [
        "[PUBLIC OBSERVABLE STATE]",
        "Derived only from the task, current observation, admissible actions, "
        "and completed public Action--Observation history.",
    ]
    completed_actions = [
        str(item.get("action"))
        for item in receipts
        if item.get("state_advanced") is True
        and isinstance(item.get("action"), str)
    ]
    task_instruction = _original_task_instruction(request.problem)
    progress = _public_transition_summary(
        task_family=task_family,
        task_instruction=task_instruction,
        observation=observation,
        receipts=receipts,
    )
    lines.append(f"Original task goal: {task_instruction}")
    latest_transition = progress["latest_transition"]
    if isinstance(latest_transition, Mapping) and latest_transition.get("action"):
        lines.append(
            "Latest Action--Observation result: "
            f"action={latest_transition.get('action')!r}; "
            f"status={latest_transition.get('observation_status')}; "
            f"precondition_failure={latest_transition.get('precondition_failure_reason')}; "
            f"observation_changed={latest_transition.get('observation_changed')}; "
            "the resulting public state is the current observation below."
        )
    if task_family.lower() == "alfworld":
        facts = _alfworld_task_facts(task_instruction)
        visible = [
            f"target_class={facts['target_class']}",
            f"destination_class={facts['destination_class']}",
            f"required_transform={facts['required_transform']}",
            f"count={facts['count']}",
        ]
        lines.append("Task facts: " + "; ".join(visible) + ".")
        target = facts.get("target_class")
        task_lower = task_instruction.lower()
        if target and re.search(r"\bit\b", task_lower):
            lines.append(f"Task coreference: `it` refers to `{target}`.")
        if target and re.search(r"\bthem\b", task_lower):
            lines.append(
                f"Task coreference: `them` refers to the required `{target}` instances."
            )
        held: list[str] = []
        for action in admissible_actions:
            if action.lower().startswith(("move ", "clean ", "heat ", "cool ")):
                obj = _alfworld_action_object(action)
                if obj and obj not in held:
                    held.append(obj)
        if held:
            lines.append("Objects implied as held by current actions: " + ", ".join(held[:6]) + ".")
        transform_actions = [
            action
            for action in completed_actions
            if action.lower().startswith(("clean ", "heat ", "cool "))
        ]
        if transform_actions:
            lines.append(
                "Completed visible transform actions: "
                + " | ".join(transform_actions[-4:])
                + "."
            )
        if target and facts.get("destination_class"):
            target_class = str(target)
            destination_class = str(facts["destination_class"])
            placed = []
            for action in completed_actions:
                match = re.match(
                    r"^move\s+(.+?)\s+to\s+(.+)$", action, flags=re.IGNORECASE
                )
                if not match:
                    continue
                if (
                    _alfworld_object_class(match.group(1)) == target_class
                    and _alfworld_object_class(match.group(2)) == destination_class
                ):
                    placed.append(match.group(1).strip())
            if placed:
                lines.append(
                    "Visible placement progress: "
                    f"{len(dict.fromkeys(placed))}/{facts['count']} distinct target instance(s); "
                    + ", ".join(dict.fromkeys(placed))
                    + "."
                )
    elif task_family.lower() == "webshop":
        constraints = _webshop_task_constraints(task_instruction)
        if constraints:
            lines.append(
                "Task constraints retained from the instruction: "
                + "; ".join(constraints)
                + "."
            )
        searches = []
        opened = []
        for action in completed_actions:
            search = re.fullmatch(r"search\[(.*)\]", action, flags=re.IGNORECASE)
            asin = re.fullmatch(
                r"click\[(b[0-9a-z]{9})\]", action, flags=re.IGNORECASE
            )
            if search and search.group(1).strip():
                searches.append(search.group(1).strip())
            if asin:
                opened.append(asin.group(1).lower())
        if searches:
            lines.append("Recent search queries: " + " | ".join(searches[-3:]) + ".")
        if opened:
            lines.append("Opened candidate ASINs: " + ", ".join(opened[-8:]) + ".")
        latest_search_outcome = progress.get("latest_search_outcome")
        if isinstance(latest_search_outcome, Mapping):
            lines.append(
                "Latest public search outcome: "
                f"query={latest_search_outcome.get('query')!r}; "
                f"outcome={latest_search_outcome.get('outcome')}."
            )
        zero_result_queries = progress.get("zero_result_queries")
        if isinstance(zero_result_queries, (list, tuple)) and zero_result_queries:
            lines.append(
                "Observed zero-result queries (do not repeat exactly): "
                + " | ".join(str(value) for value in zero_result_queries[-6:])
                + "."
            )
        current_product = progress.get("current_product")
        if isinstance(current_product, Mapping):
            title = current_product.get("title")
            if isinstance(title, str) and title:
                lines.append(f"Current product title: {title}.")
            option_groups = current_product.get("visible_option_groups")
            if isinstance(option_groups, Mapping) and option_groups:
                lines.append(
                    "Visible product option groups: "
                    + "; ".join(
                        f"{group}={', '.join(str(value) for value in values)}"
                        for group, values in option_groups.items()
                        if isinstance(values, (list, tuple))
                    )
                    + "."
                )
            selected_options = current_product.get("selected_options")
            if isinstance(selected_options, Mapping) and selected_options:
                lines.append(
                    "Current product option assignments: "
                    + "; ".join(
                        f"{group}={value}"
                        for group, value in selected_options.items()
                    )
                    + "."
                )
        purchase = progress.get("purchase_preconditions")
        if isinstance(purchase, Mapping) and purchase.get("admissible") is not True:
            missing = purchase.get("missing_option_groups", ())
            mismatched = purchase.get("mismatched_option_groups", ())
            lines.append(
                "Purchase preconditions not satisfied: "
                f"missing_option_groups={list(missing) if isinstance(missing, (list, tuple)) else []}; "
                f"mismatched_option_groups={list(mismatched) if isinstance(mismatched, (list, tuple)) else []}; "
                f"price_limit={purchase.get('price_limit')}; "
                f"product_price={purchase.get('product_price')}; "
                f"price_evidence_missing={purchase.get('price_evidence_missing')}; "
                f"price_exceeds_limit={purchase.get('price_exceeds_limit')}; "
                "satisfy the public option bindings and price bound before purchase."
            )

    recent_actions = completed_actions[-6:]
    if recent_actions:
        lines.append("Recent executed actions: " + " | ".join(recent_actions) + ".")
    previous_from_state = progress.get("previous_actions_from_current_state")
    if isinstance(previous_from_state, (list, tuple)) and previous_from_state:
        lines.append(
            "Previously executed actions from this same public state: "
            + " | ".join(str(value) for value in previous_from_state)
            + "."
        )
    no_progress = progress.get("no_progress")
    if isinstance(no_progress, Mapping) and no_progress.get("detected") is True:
        lines.append(
            "No-progress signal: reasons="
            + ",".join(str(value) for value in no_progress.get("reasons", ()))
            + "; preserve the current episode and do not repeat an action that "
            "already returned the same public state."
        )
    if receipts and receipts[-1].get("observation_status") == "parse_error":
        lines.append(
            "Format repair: the preceding response was invalid and the environment "
            "state is unchanged; copy exactly one current admissible action."
        )
    return "\n".join(lines)


def _prompt_observation(observation: str, max_observation_chars: int) -> tuple[str, bool]:
    """Apply SkillFlow's configured observation-size bound to model input only."""

    if max_observation_chars <= 0 or len(observation) <= max_observation_chars:
        return observation, False
    clipped = observation[:max_observation_chars]
    return (
        clipped
        + f"\n[OBSERVATION CLIPPED: retained first {max_observation_chars} of "
        f"{len(observation)} characters; full observation remains in the receipt.]",
        True,
    )


def _action_prompt(
    request: AgentRequest,
    *,
    task_family: str,
    observation: str,
    admissible_actions: Sequence[str],
    receipts: Sequence[Mapping[str, object]],
    turn: int,
    max_observation_chars: int = 0,
    structured_actions: bool = False,
) -> str:
    """Render SkillFlow's state/action/history boundary without a fixed role."""

    actions = "\n".join(admissible_actions)
    visible_observation, _ = _prompt_observation(
        observation, max_observation_chars
    )
    public_state = _public_state_feedback(
        request,
        task_family=task_family,
        observation=observation,
        admissible_actions=admissible_actions,
        receipts=receipts,
    )
    if task_family.lower() == "webshop" and structured_actions:
        format_instruction = (
            "Return exactly one SkillFlow StructuredAction JSON object. Use "
            "resource_id \"webshop\"; name \"search\" with arguments "
            "{\"query\":...}, or name \"click\" with arguments "
            "{\"target\":...}. Copy a click target exactly from the current "
            "admissible-action list. Return no explanation or code fence."
        )
    else:
        format_instruction = (
            "Return exactly one native WebShop action: search[keywords] or click[value]."
            if task_family.lower() == "webshop"
            else "Return exactly one native action copied from the admissible action list."
        )
    return (
        "Original task instruction:\n"
        f"{_original_task_instruction(request.problem)}\n\n"
        f"Current Agent contract:\n{request.agent.contract}\n\n"
        f"Previous environment turns:\n{_history_text(receipts)}\n\n"
        f"{public_state}\n\n"
        f"Current observation (turn {turn}):\n{visible_observation}\n\n"
        f"Admissible actions:\n{actions}\n\n"
        "Apply one ReAct control cycle: condition the next Action on the "
        "original task, current Agent contract and preceding Observation. "
        "After each Observation, update the next Action from public state. "
        "Do not repeat a previously executed Action when its public "
        "preconditions have not changed. Treat a failed Tool call as public "
        "evidence, revise the next Action, and preserve the current Agent and "
        "episode. Emit only the Action; the Runtime returns its Observation "
        "to the Director before another Canvas decision.\n\n"
        f"{format_instruction}"
        + (
            ""
            if task_family.lower() == "webshop" and structured_actions
            else " You may enclose that native action in one <action> tag. Do not "
            "return JSON, an object, a code fence, or an explanation."
        )
    )


async def _resolve(value: Union[Any, Awaitable[Any]]) -> Any:
    return await value if inspect.isawaitable(value) else value


class EnvironmentExecutionAdapter:
    """Run a bounded ALFWorld/WebShop episode or one Director-visible step."""

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        environment_backend: EnvironmentToolBackend,
        max_turns: int,
        max_action_tokens: int = 512,
        max_observation_chars: int = 0,
        stepwise_director: bool = False,
        structured_actions: bool = False,
    ) -> None:
        if not hasattr(gateway, "generate"):
            raise TypeError("gateway must implement generate")
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be a ToolRegistry")
        if not isinstance(environment_backend, EnvironmentToolBackend):
            raise TypeError("environment_backend must be an EnvironmentToolBackend")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if type(max_action_tokens) is not int or max_action_tokens < 1:
            raise ValueError("max_action_tokens must be a positive integer")
        if type(max_observation_chars) is not int or max_observation_chars < 0:
            raise ValueError("max_observation_chars must be a non-negative integer")
        if type(stepwise_director) is not bool:
            raise TypeError("stepwise_director must be bool")
        if type(structured_actions) is not bool:
            raise TypeError("structured_actions must be bool")
        if structured_actions and environment_backend.task_family != "webshop":
            raise ValueError("structured environment actions currently support WebShop")
        if environment_backend.tool_id not in tool_registry.resource_ids:
            raise ValueError("environment backend tool is absent from ToolRegistry")
        capability = tool_registry.require_capability(environment_backend.tool_id)
        if not capability.availability:
            raise ValueError("environment tool capability is unavailable")
        if not capability.supports_dataset(environment_backend.task_family):
            raise ValueError("environment tool capability has incompatible dataset scope")
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._environment_backend = environment_backend
        self._tool_id = environment_backend.tool_id
        self._max_turns = max_turns
        self._max_action_tokens = max_action_tokens
        self._max_observation_chars = max_observation_chars
        self._stepwise_director = stepwise_director
        self._structured_actions = structured_actions
        self._live_state: Optional[_EnvironmentRolloutState] = None
        self._step_lock = asyncio.Lock()
        # ``asyncio.wait_for`` may insert a Task boundary between this adapter
        # and AgentRuntime. Retain the completed public prefix until Runtime
        # has copied it into the failure receipt.
        self._cancelled_prefixes: dict[str, Mapping[str, object]] = {}

    @property
    def stepwise_director(self) -> bool:
        return self._stepwise_director

    def reset_execution_state(self) -> None:
        """Drop the task-scoped episode before a new orchestration rollout."""

        if self._step_lock.locked():
            raise EnvironmentExecutionError(
                "cannot reset an environment episode while a step is active"
            )
        self._live_state = None
        self._cancelled_prefixes.clear()

    def close(self) -> None:
        self.reset_execution_state()

    def take_cancelled_failure_metadata(
        self,
        request_id: str,
    ) -> Mapping[str, object]:
        """Consume one cancellation prefix before the Task boundary erases it."""

        return self._cancelled_prefixes.pop(request_id, MappingProxyType({}))

    async def _new_state(
        self,
        request: AgentRequest,
    ) -> tuple[_EnvironmentRolloutState, Token[Optional[_EnvironmentEpisode]]]:
        episode, token = await self._environment_backend.begin(request)
        session = episode.session
        original_task_instruction = _original_task_instruction(request.problem)
        reset_actions, _ = _admissible_actions(
            session.task_family,
            session.available_actions,
        )
        episode_id = f"{session.environment_id}:{request.run_id}"
        reset_receipt: dict[str, object] = {
            "receipt_type": "environment_reset",
            "environment_episode_id": episode_id,
            "environment_id": session.environment_id,
            "environment_revision": 0,
            "original_task_instruction": original_task_instruction,
            "observation": episode.observation,
            "admissible_actions": list(reset_actions),
            "terminal": False,
        }
        return (
            _EnvironmentRolloutState(
                episode=episode,
                episode_id=episode_id,
                original_task_instruction=original_task_instruction,
                reset_receipt=reset_receipt,
            ),
            token,
        )

    def _current_public_state(
        self,
        state: _EnvironmentRolloutState,
    ) -> dict[str, object]:
        episode = state.episode
        visible_observation, observation_clipped = _prompt_observation(
            episode.observation,
            self._max_observation_chars,
        )
        native_admissible_actions: tuple[str, ...] = ()
        if not state.terminal:
            native_admissible_actions, _ = _admissible_actions(
                episode.session.task_family,
                episode.session.available_actions,
            )
        model_visible_actions = native_admissible_actions
        if episode.session.task_family.casefold() == "webshop":
            model_visible_actions, _, _ = _webshop_model_visible_actions(
                task_instruction=state.original_task_instruction,
                observation=episode.observation,
                receipts=state.receipts,
                native_actions=native_admissible_actions,
            )
        last = state.receipts[-1] if state.receipts else None
        public_progress = _public_transition_summary(
            task_family=episode.session.task_family,
            task_instruction=state.original_task_instruction,
            observation=episode.observation,
            receipts=state.receipts,
        )
        return {
            "environment_episode_id": state.episode_id,
            "environment_id": episode.session.environment_id,
            "task_family": episode.session.task_family,
            "original_task_instruction": state.original_task_instruction,
            "environment_revision": episode.revision,
            "last_action": None if last is None else last.get("action"),
            "state_advanced": None if last is None else last.get("state_advanced"),
            "observation_status": (
                "reset" if last is None else last.get("observation_status")
            ),
            # SkillFlow applies its configured observation bound at the
            # model-input boundary.  The Director consumes this same public
            # projection; the complete observation remains losslessly stored
            # in environment/evaluator receipts and is never rewritten.
            "current_observation": visible_observation,
            "current_observation_clipped": observation_clipped,
            "current_observation_original_chars": len(episode.observation),
            # Native actions remain lossless for the environment/evaluator.
            # The model-visible projection may suppress only the currently
            # selected option value on a WebShop product page.
            "admissible_actions": list(native_admissible_actions),
            "admissible_action_count": len(native_admissible_actions),
            "model_visible_admissible_actions": list(model_visible_actions),
            "model_visible_admissible_action_count": len(model_visible_actions),
            "public_progress": public_progress,
            "turns_used": state.turns_used,
            "remaining_action_budget": max(self._max_turns - state.turns_used, 0),
            "environment_terminal": state.terminal,
            "environment_truncated": (
                not state.terminal and state.turns_used == self._max_turns
            ),
        }

    def _response(self, state: _EnvironmentRolloutState) -> AgentResponse:
        episode = state.episode
        return AgentResponse(
            episode.observation,
            {
                "execution_mode": "react",
                "environment_execution_boundary": (
                    "one_action_one_observation"
                    if self._stepwise_director
                    else "bounded_episode"
                ),
                "structured_action_format": (
                    "structured-action-json@1"
                    if self._structured_actions
                    else "native-action-text@1"
                ),
                "model_calls": list(state.model_calls),
                "environment_episode_id": state.episode_id,
                "environment_id": episode.session.environment_id,
                "task_family": episode.session.task_family,
                "environment_revision": episode.revision,
                "environment_reset_receipt": dict(state.reset_receipt),
                "environment_receipts": list(state.receipts),
                "environment_current_state": self._current_public_state(state),
                "environment_terminal": state.terminal,
                "environment_truncated": (
                    not state.terminal and state.turns_used == self._max_turns
                ),
                "environment_max_turns": self._max_turns,
                "environment_turns_used": state.turns_used,
                "environment_steps": episode.revision,
                "tool_receipts": list(state.tool_receipts),
                # Evaluator-only replay data. AgentRuntime and Director public
                # projections never render reward or environment ``info``.
                "evaluator_environment_trace": list(state.evaluator_trace),
            },
        )

    def _failure_metadata(
        self,
        state: _EnvironmentRolloutState,
        *,
        cause_error_type: str,
    ) -> dict[str, object]:
        return {
            "environment_reset_receipt": dict(state.reset_receipt),
            "environment_receipts": tuple(dict(item) for item in state.receipts),
            "environment_current_state": self._current_public_state(state),
            "evaluator_environment_trace": tuple(
                dict(item) for item in state.evaluator_trace
            ),
            "tool_receipts": tuple(dict(item) for item in state.tool_receipts),
            "model_calls": tuple(dict(item) for item in state.model_calls),
            "environment_revision": state.episode.revision,
            "environment_terminal": state.terminal,
            "cause_error_type": cause_error_type,
        }

    async def _execute_turn(
        self,
        request: AgentRequest,
        state: _EnvironmentRolloutState,
    ) -> None:
        episode = state.episode
        session = episode.session
        turn = state.turns_used + 1
        if turn > self._max_turns or state.terminal:
            return
        native_admissible_actions, has_search_bar = _admissible_actions(
            session.task_family,
            session.available_actions,
        )
        if not native_admissible_actions:
            raise EnvironmentExecutionError(
                "environment exposed no admissible actions before terminal"
            )
        model_visible_actions = native_admissible_actions
        if session.task_family.casefold() == "webshop":
            model_visible_actions, _, _ = _webshop_model_visible_actions(
                task_instruction=state.original_task_instruction,
                observation=episode.observation,
                receipts=state.receipts,
                native_actions=native_admissible_actions,
            )
        if not model_visible_actions:
            raise EnvironmentExecutionError(
                "environment exposed no model-visible actions before terminal"
            )
        prompt = _action_prompt(
            request,
            task_family=session.task_family,
            observation=episode.observation,
            admissible_actions=model_visible_actions,
            receipts=state.receipts,
            turn=turn,
            max_observation_chars=self._max_observation_chars,
            structured_actions=self._structured_actions,
        )
        public_state = _public_state_feedback(
            request,
            task_family=session.task_family,
            observation=episode.observation,
            admissible_actions=model_visible_actions,
            receipts=state.receipts,
        )
        _, observation_clipped = _prompt_observation(
            episode.observation,
            self._max_observation_chars,
        )
        model_metadata = {
            **dict(request.model.metadata),
            "max_tokens": str(self._max_action_tokens),
        }
        if self._structured_actions:
            model_metadata["response_json_schema"] = json.dumps(
                _webshop_structured_action_schema(
                    model_visible_actions,
                    has_search_bar=has_search_bar,
                ),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        model_request = replace(
            request,
            request_id=f"{request.request_id}:environment:{turn}",
            problem=prompt,
            agent=replace(
                request.agent,
                # ReAct remains the outer execution mode; a provider turn
                # emits one action and is not assigned a ReAct role.
                execution_mode="reasoning",
                contract=(
                    "Select exactly one currently admissible SkillFlow "
                    "StructuredAction."
                    if self._structured_actions
                    else "Select exactly one native action permitted by the "
                    "current admissible-action list."
                ),
                artifact_type="environment_action",
                completion_condition=(
                    "The response validates as one currently admissible "
                    "environment action."
                ),
            ),
            model=replace(request.model, metadata=model_metadata),
        )
        generated = await self._gateway.generate(model_request)
        response = (
            generated
            if isinstance(generated, AgentResponse)
            else AgentResponse(generated)
        )
        raw_action = response.text
        state.model_calls.append(
            {
                "turn": turn,
                "request_id": model_request.request_id,
                "metadata": dict(response.metadata),
                "public_state": public_state,
                "observation_clipped": observation_clipped,
                "action_format": (
                    "structured-action-json@1"
                    if self._structured_actions
                    else "native-action-text@1"
                ),
            }
        )
        if self._structured_actions:
            action, observation_status = _parse_webshop_structured_action(
                raw_action,
                admissible_actions=model_visible_actions,
                webshop_has_search_bar=has_search_bar,
            )
            # A provider may still emit a native-legal action removed from the
            # model-visible domain by a public state precondition.  Parse that
            # semantic action against the lossless native domain only to emit
            # a typed precondition receipt; never dispatch it to the Tool.
            if action is None and model_visible_actions != native_admissible_actions:
                native_candidate, native_status = _parse_webshop_structured_action(
                    raw_action,
                    admissible_actions=native_admissible_actions,
                    webshop_has_search_bar=has_search_bar,
                )
                if native_candidate is not None and (
                    _webshop_action_precondition_failure(
                        action=native_candidate,
                        task_instruction=state.original_task_instruction,
                        observation=episode.observation,
                        receipts=state.receipts,
                    )
                    is not None
                ):
                    action = native_candidate
                    observation_status = native_status
        else:
            action = _parse_action(
                raw_action,
                task_family=session.task_family,
                admissible_actions=model_visible_actions,
                webshop_has_search_bar=has_search_bar,
            )
            observation_status = "success" if action is not None else "parse_error"

        precondition_failure: Optional[str] = None
        if action is not None and session.task_family.casefold() == "webshop":
            precondition_failure = _webshop_action_precondition_failure(
                action=action,
                task_instruction=state.original_task_instruction,
                observation=episode.observation,
                receipts=state.receipts,
            )
        if action is not None and precondition_failure is not None:
            state.turns_used = turn
            state.receipts.append(
                {
                    "receipt_type": "environment_transition",
                    "environment_episode_id": state.episode_id,
                    "environment_id": session.environment_id,
                    "turn": turn,
                    "environment_revision_before": episode.revision,
                    "environment_revision_after": episode.revision,
                    "observation": episode.observation,
                    "admissible_actions": list(native_admissible_actions),
                    "model_visible_admissible_actions": list(model_visible_actions),
                    "raw_model_output": raw_action,
                    "action": action,
                    "next_observation": episode.observation,
                    "next_admissible_actions": list(native_admissible_actions),
                    "terminal": False,
                    "state_advanced": False,
                    "observation_status": "precondition_failed",
                    "precondition_failure_reason": precondition_failure,
                    "public_state": public_state,
                }
            )
            state.evaluator_trace.append(
                {
                    "step": turn - 1,
                    "observation": episode.raw_observation,
                    "legal_actions": list(native_admissible_actions),
                    "action": "<INVALID>",
                    "raw_graph_output": "",
                    "structured_action_output": (
                        raw_action if self._structured_actions else None
                    ),
                    "structured_action_status": "precondition_failed",
                    "next_observation": episode.raw_observation,
                    "feedback": (
                        "[INVALID] Public action precondition failed: "
                        f"{precondition_failure}."
                    ),
                    "reward": 0.0,
                    "done": False,
                    "state_advanced": False,
                    "parse_error": False,
                    "precondition_failed": True,
                    "precondition_failure_reason": precondition_failure,
                    "info": {
                        "precondition_failed": True,
                        "reason": precondition_failure,
                    },
                    "public_state": public_state,
                }
            )
            return

        if action is None:
            state.turns_used = turn
            state.receipts.append(
                {
                    "receipt_type": "environment_transition",
                    "environment_episode_id": state.episode_id,
                    "environment_id": session.environment_id,
                    "turn": turn,
                    "environment_revision_before": episode.revision,
                    "environment_revision_after": episode.revision,
                    "observation": episode.observation,
                    "admissible_actions": list(native_admissible_actions),
                    "model_visible_admissible_actions": list(model_visible_actions),
                    "raw_model_output": raw_action,
                    "action": None,
                    "next_observation": episode.observation,
                    "next_admissible_actions": list(native_admissible_actions),
                    "terminal": False,
                    "state_advanced": False,
                    "observation_status": observation_status,
                    "public_state": public_state,
                }
            )
            state.evaluator_trace.append(
                {
                    "step": turn - 1,
                    "observation": episode.raw_observation,
                    "legal_actions": list(native_admissible_actions),
                    "action": "<INVALID>",
                    # The native evaluator replays its established
                    # ``<action>...</action>`` wire.  Keep an invalid native
                    # wire here and persist the exact SkillFlow
                    # StructuredAction attempt separately; otherwise a valid
                    # JSON object is incorrectly interpreted as a native
                    # WebShop command during replay.
                    "raw_graph_output": "",
                    "structured_action_output": (
                        raw_action if self._structured_actions else None
                    ),
                    "structured_action_status": observation_status,
                    "next_observation": episode.raw_observation,
                    "feedback": "[INVALID] No valid <action> tag found.",
                    "reward": 0.0,
                    "done": False,
                    "state_advanced": False,
                    "parse_error": True,
                    "info": {"parse_error": True},
                    "public_state": public_state,
                }
            )
            return

        previous_revision = episode.revision
        previous_raw_observation = episode.raw_observation
        previous_observation = episode.observation
        result, tool_receipt = await self._tool_registry.ainvoke_with_receipt(
            self._tool_id,
            ToolRequest(action, {}),
        )
        state.tool_receipts.append(tool_receipt.to_value())
        if result is None:
            state.turns_used = turn
            state.receipts.append(
                {
                    "receipt_type": "environment_transition",
                    "environment_episode_id": state.episode_id,
                    "environment_id": session.environment_id,
                    "turn": turn,
                    "environment_revision_before": previous_revision,
                    "environment_revision_after": episode.revision,
                    "observation": previous_observation,
                    "admissible_actions": list(native_admissible_actions),
                    "model_visible_admissible_actions": list(model_visible_actions),
                    "raw_model_output": raw_action,
                    "action": action,
                    "next_observation": episode.observation,
                    "next_admissible_actions": list(native_admissible_actions),
                    "terminal": state.terminal,
                    "state_advanced": False,
                    "observation_status": "tool_error",
                    "tool_error_type": tool_receipt.error_type or "unknown_error",
                    "public_state": public_state,
                }
            )
            raise EnvironmentExecutionError(
                "registered environment tool failed with "
                f"{tool_receipt.error_type or 'unknown_error'}"
            )
        transition = self._environment_backend.take_transition()
        value = result.value
        if (
            not isinstance(value, dict)
            or value.get("observation") != transition.public_observation
            or value.get("terminal") is not transition.terminal
            or value.get("environment_revision") != episode.revision
        ):
            raise EnvironmentExecutionError(
                "registered environment tool returned an incompatible result"
            )
        state.turns_used = turn
        state.terminal = transition.terminal
        next_actions: tuple[str, ...] = ()
        if not state.terminal:
            next_actions, _ = _admissible_actions(
                session.task_family,
                session.available_actions,
            )
        state.receipts.append(
            {
                "receipt_type": "environment_transition",
                "environment_episode_id": state.episode_id,
                "environment_id": session.environment_id,
                "turn": turn,
                "environment_revision_before": previous_revision,
                "environment_revision_after": episode.revision,
                "observation": previous_observation,
                "admissible_actions": list(native_admissible_actions),
                "model_visible_admissible_actions": list(model_visible_actions),
                "raw_model_output": raw_action,
                "action": action,
                "next_observation": transition.public_observation,
                "next_admissible_actions": list(next_actions),
                "terminal": state.terminal,
                "state_advanced": True,
                "observation_status": observation_status,
                "public_state": public_state,
            }
        )
        state.evaluator_trace.append(
            {
                "step": turn - 1,
                "observation": previous_raw_observation,
                "legal_actions": list(native_admissible_actions),
                "action": action,
                # Preserve the native evaluator protocol while retaining the
                # exact sampled SkillFlow StructuredAction as an independent
                # trajectory field.  This is a formatting adapter only; the
                # semantic action remains the already validated ``action``.
                "raw_graph_output": (
                    f"<action>{action}</action>"
                    if self._structured_actions
                    else raw_action
                ),
                "structured_action_output": (
                    raw_action if self._structured_actions else None
                ),
                "structured_action_status": (
                    observation_status if self._structured_actions else None
                ),
                "next_observation": transition.observation,
                "reward": transition.reward,
                "done": state.terminal,
                "info": dict(transition.info),
                "state_advanced": True,
                "public_state": public_state,
            }
        )

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        if request.agent.allowed_tools != (self._tool_id,):
            raise EnvironmentExecutionError(
                "environment Agent must allow exactly its task-scoped environment tool"
            )
        async with self._step_lock:
            if self._stepwise_director and self._live_state is not None:
                state = self._live_state
                if (
                    _original_task_instruction(request.problem)
                    != state.original_task_instruction
                ):
                    raise EnvironmentExecutionError(
                        "original task instruction changed inside one environment episode"
                    )
                token = self._environment_backend.bind(state.episode)
            else:
                state, token = await self._new_state(request)
                if self._stepwise_director:
                    self._live_state = state
            try:
                if self._stepwise_director:
                    await self._execute_turn(request, state)
                else:
                    while not state.terminal and state.turns_used < self._max_turns:
                        await self._execute_turn(request, state)
                return self._response(state)
            except asyncio.CancelledError as exc:
                metadata = self._failure_metadata(
                    state,
                    cause_error_type=type(exc).__name__,
                )
                for key, value in metadata.items():
                    setattr(exc, key, value)
                self._cancelled_prefixes[request.request_id] = MappingProxyType(
                    metadata
                )
                raise
            except Exception as exc:
                cause_error_type = (
                    exc.cause_error_type
                    if isinstance(exc, EnvironmentExecutionError)
                    and exc.cause_error_type is not None
                    else type(exc).__name__
                )
                metadata = self._failure_metadata(
                    state,
                    cause_error_type=cause_error_type,
                )
                raise EnvironmentExecutionError(
                    " ".join(str(exc).split()) or "environment execution failed",
                    environment_reset_receipt=metadata["environment_reset_receipt"],
                    environment_receipts=metadata["environment_receipts"],
                    environment_current_state=metadata["environment_current_state"],
                    evaluator_environment_trace=metadata[
                        "evaluator_environment_trace"
                    ],
                    tool_receipts=metadata["tool_receipts"],
                    model_calls=metadata["model_calls"],
                    environment_revision=state.episode.revision,
                    environment_terminal=state.terminal,
                    cause_error_type=cause_error_type,
                ) from exc
            finally:
                self._environment_backend.end(token)


@dataclass(frozen=True, slots=True)
class EnvironmentExecutionResources:
    """One shared capability/registry/adapter set for AgentRuntime wiring."""

    tool_id: str
    tool_registry: ToolRegistry
    execution_adapter: EnvironmentExecutionAdapter


def build_environment_execution_resources(
    *,
    gateway: AgentGateway,
    session_factory: EnvironmentSessionFactory,
    task_family: str,
    max_turns: int,
    max_action_tokens: int = 512,
    max_observation_chars: int = 0,
    stepwise_director: bool = False,
    structured_actions: bool = False,
    tool_version: str = "skillflow.ragen_adapter.v2",
    timeout_seconds: Optional[float] = None,
) -> EnvironmentExecutionResources:
    """Create a real environment capability and its bounded execution adapter.

    The returned registry must be supplied unchanged to both AgentRuntime and
    the Director; ``execution_adapter`` is registered for the ``react`` mode.
    This factory prevents capability metadata, runtime admission, and the
    backend that actually performs ``step`` from drifting apart.
    """

    if not isinstance(task_family, str) or task_family.strip().lower() not in {
        "alfworld",
        "webshop",
    }:
        raise ValueError("task_family must be alfworld or webshop")
    if not isinstance(tool_version, str) or not tool_version.strip():
        raise ValueError("tool_version must be non-empty")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when supplied")
    family = task_family.strip().lower()
    tool_id = f"{family}.environment"
    backend = EnvironmentToolBackend(
        session_factory=session_factory,
        task_family=family,
        tool_id=tool_id,
    )
    capability = ToolCapability(
        tool_id=tool_id,
        dataset_scope=(family,),
        # The executable domain is the live session's admissible native
        # actions, not a fixed StructuredAction name catalog.
        action_schemas={},
        input_schema={
            "action": {
                "type": "string",
                "description": "one action from the current admissible action list",
            },
            "arguments": {"type": "object", "maxProperties": 0},
        },
        output_schema={
            "type": "object",
            "required": ["environment_revision", "observation", "terminal"],
            "properties": {
                "environment_revision": {"type": "integer", "minimum": 1},
                "observation": {"type": "string"},
                "terminal": {"type": "boolean"},
            },
        },
        side_effect="environment_state_transition",
        timeout_seconds=timeout_seconds,
        version=tool_version.strip(),
        availability=True,
    )
    registry = ToolRegistry(
        (ToolRegistration(tool_id, backend, capability=capability),)
    )
    adapter = EnvironmentExecutionAdapter(
        gateway=gateway,
        tool_registry=registry,
        environment_backend=backend,
        max_turns=max_turns,
        max_action_tokens=max_action_tokens,
        max_observation_chars=max_observation_chars,
        stepwise_director=stepwise_director,
        structured_actions=structured_actions,
    )
    return EnvironmentExecutionResources(tool_id, registry, adapter)


__all__ = [
    "build_environment_execution_resources",
    "evaluator_locked_ragen_session_factory",
    "EnvironmentExecutionAdapter",
    "EnvironmentExecutionError",
    "EnvironmentExecutionResources",
    "EnvironmentSession",
    "EnvironmentSessionFactory",
    "EnvironmentToolBackend",
    "RAGENEnvironmentSession",
    "RAGENEnvironmentSessionFactory",
]
