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
    reset_receipt: dict[str, object]
    receipts: list[dict[str, object]] = field(default_factory=list)
    evaluator_trace: list[dict[str, object]] = field(default_factory=list)
    tool_receipts: list[dict[str, object]] = field(default_factory=list)
    model_calls: list[dict[str, object]] = field(default_factory=list)
    turns_used: int = 0
    terminal: bool = False


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
    if task_family.lower() == "alfworld":
        facts = _alfworld_task_facts(request.problem)
        visible = [
            f"target_class={facts['target_class']}",
            f"destination_class={facts['destination_class']}",
            f"required_transform={facts['required_transform']}",
            f"count={facts['count']}",
        ]
        lines.append("Task facts: " + "; ".join(visible) + ".")
        target = facts.get("target_class")
        task_lower = request.problem.lower()
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
        constraints = _webshop_task_constraints(request.problem)
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

    recent_actions = completed_actions[-6:]
    if recent_actions:
        lines.append("Recent executed actions: " + " | ".join(recent_actions) + ".")
    if len(recent_actions) >= 4 and (
        recent_actions[-4] == recent_actions[-2]
        and recent_actions[-3] == recent_actions[-1]
    ):
        lines.append(
            "No-progress signal: the last four public actions form an A-B-A-B "
            "oscillation; reassess the task constraints and current observation."
        )
    repeated_pairs = 0
    for item in receipts:
        if (
            item.get("state_advanced") is True
            and item.get("observation") == observation
            and isinstance(item.get("action"), str)
        ):
            repeated_pairs += 1
    if repeated_pairs:
        lines.append(
            "No-progress signal: this exact public observation has already been "
            "used as an action-decision state; avoid repeating an action that did not "
            "advance the task."
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
        f"Task:\n{request.problem}\n\n"
        f"Previous environment turns:\n{_history_text(receipts)}\n\n"
        f"{public_state}\n\n"
        f"Current observation (turn {turn}):\n{visible_observation}\n\n"
        f"Admissible actions:\n{actions}\n\n"
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
            "observation": episode.observation,
            "admissible_actions": list(reset_actions),
            "terminal": False,
        }
        return (
            _EnvironmentRolloutState(
                episode=episode,
                episode_id=episode_id,
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
        admissible_actions = ()
        if not state.terminal:
            admissible_actions, _ = _admissible_actions(
                episode.session.task_family,
                episode.session.available_actions,
            )
        last = state.receipts[-1] if state.receipts else None
        return {
            "environment_episode_id": state.episode_id,
            "environment_id": episode.session.environment_id,
            "task_family": episode.session.task_family,
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
            "admissible_actions": list(admissible_actions),
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
        admissible_actions, has_search_bar = _admissible_actions(
            session.task_family,
            session.available_actions,
        )
        if not admissible_actions:
            raise EnvironmentExecutionError(
                "environment exposed no admissible actions before terminal"
            )
        prompt = _action_prompt(
            request,
            task_family=session.task_family,
            observation=episode.observation,
            admissible_actions=admissible_actions,
            receipts=state.receipts,
            turn=turn,
            max_observation_chars=self._max_observation_chars,
            structured_actions=self._structured_actions,
        )
        public_state = _public_state_feedback(
            request,
            task_family=session.task_family,
            observation=episode.observation,
            admissible_actions=admissible_actions,
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
                    admissible_actions,
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
        response = generated if isinstance(generated, AgentResponse) else AgentResponse(generated)
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
                admissible_actions=admissible_actions,
                webshop_has_search_bar=has_search_bar,
            )
        else:
            action = _parse_action(
                raw_action,
                task_family=session.task_family,
                admissible_actions=admissible_actions,
                webshop_has_search_bar=has_search_bar,
            )
            observation_status = "success" if action is not None else "parse_error"

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
                    "admissible_actions": list(admissible_actions),
                    "raw_model_output": raw_action,
                    "action": None,
                    "next_observation": episode.observation,
                    "next_admissible_actions": list(admissible_actions),
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
                    "legal_actions": list(admissible_actions),
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
                "admissible_actions": list(admissible_actions),
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
                "legal_actions": list(admissible_actions),
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
