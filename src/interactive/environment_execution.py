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
from uuid import uuid4

from .agent_runtime import AgentGateway, AgentRequest, AgentResponse, GatewayResponse
from .tool_runtime import (
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

    def close(self) -> Union[None, Awaitable[None]]: ...


EnvironmentSessionFactory = Callable[[AgentRequest], EnvironmentSession]


@dataclass(slots=True)
class _EnvironmentTransition:
    observation: str
    reward: object
    terminal: bool
    info: Mapping[str, object]


@dataclass(slots=True)
class _EnvironmentEpisode:
    session: EnvironmentSession
    observation: str
    episode_id: str
    revision: int = 0
    terminal: bool = False
    pending_transition: Optional[_EnvironmentTransition] = None
    reset_receipt: Mapping[str, object] = field(default_factory=dict)
    receipts: list[dict[str, object]] = field(default_factory=list)
    evaluator_trace: list[dict[str, object]] = field(default_factory=list)
    tool_receipts: list[dict[str, object]] = field(default_factory=list)
    model_calls: list[dict[str, object]] = field(default_factory=list)


class EnvironmentToolBackend:
    """Rollout-scoped backend for one real RAGEN environment resource.

    The backend is registered in the same :class:`ToolRegistry` exposed to the
    Director and AgentRuntime.  One backend is created by ``_runtime_for_task``
    for one rollout, and it lazily creates exactly one environment session.
    Canvas repair or re-execution therefore continues the same world revision
    instead of resetting the game.  A lock serializes all state mutations;
    independent rollouts use independent backend instances and sessions.
    Public ``ToolResult`` values intentionally omit reward and evaluator
    ``info``; the adapter consumes those fields through the private episode
    boundary solely to build deterministic evaluator replay.
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
        self._rollout_episode: Optional[_EnvironmentEpisode] = None
        self._execution_lock = asyncio.Lock()
        self._closed = False

    async def begin(
        self, request: AgentRequest
    ) -> tuple[_EnvironmentEpisode, Token[Optional[_EnvironmentEpisode]]]:
        if self._episode.get() is not None:
            raise EnvironmentExecutionError(
                "environment episode is already active in this execution context"
            )
        await self._execution_lock.acquire()
        created_session: Optional[EnvironmentSession] = None
        try:
            if self._closed:
                raise EnvironmentExecutionError("environment rollout is already closed")
            episode = self._rollout_episode
            if episode is None:
                session = self.session_factory(request)
                created_session = session
                if str(session.task_family).strip().lower() != self.task_family:
                    raise EnvironmentExecutionError(
                        "environment session task family does not match its tool capability"
                    )
                observation = await _resolve(session.reset())
                if not isinstance(observation, str):
                    raise EnvironmentExecutionError("environment reset must return text")
                if observation.startswith("[ENV_UNAVAILABLE]"):
                    raise EnvironmentExecutionError(observation)
                episode_id = f"{session.environment_id}:{uuid4()}"
                reset_actions, _ = _admissible_actions(
                    session.task_family, session.available_actions
                )
                episode = _EnvironmentEpisode(
                    session=session,
                    observation=observation,
                    episode_id=episode_id,
                    reset_receipt={
                        "receipt_type": "environment_reset",
                        "environment_episode_id": episode_id,
                        "episode_id": episode_id,
                        "environment_id": session.environment_id,
                        "environment_revision": 0,
                        "observation": observation,
                        "admissible_actions": list(reset_actions),
                        "terminal": False,
                    },
                )
                self._rollout_episode = episode
                created_session = None
            return episode, self._episode.set(episode)
        except BaseException:
            if created_session is not None:
                close = getattr(created_session, "close", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            self._execution_lock.release()
            raise

    def end(self, token: Token[Optional[_EnvironmentEpisode]]) -> None:
        self._episode.reset(token)
        self._execution_lock.release()

    def reset(self) -> None:
        """Reset the retained task-scoped episode before a new task starts."""

        if self._execution_lock.locked():
            raise EnvironmentExecutionError(
                "cannot reset an environment rollout while an execution is active"
            )
        if self._closed:
            raise EnvironmentExecutionError("environment rollout is already closed")
        episode = self._rollout_episode
        self._rollout_episode = None
        if episode is None:
            return
        close = getattr(episode.session, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                raise EnvironmentExecutionError(
                    "asynchronous environment close is unsupported by this runtime"
                )

    def close(self) -> None:
        """Close the one rollout-owned simulator session exactly once."""

        if self._closed:
            return
        if self._execution_lock.locked():
            raise EnvironmentExecutionError(
                "cannot close an environment rollout while an execution is active"
            )
        self._closed = True
        episode = self._rollout_episode
        if episode is None:
            return
        close = getattr(episode.session, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                raise EnvironmentExecutionError(
                    "asynchronous environment close is unsupported by this runtime"
                )

    async def invoke(self, request: ToolRequest) -> ToolResult:
        episode = self._episode.get()
        if episode is None:
            raise EnvironmentExecutionError(
                "environment tool requires an active request-scoped episode"
            )
        if episode.terminal:
            raise EnvironmentExecutionError(
                "environment tool cannot mutate a terminal episode"
            )
        if self.task_family == "alfworld":
            if request.action != "act" or set(request.arguments) != {"command"}:
                raise EnvironmentExecutionError(
                    "ALFWorld requires SkillFlow act(command) Tool actions"
                )
            native_action = request.arguments["command"]
            if not isinstance(native_action, str) or not native_action.strip():
                raise EnvironmentExecutionError(
                    "ALFWorld act(command) requires a non-empty command"
                )
            native_action = native_action.strip()
        else:
            if dict(request.arguments):
                raise EnvironmentExecutionError(
                    "environment actions do not accept an arguments object"
                )
            native_action = request.action
        admissible_actions, has_search_bar = _admissible_actions(
            self.task_family, episode.session.available_actions
        )
        if (
            _parse_action(
                native_action,
                task_family=self.task_family,
                admissible_actions=admissible_actions,
                webshop_has_search_bar=has_search_bar,
            )
            != native_action
        ):
            raise EnvironmentExecutionError(
                "environment tool action is not currently admissible"
            )
        transition = await _resolve(episode.session.step(native_action))
        if not isinstance(transition, tuple) or len(transition) != 4:
            raise EnvironmentExecutionError(
                "environment step must return observation, reward, terminal, and info"
            )
        observation, reward, terminal, info = transition
        if not isinstance(observation, str):
            raise EnvironmentExecutionError("environment observation must be text")
        if type(terminal) is not bool:
            raise EnvironmentExecutionError("environment terminal flag must be boolean")
        if not isinstance(info, Mapping):
            raise EnvironmentExecutionError("environment info must be a mapping")
        if observation.startswith(("[ENV_UNAVAILABLE]", "[ERROR]")):
            raise EnvironmentExecutionError(observation)
        episode.revision += 1
        episode.observation = observation
        episode.terminal = terminal
        episode.pending_transition = _EnvironmentTransition(
            observation=observation,
            reward=reward,
            terminal=terminal,
            info=MappingProxyType(dict(info)),
        )
        return ToolResult(
            {
                "environment_revision": episode.revision,
                "observation": observation,
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

    def close(self) -> None:
        """Close SkillFlow's live TextWorld session when the rollout ends."""

        environment = getattr(self.adapter, "_env", None)
        live_environment = getattr(environment, "alfred_env", None)
        close = getattr(live_environment, "close", None)
        if callable(close):
            close()


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
        return (
            tuple(
                str(action)
                for action in available_actions
                if task_family.lower() != "alfworld" or str(action) != "help"
            ),
            False,
        )
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
    # ``OpenAICompatibleGateway`` forwards request-scoped JSON Schema through
    # ``response_format``. ALFWorld uses a one-field envelope at that model
    # boundary so the schema can enumerate the current native actions; unwrap
    # it here before the unchanged native environment protocol and parser.
    try:
        structured = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        structured = None
    if isinstance(structured, Mapping):
        if set(structured) != {"action"} or not isinstance(
            structured.get("action"), str
        ):
            return None
        candidate = str(structured["action"]).strip()
        return (
            candidate
            if candidate in admissible_actions
            and candidate != "search[<your query>]"
            else None
        )
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
    # ``_workflow_problem`` appends the public environment interface after a
    # blank line.  SkillFlow parses the immutable ALFWorld instruction stored
    # on the environment, not that runtime prose.  Keep the same boundary here
    # so end-anchored official task patterns cannot be invalidated by the
    # appended Tool contract.
    instruction = str(task or "").split("\n\n", 1)[0]
    text = " ".join(instruction.lower().rstrip(".").split())
    if not text:
        return facts
    match = re.search(
        r"pick up (?:the |an? |some )?(\S+?)(?: \d)? from (\S+?)(?: \d)? "
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
        r"(?:examine|look at) (?:the |an? |some )?(\S+?)(?: \d)? "
        r"(?:with|under|by) (?:the |an? )?(?:desklamp|lamp)",
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
        r"(heat|cool|clean) (?:some |an? |the )?(\S+?)(?: \d)? and put it "
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
        r"put (?:an? |some |the )?(?:(clean|washed|cool|cold|hot|heated|cooked) )?"
        r"(\S+?)(?: \d)? (?:in/on|in|on) (\S+?)(?: \d)?$",
        text,
    )
    if match:
        adjective, target, destination = match.groups()
        if target in {"it", "them"}:
            return facts
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


def _alfworld_action_response_schema(
    admissible_actions: Sequence[str],
) -> dict[str, object]:
    """Constrain one model turn to the current public native action domain."""

    return {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": list(admissible_actions),
            }
        },
        "additionalProperties": False,
    }


def _alfworld_observation_mentions_class(
    observation: object,
    object_class: object,
) -> bool:
    """Match one visible ALFWorld class without substring aliases."""

    normalized = _alfworld_object_class(object_class)
    if not normalized:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized)}(?:\s+\d+)?(?![a-z0-9])",
            str(observation or "").lower(),
        )
    )


def _alfworld_visible_contents(
    observation: object,
) -> Optional[tuple[str, ...]]:
    """Extract only explicitly visible receptacle contents from observation."""

    compact = " ".join(str(observation or "").split())
    if not compact:
        return None
    match = re.search(
        r"\b(?:in|on)\s+(?:it|the\s+[^,.;]+),\s*you\s+see\s+"
        r"(.+?)(?:[.!]|$)",
        compact,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\byou\s+see\s+(.+?)(?:[.!]|$)",
            compact,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    visible = match.group(1).strip(" ,")
    if re.fullmatch(r"(?:nothing|no(?:thing| objects?)?)", visible, re.IGNORECASE):
        return ()
    visible = re.sub(r"\s*,?\s+and\s+", ",", visible, flags=re.IGNORECASE)
    items: list[str] = []
    for item in visible.split(","):
        normalized = re.sub(
            r"^(?:a|an|the|some)\s+", "", item.strip(), flags=re.IGNORECASE
        ).strip()
        if normalized and normalized.casefold() not in {
            value.casefold() for value in items
        }:
            items.append(normalized)
    return tuple(items) if items else None


def _alfworld_public_scene_memory(
    receipts: Sequence[Mapping[str, object]],
    *,
    target_class: object = None,
) -> list[dict[str, object]]:
    """Rebuild persistent scene memory from public Action--Observation edges.

    A ``go to`` arrival creates location memory. Later public ``open``,
    ``close`` and ``examine`` observations update the same entry rather than
    leaving stale arrival text authoritative. No simulator state, reward,
    terminal evaluator field or hidden inventory is consulted.
    """

    memory: dict[str, dict[str, object]] = {}

    def ensure(location: str) -> dict[str, object]:
        key = location.casefold()
        return memory.setdefault(
            key,
            {
                "location": location,
                "visit_turns": [],
                "evidence_turns": [],
                "open_state": "unknown",
                "contents": None,
                "contents_observed_turn": None,
                "target_evidence": "unknown",
                "target_negative_evidence_turns": [],
                "last_observation": "",
            },
        )

    def update_from_observation(
        entry: dict[str, object],
        observation: object,
        turn: object,
    ) -> None:
        compact = " ".join(str(observation or "").split())
        if not compact:
            return
        location = str(entry["location"])
        location_pattern = re.escape(location)
        if re.search(
            rf"\b(?:the\s+)?{location_pattern}\s+is\s+open\b",
            compact,
            flags=re.IGNORECASE,
        ) or re.search(
            rf"\byou\s+open\s+(?:the\s+)?{location_pattern}\b",
            compact,
            flags=re.IGNORECASE,
        ):
            entry["open_state"] = "open"
        elif re.search(
            rf"\b(?:the\s+)?{location_pattern}\s+is\s+closed\b",
            compact,
            flags=re.IGNORECASE,
        ) or re.search(
            rf"\byou\s+close\s+(?:the\s+)?{location_pattern}\b",
            compact,
            flags=re.IGNORECASE,
        ):
            entry["open_state"] = "closed"

        contents = _alfworld_visible_contents(compact)
        if contents is not None:
            entry["contents"] = list(contents)
            entry["contents_observed_turn"] = turn
            if _alfworld_object_class(target_class):
                target_visible = _alfworld_observation_mentions_class(
                    compact, target_class
                )
                entry["target_evidence"] = (
                    "present" if target_visible else "absent"
                )
                if not target_visible:
                    negative_turns = entry["target_negative_evidence_turns"]
                    if isinstance(negative_turns, list) and turn not in negative_turns:
                        negative_turns.append(turn)
        evidence_turns = entry["evidence_turns"]
        if isinstance(evidence_turns, list) and turn not in evidence_turns:
            evidence_turns.append(turn)
        entry["last_observation"] = compact[:300]

    for index, item in enumerate(receipts, start=1):
        if item.get("state_advanced") is not True:
            continue
        action = item.get("action")
        if not isinstance(action, str):
            continue
        turn = item.get("turn")
        if type(turn) is not int:
            turn = index
        result = item.get("next_observation", "")
        go = re.match(r"^\s*go\s+to\s+(.+?)\s*$", action, re.IGNORECASE)
        if go:
            location = go.group(1).strip()
            entry = ensure(location)
            visits = entry["visit_turns"]
            if isinstance(visits, list):
                visits.append(turn)
            update_from_observation(entry, result, turn)
            continue

        inspect = re.match(
            r"^\s*(open|close|examine)\s+(.+?)\s*$",
            action,
            re.IGNORECASE,
        )
        if inspect:
            operation, location = inspect.groups()
            existing = memory.get(location.casefold())
            visible_contents = _alfworld_visible_contents(result)
            visible_receptacle_state = bool(
                re.search(r"\bis\s+(?:open|closed)\b", str(result), re.IGNORECASE)
            )
            if (
                operation.casefold() != "examine"
                or existing is not None
                or visible_contents is not None
                or visible_receptacle_state
            ):
                update_from_observation(
                    existing if existing is not None else ensure(location),
                    result,
                    turn,
                )
            continue

        take = re.match(
            r"^\s*take\s+(.+?)\s+from\s+(.+?)\s*$", action, re.IGNORECASE
        )
        move = re.match(
            r"^\s*move\s+(.+?)\s+to\s+(.+?)\s*$", action, re.IGNORECASE
        )
        object_id = ""
        location = ""
        add_object = False
        if take:
            object_id, location = take.groups()
        elif move:
            object_id, location = move.groups()
            add_object = True
        entry = memory.get(location.strip().casefold()) if location else None
        if entry is None or not isinstance(entry.get("contents"), list):
            continue
        contents_list = entry["contents"]
        assert isinstance(contents_list, list)
        if add_object:
            if object_id.casefold() not in {
                str(value).casefold() for value in contents_list
            }:
                contents_list.append(object_id.strip())
        else:
            entry["contents"] = [
                value
                for value in contents_list
                if str(value).casefold() != object_id.strip().casefold()
            ]
        entry["contents_observed_turn"] = turn
        if _alfworld_object_class(target_class):
            latest_contents = entry["contents"]
            assert isinstance(latest_contents, list)
            target_visible = any(
                _alfworld_object_class(value)
                == _alfworld_object_class(target_class)
                for value in latest_contents
            )
            entry["target_evidence"] = "present" if target_visible else "absent"
            if not target_visible:
                negative_turns = entry["target_negative_evidence_turns"]
                if isinstance(negative_turns, list) and turn not in negative_turns:
                    negative_turns.append(turn)

    return [dict(entry) for entry in memory.values()]


def _alfworld_public_goal_progress(
    facts: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Project public goal milestones and turns since the latest new one."""

    target_class = _alfworld_object_class(facts.get("target_class"))
    destination_class = _alfworld_object_class(facts.get("destination_class"))
    required_transform = str(facts.get("required_transform") or "")
    acquired: dict[str, str] = {}
    held: dict[str, str] = {}
    transformed: dict[str, str] = {}
    placed: dict[str, str] = {}
    ever_placed: dict[str, str] = {}
    lamp_use_completed = False
    last_progress_turn = 0
    turns_used = 0

    for index, item in enumerate(receipts, start=1):
        raw_turn = item.get("turn")
        turn = raw_turn if type(raw_turn) is int else index
        turns_used = max(turns_used, turn)
        if item.get("state_advanced") is not True:
            continue
        action = item.get("action")
        if not isinstance(action, str):
            continue
        take = re.match(
            r"^take\s+(.+?)\s+from\s+(.+)$", action, flags=re.IGNORECASE
        )
        if take and _alfworld_object_class(take.group(1)) == target_class:
            object_id = take.group(1).strip()
            key = object_id.casefold()
            if key not in acquired:
                acquired[key] = object_id
                last_progress_turn = turn
            held[key] = object_id
            placed.pop(key, None)
            continue
        move = re.match(
            r"^move\s+(.+?)\s+to\s+(.+)$", action, flags=re.IGNORECASE
        )
        if move and _alfworld_object_class(move.group(1)) == target_class:
            object_id = move.group(1).strip()
            key = object_id.casefold()
            held.pop(key, None)
            if _alfworld_object_class(move.group(2)) == destination_class:
                placed[key] = object_id
                if key not in ever_placed:
                    ever_placed[key] = object_id
                    last_progress_turn = turn
            else:
                placed.pop(key, None)
            continue
        transform = re.match(
            r"^(clean|cool|heat)\s+(.+?)\s+with\s+",
            action,
            flags=re.IGNORECASE,
        )
        if transform and _alfworld_object_class(transform.group(2)) == target_class:
            operation = transform.group(1).casefold()
            object_id = transform.group(2).strip()
            key = object_id.casefold()
            held[key] = object_id
            if operation == required_transform.casefold():
                if key not in transformed:
                    transformed[key] = object_id
                    last_progress_turn = turn
            elif required_transform.casefold() in {"heat", "cool"} and operation in {
                "heat",
                "cool",
            }:
                # Temperature is a single public state dimension: a later
                # opposite action invalidates the earlier required state for
                # that instance. Cleaning has no corresponding public inverse.
                transformed.pop(key, None)
            continue
        if (
            facts.get("examine_with_desklamp")
            and re.fullmatch(
                r"use\s+desklamp\s+\d+",
                action.strip(),
                flags=re.IGNORECASE,
            )
            and bool(held)
            and not lamp_use_completed
        ):
            lamp_use_completed = True
            last_progress_turn = turn

    transformed_and_placed = [
        object_id for key, object_id in placed.items() if key in transformed
    ]
    return {
        "acquired_target_instances": list(acquired.values()),
        "held_target_instances": list(held.values()),
        "transformed_target_instances": list(transformed.values()),
        "placed_target_instances": list(placed.values()),
        "transformed_and_placed_target_instances": transformed_and_placed,
        "lamp_use_completed": lamp_use_completed,
        "last_goal_progress_turn": last_progress_turn,
        "turns_since_goal_progress": max(turns_used - last_progress_turn, 0),
    }


def _alfworld_public_stall_diagnostic(
    request: AgentRequest,
    *,
    observation: str,
    admissible_actions: Sequence[str],
    receipts: Sequence[Mapping[str, object]],
    environment_terminal: bool = False,
) -> dict[str, object]:
    """Build a neutral structured stall receipt from public state only."""

    facts = _alfworld_task_facts(request.problem)
    progress = _alfworld_public_goal_progress(facts, receipts)

    def state_signature(
        visible_observation: object,
        actions: object,
    ) -> tuple[str, tuple[str, ...]]:
        normalized_observation = " ".join(
            str(visible_observation or "").casefold().split()
        )
        normalized_actions = tuple(
            sorted(
                str(action).casefold()
                for action in (
                    actions
                    if isinstance(actions, Sequence)
                    and not isinstance(actions, (str, bytes))
                    else ()
                )
            )
        )
        return normalized_observation, normalized_actions

    public_states: list[tuple[str, tuple[str, ...]]] = []
    if receipts:
        first = receipts[0]
        public_states.append(
            state_signature(
                first.get("observation", ""),
                first.get("admissible_actions", ()),
            )
        )
        for index, item in enumerate(receipts):
            next_actions: object = admissible_actions
            if index + 1 < len(receipts):
                next_actions = receipts[index + 1].get("admissible_actions", ())
            public_states.append(
                state_signature(item.get("next_observation", ""), next_actions)
            )
    else:
        public_states.append(state_signature(observation, admissible_actions))

    repeated_state_count = 0
    for index in range(len(public_states) - 1, 0, -1):
        if public_states[index] != public_states[index - 1]:
            break
        repeated_state_count += 1

    completed_actions = [
        str(item["action"])
        for item in receipts
        if item.get("state_advanced") is True
        and isinstance(item.get("action"), str)
    ]
    alternating_actions: list[str] = []
    if (
        len(completed_actions) >= 4
        and completed_actions[-4] == completed_actions[-2]
        and completed_actions[-3] == completed_actions[-1]
        and completed_actions[-4] != completed_actions[-3]
    ):
        alternating_actions = completed_actions[-2:]

    turns_since_progress = int(progress["turns_since_goal_progress"])
    signals: list[str] = []
    if repeated_state_count:
        signals.append("repeated_public_state")
    if alternating_actions:
        signals.append("alternating_action_loop")
    if turns_since_progress >= 4:
        signals.append("no_goal_predicate_progress")

    latest_advanced = bool(
        receipts and receipts[-1].get("state_advanced") is True
    )
    repeated_state_stall = repeated_state_count >= 2 or (
        repeated_state_count >= 1 and latest_advanced
    )
    # ALFWorld exploration often needs more than six distinct, valid
    # transitions before the first task predicate changes.  Lack of predicate
    # progress is therefore diagnostic context, not by itself a hard stall.
    # Only an observed repeated public state or an A-B-A-B action loop blocks
    # another bare ``continue`` at the Canvas boundary.
    stalled = (
        repeated_state_stall or bool(alternating_actions)
    ) and not environment_terminal
    return {
        "schema_version": "alfworld.public-stall.v1",
        "stalled": stalled,
        "signals": signals,
        "repeated_state_count": repeated_state_count,
        "alternating_actions": alternating_actions,
        "turns_since_goal_progress": turns_since_progress,
        "goal_predicates": progress,
    }


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
    total_action_budget: Optional[int] = None,
    remaining_action_budget: Optional[int] = None,
    environment_terminal: bool = False,
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
    if total_action_budget is not None and remaining_action_budget is not None:
        lines.append(
            "Action budget: "
            f"remaining_action_budget={max(remaining_action_budget, 0)}; "
            f"total_action_budget={max(total_action_budget, 0)}; "
            f"turns_used={len(receipts)}."
        )
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
        go_targets = [
            action[6:].strip()
            for action in admissible_actions
            if action.lower().startswith("go to ")
        ]
        open_targets = [
            action[5:].strip()
            for action in admissible_actions
            if action.lower().startswith("open ")
        ]
        action_groups = (
            ("go", go_targets),
            ("open", open_targets),
            (
                "take",
                [
                    action
                    for action in admissible_actions
                    if action.lower().startswith("take ")
                ],
            ),
            (
                "move",
                [
                    action
                    for action in admissible_actions
                    if action.lower().startswith("move ")
                ],
            ),
            (
                "state",
                [
                    action
                    for action in admissible_actions
                    if action.lower().startswith(("clean ", "cool ", "heat "))
                ],
            ),
            (
                "use",
                [
                    action
                    for action in admissible_actions
                    if action.lower().startswith("use ")
                ],
            ),
        )
        action_counts = [
            f"{label}={len(values)}"
            for label, values in action_groups
            if values
        ]
        if action_counts:
            lines.append(
                "Current admissible action type counts: "
                + ", ".join(action_counts)
                + "."
            )
        if go_targets:
            lines.append(
                "Current visible go targets: "
                + ", ".join(go_targets[:18])
                + (" ..." if len(go_targets) > 18 else "")
                + "."
            )
        if open_targets:
            lines.append(
                "Current visible open targets: "
                + ", ".join(open_targets[:12])
                + (" ..." if len(open_targets) > 12 else "")
                + "."
            )
        held: list[str] = []
        for action in admissible_actions:
            if action.lower().startswith(("move ", "clean ", "heat ", "cool ")):
                obj = _alfworld_action_object(action)
                if obj and obj not in held:
                    held.append(obj)
        if held:
            lines.append("Objects implied as held by current actions: " + ", ".join(held[:6]) + ".")
        target_class = _alfworld_object_class(target)
        held_classes = {
            _alfworld_object_class(object_id)
            for object_id in held
            if _alfworld_object_class(object_id)
        }
        if target_class and held_classes and target_class not in held_classes:
            lines.append(
                "Visible entity binding conflict: held object class(es)="
                + ", ".join(sorted(held_classes))
                + f"; task target_class={target_class}."
            )
        if target_class:
            target_mentions = []
            for action in admissible_actions:
                object_id = _alfworld_action_object(action)
                target_go = (
                    action.lower().startswith("go to ")
                    and _alfworld_object_class(action[6:].strip()) == target_class
                )
                if (
                    object_id
                    and _alfworld_object_class(object_id) == target_class
                ) or target_go:
                    target_mentions.append(action)
            if target_mentions:
                lines.append(
                    "Current admissible strings mentioning the target class: "
                    + " | ".join(target_mentions[:10])
                    + (" ..." if len(target_mentions) > 10 else "")
                    + "."
                )
        destination_class = _alfworld_object_class(
            facts.get("destination_class")
        )
        if destination_class:
            destination_mentions = []
            for action in admissible_actions:
                go_match = re.match(
                    r"^go\s+to\s+(.+)$",
                    action,
                    flags=re.IGNORECASE,
                )
                move_match = re.match(
                    r"^move\s+.+?\s+to\s+(.+)$",
                    action,
                    flags=re.IGNORECASE,
                )
                if (
                    go_match
                    and _alfworld_object_class(go_match.group(1))
                    == destination_class
                ) or (
                    move_match
                    and _alfworld_object_class(move_match.group(1))
                    == destination_class
                ):
                    destination_mentions.append(action)
            if destination_mentions:
                lines.append(
                    "Current admissible strings mentioning the destination class: "
                    + " | ".join(destination_mentions[:10])
                    + (" ..." if len(destination_mentions) > 10 else "")
                    + "."
                )
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
        required_transform = facts.get("required_transform")
        transformed_target_ids: set[str] = set()
        if required_transform in {"clean", "cool", "heat"} and target_class:
            for action in transform_actions:
                match = re.match(
                    rf"^{re.escape(str(required_transform))}\s+(.+?)\s+with\s+",
                    action,
                    flags=re.IGNORECASE,
                )
                if (
                    match
                    and _alfworld_object_class(match.group(1)) == target_class
                ):
                    transformed_target_ids.add(match.group(1).strip().casefold())
            lines.append(
                "Visible required transform progress: "
                + (
                    f"{required_transform} observed for "
                    + ", ".join(sorted(transformed_target_ids))
                    if transformed_target_ids
                    else (
                        f"no completed {required_transform} action for "
                        f"target_class={target_class}"
                    )
                )
                + "."
            )
            appliance = {
                "clean": "sinkbasin",
                "cool": "fridge",
                "heat": "microwave",
            }[str(required_transform)]
            transform_mentions = [
                action
                for action in admissible_actions
                if (
                    action.lower().startswith(f"{required_transform} ")
                    and _alfworld_object_class(
                        _alfworld_action_object(action)
                    )
                    == target_class
                )
                or (
                    action.lower().startswith("go to ")
                    and _alfworld_object_class(action[6:].strip()) == appliance
                )
                or appliance in action.lower()
            ]
            if transform_mentions:
                lines.append(
                    f"Current admissible strings for visible {required_transform} "
                    "state/appliance: "
                    + " | ".join(transform_mentions[:8])
                    + (" ..." if len(transform_mentions) > 8 else "")
                    + "."
                )
        if facts.get("examine_with_desklamp"):
            lamp_mentions = [
                action
                for action in admissible_actions
                if "desklamp" in action.lower()
                or action.lower().startswith("use ")
            ]
            if lamp_mentions:
                lines.append(
                    "Current admissible strings mentioning desklamp/use actions: "
                    + " | ".join(lamp_mentions[:8])
                    + (" ..." if len(lamp_mentions) > 8 else "")
                    + "."
                )
        if target and facts.get("destination_class"):
            # This is current public placement state, not an ever-completed
            # counter. A later ``take X from destination`` visibly retracts
            # the earlier subgoal and must therefore remove X from progress.
            placed: dict[str, str] = {}
            for action in completed_actions:
                take = re.match(
                    r"^take\s+(.+?)\s+from\s+(.+)$",
                    action,
                    flags=re.IGNORECASE,
                )
                if take and _alfworld_object_class(take.group(1)) == target_class:
                    placed.pop(take.group(1).strip().casefold(), None)
                    continue
                match = re.match(
                    r"^move\s+(.+?)\s+to\s+(.+)$", action, flags=re.IGNORECASE
                )
                if not match:
                    continue
                if _alfworld_object_class(match.group(1)) != target_class:
                    continue
                object_id = match.group(1).strip()
                if _alfworld_object_class(match.group(2)) == destination_class:
                    placed[object_id.casefold()] = object_id
                else:
                    placed.pop(object_id.casefold(), None)
            placement_detail = (
                "; " + ", ".join(placed.values()) if placed else ""
            )
            lines.append(
                "Visible placement progress: "
                f"{len(placed)}/{facts['count']} distinct target instance(s)"
                f"{placement_detail}."
            )
            if required_transform in {"clean", "cool", "heat"}:
                transformed_and_placed = [
                    object_id
                    for key, object_id in placed.items()
                    if key in transformed_target_ids
                ]
                lines.append(
                    "Visible transformed-and-placed progress: "
                    f"{len(transformed_and_placed)}/{facts['count']} distinct "
                    "target instance(s)"
                    + (
                        "; " + ", ".join(transformed_and_placed)
                        if transformed_and_placed
                        else ""
                    )
                    + "."
                )

        # Thin adaptation of SkillFlow visible memory. In addition to arrival
        # observations, retain later public open/examine evidence so a known
        # open or empty receptacle cannot silently regress to stale "closed"
        # arrival text. No prescriptive next-action rule is copied.
        scene_memory = _alfworld_public_scene_memory(
            receipts,
            target_class=target_class,
        )
        visited = {
            str(entry["location"]).casefold(): entry for entry in scene_memory
        }
        if scene_memory:
            lines.append(
                "Visited receptacles from public Action--Observation history "
                "(persistent scene memory):"
            )
            for entry in scene_memory:
                visit_turns = entry.get("visit_turns", [])
                evidence_turns = entry.get("evidence_turns", [])
                contents = entry.get("contents")
                contents_text = (
                    "unknown"
                    if contents is None
                    else json.dumps(contents, ensure_ascii=False)
                )
                lines.append(
                    f"- {entry['location']}: turns={visit_turns[-6:]}; "
                    f"evidence_turns={evidence_turns[-6:]}; "
                    f"open_state={entry['open_state']}; contents={contents_text}; "
                    f"target_evidence={entry['target_evidence']}; "
                    "target_negative_evidence_turns="
                    f"{entry['target_negative_evidence_turns'][-6:]}; "
                    f"last_observation={entry['last_observation']}"
                )
            current_locations = []
            for action in admissible_actions:
                match = re.match(
                    r"^\s*go\s+to\s+(.+?)\s*$", action, re.IGNORECASE
                )
                if match:
                    current_locations.append(match.group(1).strip())
            unvisited = [
                location
                for location in current_locations
                if location.casefold() not in visited
            ]
            if unvisited:
                lines.append(
                    "Currently admissible unvisited receptacles: "
                    + ", ".join(unvisited[:25])
                    + "."
                )
        stall_diagnostic = _alfworld_public_stall_diagnostic(
            request,
            observation=observation,
            admissible_actions=admissible_actions,
            receipts=receipts,
            environment_terminal=environment_terminal,
        )
        lines.append(
            "[PUBLIC STALL DIAGNOSTIC] "
            + json.dumps(
                stall_diagnostic,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
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
            "Repeated public action pattern observed: the last four executed "
            "actions form A-B-A-B. Interpret it together with the current "
            "observation and admissible actions."
        )
    if receipts and receipts[-1].get("observation_status") == "parse_error":
        lines.append(
            "Format repair: the preceding response was invalid and the environment "
            "state is unchanged; copy exactly one current admissible action."
        )
    failed_transitions = [
        item for item in receipts if item.get("state_advanced") is False
    ]
    if failed_transitions:
        lines.append(
            "Failed public transitions with environment state preserved: "
            + " | ".join(
                "turn={turn}, status={status}, revision={revision}".format(
                    turn=item.get("turn"),
                    status=item.get("observation_status"),
                    revision=item.get("environment_revision_after"),
                )
                for item in failed_transitions[-6:]
            )
            + "."
        )
    return "\n".join(lines)


def _public_action_observation_history(
    receipts: Sequence[Mapping[str, object]],
    *,
    max_result_chars: int = 600,
) -> list[dict[str, object]]:
    """Project every public transition without evaluator or hidden state."""

    result: list[dict[str, object]] = []
    for item in receipts:
        raw_result = str(item.get("next_observation", ""))
        clipped = max_result_chars > 0 and len(raw_result) > max_result_chars
        observation_result = (
            raw_result[:max_result_chars] + "..." if clipped else raw_result
        )
        result.append(
            {
                "turn": item.get("turn"),
                "environment_revision_before": item.get(
                    "environment_revision_before"
                ),
                "environment_revision_after": item.get(
                    "environment_revision_after"
                ),
                "raw_action": item.get("raw_model_output"),
                "action": item.get("action"),
                "observation_result": observation_result,
                "observation_result_clipped": clipped,
                "observation_status": item.get("observation_status"),
                "state_advanced": item.get("state_advanced"),
                "environment_terminal": item.get("terminal"),
            }
        )
    return result


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
    total_action_budget: Optional[int] = None,
) -> str:
    """Render the same SkillFlow ReAct prompt used by the Direct condition.

    ``task_evaluator._environment_prompt`` is the existing thin copy of
    SkillFlow ``training/react_prompts.py``.  Reusing it here keeps Direct and
    AgentGraph on the same action syntax, observation, and bounded history
    protocol.  The Director-authored Agent contract remains on ``request``;
    this function supplies no fixed role or topology.
    """

    visible_observation, _ = _prompt_observation(
        observation, max_observation_chars
    )
    remaining_action_budget = (
        None
        if total_action_budget is None
        else max(total_action_budget - len(receipts), 0)
    )
    if task_family.lower() != "alfworld":
        actions = "\n".join(admissible_actions)
        public_state = _public_state_feedback(
            request,
            task_family=task_family,
            observation=observation,
            admissible_actions=admissible_actions,
            receipts=receipts,
            total_action_budget=total_action_budget,
            remaining_action_budget=remaining_action_budget,
        )
        return (
            f"Task:\n{request.problem}\n\n"
            f"Previous environment turns:\n{_history_text(receipts)}\n\n"
            f"{public_state}\n\n"
            f"Current observation (turn {turn}):\n{visible_observation}\n\n"
            f"Admissible actions:\n{actions}\n\n"
            "Return exactly one native WebShop action: search[keywords] or "
            "click[value]. You may enclose that native action in one <action> "
            "tag. Do not return JSON, an object, a code fence, or an explanation."
        )

    # Local import avoids making the generic environment resource import the
    # evaluator (and its optional dependencies) until ALFWorld is enabled.
    from .task_evaluator import _environment_prompt

    instruction = str(request.problem).split("\n\n", 1)[0].strip()
    trace: list[dict[str, object]] = []
    for index, receipt in enumerate(receipts):
        parse_error = receipt.get("observation_status") == "parse_error"
        trace.append(
            {
                "step": index,
                "observation": str(receipt.get("observation", "")),
                "action": (
                    "<INVALID>"
                    if parse_error
                    else str(receipt.get("action", ""))
                ),
                "next_observation": str(receipt.get("next_observation", "")),
                "feedback": (
                    "[INVALID] No valid <action> tag found."
                    if parse_error
                    else str(receipt.get("next_observation", ""))
                ),
            }
        )
    public_state = _public_state_feedback(
        request,
        task_family=task_family,
        observation=observation,
        admissible_actions=admissible_actions,
        receipts=receipts,
        total_action_budget=total_action_budget,
        remaining_action_budget=remaining_action_budget,
    )
    prompt = _environment_prompt(
        dataset=task_family.lower(),
        task_description=instruction,
        observation=visible_observation,
        legal_actions=admissible_actions,
        trace=trace,
        step_index=turn - 1,
        public_state=public_state,
    )
    return (
        prompt
        + "\n\nConstrained response envelope: return one JSON object with "
        "exactly the field `action`; its value must be one exact string from "
        "the current admissible-actions list. The runtime unwraps that field "
        "to the unchanged native ALFWorld action before environment execution."
    )


async def _resolve(value: Union[Any, Awaitable[Any]]) -> Any:
    return await value if inspect.isawaitable(value) else value


class EnvironmentExecutionAdapter:
    """Run a bounded episode or one Director-visible environment step."""

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
        # SkillFlow carries this bound as RolloutDecoding.max_action_tokens.
        # Native environment turns return one short action, so retaining the
        # generic Executor completion budget can make input+output exceed the
        # local Qwen context window after several long WebShop observations.
        self._max_action_tokens = max_action_tokens
        self._max_observation_chars = max_observation_chars
        self._stepwise_director = stepwise_director
        # ``asyncio.wait_for`` may insert a Task boundary between this adapter
        # and AgentRuntime.  Task cancellation intentionally normalizes the
        # raised ``CancelledError``, so retain the completed public prefix in
        # a request-keyed handoff until Runtime has recorded it.
        self._cancelled_prefixes: dict[str, Mapping[str, object]] = {}

    @property
    def stepwise_director(self) -> bool:
        return self._stepwise_director

    def reset_execution_state(self) -> None:
        """Drop one retained episode at the task-runtime reset boundary."""

        self._environment_backend.reset()
        self._cancelled_prefixes.clear()

    def take_cancelled_failure_metadata(
        self,
        request_id: str,
    ) -> Mapping[str, object]:
        """Consume one cancellation prefix before the Task boundary erases it."""

        return self._cancelled_prefixes.pop(request_id, MappingProxyType({}))

    def close(self) -> None:
        """Close the rollout-scoped environment backend."""

        self._environment_backend.close()

    def _current_public_state(
        self,
        episode: _EnvironmentEpisode,
        request: AgentRequest,
    ) -> dict[str, object]:
        """Project only the public next-step state returned to the Director."""

        admissible_actions: tuple[str, ...] = ()
        if not episode.terminal:
            admissible_actions, _ = _admissible_actions(
                episode.session.task_family,
                episode.session.available_actions,
            )
        last = episode.receipts[-1] if episode.receipts else None
        turns_used = len(episode.receipts)
        remaining_action_budget = max(self._max_turns - turns_used, 0)
        action_observation_history = _public_action_observation_history(
            episode.receipts
        )
        public_state = _public_state_feedback(
            request,
            task_family=episode.session.task_family,
            observation=episode.observation,
            admissible_actions=admissible_actions,
            receipts=episode.receipts,
            total_action_budget=self._max_turns,
            remaining_action_budget=remaining_action_budget,
            environment_terminal=episode.terminal,
        )
        alfworld_state: dict[str, object] = {}
        if episode.session.task_family.lower() == "alfworld":
            facts = _alfworld_task_facts(request.problem)
            alfworld_state = {
                "public_scene_memory": _alfworld_public_scene_memory(
                    episode.receipts,
                    target_class=facts.get("target_class"),
                ),
                "stall_diagnostic": _alfworld_public_stall_diagnostic(
                    request,
                    observation=episode.observation,
                    admissible_actions=admissible_actions,
                    receipts=episode.receipts,
                    environment_terminal=episode.terminal,
                ),
            }
        return {
            "environment_episode_id": episode.episode_id,
            "environment_id": episode.session.environment_id,
            "task_family": episode.session.task_family,
            "environment_revision": episode.revision,
            "last_action": None if last is None else last.get("action"),
            "state_advanced": (
                None if last is None else last.get("state_advanced")
            ),
            "observation_status": (
                "reset" if last is None else last.get("observation_status")
            ),
            "current_observation": episode.observation,
            "admissible_actions": list(admissible_actions),
            "public_state": public_state,
            "latest_action_observation": (
                action_observation_history[-1]
                if action_observation_history
                else None
            ),
            "action_observation_history": action_observation_history,
            "turns_used": turns_used,
            "remaining_action_budget": remaining_action_budget,
            "total_action_budget": self._max_turns,
            "environment_terminal": episode.terminal,
            "environment_truncated": (
                not episode.terminal and turns_used >= self._max_turns
            ),
            **alfworld_state,
        }

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        if request.agent.allowed_tools != (self._tool_id,):
            raise EnvironmentExecutionError(
                "environment Agent must allow exactly its request-scoped environment tool"
            )
        episode, token = await self._environment_backend.begin(request)
        session = episode.session
        observation = episode.observation

        revision = episode.revision
        terminal = episode.terminal
        receipts = episode.receipts
        evaluator_trace = episode.evaluator_trace
        tool_receipts = episode.tool_receipts
        model_calls = episode.model_calls
        reset_receipt = dict(episode.reset_receipt)

        try:
            # SkillFlow's episode limit applies to the complete rollout-owned
            # session.  A Canvas retry continues from the first unconsumed
            # turn rather than resetting either the simulator or the budget.
            for turn in range(len(receipts) + 1, self._max_turns + 1):
                if terminal:
                    break
                admissible_actions, has_search_bar = _admissible_actions(
                    session.task_family, session.available_actions
                )
                if not admissible_actions:
                    raise EnvironmentExecutionError(
                        "environment exposed no admissible actions before terminal"
                    )
                prompt = _action_prompt(
                    request,
                    task_family=session.task_family,
                    observation=observation,
                    admissible_actions=admissible_actions,
                    receipts=receipts,
                    turn=turn,
                    max_observation_chars=self._max_observation_chars,
                    total_action_budget=self._max_turns,
                )
                remaining_action_budget = max(
                    self._max_turns - len(receipts), 0
                )
                public_state = _public_state_feedback(
                    request,
                    task_family=session.task_family,
                    observation=observation,
                    admissible_actions=admissible_actions,
                    receipts=receipts,
                    total_action_budget=self._max_turns,
                    remaining_action_budget=remaining_action_budget,
                )
                _, observation_clipped = _prompt_observation(
                    observation, self._max_observation_chars
                )
                model_metadata = {
                    **dict(request.model.metadata),
                    "max_tokens": str(self._max_action_tokens),
                    "environment_total_action_budget": str(self._max_turns),
                    "environment_remaining_action_budget": str(
                        remaining_action_budget
                    ),
                }
                if session.task_family.lower() == "alfworld":
                    model_metadata.update(
                        {
                            "response_json_schema": json.dumps(
                                _alfworld_action_response_schema(
                                    admissible_actions
                                ),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "response_json_schema_version": (
                                "alfworld.native-action-enum.v1"
                            ),
                        }
                    )
                model_request = replace(
                    request,
                    request_id=f"{request.request_id}:environment:{turn}",
                    problem=prompt,
                    # RAGEN still receives one native action. The model-only
                    # response envelope is unwrapped by ``_parse_action``.
                    agent=replace(
                        request.agent,
                        execution_mode="reasoning",
                    ),
                    model=replace(
                        request.model,
                        metadata=model_metadata,
                    ),
                )
                generated = await self._gateway.generate(model_request)
                response = (
                    generated
                    if isinstance(generated, AgentResponse)
                    else AgentResponse(generated)
                )
                raw_action = response.text
                model_calls.append(
                    {
                        "turn": turn,
                        "request_id": model_request.request_id,
                        "metadata": dict(response.metadata),
                        "public_state": public_state,
                        "remaining_action_budget": remaining_action_budget,
                        "total_action_budget": self._max_turns,
                        **(
                            {
                                "stall_diagnostic": (
                                    _alfworld_public_stall_diagnostic(
                                        request,
                                        observation=observation,
                                        admissible_actions=admissible_actions,
                                        receipts=receipts,
                                    )
                                )
                            }
                            if session.task_family.lower() == "alfworld"
                            else {}
                        ),
                        "observation_clipped": observation_clipped,
                    }
                )
                action = _parse_action(
                    raw_action,
                    task_family=session.task_family,
                    admissible_actions=admissible_actions,
                    webshop_has_search_bar=has_search_bar,
                )
                if action is None:
                    receipts.append(
                        {
                            "receipt_type": "environment_transition",
                            "environment_episode_id": episode.episode_id,
                            "episode_id": episode.episode_id,
                            "environment_id": session.environment_id,
                            "turn": turn,
                            "environment_revision_before": revision,
                            "environment_revision_after": revision,
                            "observation": observation,
                            "admissible_actions": list(admissible_actions),
                            "raw_model_output": raw_action,
                            "action": None,
                            "next_observation": observation,
                            "terminal": False,
                            "state_advanced": False,
                            "observation_status": "parse_error",
                            "public_state": public_state,
                        }
                    )
                    evaluator_trace.append(
                        {
                            "environment_episode_id": episode.episode_id,
                            "episode_id": episode.episode_id,
                            "step": turn - 1,
                            "observation": observation,
                            "legal_actions": list(admissible_actions),
                            "action": "<INVALID>",
                            "raw_graph_output": raw_action,
                            "next_observation": observation,
                            "feedback": "[INVALID] No valid <action> tag found.",
                            "reward": 0.0,
                            "done": False,
                            "state_advanced": False,
                            "parse_error": True,
                            "info": {"parse_error": True},
                            "public_state": public_state,
                        }
                    )
                    if self._stepwise_director:
                        break
                    continue

                previous_revision = revision
                tool_request = (
                    ToolRequest("act", {"command": action})
                    if session.task_family.lower() == "alfworld"
                    else ToolRequest(action, {})
                )
                result, tool_receipt = await self._tool_registry.ainvoke_with_receipt(
                    self._tool_id, tool_request
                )
                tool_receipts.append(tool_receipt.to_value())
                if result is None:
                    raise EnvironmentExecutionError(
                        "registered environment tool failed with "
                        f"{tool_receipt.error_type or 'unknown_error'}"
                    )
                transition = self._environment_backend.take_transition()
                value = result.value
                if (
                    not isinstance(value, dict)
                    or value.get("observation") != transition.observation
                    or value.get("terminal") is not transition.terminal
                    or value.get("environment_revision") != episode.revision
                ):
                    raise EnvironmentExecutionError(
                        "registered environment tool returned an incompatible result"
                    )
                revision = episode.revision
                next_observation = transition.observation
                done = transition.terminal
                receipts.append(
                    {
                        "receipt_type": "environment_transition",
                        "environment_episode_id": episode.episode_id,
                        "episode_id": episode.episode_id,
                        "environment_id": session.environment_id,
                        "turn": turn,
                        "environment_revision_before": previous_revision,
                        "environment_revision_after": revision,
                        "observation": observation,
                        "admissible_actions": list(admissible_actions),
                        "raw_model_output": raw_action,
                        "action": action,
                        "next_observation": next_observation,
                        "terminal": done,
                        "state_advanced": True,
                        "observation_status": "success",
                        "public_state": public_state,
                    }
                )
                evaluator_trace.append(
                    {
                        "environment_episode_id": episode.episode_id,
                        "episode_id": episode.episode_id,
                        "step": turn - 1,
                        "observation": observation,
                        "legal_actions": list(admissible_actions),
                        "action": action,
                        "raw_graph_output": raw_action,
                        "next_observation": next_observation,
                        "reward": transition.reward,
                        "done": done,
                        "info": dict(transition.info),
                        "state_advanced": True,
                        "public_state": public_state,
                    }
                )
                observation = next_observation
                terminal = done
                if terminal or self._stepwise_director:
                    break

            return AgentResponse(
                observation,
                {
                    "execution_mode": "react",
                    "environment_execution_boundary": (
                        "one_action_one_observation"
                        if self._stepwise_director
                        else "bounded_episode"
                    ),
                    "model_calls": list(model_calls),
                    "environment_episode_id": episode.episode_id,
                    "episode_id": episode.episode_id,
                    "environment_id": session.environment_id,
                    "task_family": session.task_family,
                    "environment_revision": revision,
                    "environment_reset_receipt": reset_receipt,
                    "environment_receipts": list(receipts),
                    "environment_current_state": self._current_public_state(
                        episode,
                        request,
                    ),
                    "environment_terminal": terminal,
                    "environment_truncated": (
                        not terminal and len(receipts) >= self._max_turns
                    ),
                    "environment_max_turns": self._max_turns,
                    "environment_turns_used": len(receipts),
                    "environment_steps": revision,
                    "tool_receipts": list(tool_receipts),
                    # Evaluator-only replay data. AgentRuntime never renders this
                    # metadata into downstream prompts or Director feedback.
                    "evaluator_environment_trace": list(evaluator_trace),
                },
            )
        except asyncio.CancelledError as exc:
            # ``asyncio`` must still observe a real cancellation so the
            # Runtime's fail-fast scheduler cannot mistake this invocation for
            # a completed Agent.  Publish SkillFlow's already-completed
            # Action--Observation prefix on the in-flight exception; the
            # enclosing AgentRuntime consumes it before the Task boundary
            # normalizes ``CancelledError``.
            exc.environment_reset_receipt = dict(reset_receipt)
            exc.environment_receipts = tuple(dict(item) for item in receipts)
            exc.evaluator_environment_trace = tuple(
                dict(item) for item in evaluator_trace
            )
            exc.tool_receipts = tuple(dict(item) for item in tool_receipts)
            exc.model_calls = tuple(dict(item) for item in model_calls)
            exc.environment_revision = revision
            exc.environment_terminal = terminal
            exc.cause_error_type = type(exc).__name__
            self._cancelled_prefixes[request.request_id] = MappingProxyType(
                {
                    "environment_reset_receipt": dict(reset_receipt),
                    "environment_receipts": tuple(
                        dict(item) for item in receipts
                    ),
                    "evaluator_environment_trace": tuple(
                        dict(item) for item in evaluator_trace
                    ),
                    "tool_receipts": tuple(dict(item) for item in tool_receipts),
                    "model_calls": tuple(dict(item) for item in model_calls),
                    "environment_revision": revision,
                    "environment_terminal": terminal,
                    "cause_error_type": type(exc).__name__,
                }
            )
            raise
        except Exception as exc:
            cause_error_type = (
                exc.cause_error_type
                if isinstance(exc, EnvironmentExecutionError)
                and exc.cause_error_type is not None
                else type(exc).__name__
            )
            raise EnvironmentExecutionError(
                " ".join(str(exc).split()) or "environment execution failed",
                environment_reset_receipt=reset_receipt,
                environment_receipts=receipts,
                evaluator_environment_trace=evaluator_trace,
                tool_receipts=tool_receipts,
                model_calls=model_calls,
                environment_revision=revision,
                environment_terminal=terminal,
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

    def close(self) -> None:
        """Release the rollout-owned environment session, if it was opened."""

        self.execution_adapter.close()


def build_environment_execution_resources(
    *,
    gateway: AgentGateway,
    session_factory: EnvironmentSessionFactory,
    task_family: str,
    max_turns: int,
    max_action_tokens: int = 512,
    max_observation_chars: int = 0,
    stepwise_director: bool = False,
    tool_version: str = "skillflow.ragen_adapter.v2",
    timeout_seconds: Optional[float] = None,
) -> EnvironmentExecutionResources:
    """Create a real environment capability and its request-scoped adapter.

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
    if type(stepwise_director) is not bool:
        raise TypeError("stepwise_director must be bool")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when supplied")
    family = task_family.strip().lower()
    # The stepwise AgentGraph capability is explicitly task-scoped.  Legacy
    # ALFWorld conditions retain the historical ``alfworld`` resource ID;
    # stepwise conditions expose ``alfworld.environment`` while continuing to
    # dispatch SkillFlow's exact ``act(command)`` action internally.
    tool_id = (
        f"{family}.environment"
        if stepwise_director or family != "alfworld"
        else "alfworld"
    )
    backend = EnvironmentToolBackend(
        session_factory=session_factory,
        task_family=family,
        tool_id=tool_id,
    )
    alfworld_arguments_schema = {
        "type": "object",
        "required": ["command"],
        "additionalProperties": False,
        "properties": {
            "command": {
                "type": "string",
                "minLength": 1,
                "description": "one current ALFWorld admissible command",
            }
        },
    }
    action_schemas = (
        {"act": alfworld_arguments_schema} if family == "alfworld" else {}
    )
    input_schema = (
        {
            "type": "object",
            "required": ["action", "arguments"],
            "additionalProperties": False,
            "properties": {
                "action": {"const": "act"},
                "arguments": alfworld_arguments_schema,
            },
        }
        if family == "alfworld"
        else {
            "action": {
                "type": "string",
                "description": "one action from the current admissible action list",
            },
            "arguments": {"type": "object", "maxProperties": 0},
        }
    )
    capability = ToolCapability(
        tool_id=tool_id,
        dataset_scope=(family,),
        action_schemas=action_schemas,
        input_schema=input_schema,
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
