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
    if task_family.lower() != "alfworld":
        actions = "\n".join(admissible_actions)
        public_state = _public_state_feedback(
            request,
            task_family=task_family,
            observation=observation,
            admissible_actions=admissible_actions,
            receipts=receipts,
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
    return _environment_prompt(
        dataset=task_family.lower(),
        task_description=instruction,
        observation=visible_observation,
        legal_actions=admissible_actions,
        trace=trace,
        step_index=turn - 1,
    )


async def _resolve(value: Union[Any, Awaitable[Any]]) -> Any:
    return await value if inspect.isawaitable(value) else value


class EnvironmentExecutionAdapter:
    """Run one bounded model-driven ALFWorld or WebShop environment episode."""

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        environment_backend: EnvironmentToolBackend,
        max_turns: int,
        max_action_tokens: int = 512,
        max_observation_chars: int = 0,
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
        # ``asyncio.wait_for`` may insert a Task boundary between this adapter
        # and AgentRuntime.  Task cancellation intentionally normalizes the
        # raised ``CancelledError``, so retain the completed public prefix in
        # a request-keyed handoff until Runtime has recorded it.
        self._cancelled_prefixes: dict[str, Mapping[str, object]] = {}

    def take_cancelled_failure_metadata(
        self,
        request_id: str,
    ) -> Mapping[str, object]:
        """Consume one cancellation prefix before the Task boundary erases it."""

        return self._cancelled_prefixes.pop(request_id, MappingProxyType({}))

    def close(self) -> None:
        """Close the rollout-scoped environment backend."""

        self._environment_backend.close()

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
                )
                public_state = _public_state_feedback(
                    request,
                    task_family=session.task_family,
                    observation=observation,
                    admissible_actions=admissible_actions,
                    receipts=receipts,
                )
                _, observation_clipped = _prompt_observation(
                    observation, self._max_observation_chars
                )
                model_request = replace(
                    request,
                    request_id=f"{request.request_id}:environment:{turn}",
                    problem=prompt,
                    # RAGEN exposes native admissible actions rather than the
                    # generic StructuredAction protocol.  Preserve the
                    # Director-authored free-text contract; the prompt above
                    # appends only SkillFlow's state-dependent action grammar.
                    agent=replace(
                        request.agent,
                        execution_mode="reasoning",
                    ),
                    model=replace(
                        request.model,
                        metadata={
                            **dict(request.model.metadata),
                            "max_tokens": str(self._max_action_tokens),
                        },
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
                if terminal:
                    break

            return AgentResponse(
                observation,
                {
                    "execution_mode": "react",
                    "model_calls": list(model_calls),
                    "episode_id": episode.episode_id,
                    "environment_id": session.environment_id,
                    "task_family": session.task_family,
                    "environment_revision": revision,
                    "environment_reset_receipt": reset_receipt,
                    "environment_receipts": list(receipts),
                    "environment_terminal": terminal,
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
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when supplied")
    family = task_family.strip().lower()
    # DIRECT_REUSE: SkillFlow's public ALFWorld item registers resource_id
    # ``alfworld`` and exposes exactly one StructuredAction ``act(command)``.
    # WebShop retains the repository's existing dynamic-action compatibility
    # surface because it is outside this ALFWorld-only adaptation.
    tool_id = "alfworld" if family == "alfworld" else f"{family}.environment"
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
