"""Bounded ReAct execution for RAGEN-compatible benchmark environments.

The control flow is a thin adaptation of SkillFlow's ``BoundedAgent`` and
``RolloutEnvironmentSession`` contracts: one model turn selects one public
environment action, the action is executed, and only the resulting public
observation is returned to the next model turn.  The concrete ALFWorld and
WebShop bridge remains the deployed ``RAGENAdapter``; evaluator labels and
terminal success predicates are deliberately outside this module.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Any, Optional, Protocol, Union

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
    observation: str
    reward: object
    terminal: bool
    info: Mapping[str, object]


@dataclass(slots=True)
class _EnvironmentEpisode:
    session: EnvironmentSession
    observation: str
    revision: int = 0
    pending_transition: Optional[_EnvironmentTransition] = None


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
        observation = await _resolve(session.reset())
        if not isinstance(observation, str):
            raise EnvironmentExecutionError("environment reset must return text")
        if observation.startswith("[ENV_UNAVAILABLE]"):
            raise EnvironmentExecutionError(observation)
        episode = _EnvironmentEpisode(session=session, observation=observation)
        return episode, self._episode.set(episode)

    def end(self, token: Token[Optional[_EnvironmentEpisode]]) -> None:
        self._episode.reset(token)

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
        lines.append(
            "[Turn {turn}: observation={observation!r}, action={action!r}, "
            "next_observation={next_observation!r}]".format(**receipt)
        )
    return "\n".join(lines)


def _action_prompt(
    request: AgentRequest,
    *,
    task_family: str,
    observation: str,
    admissible_actions: Sequence[str],
    receipts: Sequence[Mapping[str, object]],
    turn: int,
) -> str:
    """Render SkillFlow's state/action/history boundary without a fixed role."""

    actions = "\n".join(admissible_actions)
    format_instruction = (
        "Return exactly one native WebShop action: search[keywords] or click[value]."
        if task_family.lower() == "webshop"
        else "Return exactly one native action copied from the admissible action list."
    )
    return (
        f"Task:\n{request.problem}\n\n"
        f"Previous environment turns:\n{_history_text(receipts)}\n\n"
        f"Current observation (turn {turn}):\n{observation}\n\n"
        f"Admissible actions:\n{actions}\n\n"
        f"{format_instruction} You may enclose that native action in one <action> "
        "tag. Do not return JSON, an object, a code fence, or an explanation."
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

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        if request.agent.allowed_tools != (self._tool_id,):
            raise EnvironmentExecutionError(
                "environment Agent must allow exactly its request-scoped environment tool"
            )
        episode, token = await self._environment_backend.begin(request)
        session = episode.session
        observation = episode.observation

        revision = 0
        terminal = False
        receipts: list[dict[str, object]] = []
        evaluator_trace: list[dict[str, object]] = []
        tool_receipts: list[dict[str, object]] = []
        model_calls: list[dict[str, object]] = []
        reset_actions, _ = _admissible_actions(
            session.task_family, session.available_actions
        )
        reset_receipt: dict[str, object] = {
            "receipt_type": "environment_reset",
            "environment_id": session.environment_id,
            "environment_revision": revision,
            "observation": observation,
            "admissible_actions": list(reset_actions),
            "terminal": False,
        }

        try:
            for turn in range(1, self._max_turns + 1):
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
                )
                model_request = replace(
                    request,
                    request_id=f"{request.request_id}:environment:{turn}",
                    problem=prompt,
                    # RAGEN exposes native admissible actions rather than the
                    # generic StructuredAction protocol.  Like FlowSteer's Format
                    # Operator boundary, the Canvas-authored contract remains in
                    # the trajectory but cannot override the executor's native
                    # action grammar for this provider turn.
                    agent=replace(
                        request.agent,
                        execution_mode="reasoning",
                        contract=(
                            "Select exactly one native action permitted by the "
                            "current admissible-action list."
                        ),
                        artifact_type="environment_action",
                        completion_condition=(
                            "The response parses as one currently admissible native "
                            "environment action."
                        ),
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
                        }
                    )
                    evaluator_trace.append(
                        {
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
                        }
                    )
                    continue

                previous_revision = revision
                result, tool_receipt = await self._tool_registry.ainvoke_with_receipt(
                    self._tool_id, ToolRequest(action, {})
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
                    }
                )
                evaluator_trace.append(
                    {
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
                    "model_calls": model_calls,
                    "environment_id": session.environment_id,
                    "task_family": session.task_family,
                    "environment_revision": revision,
                    "environment_reset_receipt": reset_receipt,
                    "environment_receipts": receipts,
                    "environment_terminal": terminal,
                    "environment_turns_used": len(receipts),
                    "environment_steps": revision,
                    "tool_receipts": tool_receipts,
                    # Evaluator-only replay data. AgentRuntime never renders this
                    # metadata into downstream prompts or Director feedback.
                    "evaluator_environment_trace": evaluator_trace,
                },
            )
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
