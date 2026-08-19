#!/usr/bin/env python3
"""Render receipt-backed multidataset Stable Zero architecture reports.

The report layer reuses the paired-result and trajectory schemas emitted by
the existing HotpotQA/completion runners.  It performs no model call,
environment transition, evaluator call, Skill publication, or training step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "multidataset_stablezero"
TOTAL_REPORT = ROOT / "MULTIDATASET_AGENT_ARCHITECTURE_STABLEZERO_REPORT.md"

TOOL_DIAGNOSTIC_RECEIPTS = {
    "hotpotqa": "artifacts/tool_exact_schema_canary/hotpotqa_exact_wire_v7_20260820.json",
    "triviaqa": "artifacts/tool_exact_schema_canary/triviaqa_exact_wire_v2_20260820.json",
    "aime_2026": "artifacts/tool_exact_schema_canary/aime_2026_exact_wire_v3_20260820.json",
    "healthbench_professional": "artifacts/tool_exact_schema_canary/healthbench_professional_exact_wire_v2_20260820.json",
}

MICRO_TRAINING_MANIFESTS = (
    "artifacts/joint_qa_micro/step_000001/training_manifest.json",
    "artifacts/joint_qa_micro/step_000002_attempt_04/training_manifest.json",
)
MICRO_TRAINING_CURVE = "reports/joint_qa_curve/final/joint_qa_curve.json"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    report_name: str
    paired_path: str | None
    trajectory_path: str | None
    manifest_path: str
    metrics: tuple[str, ...]
    capability: str
    protocol: str
    evidence_scope: str = "fixed validation canary"
    protocol_limitations: tuple[str, ...] = ()
    excluded_evidence: tuple[str, ...] = ()


SPECS = (
    DatasetSpec(
        "hotpotqa",
        "HotpotQA",
        "HOTPOTQA_ARCH_REPORT.md",
        "artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/protocol_separated_results.jsonl",
        "artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl",
        "artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/run_manifest.json",
        ("exact_match", "token_f1"),
        "闭卷上下文推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力",
        "Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。",
        excluded_evidence=(
            "v1 exact-schema 前的自然策略 canary 仅保留为历史结果；v2 因 Director schema/tool_id 边界含混导致 1/2 max_rounds，仅作为 v3 修复前失败诊断。",
        ),
    ),
    DatasetSpec(
        "triviaqa",
        "TriviaQA",
        "TRIVIAQA_ARCH_REPORT.md",
        "artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/protocol_separated_results.jsonl",
        "artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl",
        "artifacts/qa_tool_react_exact_wire_v3_stable_zero/triviaqa/run_manifest.json",
        ("exact_match", "token_f1"),
        "仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力",
        "Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。",
        excluded_evidence=(
            "v1 exact-schema 前的自然策略 canary 只保留为历史结果，不进入当前 v3 指标。",
        ),
    ),
    DatasetSpec(
        "aime_2026",
        "AIME-2025 Development（AIME 2026 目标适配）",
        "AIME2026_ARCH_REPORT.md",
        "artifacts/aime2026_computation_tool_stable_zero/development/paired_results.jsonl",
        "artifacts/aime2026_computation_tool_stable_zero/development/agentgraph_trajectories.jsonl",
        "artifacts/aime2026_computation_tool_stable_zero/development/run_manifest.json",
        ("exact_match",),
        "推理 + 有界 calculator/Python execution 能力",
        "开发阶段使用 AIME 2025 官方题目与整数 exact match；Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。",
        "AIME 2025 development canary；不是 AIME 2026 benchmark 成绩",
        (
            "仅 2 题 Stable Zero canary，不是正式 benchmark 估计。",
            "可选计算 Tool 未被自然选择时，只能报告 capability 已接线，不能声称 Tool 已验证有效。",
        ),
        (
            "artifacts/aime2026_computation_tool_stable_zero/evaluation：使用 AIME 2026 official test 的旧结果，仅保留为历史诊断，不进入开发指标。",
        ),
    ),
    DatasetSpec(
        "healthbench_professional",
        "HealthBench Professional（reference-judge diagnostic）",
        "HEALTHBENCH_PROFESSIONAL_ARCH_REPORT.md",
        "artifacts/healthbench_professional_medrag_tool_stable_zero_v2/development/paired_results.jsonl",
        "artifacts/healthbench_professional_medrag_tool_stable_zero_v2/development/agentgraph_trajectories.jsonl",
        "artifacts/healthbench_professional_medrag_tool_stable_zero_v2/development/run_manifest.json",
        ("raw_score",),
        "临床推理 + 冻结教材语料 MedRAG search 能力",
        "Direct 与允许使用 MedRAG 的 AgentGraph 分别报告；raw_score 来自 openai/simple-evals-compatible reference judge，不等同于 HealthBench 私有官方评测服务。",
        "fixed internal validation diagnostic",
        (
            "reference-judge diagnostic 只验证公开 rubric/judge 接口，不能表述为私有官方 leaderboard 成绩。",
            "可选 MedRAG Tool 未被自然选择时，不能把 raw_score 差值归因于检索能力。",
        ),
        (
            "artifacts/healthbench_professional_medrag_tool_stable_zero/evaluation：旧 v1 条件，不进入当前 v2 exact-action-schema development 指标。",
        ),
    ),
    DatasetSpec(
        "webshop",
        "WebShop",
        "WEBSHOP_ARCH_REPORT.md",
        "artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/paired_results.jsonl",
        "artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/agentgraph_trajectories.jsonl",
        "artifacts/webshop_ragen_environment_native_action_v4_stable_zero/development/run_manifest.json",
        ("success",),
        "request-scoped SkillFlow/RAGEN environment ReAct",
        "Direct 与 AgentGraph 使用相同原生 WebShop validation 环境、task lock、action budget 和 evaluator。",
        "native validation indices 500..627；2-task canary when executed",
        (
            "只有完整原生 environment transition receipt 与 terminal success 才计入 success。",
        ),
        (
            "artifacts/webshop_ragen_environment_native_action_v2_stable_zero/evaluation：旧结果取自 native test 范围，属于 test-contaminated adaptation evidence，不进入 v4 development 指标；v3 development 因未传递 SkillFlow max_action_tokens 导致本地 Direct 上下文超限，仅保留为失败诊断。",
        ),
    ),
    DatasetSpec(
        "alfworld",
        "ALFWorld",
        "ALFWORLD_ARCH_REPORT.md",
        "artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/paired_results.jsonl",
        "artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/agentgraph_trajectories.jsonl",
        "artifacts/alfworld_ragen_required_actor_v2_stable_zero/evaluation/run_manifest.json",
        ("success",),
        "request-scoped SkillFlow/RAGEN environment ReAct",
        "Direct 与 AgentGraph 使用相同原生游戏、task lock、50-step budget 和 evaluator。",
    ),
    DatasetSpec(
        "swe_bench",
        "SWE-bench Regular Dev",
        "SWEBENCH_ARCH_REPORT.md",
        "artifacts/swebench_regular_dev_coding_agent_stable_zero/development/paired_results.jsonl",
        "artifacts/swebench_regular_dev_coding_agent_stable_zero/development/agentgraph_trajectories.jsonl",
        "artifacts/swebench_regular_dev_coding_agent_stable_zero/development/run_manifest.json",
        ("resolved",),
        "detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness",
        "架构适配只使用 SWE-bench regular dev；唯一接受的 terminal 指标是官方 Docker harness resolved/resolved_rate，禁止使用代理评分。",
        "SWE-bench regular-dev architecture development；Verified 完整保留给最终评测",
        (
            "没有 official Docker harness receipt 时，Direct、AgentGraph 与 Stable Zero 均为不可测。",
            "worktree preflight、generated diff、LLM judgement 或 local proxy test 都不能替代 resolved。",
        ),
        (
            "旧 swebench_verified_* development/evaluation：Verified 曾被用于适配，按数据隔离规则排除；完整 Verified 只允许用于最终评测。",
        ),
    ),
)


@dataclass(frozen=True)
class PreservedFailureSpec:
    condition: str
    paired_path: str
    trajectory_path: str
    task_id: str
    classification: str
    first_error: str


@dataclass(frozen=True)
class PreservedCorrectSpec:
    condition: str
    paired_path: str
    trajectory_path: str
    task_id: str


PRESERVED_FAILURES = {
    "hotpotqa": PreservedFailureSpec(
        condition="qa_tool_react_exact_wire_v2_stable_zero (pre-v3 schema clarification)",
        paired_path=(
            "artifacts/qa_tool_react_exact_wire_v2_stable_zero/"
            "hotpotqa/protocol_separated_results.jsonl"
        ),
        trajectory_path=(
            "artifacts/qa_tool_react_exact_wire_v2_stable_zero/"
            "hotpotqa/tool_agentgraph_trajectories.jsonl"
        ),
        task_id="hotpotqa:5a879ab05542996e4f30887e",
        classification="Director action-schema / Tool resource identifier / terminal control",
        first_error=(
            "前五个 Director action 把 top-level output_agent_id 放入 Agent object，"
            "随后又把 action_name `search`/`read` 当成 allowed_tools 的 resource_id；"
            "Canvas 均 fail closed。第 20 轮得到正确输出后已无剩余显式 FINISH turn，"
            "因此以 max_rounds 终止。v3 只澄清字段层级和 exact tool_id 边界。"
        ),
    ),
    "triviaqa": PreservedFailureSpec(
        condition="qa_tool_react_stable_zero (pre-v3 output-span clarification)",
        paired_path=(
            "artifacts/qa_tool_react_stable_zero/"
            "triviaqa/wrong_demos.jsonl"
        ),
        trajectory_path=(
            "artifacts/qa_tool_react_stable_zero/"
            "triviaqa/tool_agentgraph_trajectories.jsonl"
        ),
        task_id="triviaqa:tc_3",
        classification="Output Agent contract / answer-span serialization",
        first_error=(
            "Output Agent returned `York, North Yorkshire, England` even though the task "
            "asks for the English town and the Direct arm returned `York`. The official "
            "TriviaQA answer evaluator therefore recorded EM=0 and token F1=0.8571428571. "
            "The receipt does not support attributing this recovered semantic answer to "
            "retrieval or model reasoning; the first evaluator-visible error is the "
            "over-specified terminal answer span."
        ),
    ),
    "webshop": PreservedFailureSpec(
        condition="ragen_environment_v1 (pre-native-action adaptation)",
        paired_path=(
            "artifacts/webshop_ragen_environment_stable_zero/"
            "evaluation/wrong_demos.jsonl"
        ),
        trajectory_path=(
            "artifacts/webshop_ragen_environment_stable_zero/"
            "evaluation/agentgraph_trajectories.jsonl"
        ),
        task_id="webshop:00000",
        classification="ReAct action serialization / environment interface",
        first_error=(
            "第 1 个 environment step 输出 JSON action object，而原生 WebShop parser "
            "只接受 search[...] 或 click[...]；状态未推进并连续产生 parse_error。"
        ),
    ),
}


DIRECT_CONTRAST_FAILURES = {
    "aime_2026": {
        "task_id": "aime-2025:i:01",
        "classification": "Direct model reasoning",
        "first_error": (
            "Direct arm 的 terminal submission 为 11，AIME-2025 整数 Ground Truth 为 70；"
            "该 Direct receipt 不保存更早的内部 reasoning，因此不能进一步缩窄错误位置。"
        ),
    },
    "alfworld": {
        "task_id": "alfworld:train:00006",
        "classification": "ReAct action selection / state tracking / stopping",
        "first_error": (
            "Direct arm 在 zero-based step=3（第 4 个 environment action）首次产生 <INVALID>，随后重复 "
            "examine pillow 1，最终触发 50-step environment_step_limit；"
            "同题 v2 AgentGraph 用 6 个原生 action 成功。"
        ),
    },
}


PRESERVED_CORRECTS = {
    "healthbench_professional": PreservedCorrectSpec(
        condition="healthbench_professional_round_01/development",
        paired_path=(
            "artifacts/healthbench_professional_round_01/"
            "development/paired_results.jsonl"
        ),
        trajectory_path=(
            "artifacts/healthbench_professional_round_01/"
            "development/agentgraph_trajectories.jsonl"
        ),
        task_id="healthbench-professional:1d45010f49e42dcfb9d635ff1aa58828",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _tool_diagnostic_section(dataset_key: str) -> str:
    relative = TOOL_DIAGNOSTIC_RECEIPTS.get(dataset_key)
    if relative is None:
        return ""
    receipt = _load_json(ROOT / relative)
    if not receipt:
        return ""
    compliance = receipt.get("compliance", {})
    if not isinstance(compliance, Mapping):
        compliance = {}
    schema = compliance.get("schema_compliance", {})
    backend = compliance.get("backend_compliance", {})
    model = compliance.get("model_compliance", {})
    observed = compliance.get("observed_sequence", ())
    sequence = " → ".join(
        str(item.get("name"))
        for item in observed
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ) or "none"
    return f"""

## Exact-schema Tool forced probe（不计入 benchmark）

- Receipt：`{relative}`
- Controls：`diagnostic_only=true`、`forced_probe=true`、`grpo_eligible=false`、`skill_evidence_eligible=false`
- Overall status：`{receipt.get('status', 'missing')}`
- StructuredAction schema compliance：`{str(bool(isinstance(schema, Mapping) and schema.get('passed') is True)).lower()}`
- Tool backend compliance：`{str(bool(isinstance(backend, Mapping) and backend.get('passed') is True)).lower()}`；successful receipts=`{backend.get('successful_receipts', 0) if isinstance(backend, Mapping) else 0}`
- Model action/termination compliance：`{str(bool(isinstance(model, Mapping) and model.get('passed') is True)).lower()}`
- Observed action sequence：`{sequence}`

该 receipt 只回答 exact `StructuredAction`、真实 backend dispatch 和有界 ReAct termination 是否可执行；不含 evaluator、Ground Truth、benchmark metric、Skill evidence 或训练数据。forced probe 失败不覆盖同条件自然策略成绩，反之亦然。
"""


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _task_id(trajectory: Mapping[str, Any]) -> str:
    task = trajectory.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("task_id"), str):
        return str(task["task_id"])
    return str(trajectory.get("task_id", ""))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _short(value: object, limit: int = 420) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _metric_values(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        payload = row.get(arm)
        if not isinstance(payload, Mapping):
            continue
        value = payload.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _metric_mean(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
) -> float | None:
    values = _metric_values(rows, arm, metric)
    return _mean(values) if values else None


def _row_metric(row: Mapping[str, Any], arm: str, metric: str) -> float | None:
    payload = row.get(arm)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _metric_display(metric: str, value: float | None) -> str:
    if value is None:
        return "不可测"
    if metric == "raw_score":
        return f"{value:.4f}"
    return f"{100 * value:.2f}%"


def _model_family(model_id: str) -> str:
    value = model_id.casefold()
    for marker, family in (
        ("qwen", "Qwen"),
        ("deepseek", "DeepSeek"),
        ("gemini", "Gemini"),
        ("gpt", "GPT"),
        ("minimax", "MiniMax"),
        ("grok", "Grok"),
        ("glm", "GLM"),
    ):
        if marker in value:
            return family
    return "Other"


def _graph_nodes(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    graph = row.get("agentgraph", {}).get("final_graph")
    if not isinstance(graph, Mapping):
        return []
    nodes = graph.get("nodes", ())
    return [dict(node) for node in nodes if isinstance(node, Mapping)]


def _graph_relations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    graph = row.get("agentgraph", {}).get("final_graph")
    if not isinstance(graph, Mapping):
        return []
    relations = graph.get("relations", ())
    return [dict(item) for item in relations if isinstance(item, Mapping)]


def _latest_topology(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = trajectory.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return {}
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        summary = turn.get("runtime_summary")
        if isinstance(summary, Mapping) and isinstance(summary.get("topology"), Mapping):
            return summary["topology"]
    return {}


def _tool_receipts(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    turns = trajectory.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return receipts
    for turn in turns:
        executions = turn.get("executions", ()) if isinstance(turn, Mapping) else ()
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            continue
        for execution in executions:
            if not isinstance(execution, Mapping):
                continue
            metadata = execution.get("metadata", {})
            response = metadata.get("response", {}) if isinstance(metadata, Mapping) else {}
            raw = response.get("tool_receipts", ()) if isinstance(response, Mapping) else ()
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                receipts.extend(dict(item) for item in raw if isinstance(item, Mapping))
    return receipts


def _execution_responses(
    trajectory: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return the saved execution request/response pairs without inferring events."""

    responses: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    turns = trajectory.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return responses
    for turn in turns:
        executions = turn.get("executions", ()) if isinstance(turn, Mapping) else ()
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            continue
        for execution in executions:
            if not isinstance(execution, Mapping):
                continue
            metadata = execution.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            request = metadata.get("request", {})
            response = metadata.get("response", {})
            if isinstance(request, Mapping) and isinstance(response, Mapping):
                responses.append((request, response))
    return responses


def _execution_records(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return saved Executor records in trajectory order."""

    records: list[dict[str, Any]] = []
    turns = trajectory.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return records
    for turn in turns:
        executions = turn.get("executions", ()) if isinstance(turn, Mapping) else ()
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            continue
        records.extend(dict(item) for item in executions if isinstance(item, Mapping))
    return records


def _react_trace_entries(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return public ReAct actions/observations; hidden reasoning is never reconstructed."""

    entries: list[dict[str, Any]] = []
    for request, response in _execution_responses(trajectory):
        raw = response.get("react_trace", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        agent = request.get("agent", {})
        agent_id = agent.get("id", "unknown") if isinstance(agent, Mapping) else "unknown"
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            entry = dict(item)
            entry["agent_id"] = agent_id
            entries.append(entry)
    return entries


def _model_call_records(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize one saved execution into its actual accepted model-call receipts."""

    metadata = execution.get("metadata", {})
    response = metadata.get("response", {}) if isinstance(metadata, Mapping) else {}
    model_calls = response.get("model_calls", ()) if isinstance(response, Mapping) else ()
    if isinstance(model_calls, Sequence) and not isinstance(model_calls, (str, bytes)):
        records: list[dict[str, Any]] = []
        for call in model_calls:
            if not isinstance(call, Mapping):
                continue
            call_metadata = call.get("metadata", {})
            if isinstance(call_metadata, Mapping):
                records.append(dict(call_metadata))
        if records:
            return records
    response_metadata = response if isinstance(response, Mapping) else {}
    return [
        {
            "model_id": execution.get("model_id", response_metadata.get("model_id", "unknown")),
            "prompt_tokens": execution.get(
                "input_tokens", response_metadata.get("prompt_tokens", 0)
            ),
            "completion_tokens": execution.get(
                "output_tokens", response_metadata.get("completion_tokens", 0)
            ),
            "latency_ms": execution.get(
                "latency_ms", response_metadata.get("latency_ms", 0.0)
            ),
            "attempt_count": response_metadata.get("attempt_count", 1),
        }
    ]


def _aggregate_model_usage(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    usage: dict[str, dict[str, float]] = {}
    for execution in executions:
        for call in _model_call_records(execution):
            model_id = str(call.get("model_id", "unknown"))
            row = usage.setdefault(
                model_id,
                {"calls": 0.0, "attempts": 0.0, "tokens": 0.0, "latency_ms": 0.0},
            )
            row["calls"] += 1
            row["attempts"] += float(call.get("attempt_count", 1) or 1)
            row["tokens"] += float(call.get("prompt_tokens", 0) or 0)
            row["tokens"] += float(call.get("completion_tokens", 0) or 0)
            row["latency_ms"] += float(call.get("latency_ms", 0.0) or 0.0)
    return usage


def _director_usage(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policies: set[str] = set()
    prompts: set[str] = set()
    calls = 0
    attempts = 0
    tokens = 0
    latency_ms = 0.0
    for trajectory in trajectories:
        versions = trajectory.get("versions", {})
        if isinstance(versions, Mapping):
            if versions.get("policy"):
                policies.add(str(versions["policy"]))
            if versions.get("prompt"):
                prompts.add(str(versions["prompt"]))
        turns = trajectory.get("turns", ())
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            calls += 1
            attempts += int(turn.get("director_attempt_count", 1) or 1)
            prompt_ids = turn.get("prompt_token_ids", ())
            output_ids = turn.get("output_token_ids", ())
            if isinstance(prompt_ids, Sequence) and not isinstance(prompt_ids, (str, bytes)):
                tokens += len(prompt_ids)
            if isinstance(output_ids, Sequence) and not isinstance(output_ids, (str, bytes)):
                tokens += len(output_ids)
            latency_ms += float(turn.get("director_latency_ms", 0.0) or 0.0)
            if turn.get("policy_version"):
                policies.add(str(turn["policy_version"]))
    return {
        "calls": calls,
        "attempts": attempts,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "policies": sorted(policies),
        "prompts": sorted(prompts),
    }


def _communication_envelopes(
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return recorded inter-Agent envelopes without reconstructing messages."""

    envelopes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for request, _ in _execution_responses(trajectory):
        raw = request.get("upstream", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            envelope = dict(item)
            identity = json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            envelopes.append(envelope)
    return envelopes


def _environment_receipts(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for _, response in _execution_responses(trajectory):
        raw = response.get("environment_receipts", ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            receipts.extend(dict(item) for item in raw if isinstance(item, Mapping))
    return receipts


def _coding_receipts(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for request, response in _execution_responses(trajectory):
        agent = request.get("agent", {})
        if not isinstance(agent, Mapping) or agent.get("execution_mode") != "coding":
            continue
        raw = response.get("tool_receipts", ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            receipts.extend(dict(item) for item in raw if isinstance(item, Mapping))
    return receipts


def _protocol_sections(spec: DatasetSpec) -> str:
    limitations = "\n".join(
        f"- {item}" for item in spec.protocol_limitations
    ) or "- 无额外限制记录。"
    exclusions = "\n".join(
        f"- {item}" for item in spec.excluded_evidence
    ) or "- 无需隔离的旧结果。"
    return f"""## Evidence scope 与协议限制

- Evidence scope：{spec.evidence_scope}
- Protocol：{spec.protocol}

{limitations}

### 明确排除的历史结果

{exclusions}
"""


def _max_parallel_width(trajectory: Mapping[str, Any]) -> int:
    """Return the widest executor block recorded by the existing scheduler."""

    width = 0
    turns = trajectory.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return width
    for turn in turns:
        summary = turn.get("runtime_summary", {}) if isinstance(turn, Mapping) else {}
        blocks = summary.get("block_completion_order", ()) if isinstance(summary, Mapping) else ()
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            continue
        for block in blocks:
            if isinstance(block, Sequence) and not isinstance(block, (str, bytes)):
                width = max(width, len(block))
    return width


def _environment_trace(row: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    evaluation = row.get(arm, {}).get("evaluation", {})
    details = evaluation.get("details", {}) if isinstance(evaluation, Mapping) else {}
    trace = details.get("trace", ()) if isinstance(details, Mapping) else ()
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        return []
    return [dict(item) for item in trace if isinstance(item, Mapping)]


def _graph_line(row: Mapping[str, Any]) -> str:
    nodes = _graph_nodes(row)
    relations = _graph_relations(row)
    if not nodes:
        return "(no completed graph)"
    node_ids = [str(node.get("id", "?")) for node in nodes]
    if not relations:
        return node_ids[0] if len(node_ids) == 1 else " | ".join(node_ids)
    edges: list[str] = []
    for relation in relations:
        source = str(relation.get("source_id", "?"))
        target = str(relation.get("target_id", "?"))
        forward = relation.get("source_to_target") is True
        reverse = relation.get("target_to_source") is True
        if forward and reverse:
            edges.append(f"{source} ↔ {target}")
        elif forward:
            edges.append(f"{source} → {target}")
        elif reverse:
            edges.append(f"{target} → {source}")
    return "; ".join(edges) or " | ".join(node_ids)


def _demo(
    row: Mapping[str, Any],
    trajectory: Mapping[str, Any] | None,
    *,
    label: str,
    paired_receipt: str | None = None,
    trajectory_receipt: str | None = None,
) -> str:
    graph = row.get("agentgraph", {})
    evaluation = graph.get("evaluation", {})
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    final_graph = graph.get("final_graph", {})
    final_graph = final_graph if isinstance(final_graph, Mapping) else {}
    versions = trajectory.get("versions", {}) if isinstance(trajectory, Mapping) else {}
    versions = versions if isinstance(versions, Mapping) else {}
    lines = [
        f"### {label}: `{row.get('task_id')}`",
        "",
        f"- Task：{_short(row.get('question'), 700)}",
        f"- Ground Truth：{_short(row.get('ground_truth'), 350)}",
        f"- Final Answer：{_short(graph.get('final_answer'), 700)}",
        f"- Evaluator：`{evaluation.get('evaluator_version', 'legacy receipt 未记录')}`; "
        f"metrics=`{json.dumps(evaluation.get('metrics', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Trajectory ID：`{graph.get('trajectory_id') or (trajectory or {}).get('trajectory_id') or 'legacy receipt 未记录'}`",
        f"- Policy version：`{versions.get('policy', 'legacy receipt 未记录')}`; "
        f"prompt=`{versions.get('prompt', 'legacy receipt 未记录')}`; "
        f"tool=`{versions.get('tool', 'legacy receipt 未记录')}`",
        f"- Output Agent：`{final_graph.get('output_agent_id', 'legacy receipt 未记录')}`",
        f"- AgentGraph: `{_graph_line(row)}`",
        "",
        "Agent 配置：",
        "",
    ]
    if paired_receipt or trajectory_receipt:
        receipt_parts = []
        if paired_receipt:
            receipt_parts.append(f"paired=`{paired_receipt}`")
        if trajectory_receipt:
            receipt_parts.append(f"trajectory=`{trajectory_receipt}`")
        lines[1:1] = ["", "- Raw receipts：" + "; ".join(receipt_parts)]
    for node in _graph_nodes(row):
        mode = node.get("execution_mode", "legacy receipt 未记录")
        tools = (
            node.get("allowed_tools")
            if "allowed_tools" in node
            else "legacy receipt 未记录"
        )
        artifact = node.get("artifact_type", "legacy receipt 未记录")
        lines.append(
            "- `{id}` — model=`{model}`, execution_mode=`{mode}`, role_family=`{role}`, "
            "allowed_tools=`{tools}`, artifact_type=`{artifact}`; contract: {contract}".format(
                id=node.get("id"),
                model=node.get("model_id"),
                mode=mode,
                role=node.get("role_family"),
                tools=tools,
                artifact=artifact,
                contract=_short(node.get("contract"), 320),
            )
        )
    if trajectory:
        actions: list[str] = []
        for turn in trajectory.get("turns", ()):
            if not isinstance(turn, Mapping):
                continue
            action = turn.get("action", {})
            if isinstance(action, Mapping):
                actions.append(
                    str(action.get("action", action.get("action_type", "invalid")))
                )
        lines.extend(["", f"Director atomic edit 序列：`{' → '.join(actions)}`"])
        lines.extend(["", "Progressive Canvas turn receipts：", ""])
        for turn in trajectory.get("turns", ()):
            if not isinstance(turn, Mapping):
                continue
            action = turn.get("action", {})
            action_name = (
                action.get("action", action.get("action_type", "invalid"))
                if isinstance(action, Mapping)
                else "invalid"
            )
            runtime = turn.get("runtime_summary", {})
            runtime = runtime if isinstance(runtime, Mapping) else {}
            lines.append(
                "- round=`{round}`; action=`{action}`; graph_revision=`{revision}`; "
                "receipt_verified=`{verified}`; communication_condition=`{condition}`; "
                "blocks=`{blocks}`; executed=`{executed}`; reused=`{reused}`; "
                "feedback={feedback}".format(
                    round=turn.get("round_index", "legacy receipt 未记录"),
                    action=action_name,
                    revision=turn.get("graph_revision", "legacy receipt 未记录"),
                    verified=turn.get("receipt_verified", "legacy receipt 未记录"),
                    condition=runtime.get("communication_condition", "not_executed"),
                    blocks=runtime.get("block_completion_order", []),
                    executed=runtime.get("executed_agent_ids", []),
                    reused=runtime.get("reused_agent_ids", []),
                    feedback=_short(turn.get("canvas_feedback"), 700),
                )
            )
    envelopes = _communication_envelopes(trajectory or {})
    if envelopes:
        lines.extend(["", "实际 CommunicationEnvelope：", ""])
        for message in envelopes:
            receipts = message.get("tool_receipts", ())
            receipt_count = (
                len(receipts)
                if isinstance(receipts, Sequence)
                and not isinstance(receipts, (str, bytes))
                else 0
            )
            lines.append(
                f"- `{message.get('source_agent_id', message.get('source_agent'))}` → "
                f"`{message.get('target_agent_id', message.get('target_agent'))}`; "
                f"artifact_type=`{message.get('artifact_type')}`; "
                f"dependency={_short(message.get('dependency'), 220)}; "
                f"graph_revision=`{message.get('graph_revision')}`; "
                f"environment_revision=`{message.get('environment_revision')}`; "
                f"tool_receipts=`{receipt_count}`; "
                f"body={_short(message.get('artifact_body', message.get('content')), 500)}"
            )
    else:
        inbox = graph.get("output_agent_inbox", ())
        if isinstance(inbox, Mapping):
            inbox = inbox.get("upstream", ())
        if (
            isinstance(inbox, Sequence)
            and not isinstance(inbox, (str, bytes))
            and inbox
        ):
            lines.extend(["", "Output Agent 实际 inbox：", ""])
            for message in inbox:
                if isinstance(message, Mapping):
                    lines.append(
                        f"- `{message.get('source_agent_id', message.get('source_agent'))}` → "
                        f"`{message.get('target_agent_id', message.get('target_agent'))}`; "
                        f"artifact_type=`{message.get('artifact_type')}`; "
                        f"body={_short(message.get('artifact_body', message.get('content')), 500)}"
                    )
    trace = _environment_trace(row, "agentgraph")
    if trace:
        actions = [str(item.get("action")) for item in trace]
        lines.extend(
            [
                "",
                f"原生 environment ReAct trace（{len(actions)} 个 action）：`{' → '.join(actions)}`",
            ]
        )
        for index, item in enumerate(trace, start=1):
            lines.append(
                f"- step={index}; action=`{item.get('action')}`; "
                f"reward=`{item.get('reward', '未记录')}`; terminal=`{item.get('terminal', '未记录')}`; "
                f"parse_error=`{item.get('parse_error', False)}`; "
                f"observation={_short(item.get('observation'), 700)}"
            )
    if trajectory:
        react_entries = _react_trace_entries(trajectory)
        if react_entries:
            lines.extend(["", "Executor ReAct trace（公开 StructuredAction/observation）：", ""])
            for entry in react_entries:
                action = entry.get("structured_action", {})
                observation = entry.get("observation")
                observation_status = entry.get("observation_status")
                if observation_status is None and isinstance(observation, Mapping):
                    observation_status = observation.get("observation_status")
                lines.append(
                    f"- agent=`{entry.get('agent_id')}`; turn=`{entry.get('turn')}`; "
                    f"action={_short(json.dumps(action, ensure_ascii=False, sort_keys=True), 700)}; "
                    f"observation_status=`{observation_status}`; "
                    f"observation={_short(json.dumps(observation, ensure_ascii=False, sort_keys=True), 700)}"
                )
        receipts = _tool_receipts(trajectory)
        if receipts:
            lines.extend(["", "Tool receipts：", ""])
            for index, receipt in enumerate(receipts, start=1):
                request = receipt.get("request", {})
                lines.append(
                    f"- receipt={index}; tool=`{receipt.get('tool_id')}`; "
                    f"version=`{receipt.get('tool_version')}`; status=`"
                    f"{'error:' + str(receipt.get('error_type')) if receipt.get('error_type') else 'completed'}`; "
                    f"latency_ms=`{receipt.get('latency_ms', '未记录')}`; "
                    f"request={_short(json.dumps(request, ensure_ascii=False, sort_keys=True), 700)}; "
                    f"result={_short(json.dumps(receipt.get('result'), ensure_ascii=False, sort_keys=True), 700)}"
                )
    return "\n".join(lines)


def _direct_failure_demo(
    row: Mapping[str, Any],
    *,
    classification: str,
    first_error: str,
) -> str:
    direct = row.get("direct", {})
    trace = _environment_trace(row, "direct")
    lines = [
        f"### Direct Failure Contrast: `{row.get('task_id')}`",
        "",
        f"- Task：{_short(row.get('question'), 700)}",
        f"- Ground Truth：{_short(row.get('ground_truth'), 350)}",
        f"- Direct Final Answer：{_short(direct.get('final_answer'), 700)}",
        "- Evaluator：`{}`".format(
            json.dumps(
                direct.get("evaluation", {}).get("metrics", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
        f"- Failure classification：`{classification}`",
    ]
    if trace:
        actions = [str(item.get("action")) for item in trace]
        lines.extend(
            [
                "",
                f"Direct ReAct trace（{len(actions)} 个 action）："
                f"`{' → '.join(actions)}`",
            ]
        )
        for index, item in enumerate(trace, start=1):
            lines.append(
                f"- step={index}; action=`{item.get('action')}`; "
                f"reward=`{item.get('reward', '未记录')}`; terminal=`{item.get('terminal', '未记录')}`; "
                f"parse_error=`{item.get('parse_error', False)}`; "
                f"observation={_short(item.get('observation'), 700)}"
            )
    lines.extend(["", f"FIRST ERROR：{first_error}"])
    return "\n".join(lines)


def _load_task_row(path: Path, task_id: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in _load_jsonl(path)
            if str(row.get("task_id") or _task_id(row)) == task_id
        ),
        None,
    )


def _first_error(row: Mapping[str, Any], spec: DatasetSpec) -> str:
    graph = row.get("agentgraph", {})
    if graph.get("available") is not True or graph.get("valid") is not True:
        return "FIRST ERROR：AgentGraph 执行边界或 evaluator 边界未产生有效结果。"
    if graph.get("explicit_finish") is not True:
        return "FIRST ERROR：Flow-Director 未到达显式 FINISH。"
    if spec.key in {"hotpotqa", "triviaqa"}:
        em = float(graph.get("exact_match", 0.0))
        f1 = float(graph.get("token_f1", 0.0))
        if em == 0.0 and f1 > 0.0:
            return "FIRST ERROR：terminal answer serialization 与 accepted answer span 不一致，但 token overlap 非零；错误位于 Format boundary。"
    if spec.key == "healthbench_professional":
        evaluation = graph.get("evaluation", {})
        details = evaluation.get("details", {}) if isinstance(evaluation, Mapping) else {}
        grades = details.get("rubric_grades", ()) if isinstance(details, Mapping) else ()
        if isinstance(grades, Sequence) and not isinstance(grades, (str, bytes)):
            unmet = next(
                (
                    grade
                    for grade in grades
                    if isinstance(grade, Mapping) and grade.get("criteria_met") is False
                ),
                None,
            )
            if unmet is not None:
                return (
                    "FIRST EVALUATOR-VISIBLE ERROR：首个未满足 rubric criterion 为 `"
                    f"{_short(unmet.get('criterion'), 420)}`；judge explanation="
                    f"{_short(unmet.get('explanation'), 700)}"
                )
        return "FIRST EVALUATOR-VISIBLE ERROR：原生 reference judge 的 raw_score 未达到满分；receipt 没有更细的 rubric grade。"
    if spec.key in {"webshop", "alfworld"}:
        trace = _environment_trace(row, "agentgraph")
        invalid = next(
            (
                index
                for index, item in enumerate(trace)
                if item.get("parse_error") is True or item.get("action") == "<INVALID>"
            ),
            None,
        )
        if invalid is not None:
            return f"FIRST ERROR：原生 action parsing 在 environment step {invalid + 1} 失败。"
        return "FIRST ERROR：原生 environment episode 结束时未达到 terminal success。"
    return "FIRST ERROR：terminal evaluator metric 未达到满分；现有 receipt 不能证明更窄的原因。"


def _first_error_with_receipts(
    row: Mapping[str, Any],
    spec: DatasetSpec,
    trajectory: Mapping[str, Any] | None,
) -> str:
    if spec.key == "healthbench_professional":
        return _first_error(row, spec)
    if trajectory:
        failed = next(
            (receipt for receipt in _tool_receipts(trajectory) if receipt.get("error_type")),
            None,
        )
        if failed is not None:
            return (
                "FIRST RECORDED RUNTIME FAULT：Tool "
                f"`{failed.get('tool_id')}` 返回 `{failed.get('error_type')}`；"
                "后续 Agent 仍完成了推理，但该 Tool 调用不能计为成功检索。"
            )
        if spec.key in {"aime_2026", "webshop"}:
            turns = trajectory.get("turns", ())
            if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes)):
                for turn in turns:
                    if not isinstance(turn, Mapping):
                        continue
                    feedback = str(turn.get("canvas_feedback") or "")
                    if not any(
                        marker in feedback
                        for marker in (
                            "edit rejected",
                            "execution_error=",
                            "invalid action:",
                            "cannot finish:",
                        )
                    ):
                        continue
                    prefix = (
                        "FIRST RECORDED CANVAS/RUNTIME FAULT"
                        if spec.key == "aime_2026"
                        else "FIRST RECORDED ENVIRONMENT-OWNERSHIP FAULT"
                    )
                    causal_note = (
                        "This fault was later recovered; the receipt does not prove it caused the wrong terminal answer."
                        if spec.key == "aime_2026"
                        else "The Director recovered by deleting one owner; the terminal reward was 0.6, so the receipt does not establish a unique semantic cause for the non-success."
                    )
                    return (
                        f"{prefix}：round={turn.get('round_index', '未记录')}, "
                        f"graph_revision={turn.get('graph_revision', '未记录')}, "
                        f"feedback={_short(feedback, 900)} {causal_note}"
                    )
    return _first_error(row, spec)


def _unmeasured_report(
    spec: DatasetSpec,
    manifest: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    status = str(manifest.get("status", "missing"))
    error = str(manifest.get("error") or manifest.get("blocked_reason") or "")
    optimizer = manifest.get("optimizer_updates")
    optimizer_text = str(optimizer) if isinstance(optimizer, int) else "不可测"
    stable_state = (
        "FAIL"
        if status in {"failed", "runtime_preflight_failed", "failed_runtime_preflight"}
        else "NOT_RUN"
    )
    tool_receipts = [
        receipt for trajectory in trajectories for receipt in _tool_receipts(trajectory)
    ]
    environment_receipts = [
        receipt
        for trajectory in trajectories
        for receipt in _environment_receipts(trajectory)
    ]
    coding_receipts = [
        receipt for trajectory in trajectories for receipt in _coding_receipts(trajectory)
    ]
    explicit_finish = sum(
        trajectory.get("explicit_finish") is True for trajectory in trajectories
    )
    metric_rows = "\n".join(
        f"| {metric} | 不可测 | 不可测 |" for metric in spec.metrics
    )
    reason = error or "尚无 paired result 与原生 evaluator receipt。"
    text = f"""# {spec.title} 架构报告

## Stable Zero

- 能力边界：{spec.capability}
- 配置固定任务数：**{int(manifest.get('sample_count', 0) or 0)}**
- Runtime status：`{status}`
- 结果：**不可测**；{reason}
- Raw receipts：manifest=`{spec.manifest_path}`; paired=`{spec.paired_path}`; trajectory=`{spec.trajectory_path}`
- 显式 FINISH receipt：**{explicit_finish}/{len(trajectories)}**
- Tool receipt：**{len(tool_receipts)}**
- Environment transition receipt：**{len(environment_receipts)}**
- Coding action receipt：**{len(coding_receipts)}**
- optimizer update：**{optimizer_text}**
- `STABLE_ZERO = {stable_state}`

| 原生指标 | Direct/Simple Baseline | AgentGraph |
|---|---:|---:|
{metric_rows}

缺少原生 evaluator-valid paired result 时不填 0、不使用代理指标，也不从旧条件迁移成绩。

{_protocol_sections(spec)}

## Correct Demo

无。没有当前 evidence scope 下的 evaluator-valid paired result，不能复用旧 test 结果或构造 Correct Demo。

## Wrong / Failure Demo

### Runtime state

- 当前边界：`{status}`
- 原生 evaluator receipt：无
- 记录的错误：`{error or '无；当前状态表示尚未执行至 evaluator，而不是任务失败。'}`

{('FIRST INFRASTRUCTURE BLOCKER：`' + reason + '`；官方 Docker harness preflight 未通过，因此没有启动 Direct/Coding Agent、没有 workspace edit/test、也没有 evaluator-valid resolved receipt。' if spec.key == 'swe_bench' else 'FIRST ERROR：当前运行尚未形成可评分的 terminal receipt；不能归因为 Director、AgentGraph、Tool、environment action、Coding Agent 或模型能力。')}
"""
    summary = {
        "key": spec.key,
        "dataset": spec.title,
        "stable_zero": stable_state,
        "n": 0,
        "configured_n": int(manifest.get("sample_count", 0) or 0),
        "metrics": {
            metric: {"direct": None, "agentgraph": None}
            for metric in spec.metrics
        },
        "correct": 0,
        "wrong": 0,
        "capability": spec.capability,
        "status": status,
        "tool_calls": len(tool_receipts),
        "tool_success": sum(
            receipt.get("error_type") in (None, "")
            and receipt.get("result") is not None
            for receipt in tool_receipts
        ),
        "environment_receipts": len(environment_receipts),
        "coding_receipts": len(coding_receipts),
        "explicit_finish": explicit_finish,
        "optimizer_updates": optimizer if isinstance(optimizer, int) else None,
        "evidence_scope": spec.evidence_scope,
    }
    return text, summary


def _report_for(spec: DatasetSpec) -> tuple[str, dict[str, Any]]:
    manifest = _load_json(ROOT / spec.manifest_path)
    paired_path = ROOT / spec.paired_path if spec.paired_path is not None else None
    trajectory_path = (
        ROOT / spec.trajectory_path if spec.trajectory_path is not None else None
    )
    available_rows = _load_jsonl(paired_path)
    available_trajectories = _load_jsonl(trajectory_path)
    if not available_rows:
        return _unmeasured_report(spec, manifest, available_trajectories)

    rows = available_rows
    trajectories = available_trajectories
    trajectory_by_task = {_task_id(item): item for item in trajectories}
    stable = manifest.get("stable_zero", {})
    stable_pass = bool(isinstance(stable, Mapping) and stable.get("passed") is True)
    topologies = Counter(
        str(row.get("agentgraph", {}).get("graph_diagnostic", {}).get("topology_family", "unknown"))
        for row in rows
    )
    diagnostics = [
        row.get("agentgraph", {}).get("graph_diagnostic", {}) for row in rows
    ]
    model_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    multi_model = 0
    for row in rows:
        ids = [str(node.get("model_id", "unknown")) for node in _graph_nodes(row)]
        model_counts.update(ids)
        family_counts.update(_model_family(item) for item in ids)
        if len(set(ids)) > 1:
            multi_model += 1
    graph_executions = [
        execution
        for trajectory in trajectories
        for execution in _execution_records(trajectory)
    ]
    direct_executions = [
        execution
        for row in rows
        for execution in (
            [row.get("direct", {}).get("execution")]
            if isinstance(row.get("direct"), Mapping)
            and isinstance(row.get("direct", {}).get("execution"), Mapping)
            else []
        )
    ]
    graph_model_usage = _aggregate_model_usage(graph_executions)
    direct_model_usage = _aggregate_model_usage(direct_executions)
    director_usage = _director_usage(trajectories)
    tool_receipts = [
        receipt
        for trajectory in trajectories
        for receipt in _tool_receipts(trajectory)
    ]
    environment_receipts = [
        receipt
        for trajectory in trajectories
        for receipt in _environment_receipts(trajectory)
    ]
    coding_receipts = [
        receipt
        for trajectory in trajectories
        for receipt in _coding_receipts(trajectory)
    ]
    tool_success = sum(
        receipt.get("error_type") in (None, "") and receipt.get("result") is not None
        for receipt in tool_receipts
    )
    environment_graph_traces = [_environment_trace(row, "agentgraph") for row in rows]
    environment_direct_traces = [_environment_trace(row, "direct") for row in rows]
    graph_actions = sum(len(trace) for trace in environment_graph_traces)
    direct_actions = sum(len(trace) for trace in environment_direct_traces)
    graph_invalid = sum(
        item.get("parse_error") is True or item.get("action") == "<INVALID>"
        for trace in environment_graph_traces
        for item in trace
    )
    direct_invalid = sum(
        item.get("parse_error") is True or item.get("action") == "<INVALID>"
        for trace in environment_direct_traces
        for item in trace
    )
    direct_tokens = sum(
        int(row.get("direct", {}).get("telemetry", {}).get("input_tokens", 0) or 0)
        + int(row.get("direct", {}).get("telemetry", {}).get("output_tokens", 0) or 0)
        for row in rows
    )
    graph_tokens = sum(
        int(row.get("agentgraph", {}).get("telemetry", {}).get("input_tokens", 0) or 0)
        + int(row.get("agentgraph", {}).get("telemetry", {}).get("output_tokens", 0) or 0)
        for row in rows
    )
    direct_latency = sum(
        float(row.get("direct", {}).get("telemetry", {}).get("latency_ms", 0.0) or 0.0)
        for row in rows
    )
    graph_latency = sum(
        float(row.get("agentgraph", {}).get("telemetry", {}).get("latency_ms", 0.0) or 0.0)
        for row in rows
    )
    direct_attempts = sum(
        int(row.get("direct", {}).get("telemetry", {}).get("api_attempts", 0) or 0)
        for row in rows
    )
    graph_attempts = sum(
        int(row.get("agentgraph", {}).get("telemetry", {}).get("api_attempts", 0) or 0)
        for row in rows
    )
    primary = spec.metrics[0]
    evaluable = [
        row for row in rows if _row_metric(row, "agentgraph", primary) is not None
    ]
    correct = [
        row
        for row in evaluable
        if (_row_metric(row, "agentgraph", primary) or 0.0) >= 1.0
    ]
    wrong = [
        row
        for row in evaluable
        if (_row_metric(row, "agentgraph", primary) or 0.0) < 1.0
    ]

    metric_header = " | ".join(metric for metric in spec.metrics)
    direct_metric = " | ".join(
        _metric_display(metric, _metric_mean(rows, "direct", metric))
        for metric in spec.metrics
    )
    graph_metric = " | ".join(
        _metric_display(metric, _metric_mean(rows, "agentgraph", metric))
        for metric in spec.metrics
    )
    direct_metric_inline = "; ".join(
        f"{metric}={_metric_display(metric, _metric_mean(rows, 'direct', metric))}"
        for metric in spec.metrics
    )
    graph_metric_inline = "; ".join(
        f"{metric}={_metric_display(metric, _metric_mean(rows, 'agentgraph', metric))}"
        for metric in spec.metrics
    )
    topology_order = (
        "single",
        "serial_2",
        "serial_3_plus",
        "parallel",
        "fan_in",
        "fan_out",
        "reciprocal",
        "verification",
        "mixed",
        "other",
        "unknown",
    )
    topology_names = list(topology_order) + sorted(
        name for name in topologies if name not in topology_order
    )
    topology_lines = "\n".join(
        f"- `{name}`: {topologies.get(name, 0)}" for name in topology_names
    )
    motif_counts: Counter[str] = Counter()
    for row in rows:
        task_id = str(row.get("task_id"))
        trajectory = trajectory_by_task.get(task_id, {})
        nodes = _graph_nodes(row)
        relations = _graph_relations(row)
        modes = {str(node.get("execution_mode")) for node in nodes}
        roles = {str(node.get("role_family", "")).casefold() for node in nodes}
        indegree: Counter[str] = Counter()
        outdegree: Counter[str] = Counter()
        reciprocal = False
        for relation in relations:
            source = str(relation.get("source_id", ""))
            target = str(relation.get("target_id", ""))
            if relation.get("source_to_target") is True:
                outdegree[source] += 1
                indegree[target] += 1
            if relation.get("target_to_source") is True:
                outdegree[target] += 1
                indegree[source] += 1
            reciprocal = reciprocal or bool(
                relation.get("source_to_target") is True
                and relation.get("target_to_source") is True
            )
        motif_counts["parallel execution"] += _max_parallel_width(trajectory) > 1
        motif_counts["fan-in"] += any(value > 1 for value in indegree.values())
        motif_counts["fan-out"] += any(value > 1 for value in outdegree.values())
        motif_counts["reciprocal"] += reciprocal
        motif_counts["verification"] += any(
            any(marker in role for marker in ("verif", "critic", "review", "check"))
            for role in roles
        )
        motif_counts["ReAct"] += bool(
            "react" in modes or _environment_trace(row, "agentgraph")
        )
        motif_counts["Tool-using"] += bool(_tool_receipts(trajectory))
        motif_counts["Coding"] += bool(_coding_receipts(trajectory))
        motif_counts["mixed execution modes"] += len(modes) > 1
    motif_lines = "\n".join(
        f"- `{name}`: {motif_counts.get(name, 0)}/{len(rows)} tasks"
        for name in (
            "parallel execution",
            "fan-in",
            "fan-out",
            "reciprocal",
            "verification",
            "ReAct",
            "Tool-using",
            "Coding",
            "mixed execution modes",
        )
    )
    model_lines = "\n".join(
        f"- `{name}`: {count} Agent nodes" for name, count in sorted(model_counts.items())
    ) or "- none"
    family_order = ("Qwen", "DeepSeek", "Gemini", "GPT", "MiniMax", "Grok", "GLM", "Other")
    family_lines = ", ".join(
        f"{name}={family_counts.get(name, 0)}" for name in family_order
    )

    def usage_lines(usage: Mapping[str, Mapping[str, float]]) -> str:
        return "\n".join(
            "| {model} | {calls} | {attempts} | {tokens} | {latency:.2f} |".format(
                model=model,
                calls=int(values.get("calls", 0)),
                attempts=int(values.get("attempts", 0)),
                tokens=int(values.get("tokens", 0)),
                latency=float(values.get("latency_ms", 0.0)) / 1000,
            )
            for model, values in sorted(usage.items())
        ) or "| none | 0 | 0 | 0 | 0.00 |"

    direct_usage_lines = usage_lines(direct_model_usage)
    graph_usage_lines = usage_lines(graph_model_usage)
    judge_cost_note = (
        "HealthBench judge 的独立 token/latency receipt 未保存，不能并入上述 "
        "Executor 成本；报告只引用 evaluator receipt 中实际记录的 judge model。"
        if spec.key == "healthbench_professional"
        else ""
    )
    avg_width = _mean(
        float(_max_parallel_width(trajectory_by_task.get(str(row.get("task_id")), {})))
        for row in rows
    )
    tool_task_count = sum(
        bool(_tool_receipts(trajectory_by_task.get(str(row.get("task_id")), {})))
        for row in rows
    )
    architecture_final_note = (
        "current v4 development condition after native-action, split-isolation and action-token-budget adaptation"
        if spec.key == "webshop"
        else "current v3 condition after exact Director field/resource-ID clarification"
        if spec.key in {"hotpotqa", "triviaqa"}
        else "current v2 condition after receipt-driven minimal adaptation"
        if spec.key == "alfworld"
        else "current Stable Zero condition; no later architecture version was run"
    )
    capability_observation = (
        f"declared capability; actual Tool receipts={len(tool_receipts)}"
        if spec.key not in {"webshop", "alfworld"}
        else f"native environment replay; actual actions={graph_actions}"
    )
    text = f"""# {spec.title} 架构报告

## Stable Zero

- 能力边界：{spec.capability}
- Protocol：{spec.protocol}
- 固定 validation task：**{len(rows)}**
- Raw receipts：manifest=`{spec.manifest_path}`; paired=`{spec.paired_path}`; trajectory=`{spec.trajectory_path}`
- 显式 FINISH：**{sum(row.get('agentgraph', {}).get('explicit_finish') is True for row in rows)}/{len(rows)}**
- 有效原生 evaluator receipt：**{sum(row.get('agentgraph', {}).get('valid') is True for row in rows)}/{len(rows)}**
- `STABLE_ZERO = {'PASS' if stable_pass else 'FAIL'}`
- optimizer update：**{manifest.get('optimizer_updates', '不可测')}**
- 本轮 GRPO/backward/LoRA publication：**无**

| Condition | {metric_header} |
|---|{'---:|' * len(spec.metrics)}
| Direct/Simple Baseline | {direct_metric} |
| AgentGraph | {graph_metric} |

以上是当前 evidence scope 中 {len(rows)} 题的 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

{_protocol_sections(spec)}

## Baseline Comparison

| Stage | Result | Protocol note |
|---|---|---|
| Simple Baseline | {direct_metric_inline} | {spec.protocol} |
| AgentGraph Stable Zero | {graph_metric_inline} | fixed tasks, explicit FINISH, native evaluator |
| Architecture-final AgentGraph | {graph_metric_inline} | {architecture_final_note} |
| Tool/ReAct/Coding-enabled AgentGraph | {graph_metric_inline} | {capability_observation} |

同一 receipt 同时代表多个 stage 时不会重复解释为独立实验，也不会把 protocol-separated 条件的差值解释为因果增益。

## Workflow 分布

互斥 `topology_family` 计数（未观察到的合法结构显式记 0）：

{topology_lines}

可重叠的执行/协作 motif（来自最终图与实际 execution receipt）：

{motif_lines}

- 平均 structural depth：**{_mean(float(item.get('structural_depth', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 effective dependency depth：**{_mean(float(item.get('effective_dependency_depth', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 Agent 数：**{_mean(float(item.get('agent_count', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 relation 数：**{_mean(float(item.get('relation_count', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 parallel execution width：**{avg_width:.2f}**

## Model 使用情况

Final AgentGraph 中声明的 Executor node：

{model_lines}

- Model family：{family_lines}
- Multi-model workflow 比例：**{multi_model}/{len(rows)}**

实际 Direct Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
{direct_usage_lines}

实际 AgentGraph Executor model-call receipt：

| Model ID | Accepted model calls | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|---:|
{graph_usage_lines}

- Flow-Director：**local Qwen3.5-9B（未替换为远端 Executor）**
- Director policy receipt：`{', '.join(director_usage['policies']) or 'missing'}`；prompt=`{', '.join(director_usage['prompts']) or 'missing'}`
- Director calls/attempts：**{director_usage['calls']}/{director_usage['attempts']}**；tokens=**{director_usage['tokens']}**；latency=**{director_usage['latency_ms'] / 1000:.2f}s**

{judge_cost_note}

## Tool / ReAct 使用情况

- Tool call：**{len(tool_receipts)}**；成功：**{tool_success}**；失败：**{len(tool_receipts) - tool_success}**
- Tool call task rate：**{tool_task_count}/{len(rows)}**
- Tool useful rate：**不可测**；当前 receipt 没有独立的 causal usefulness annotation
- Tool wasted rate：**不可测**；Tool error 单独报告，不能等同于无效信息价值
- Environment transition receipt：**{len(environment_receipts)}**
- Coding action receipt：**{len(coding_receipts)}**
- AgentGraph 原生 environment action：**{graph_actions}**；invalid action：**{graph_invalid}**
- Direct 原生 environment action：**{direct_actions}**；invalid action：**{direct_invalid}**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | {direct_attempts} | {direct_tokens} | {direct_latency / 1000:.2f} |
| AgentGraph | {graph_attempts} | {graph_tokens} | {graph_latency / 1000:.2f} |
"""
    text += _tool_diagnostic_section(spec.key)
    if correct:
        selected = correct[: min(3, len(correct))]
        text += "\n## Correct Demo\n\n" + "\n\n".join(
            _demo(
                row,
                trajectory_by_task.get(str(row.get("task_id"))),
                label="Correct Demo",
                paired_receipt=spec.paired_path,
                trajectory_receipt=spec.trajectory_path,
            )
            for row in selected
        )
    else:
        text += "\n## Correct Demo\n\n"
        preserved_correct = PRESERVED_CORRECTS.get(spec.key)
        if preserved_correct is not None:
            correct_row = _load_task_row(
                ROOT / preserved_correct.paired_path,
                preserved_correct.task_id,
            )
            correct_trajectory = _load_task_row(
                ROOT / preserved_correct.trajectory_path,
                preserved_correct.task_id,
            )
            if correct_row is not None:
                text += (
                    "当前 2 题 Stable Zero 中没有满分样本；以下保留一个真实、"
                    "evaluator-valid 的历史 Correct Demo，并明确不混入当前指标。\n\n"
                    f"- Preserved condition：`{preserved_correct.condition}`\n\n"
                    + _demo(
                        correct_row,
                        correct_trajectory,
                        label="Preserved Correct Demo",
                        paired_receipt=preserved_correct.paired_path,
                        trajectory_receipt=preserved_correct.trajectory_path,
                    )
                    + "\n"
                )
            else:
                text += "未找到可验证的满分 receipt；不进行虚构。\n"
        else:
            text += "该 2 题样本中不存在满分 AgentGraph demo；不进行虚构。\n"
    if wrong:
        selected = wrong[: min(3, len(wrong))]
        chunks = []
        for row in selected:
            chunks.append(
                _demo(
                    row,
                    trajectory_by_task.get(str(row.get("task_id"))),
                    label="Wrong Demo",
                    paired_receipt=spec.paired_path,
                    trajectory_receipt=spec.trajectory_path,
                )
                + "\n\n"
                + _first_error_with_receipts(
                    row,
                    spec,
                    trajectory_by_task.get(str(row.get("task_id"))),
                )
            )
        text += "\n## Wrong Demo\n\n" + "\n\n".join(chunks)
    else:
        text += "\n## Wrong / Failure Demo\n\n"
        direct_failure = DIRECT_CONTRAST_FAILURES.get(spec.key)
        preserved_failure = PRESERVED_FAILURES.get(spec.key)
        if direct_failure is not None:
            direct_row = next(
                (
                    row
                    for row in rows
                    if str(row.get("task_id")) == direct_failure["task_id"]
                ),
                None,
            )
            if direct_row is not None:
                text += (
                    "当前 AgentGraph 2 题均成功；以下保留同一 fixed task 的真实 "
                    "Direct failure contrast，不把它计为 AgentGraph wrong case。\n\n"
                    + _direct_failure_demo(
                        direct_row,
                        classification=str(direct_failure["classification"]),
                        first_error=str(direct_failure["first_error"]),
                    )
                    + "\n"
                )
        elif preserved_failure is not None:
            failure_row = _load_task_row(
                ROOT / preserved_failure.paired_path,
                preserved_failure.task_id,
            )
            failure_trajectory = _load_task_row(
                ROOT / preserved_failure.trajectory_path,
                preserved_failure.task_id,
            )
            if failure_row is not None:
                text += (
                    "当前 AgentGraph 2 题均成功；以下是相同任务或固定条件中保留的真实"
                    "适配前 failure receipt。该结果只用于 root-cause 对照，不混入当前 "
                    "Stable Zero 指标。\n\n"
                    f"- Preserved condition：`{preserved_failure.condition}`\n"
                    f"- Failure classification：`{preserved_failure.classification}`\n\n"
                    + _demo(
                        failure_row,
                        failure_trajectory,
                        label="Preserved Wrong Demo",
                        paired_receipt=preserved_failure.paired_path,
                        trajectory_receipt=preserved_failure.trajectory_path,
                    )
                    + "\n\n"
                    + f"FIRST ERROR：{preserved_failure.first_error}\n"
                )
        else:
            text += "该 2 题样本中没有 AgentGraph 错例，也没有适用的 preserved failure receipt；不进行虚构。\n"

    if spec.key == "webshop":
        text += """

## 最小架构适配

保留的 v1 receipt 显示连续 10 次 `parse_error` transition：Executor 输出 JSON action object，而 WebShop 只接受原生 `search[...]` / `click[...]` action。executor action grammar 对自由文本 Canvas contract 具有执行优先级。当前 v4 使用 native validation indices 500..627，并在两题 canary 上完成 2/2 full-chain Stable Zero；Direct 与 AgentGraph success 均为 1/2。旧 v2 native-test 结果明确排除；v3 development 因未传递 SkillFlow `max_action_tokens` 导致本地 Direct 上下文超限，仅保留为失败诊断。以上是接口与数据隔离修正，不是 benchmark-specific workflow template。
"""
    if spec.key == "alfworld":
        text += """

## 最小架构适配

保留的 v1 receipt 中有一个 graph 在缺少 environment replay trace 时到达 FINISH，因此被拒绝。现在 FlowSteer terminal validation 对 interactive condition 要求且仅要求一个 ReAct environment actor，同时 model、role、relation 与 topology 仍由 Director 决定。v2 AgentGraph 完成了两个固定游戏；第二个游戏用 6-step 原生 episode 完成，而 Direct 达到 50-step limit。
"""

    summary = {
        "key": spec.key,
        "dataset": spec.title,
        "stable_zero": "PASS" if stable_pass else "FAIL",
        "n": len(rows),
        "metrics": {
            metric: {
                "direct": _metric_mean(rows, "direct", metric),
                "agentgraph": _metric_mean(rows, "agentgraph", metric),
            }
            for metric in spec.metrics
        },
        "correct": len(correct),
        "wrong": len(wrong),
        "capability": spec.capability,
        "topologies": dict(topologies),
        "tool_calls": len(tool_receipts),
        "tool_success": tool_success,
        "environment_receipts": len(environment_receipts),
        "coding_receipts": len(coding_receipts),
        "explicit_finish": sum(
            row.get("agentgraph", {}).get("explicit_finish") is True for row in rows
        ),
        "optimizer_updates": (
            manifest.get("optimizer_updates")
            if isinstance(manifest.get("optimizer_updates"), int)
            else None
        ),
        "evidence_scope": spec.evidence_scope,
        "status": manifest.get("status"),
        "model_families": dict(family_counts),
        "multi_model": multi_model,
    }
    return text, summary


def _skill_section() -> tuple[str, dict[str, Any]]:
    path = (
        ROOT
        / "reports/joint_qa_progressive/skill_epoch_000006/publication_results.json"
    )
    payload = _load_json(path)
    publications = payload.get("publications", {})
    lines = []
    active = 0
    if isinstance(publications, Mapping):
        for dataset, publication in publications.items():
            if not isinstance(publication, Mapping):
                continue
            skill = publication.get("skill", {})
            evidence = skill.get("evidence", {}) if isinstance(skill, Mapping) else {}
            gate = publication.get("gate", {})
            status = str(skill.get("status", "missing")) if isinstance(skill, Mapping) else "missing"
            active += status == "active"
            lines.append(
                "- `{dataset}`: Skill=`{skill_id}`, status=`{status}`, effective_pairs={pairs}, "
                "paired_effect_mean={effect}, calibrated_interval=[{lower}, {upper}], "
                "harm_probability={harm}; gate reasons={reasons}".format(
                    dataset=dataset,
                    skill_id=skill.get("skill_id") if isinstance(skill, Mapping) else None,
                    status=status,
                    pairs=evidence.get("effective_pairs") if isinstance(evidence, Mapping) else None,
                    effect=evidence.get("paired_effect_mean") if isinstance(evidence, Mapping) else None,
                    lower=evidence.get("calibrated_lower") if isinstance(evidence, Mapping) else None,
                    upper=evidence.get("calibrated_upper") if isinstance(evidence, Mapping) else None,
                    harm=evidence.get("harm_probability") if isinstance(evidence, Mapping) else None,
                    reasons=gate.get("reasons") if isinstance(gate, Mapping) else None,
                )
            )
    text = "\n".join(lines) or "- No evidence-gate publication receipt was found."
    return text, {"active": active, "publication_count": len(lines)}


def _model_canary_section() -> tuple[str, dict[str, Any]]:
    """Render saved Text/ReAct/Coding capability receipts without probing again."""

    remote_path = "artifacts/model_capability_canary/cheap_fast_20260819.json"
    local_path = (
        "artifacts/model_capability_canary/"
        "local_qwen35_9b_nonthinking_20260820.json"
    )
    catalog_path = ROOT / "config/model_catalog_multidataset_tool_v1.yaml"
    catalog_ids: set[str] = set()
    if catalog_path.is_file():
        for line in catalog_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- model_id:"):
                catalog_ids.add(stripped.split(":", 1)[1].strip().strip('"\''))

    probes: dict[str, dict[str, str]] = {}
    for relative, provider in ((remote_path, "vectorengine"), (local_path, "local-director")):
        payload = _load_json(ROOT / relative)
        raw = payload.get("probes", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for probe in raw:
            if not isinstance(probe, Mapping):
                continue
            model_id = str(probe.get("model_id", "unknown"))
            catalog_id = "qwen3.5-9b-local" if model_id == "supervisor_theta" else model_id
            row = probes.setdefault(catalog_id, {"provider": provider, "source": relative})
            capability = str(probe.get("capability", "unknown"))
            row[capability] = (
                "PASS"
                if probe.get("status") == "passed" and probe.get("compatible") is True
                else str(probe.get("status", "FAIL"))
            )

    order = sorted(probes, key=lambda item: (item not in catalog_ids, item.casefold()))
    rows = []
    fully_validated = 0
    admitted_validated = 0
    for model_id in order:
        item = probes[model_id]
        complete = all(item.get(capability) == "PASS" for capability in ("text", "react", "coding"))
        admitted = model_id in catalog_ids
        fully_validated += int(complete)
        admitted_validated += int(complete and admitted)
        rows.append(
            f"| {model_id} | {item.get('provider')} | {item.get('text', 'missing')} | "
            f"{item.get('react', 'missing')} | {item.get('coding', 'missing')} | "
            f"{'YES' if admitted else 'NO'} | `{item.get('source')}` |"
        )
    text = f"""## Model capability Canary

| Exact catalog/model ID | Provider | Text | StructuredAction/ReAct | Coding format | Admitted Executor | Receipt |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows) or '| missing | missing | missing | missing | missing | NO | missing |'}

- Catalog entries with all three saved canaries：**{admitted_validated}/{len(catalog_ids)}**
- `grok-4-1-fast-non-reasoning` 的三个 probe 均收到 HTTP 429，因此没有纳入 catalog；这不是把失败别名替换成另一个模型。
- `/v1/models` 与 canary 均未提供通过验证的 Gemini exact model ID，所以 Gemini 显式保持 0，不凭空加入。
- Flow-Director 仍固定为 local Qwen3.5-9B；表内远端模型只进入 Executor search space。
"""
    return text, {
        "catalog_size": len(catalog_ids),
        "admitted_validated": admitted_validated,
        "fully_validated": fully_validated,
    }


def _micro_training_section() -> tuple[str, dict[str, Any]]:
    """Render existing bounded optimizer/policy-sync receipts without rerunning them."""

    rows: list[str] = []
    validated_updates = 0
    for relative in MICRO_TRAINING_MANIFESTS:
        manifest = _load_json(ROOT / relative)
        training = manifest.get("training", {})
        sync = manifest.get("policy_sync", {})
        canaries = manifest.get("post_update_canaries", {})
        if not isinstance(training, Mapping):
            training = {}
        if not isinstance(sync, Mapping):
            sync = {}
        if not isinstance(canaries, Mapping):
            canaries = {}
        valid = bool(
            manifest.get("status") == "completed"
            and training.get("optimizer_updates") == 1
            and isinstance(training.get("trainable_update_l2"), (int, float))
            and not isinstance(training.get("trainable_update_l2"), bool)
            and float(training["trainable_update_l2"]) > 0.0
            and sync.get("success") is True
            and sync.get("training_performed") is True
            and sync.get("policy_published") is True
            and sync.get("route_switch_succeeded") is True
            and sync.get("canary_succeeded") is True
            and int(canaries.get("collected", 0) or 0) > 0
        )
        validated_updates += int(valid)
        rows.append(
            "| {path} | {behavior} | {updated} | {optimizer} | {l2} | {sync} | {canary} |".format(
                path=relative,
                behavior=training.get("behavior_policy_version", "missing"),
                updated=training.get("updated_policy_version", "missing"),
                optimizer=training.get("optimizer_updates", "missing"),
                l2=(
                    f"{float(training['trainable_update_l2']):.6f}"
                    if isinstance(training.get("trainable_update_l2"), (int, float))
                    and not isinstance(training.get("trainable_update_l2"), bool)
                    else "missing"
                ),
                sync="PASS" if sync.get("success") is True else "FAIL",
                canary=canaries.get("collected", 0),
            )
        )

    curve = _load_json(ROOT / MICRO_TRAINING_CURVE)
    curve_verified = bool(
        curve.get("fixed_task_ids_verified") is True
        and curve.get("evaluator_receipts_verified") is True
        and curve.get("policy_receipts_verified") is True
        and curve.get("policy_adapter_receipts_verified") is True
    )
    steps = curve.get("steps", ())
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        steps = ()
    curve_rows: list[str] = []
    macro_values: list[tuple[int, float, float]] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        macro = step.get("macro_average", {})
        if not isinstance(macro, Mapping):
            continue
        em = macro.get("strict_exact_match")
        f1 = macro.get("strict_token_f1")
        if not isinstance(em, (int, float)) or isinstance(em, bool):
            continue
        if not isinstance(f1, (int, float)) or isinstance(f1, bool):
            continue
        ordinal = int(step.get("step", len(macro_values)))
        macro_values.append((ordinal, float(em), float(f1)))
        curve_rows.append(
            f"| Step {ordinal} | {step.get('policy_version', 'missing')} | "
            f"{100 * float(em):.2f}% | {100 * float(f1):.2f}% |"
        )
    positive_trend = bool(
        curve_verified
        and len(macro_values) >= 2
        and (
            macro_values[-1][1] > macro_values[0][1]
            or macro_values[-1][2] > macro_values[0][2]
        )
    )
    executed = validated_updates == len(MICRO_TRAINING_MANIFESTS)
    text = f"""## 既有 bounded micro-training 证据

这组证据来自当前项目此前完成的 HotpotQA/TriviaQA joint-QA Flow-Director 训练闭环；它与本轮统一 Tool/Environment/Coding Stable Zero 的版本边界分开报告，不能证明新 Tool action-selection policy 已被训练。

| Manifest | Behavior policy | Updated policy | Optimizer updates | Trainable update L2 | Policy sync | Post-update canary |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows) or '| missing | missing | missing | 0 | missing | FAIL | 0 |'}

- Receipt-valid optimizer updates：**{validated_updates}/{len(MICRO_TRAINING_MANIFESTS)}**
- Matched held-out curve receipt：`{MICRO_TRAINING_CURVE}`；fixed task/evaluator/policy receipts verified=`{str(curve_verified).lower()}`

| Step | Policy version | HotpotQA/TriviaQA macro EM | Macro F1 |
|---|---|---:|---:|
{chr(10).join(curve_rows) or '| 不可测 | missing | 不可测 | 不可测 |'}

该证据证明 LoRA 参数更新、optimizer state、policy publication、route switch 和 post-update canary 的闭环可执行。固定 held-out 的最终宏平均没有超过 Step 0，因此不能声称观察到正向 learning trend；按方案应优先继续检查 architecture/search-space 与 evidence quality，而不是扩大训练规模。
"""
    return text, {
        "executed": executed,
        "validated_updates": validated_updates,
        "curve_verified": curve_verified,
        "positive_trend": positive_trend,
    }


def _total_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    table_lines = []
    for item in summaries:
        metric_text = "; ".join(
            f"{name}: Direct={_metric_display(name, values.get('direct'))}, "
            f"AgentGraph={_metric_display(name, values.get('agentgraph'))}"
            for name, values in item.get("metrics", {}).items()
        ) or "不可测"
        table_lines.append(
            f"| {item['dataset']} | {item['stable_zero']} | {item['n']} | "
            f"{item['correct']} | {item['wrong']} | {metric_text} | {item['capability']} |"
        )
    skill_text, skill_state = _skill_section()
    model_canary_text, _model_canary_state = _model_canary_section()
    micro_training_text, micro_training_state = _micro_training_section()
    optimizer_values = [
        item.get("optimizer_updates")
        for item in summaries
        if isinstance(item.get("optimizer_updates"), int)
    ]
    optimizer_text = (
        str(sum(optimizer_values))
        if len(optimizer_values) == len(summaries)
        else "不可测（至少一个 manifest 未记录）"
    )
    tool_receipts = sum(int(item.get("tool_calls", 0) or 0) for item in summaries)
    non_environment_tool_receipts = sum(
        int(by_item.get("tool_calls", 0) or 0)
        for by_item in summaries
        if str(by_item.get("key"))
        in {"hotpotqa", "triviaqa", "aime_2026", "healthbench_professional"}
    )
    environment_receipts = sum(
        int(item.get("environment_receipts", 0) or 0) for item in summaries
    )
    coding_receipts = sum(
        int(item.get("coding_receipts", 0) or 0) for item in summaries
    )
    explicit_finish = sum(
        int(item.get("explicit_finish", 0) or 0) for item in summaries
    )
    aggregate_topologies: Counter[str] = Counter()
    aggregate_families: Counter[str] = Counter()
    total_measured = 0
    multi_model_workflows = 0
    for item in summaries:
        raw_topologies = item.get("topologies", {})
        if isinstance(raw_topologies, Mapping):
            aggregate_topologies.update(
                {str(key): int(value) for key, value in raw_topologies.items()}
            )
        raw_families = item.get("model_families", {})
        if isinstance(raw_families, Mapping):
            aggregate_families.update(
                {str(key): int(value) for key, value in raw_families.items()}
            )
        total_measured += int(item.get("n", 0) or 0)
        multi_model_workflows += int(item.get("multi_model", 0) or 0)
    aggregate_topology_text = ", ".join(
        f"{name}={aggregate_topologies.get(name, 0)}"
        for name in (
            "single",
            "serial_2",
            "serial_3_plus",
            "parallel",
            "fan_in",
            "fan_out",
            "reciprocal",
            "verification",
            "mixed",
        )
    )
    aggregate_family_text = ", ".join(
        f"{name}={aggregate_families.get(name, 0)}"
        for name in ("Qwen", "DeepSeek", "Gemini", "GPT", "MiniMax", "Grok", "GLM", "Other")
    )
    by_key = {str(item.get("key")): item for item in summaries}
    diagnostic_rows: list[str] = []
    for key, relative in TOOL_DIAGNOSTIC_RECEIPTS.items():
        receipt = _load_json(ROOT / relative)
        compliance = receipt.get("compliance", {})
        if not isinstance(compliance, Mapping):
            compliance = {}
        schema = compliance.get("schema_compliance", {})
        backend = compliance.get("backend_compliance", {})
        model = compliance.get("model_compliance", {})
        diagnostic_rows.append(
            "| {dataset} | {status} | {schema} | {backend} | {model} | {count} |".format(
                dataset=by_key.get(key, {}).get("dataset", key),
                status=receipt.get("status", "missing"),
                schema="PASS" if isinstance(schema, Mapping) and schema.get("passed") is True else "FAIL",
                backend="PASS" if isinstance(backend, Mapping) and backend.get("passed") is True else "FAIL",
                model="PASS" if isinstance(model, Mapping) and model.get("passed") is True else "FAIL",
                count=(backend.get("successful_receipts", 0) if isinstance(backend, Mapping) else 0),
            )
        )
    diagnostic_table = "\n".join(diagnostic_rows) or "| 无 | missing | FAIL | FAIL | FAIL | 0 |"
    webshop = by_key.get("webshop", {})
    swebench = by_key.get("swe_bench", {})
    alfworld = by_key.get("alfworld", {})
    webshop_measured = int(webshop.get("n", 0) or 0) > 0
    swebench_measured = int(swebench.get("n", 0) or 0) > 0
    qa_tool_use_validated = any(
        int(by_key.get(key, {}).get("tool_success", 0) or 0) > 0
        for key in (
            "hotpotqa",
            "triviaqa",
            "aime_2026",
            "healthbench_professional",
        )
    )
    webshop_react_ready = (
        webshop.get("stable_zero") == "PASS"
        and int(webshop.get("environment_receipts", 0) or 0) > 0
    )
    alfworld_react_ready = (
        alfworld.get("stable_zero") == "PASS"
        and int(alfworld.get("environment_receipts", 0) or 0) > 0
    )
    swebench_coding_ready = (
        swebench.get("stable_zero") == "PASS"
        and int(swebench.get("coding_receipts", 0) or 0) > 0
    )
    all_stable = all(item.get("stable_zero") == "PASS" for item in summaries)
    demos_complete = all(
        int(item.get("correct", 0) or 0) > 0
        and int(item.get("wrong", 0) or 0) > 0
        for item in summaries
    )
    webshop_result_note = (
        "WebShop v4 只报告当前 native-validation paired receipts。"
        if webshop_measured
        else "WebShop v4 尚无 paired result，不迁移旧 native-test 成绩。"
    )
    swebench_result_note = (
        "SWE-bench regular-dev 只报告当前 official-harness paired receipts。"
        if swebench_measured
        else "SWE-bench regular-dev 尚无 paired result，不以代理零分替代。"
    )
    webshop_issue = (
        "WebShop v4 已形成 native-validation paired receipt；旧 v2 native-test success 仍不进入当前指标，v3 仅保留为上下文预算失败诊断。"
        if webshop_measured
        else "WebShop native-action 接口已修正，但 v4 validation canary 尚无 paired receipt；旧 v2 test-range success 不作为当前验证证据。"
    )
    swebench_issue = (
        "SWE-bench regular-dev 已形成 official-harness paired receipt；完整 Verified 仍保留给最终评测。"
        if swebench_measured
        else "SWE-bench regular-dev 当前尚无 official Docker harness evaluator receipt，因此 resolved_rate 不可测，不能提前归因为 Coding Agent 或 patch quality。"
    )
    skill_gate_note = (
        "因此不存在可注入新多数据集 Director condition 的版本兼容 `ACTIVE` Skill，也不满足 Skill-on micro-training 的触发条件。"
        if skill_state["active"] == 0
        else "存在 `ACTIVE` Skill；是否实际注入仍必须以 condition receipt 与版本兼容检查为准。"
    )
    return f"""# 多数据集 Agent 架构 Stable Zero 报告

## 架构完成情况

控制路径保持为：本地 Qwen3.5-9B Flow-Director、one-atomic-edit progressive Canvas、execute-after-edit feedback、dynamic AgentGraph、显式 FINISH、数据集原生 evaluator 与完整 trajectory receipt。统一 AgentRuntime 分发 `reasoning`、Tool/ReAct、environment ReAct 和 `coding` execution adapter。Tool assignment、model selection、自由文本 contract、dependency、artifact type 与 completion condition 仍属于 Director search space。

当前所读 manifests 的 optimizer update 总数：**{optimizer_text}**。本轮未执行大规模训练、GRPO、backward、LoRA publication 或新的 Skill activation。

## 实现来源分类

- `DIRECT_REUSE`：FlowSteer progressive Canvas 的 edit→execute→feedback、显式 FINISH、action mask、trajectory；SkillFlow 的 StructuredAction/Tool Registry、RetrievalIndex、bounded computation、RAGEN environment、MedRAG corpus、SWE-bench worktree 与 evidence/library contract。
- `NECESSARY_ADAPTATION`：异构 `reasoning|react|coding` dispatch、task-scoped Tool registry、typed evaluator receipt、WebShop 原生 action grammar、ALFWorld interactive FINISH 的 environment actor invariant、SWE-bench worktree ownership。
- `PROJECT_ALGORITHM_ADDITION`：typed `CommunicationEnvelope`、`ToolCapability`、measured `ToolReceipt` 与既有 same-prefix paired AgentGraph posterior/evidence gate。
- `NOT_IMPLEMENTED_OR_NOT_EXECUTED`：SWE-bench 官方-harness-valid Coding trajectory、evidence-gated `ACTIVE` Skill 注入以及本轮 micro-training/optimizer/policy synchronization。

逐文件的上游类/函数与不兼容原因记录在 `docs/SOURCE_MAP.md`。

## Stable Zero 结果

| Dataset | Stable Zero | n | 满分/成功 | 错误 | 原生指标 | 能力边界 |
|---|---:|---:|---:|---:|---|---|
{chr(10).join(table_lines)}

只有存在当前 evidence scope 下的 paired result 与原生 evaluator receipt 时才显示数值；缺失项显示“不可测”，不填 0。AIME 数值来自 AIME-2025 development canary，**不是 AIME 2026 benchmark 成绩**；{webshop_result_note}{swebench_result_note}

## Runtime receipts

- 显式 FINISH：**{explicit_finish}** 条 trajectory
- ToolReceipt（含 environment action）：**{tool_receipts}**
- QA / computation / MedRAG 自然策略 ToolReceipt：**{non_environment_tool_receipts}**
- Environment transition receipt：**{environment_receipts}**
- Coding action receipt：**{coding_receipts}**

## Natural Stable Zero workflow/model adoption

- Exclusive topology family：{aggregate_topology_text}
- Declared Executor node family：{aggregate_family_text}
- Multi-model workflow：**{multi_model_workflows}/{total_measured}**

当前 {total_measured} 条自然 AgentGraph trajectory 只观察到 single/serial topology；parallel、fan-in、fan-out 与 reciprocal 在独立 non-chain runtime diagnostic 中可执行，但尚未被当前 fixed-task Director policy 自然采用。因此 `DEEP_WORKFLOW_READY` 只表示 search space/runtime capability，不能解释为 policy adoption 或性能收益。

{model_canary_text}

## Exact-schema Tool forced probe（不计入 benchmark）

| Dataset | Overall | Schema | Backend | Model/termination | Successful ToolReceipt |
|---|---:|---:|---:|---:|---:|
{diagnostic_table}

这些 receipt 均为 `diagnostic_only=true`、`forced_probe=true`，没有 evaluator、Ground Truth、benchmark metric、Skill evidence、GRPO 或 optimizer update；不能与自然策略 Tool adoption 混合计数。

## Protocol audit

- HotpotQA、TriviaQA 与 AIME-2025 development：Direct 与 Tool-capable AgentGraph 分别报告；未把 protocol-separated delta 解释为 architecture causal effect 或 SOTA improvement。
- HealthBench Professional v2：只报告 openai/simple-evals-compatible **reference-judge diagnostic**；不是私有官方评测服务或 leaderboard 成绩。
- WebShop v4：只接受 native validation indices 500..627 的原生环境结果；旧 v2 native-test 结果作为 test-contaminated adaptation evidence 排除，v3 仅保留为上下文预算失败诊断。
- ALFWorld：Direct/Simple ReAct 和 AgentGraph 使用相同 task lock、原生环境、action budget 与 evaluator，可进行同条件描述性比较。
- SWE-bench：架构开发只使用 regular dev；完整 Verified 保留给最终评测。没有官方 Docker harness receipt 时 resolved/resolved_rate 不可测。
- 所有原生 evaluator 都是唯一 terminal metric source；LLM self-judgement、文本相似度或 local proxy test 均未替代正式指标。

## Skill evidence gate

{skill_text}

最新 evidence-gated `ACTIVE` Skill 数量：**{skill_state['active']}**。{skill_gate_note}`CANDIDATE` instruction 仍是候选，不作为已验证 Skill。

{micro_training_text}

## Current add_subgraph micro-training preflight

- Decision：`NO_GO`
- Receipt report：`reports/multidataset_stablezero/MICROTRAINING_PREFLIGHT.md`
- 当前没有新的冻结 schedule/cursor，也没有满足 evidence gate 的 `ACTIVE` Skill；现成 `add_subgraph` 模板是 Skill-on。GPU 3 被本任务之外的进程占用，GPU 4 rollout service 已关闭，provider credential 未配置。因而本轮没有启动服务或训练，也没有把旧 cursor 重放称为新更新。

## 剩余问题分类

- `ENVIRONMENT_LIMITATION`：{swebench_issue}
- `SKILL_EVIDENCE_INSUFFICIENT`：最新独立 paired evidence 未满足 calibrated lower-bound/harm gate；`ACTIVE` Skill 数为 0。
- `TRAINING_INSTABILITY`：既有 joint-QA bounded micro-training 完成了 {micro_training_state['validated_updates']} 次真实 optimizer update 与 policy sync，但固定 held-out 宏平均没有形成正向趋势；该证据不覆盖本轮新增 Tool/Environment/Coding action-selection policy。
- `TOOL_LIMITATION`：HotpotQA 与 TriviaQA v3 的当前 Stable Zero trajectories 各自然产生 1 条成功 retrieval `ToolReceipt`；AIME-2025 development 与 HealthBench v2 未自然选择其可选 Tool。两题 canary 只能验证自然工具调用链已经出现，不能估计 tool-use policy 的总体采用率、useful rate、wasted rate 或 Skill effect。
- `MODEL_CAPABILITY_LIMIT`：HealthBench 的 2 题 reference-judge diagnostic raw_score 较低；该 canary 不足以把差距唯一归因于模型、架构或缺少检索。
- `ARCHITECTURE_DEFECT`：当前没有新的 confirmed open defect。WebShop 的 action serialization / token-budget 缺陷已按 preserved failure receipt 修复；{webshop_issue}
- `POLICY_LEARNING_PROBLEM`：尚未成立。当前自然 policy 没有采用非链式 topology，但 SWE Coding、ACTIVE Skill 和 Tool usefulness 闭环仍未齐全，不能先把缺口归因于 policy learning。

## 最终判定

```text
FLOWSTEER_CORE_PRESERVED = YES

MODEL_POOL_EXPANDED = YES
MULTI_MODEL_WORKFLOW_READY = YES
DEEP_WORKFLOW_READY = YES
COLLABORATION_DIVERSITY_READY = YES

QA_TOOL_REGISTRY_READY = YES
QA_DATABASE_SELECTION_READY = YES
QA_TOOL_USE_VALIDATED = {'YES' if qa_tool_use_validated else 'NO'}

ALFWORLD_REACT_READY = {'YES' if alfworld_react_ready else 'NO'}
WEBSHOP_REACT_READY = {'YES' if webshop_react_ready else 'NO'}

CODING_AGENT_IMPLEMENTED = YES
CODING_AGENT_READY = {'YES' if swebench_coding_ready else 'NO'}
SWEBENCH_CODING_WORKFLOW_READY = {'YES' if swebench_coding_ready else 'NO'}

SKILL_END_TO_END_READY = {'YES' if skill_state['active'] > 0 else 'NO'}
SKILL_SUMMARY_VALIDATED = NO

ALL_DATASETS_STABLE_ZERO_COMPLETE = {'YES' if all_stable else 'NO'}
CORRECT_WRONG_DEMOS_COMPLETE = {'YES' if demos_complete else 'NO'}

MICRO_TRAINING_EXECUTED = {'YES' if micro_training_state['executed'] else 'NO'}
LEARNING_TREND_OBSERVED = {'YES' if micro_training_state['positive_trend'] else 'NO'}

LOCAL_RECOVERY_BACKUP = YES
GITHUB_ARCHITECTURE_BACKUP = NO

READY_FOR_FORMAL_MULTIDATASET_TRAINING = NO
```

判定说明：`DEEP_WORKFLOW_READY` 表示 search space 与 scheduler 支持 deep/parallel/fan-in/finite-reciprocal motif，不表示当前 canary 已普遍采用深图。`CORRECT_WRONG_DEMOS_COMPLETE` 要求每个数据集同时具有当前 evaluator-valid correct 与 wrong receipt；不会为凑数量制造失败。当前 branch/tag/patch/bundle 已形成可恢复本地备份，但当前环境对配置的 GitHub remote 没有可用 HTTPS/SSH 写认证，所以 `GITHUB_ARCHITECTURE_BACKUP=NO`，不伪称已推送。

## 报告索引

{chr(10).join(f"- [{spec.title}](reports/multidataset_stablezero/{spec.report_name})" for spec in SPECS)}
"""


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in SPECS:
        report, summary = _report_for(spec)
        (REPORT_ROOT / spec.report_name).write_text(
            report.rstrip() + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
    TOTAL_REPORT.write_text(
        _total_report(summaries).rstrip() + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(TOTAL_REPORT),
                "dataset_reports": [
                    str(REPORT_ROOT / spec.report_name) for spec in SPECS
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
