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


SPECS = (
    DatasetSpec(
        "hotpotqa",
        "HotpotQA",
        "HOTPOTQA_ARCH_REPORT.md",
        "artifacts/qa_tool_react_stable_zero/hotpotqa/protocol_separated_results.jsonl",
        "artifacts/qa_tool_react_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl",
        "artifacts/qa_tool_react_stable_zero/hotpotqa/run_manifest.json",
        ("exact_match", "token_f1"),
        "闭卷上下文推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力",
        "Direct 使用给定上下文；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。",
    ),
    DatasetSpec(
        "triviaqa",
        "TriviaQA",
        "TRIVIAQA_ARCH_REPORT.md",
        "artifacts/qa_tool_react_stable_zero/triviaqa/protocol_separated_results.jsonl",
        "artifacts/qa_tool_react_stable_zero/triviaqa/tool_agentgraph_trajectories.jsonl",
        "artifacts/qa_tool_react_stable_zero/triviaqa/run_manifest.json",
        ("exact_match", "token_f1"),
        "仅问题推理 + 冻结 Wikipedia RetrievalIndex/ReAct 能力",
        "Direct 仅接收问题；AgentGraph 允许使用检索 Tool。两者属于 protocol-separated 条件，差值只作描述性统计。",
    ),
    DatasetSpec(
        "aime_2026",
        "AIME 2026",
        "AIME2026_ARCH_REPORT.md",
        "artifacts/aime2026_computation_tool_stable_zero/evaluation/paired_results.jsonl",
        "artifacts/aime2026_computation_tool_stable_zero/evaluation/agentgraph_trajectories.jsonl",
        "artifacts/aime2026_computation_tool_stable_zero/evaluation/run_manifest.json",
        ("exact_match",),
        "推理 + 有界 calculator/Python execution 能力",
        "Direct 与允许使用计算 Tool 的 AgentGraph 分别报告，不作 protocol-equivalent 因果比较。",
    ),
    DatasetSpec(
        "healthbench_professional",
        "HealthBench Professional",
        "HEALTHBENCH_PROFESSIONAL_ARCH_REPORT.md",
        "artifacts/healthbench_professional_medrag_tool_stable_zero/evaluation/paired_results.jsonl",
        "artifacts/healthbench_professional_medrag_tool_stable_zero/evaluation/agentgraph_trajectories.jsonl",
        "artifacts/healthbench_professional_medrag_tool_stable_zero/evaluation/run_manifest.json",
        ("raw_score",),
        "临床推理 + 冻结教材语料 MedRAG search 能力",
        "Direct 与允许使用 MedRAG 的 AgentGraph 分别报告；raw_score 来自配置的 reference judge。",
    ),
    DatasetSpec(
        "webshop",
        "WebShop",
        "WEBSHOP_ARCH_REPORT.md",
        "artifacts/webshop_ragen_environment_native_action_v2_stable_zero/evaluation/paired_results.jsonl",
        "artifacts/webshop_ragen_environment_native_action_v2_stable_zero/evaluation/agentgraph_trajectories.jsonl",
        "artifacts/webshop_ragen_environment_native_action_v2_stable_zero/evaluation/run_manifest.json",
        ("success",),
        "request-scoped SkillFlow/RAGEN environment ReAct",
        "Direct 与 AgentGraph 使用相同原生环境、task lock、action budget 和 evaluator。",
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
        "SWE-bench Verified",
        "SWEBENCH_ARCH_REPORT.md",
        None,
        None,
        "artifacts/swebench_verified_coding_agent_stable_zero/evaluation/run_manifest.json",
        ("resolved",),
        "detached task-pinned worktree + iterative CodingExecutionAdapter + official Docker harness",
        "唯一接受的指标是官方 resolved_rate；禁止使用代理评分。",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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


def _metric_mean(rows: Sequence[Mapping[str, Any]], arm: str, metric: str) -> float:
    return _mean(float(row.get(arm, {}).get(metric, 0.0)) for row in rows)


def _metric_display(metric: str, value: float) -> str:
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
) -> str:
    graph = row.get("agentgraph", {})
    lines = [
        f"### {label}: `{row.get('task_id')}`",
        "",
        f"- Task：{_short(row.get('question'), 700)}",
        f"- Ground Truth：{_short(row.get('ground_truth'), 350)}",
        f"- Final Answer：{_short(graph.get('final_answer'), 700)}",
        f"- Evaluator: `{json.dumps(graph.get('evaluation', {}).get('metrics', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- AgentGraph: `{_graph_line(row)}`",
        "",
        "Agent 配置：",
        "",
    ]
    for node in _graph_nodes(row):
        lines.append(
            "- `{id}` — model=`{model}`, execution_mode=`{mode}`, role_family=`{role}`, "
            "allowed_tools=`{tools}`, artifact_type=`{artifact}`; contract: {contract}".format(
                id=node.get("id"),
                model=node.get("model_id"),
                mode=node.get("execution_mode"),
                role=node.get("role_family"),
                tools=node.get("allowed_tools", []),
                artifact=node.get("artifact_type"),
                contract=_short(node.get("contract"), 320),
            )
        )
    if trajectory:
        actions = []
        for turn in trajectory.get("turns", ()):
            if not isinstance(turn, Mapping):
                continue
            action = turn.get("action", {})
            if isinstance(action, Mapping):
                actions.append(str(action.get("action", action.get("action_type", "?"))))
        lines.extend(["", f"Director atomic edit 序列：`{' → '.join(actions)}`"])
    inbox = graph.get("output_agent_inbox", ())
    if isinstance(inbox, Mapping):
        inbox = inbox.get("upstream", ())
    if isinstance(inbox, Sequence) and not isinstance(inbox, (str, bytes)) and inbox:
        lines.extend(["", "Output Agent 实际 inbox：", ""])
        for message in inbox[:4]:
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
                f"原生 ReAct trace（{len(actions)} 个 action）：`{' → '.join(actions[:12])}`"
                + (" …" if len(actions) > 12 else ""),
            ]
        )
    if trajectory:
        receipts = _tool_receipts(trajectory)
        if receipts:
            lines.extend(["", "Tool receipts：", ""])
            for receipt in receipts[:6]:
                request = receipt.get("request", {})
                lines.append(
                    f"- tool=`{receipt.get('tool_id')}`, status=`"
                    f"{'error:' + str(receipt.get('error_type')) if receipt.get('error_type') else 'completed'}`; "
                    f"request={_short(json.dumps(request, ensure_ascii=False, sort_keys=True), 360)}"
                )
    return "\n".join(lines)


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
        return "FIRST ERROR：原生 reference judge 的 raw_score 未达到满分；需要结合 trajectory 中的 Tool receipt 判断是否存在更早的运行时失败。"
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
    return _first_error(row, spec)


def _report_for(spec: DatasetSpec) -> tuple[str, dict[str, Any]]:
    manifest = _load_json(ROOT / spec.manifest_path)
    if spec.paired_path is None:
        error = str(manifest.get("error", "official evaluator unavailable"))
        text = f"""# {spec.title} 架构报告

## Stable Zero

- 能力边界：{spec.capability}
- Protocol：{spec.protocol}
- 冻结任务数：**{int(manifest.get('sample_count', 0) or 0)}**
- Runtime status：`{manifest.get('status', 'missing')}`
- 官方指标：`resolved_rate`
- 结果：**不可测**；`{error}`
- `STABLE_ZERO = FAIL`

两个固定 `astropy/astropy` base commit 的 repository/worktree preflight 已通过，但官方 Docker harness 不可用。fail-closed preflight 后没有执行模型/API 调用或 Coding Agent trajectory，也没有报告代理指标。

## Coding trace

不存在通过官方 evaluator 验证的 coding trajectory，因此不虚构 inspected files、edits、commands、tests、revisions 或 resolved status。

## 问题分类

`ENVIRONMENT_LIMITATION`：当前 runtime 中官方 SWE-bench Docker harness 无法连接 Docker daemon。
"""
        summary = {
            "dataset": spec.title,
            "stable_zero": "FAIL",
            "n": 0,
            "metrics": {},
            "correct": 0,
            "wrong": 0,
            "capability": spec.capability,
            "status": manifest.get("status"),
        }
        return text, summary

    rows = _load_jsonl(ROOT / spec.paired_path)
    trajectories = _load_jsonl(ROOT / spec.trajectory_path) if spec.trajectory_path else []
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
    tool_receipts = [
        receipt
        for trajectory in trajectories
        for receipt in _tool_receipts(trajectory)
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
    correct = [row for row in rows if float(row.get("agentgraph", {}).get(primary, 0.0)) >= 1.0]
    wrong = [row for row in rows if float(row.get("agentgraph", {}).get(primary, 0.0)) < 1.0]

    metric_header = " | ".join(metric for metric in spec.metrics)
    direct_metric = " | ".join(
        _metric_display(metric, _metric_mean(rows, "direct", metric))
        for metric in spec.metrics
    )
    graph_metric = " | ".join(
        _metric_display(metric, _metric_mean(rows, "agentgraph", metric))
        for metric in spec.metrics
    )
    topology_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in sorted(topologies.items())
    ) or "- none"
    model_lines = "\n".join(
        f"- `{name}`: {count} Agent nodes" for name, count in sorted(model_counts.items())
    ) or "- none"
    family_lines = ", ".join(f"{name}={count}" for name, count in sorted(family_counts.items())) or "none"
    avg_width = _mean(
        float(_max_parallel_width(trajectory_by_task.get(str(row.get("task_id")), {})))
        for row in rows
    )
    text = f"""# {spec.title} 架构报告

## Stable Zero

- 能力边界：{spec.capability}
- Protocol：{spec.protocol}
- 固定 validation task：**{len(rows)}**
- 显式 FINISH：**{sum(row.get('agentgraph', {}).get('explicit_finish') is True for row in rows)}/{len(rows)}**
- 有效原生 evaluator receipt：**{sum(row.get('agentgraph', {}).get('valid') is True for row in rows)}/{len(rows)}**
- `STABLE_ZERO = {'PASS' if stable_pass else 'FAIL'}`
- 本轮 training/optimizer/LoRA publication：**无**

| Condition | {metric_header} |
|---|{'---:|' * len(spec.metrics)}
| Direct/Simple Baseline | {direct_metric} |
| AgentGraph | {graph_metric} |

以上是固定 2 题 Stable Zero 行为结果，不是正式 benchmark 或 SOTA 声明。

## Workflow 分布

{topology_lines}

- 平均 structural depth：**{_mean(float(item.get('structural_depth', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 effective dependency depth：**{_mean(float(item.get('effective_dependency_depth', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 Agent 数：**{_mean(float(item.get('agent_count', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 relation 数：**{_mean(float(item.get('relation_count', 0.0) or 0.0) for item in diagnostics):.2f}**
- 平均 parallel execution width：**{avg_width:.2f}**

## Model 使用情况

{model_lines}

- Model family：{family_lines}
- Multi-model workflow 比例：**{multi_model}/{len(rows)}**

## Tool / ReAct 使用情况

- Tool call：**{len(tool_receipts)}**；成功：**{tool_success}**；失败：**{len(tool_receipts) - tool_success}**
- AgentGraph 原生 environment action：**{graph_actions}**；invalid action：**{graph_invalid}**
- Direct 原生 environment action：**{direct_actions}**；invalid action：**{direct_invalid}**

## 成本

| Arm | API attempts | Tokens | Latency (s) |
|---|---:|---:|---:|
| Direct | {direct_attempts} | {direct_tokens} | {direct_latency / 1000:.2f} |
| AgentGraph | {graph_attempts} | {graph_tokens} | {graph_latency / 1000:.2f} |
"""
    if correct:
        selected = correct[: min(3, len(correct))]
        text += "\n## Correct Demo\n\n" + "\n\n".join(
            _demo(
                row,
                trajectory_by_task.get(str(row.get("task_id"))),
                label="Correct Demo",
            )
            for row in selected
        )
    else:
        text += "\n## Correct Demo\n\n该 2 题样本中不存在满分 AgentGraph demo；不进行虚构。\n"
    if wrong:
        selected = wrong[: min(3, len(wrong))]
        chunks = []
        for row in selected:
            chunks.append(
                _demo(
                    row,
                    trajectory_by_task.get(str(row.get("task_id"))),
                    label="Wrong Demo",
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
        text += "\n## Wrong Demo\n\n该 2 题样本中没有 AgentGraph 错例；不进行虚构。\n"

    if spec.key == "webshop":
        text += """

## 最小架构适配

保留的 v1 receipt 显示连续 10 次 `parse_error` transition：Executor 输出 JSON action object，而 WebShop 只接受原生 `search[...]` / `click[...]` action。修正后，executor action grammar 对自由文本 Canvas contract 具有执行优先级；在相同 2 个固定任务上，AgentGraph success 从 **1/2 变为 2/2**。这是接口修正，不是 benchmark-specific workflow template。
"""
    if spec.key == "alfworld":
        text += """

## 最小架构适配

保留的 v1 receipt 中有一个 graph 在缺少 environment replay trace 时到达 FINISH，因此被拒绝。现在 FlowSteer terminal validation 对 interactive condition 要求且仅要求一个 ReAct environment actor，同时 model、role、relation 与 topology 仍由 Director 决定。v2 AgentGraph 完成了两个固定游戏；第二个游戏用 6-step 原生 episode 完成，而 Direct 达到 50-step limit。
"""

    summary = {
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


def _total_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    table_lines = []
    for item in summaries:
        metric_text = "; ".join(
            f"{name}: Direct={_metric_display(name, float(values['direct']))}, "
            f"AgentGraph={_metric_display(name, float(values['agentgraph']))}"
            for name, values in item.get("metrics", {}).items()
        ) or "不可测"
        table_lines.append(
            f"| {item['dataset']} | {item['stable_zero']} | {item['n']} | "
            f"{item['correct']} | {item['wrong']} | {metric_text} | {item['capability']} |"
        )
    skill_text, skill_state = _skill_section()
    return f"""# 多数据集 Agent 架构 Stable Zero 报告

## 架构完成情况

控制路径保持为：本地 Qwen3.5-9B Flow-Director、one-atomic-edit progressive Canvas、execute-after-edit feedback、dynamic AgentGraph、显式 FINISH、数据集原生 evaluator 与完整 trajectory receipt。统一 AgentRuntime 分发 `reasoning`、Tool/ReAct、environment ReAct 和 `coding` execution adapter。Tool assignment、model selection、自由文本 contract、dependency、artifact type 与 completion condition 仍属于 Director search space。

本轮未执行大规模训练、GRPO、backward、optimizer update、LoRA publication 或新的 Skill activation。

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

以上均为固定 2 题 Stable Zero 行为结果（AIME 从 30 个官方任务中固定选取），不是正式 benchmark 或 SOTA 估计。SWE-bench 标记为不可测，不以代理零分替代。

## Skill evidence gate

{skill_text}

最新 evidence-gated `ACTIVE` Skill 数量：**{skill_state['active']}**。因此不存在可注入新多数据集 Director condition 的版本兼容 `ACTIVE` Skill，也不满足 Skill-on micro-training 的触发条件。`CANDIDATE` instruction 仍是候选，不作为已验证 Skill。

## 剩余问题分类

- `ENVIRONMENT_LIMITATION`：SWE-bench 官方 Docker harness 无法访问 Docker daemon，官方 `resolved_rate` 不可测。
- `SKILL_EVIDENCE_INSUFFICIENT`：最新独立 paired evidence 未满足 calibrated lower-bound/harm gate；`ACTIVE` Skill 数为 0。
- `POLICY_LEARNING_PROBLEM`：Tool 能力已经接线，但 HotpotQA、TriviaQA 与 AIME 的 Stable Zero graph 没有自然选择可选 retrieval/computation Tool。
- `MODEL_CAPABILITY_LIMIT`：HealthBench 的 2 题样本在 evaluator-valid execution 下 raw_score 仍低；该样本不足以证明更窄的架构缺陷。
- `ARCHITECTURE_DEFECT`（已修复）：WebShop JSON/native-action 不匹配和 ALFWorld 缺少 environment actor 的 terminal condition 已通过最小 executor/terminal adaptation 修正并独立复跑。

## 最终判定

```text
FLOWSTEER_CORE_PRESERVED = YES

MODEL_POOL_EXPANDED = YES
MULTI_MODEL_WORKFLOW_READY = YES
DEEP_WORKFLOW_READY = YES
COLLABORATION_DIVERSITY_READY = YES

QA_TOOL_REGISTRY_READY = YES
QA_DATABASE_SELECTION_READY = YES
QA_TOOL_USE_VALIDATED = NO

ALFWORLD_REACT_READY = YES
WEBSHOP_REACT_READY = YES

CODING_AGENT_READY = YES
SWEBENCH_CODING_WORKFLOW_READY = NO

SKILL_END_TO_END_READY = YES
SKILL_SUMMARY_VALIDATED = NO

ALL_DATASETS_STABLE_ZERO_COMPLETE = NO
CORRECT_WRONG_DEMOS_COMPLETE = NO

MICRO_TRAINING_EXECUTED = NO
LEARNING_TREND_OBSERVED = NO

GITHUB_ARCHITECTURE_BACKUP = NO

READY_FOR_FORMAL_MULTIDATASET_TRAINING = NO
```

`GITHUB_ARCHITECTURE_BACKUP = NO` 表示当前 branch 在本地可恢复，但生成报告时远端认证不可用；正常认证的 `git push` 成功前，不得表述为已推送。

## 备份状态

- 当前架构 commit：`9f38701`（`architecture: validate multidataset stable-zero runtime`）。
- 本地统一备份 branch/tag：`backup/multidataset-stablezero-arch-20260820`。
- 数据集恢复 branch：HotpotQA、TriviaQA、AIME 2026、HealthBench Professional、WebShop v2、ALFWorld v2、SWE-bench coding preflight 各有独立命名的本地 branch。
- Patch：`/ssd1/iclr/1/0001-architecture-validate-multidataset-stable-zero-runti.patch`。
- Bundle：`/ssd1/iclr/1/FlowSteer-multidataset-stablezero-20260820.bundle`。
- GitHub push：`BLOCKED`；使用仓库现有配置执行非交互式 push 时，没有可用的 GitHub credential。未把 token 写入命令、remote、日志或 commit。

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
