#!/usr/bin/env python3
"""Run the fixed TriviaQA Direct-vs-AgentGraph architecture validation.

This is an inference/evaluation runner.  It reuses SkillFlow's public
``RetrievalIndex.search/read`` implementation and official TriviaQA answer
scorer, then reuses FlowSteer's existing progressive Canvas collector.  It
does not run training, GRPO, backward, optimizer, policy sync, or Skill update.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_hotpotqa_round import (
    _aggregate,
    _atomic_jsonl,
    _git_state,
    _paired_rows,
    _read_jsonl,
    _stable_zero_check,
    _collect_graph,
)
from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _write_json,
)
from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentCallRecord,
    AgentRequest,
    ExecutionPhase,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.graph_diagnostics import aggregate_trajectory_diagnostics
from src.interactive.qa_retrieval import (
    QARetrievalReceipt,
    SkillFlowQARetriever,
    augment_task_with_retrieval,
    build_keyword_query,
    receipt_from_mapping,
)
from src.interactive.records import TaskRecord
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import (
    TRIVIAQA_ANSWER_EVALUATOR_VERSION,
    evaluate_task,
)


class TriviaRoundError(RuntimeError):
    """The fixed TriviaQA evaluation condition could not complete."""


DIRECT_CONTRACT = (
    "Answer the TriviaQA question using the supplied public retrieval observations. "
    "Return only the shortest name, title, date, number, location, or phrase that "
    "answers the question, enclosed in <answer> and </answer>."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def validate_trivia_config(config: Mapping[str, Any]) -> None:
    validate_agent_graph_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    bounded = _mapping(config.get("triviaqa_evaluation"), "triviaqa_evaluation")
    retrieval = _mapping(bounded.get("retrieval"), "triviaqa_evaluation.retrieval")
    director = _mapping(config.get("director"), "director")
    grpo = _mapping(config.get("grpo"), "grpo")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")
    deployment = _mapping(config.get("deployment"), "deployment")
    gpu = _mapping(config.get("gpu"), "gpu")
    skills_enabled = skills.get("enabled") is True
    skill_evaluation_mode = (
        skills.get("enabled") is False
        or (
            skills_enabled
            and isinstance(skills.get("store_path"), str)
            and bool(str(skills.get("store_path")).strip())
            and type(skills.get("retrieval_top_k")) is int
            and int(skills.get("retrieval_top_k")) > 0
            and type(skills.get("current_epoch")) is int
            and int(skills.get("current_epoch")) >= 1
            and deployment.get("exploration_beta") == 0.0
            and deployment.get("allow_forced_probes") is False
            and deployment.get("active_skills_only") is True
            and deployment.get("require_version_compatible_skills") is True
        )
    )
    checks = {
        "experiment.phase": experiment.get("phase") == "triviaqa_evaluation",
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "dataset_key": bounded.get("dataset_key") == "triviaqa",
        "split": bounded.get("split") == "validation",
        "selection": bounded.get("selection") == "sequential",
        "rollouts_per_task": bounded.get("rollouts_per_task") == 1,
        "direct_model_id": bounded.get("direct_model_id") == "qwen3.5-9b-local",
        "retrieval.enabled": retrieval.get("enabled") is True,
        "retrieval.mode": retrieval.get("mode")
        == "deterministic_question_query_prefetch",
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "director.execute_on_edit": director.get("execute_on_edit") is True,
        "grpo.enabled": grpo.get("enabled") is False,
        "optimizer_passes": grpo.get("optimization_passes_per_rollout_batch") == 0,
        "optimizer_updates": grpo.get("max_optimizer_updates") == 0,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.evaluation_mode": skill_evaluation_mode,
        "gpu.training_enabled": gpu.get("training_enabled") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ConfigurationError(
            "TriviaQA round violates fixed evaluation bounds: " + ", ".join(failed)
        )
    sample_count = bounded.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= 128
    ):
        raise ConfigurationError(
            "triviaqa_evaluation.sample_count must be between 1 and 128"
        )
    concurrency = bounded.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError("triviaqa_evaluation.concurrency must be positive")
    terminal = _mapping(config["agent_graph"], "agent_graph").get(
        "terminal_protocol_by_source", {}
    )
    if not isinstance(terminal, Mapping) or terminal.get("triviaqa") != "exact_single_answer_tag":
        raise ConfigurationError(
            "TriviaQA requires terminal_protocol_by_source.triviaqa="
            "exact_single_answer_tag"
        )


def _paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config["storage"], "storage")
    names = {
        "selected": "selected_tasks_path",
        "retrieval": "retrieval_receipts_path",
        "direct": "direct_predictions_path",
        "trajectories": "trajectories_path",
        "failures": "failures_path",
        "paired": "paired_results_path",
        "wrong": "wrong_demos_path",
        "manifest": "manifest_path",
        "preflight": "preflight_receipt_path",
        "report_json": "report_json_path",
        "report_markdown": "report_markdown_path",
    }
    return {name: _resolve(root, str(storage[field])) for name, field in names.items()}


def _select_tasks(
    config: Mapping[str, Any], root: Path, selected_path: Path
) -> tuple[TaskRecord, ...]:
    data = _mapping(config["data"], "data")
    bounded = _mapping(config["triviaqa_evaluation"], "triviaqa_evaluation")
    source = _resolve(root, str(data["validation_path"]))
    candidates = tuple(
        record
        for record in iter_task_records(source, expected_split="validation")
        if _dataset_key(record) == "triviaqa"
    )
    count = int(bounded["sample_count"])
    if len(candidates) < count:
        raise TriviaRoundError(
            f"aligned validation contains {len(candidates)} TriviaQA tasks, expected {count}"
        )
    candidates = candidates[:count]
    if selected_path.exists():
        frozen = tuple(
            iter_task_records(selected_path, expected_split="validation")
        )
        if len(frozen) != len(candidates):
            raise TriviaRoundError("frozen TriviaQA selection has the wrong size")
        for expected, actual in zip(candidates, frozen, strict=True):
            if (
                expected.task_id != actual.task_id
                or expected.question != actual.question
                or expected.ground_truth != actual.ground_truth
            ):
                raise TriviaRoundError(
                    "frozen TriviaQA selection differs from aligned validation"
                )
        return frozen
    _atomic_jsonl(
        selected_path,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **record.to_dict()}
            for record in candidates
        ],
    )
    return candidates


def _retrieval_by_task(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in _read_jsonl(path):
        task_id = value.get("task_id")
        if isinstance(task_id, str) and task_id not in result:
            result[task_id] = value
    return result


def _prepare_retrieval(
    tasks: Sequence[TaskRecord],
    config: Mapping[str, Any],
    path: Path,
) -> dict[str, QARetrievalReceipt]:
    bounded = _mapping(config["triviaqa_evaluation"], "triviaqa_evaluation")
    retrieval = _mapping(bounded["retrieval"], "triviaqa_evaluation.retrieval")
    cached_rows = _retrieval_by_task(path)
    receipts: dict[str, QARetrievalReceipt] = {}
    rows: dict[str, dict[str, Any]] = {}
    for task in tasks:
        cached = cached_rows.get(task.task_id)
        if cached is None:
            continue
        if cached.get("question") != task.question:
            raise TriviaRoundError(
                f"cached retrieval question differs for {task.task_id}"
            )
        receipt_value = cached.get("retrieval")
        if not isinstance(receipt_value, Mapping):
            raise TriviaRoundError(f"cached retrieval receipt is malformed: {task.task_id}")
        restored = receipt_from_mapping(receipt_value)
        if restored.query != build_keyword_query(task.question):
            continue
        receipts[task.task_id] = restored
        rows[task.task_id] = cached

    missing = [task for task in tasks if task.task_id not in receipts]
    if missing:
        with SkillFlowQARetriever(
            index_path=str(retrieval["index_path"]),
            skillflow_source=str(retrieval["skillflow_source"]),
            search_limit=int(retrieval["search_limit"]),
        ) as retriever:
            for ordinal, task in enumerate(missing, start=1):
                receipt = retriever.retrieve(build_keyword_query(task.question))
                receipts[task.task_id] = receipt
                rows[task.task_id] = {
                    "schema_version": "flowsteer.triviaqa.public_retrieval.v1",
                    "task_id": task.task_id,
                    "question": task.question,
                    "retrieval": receipt.to_dict(),
                    "created_at": _utc_now(),
                }
                _atomic_jsonl(
                    path,
                    [rows[item.task_id] for item in tasks if item.task_id in rows],
                )
                print(
                    f"retrieval {len(receipts)}/{len(tasks)} "
                    f"({ordinal}/{len(missing)} new): {task.task_id}",
                    flush=True,
                )
    return receipts


async def _direct_one(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    index: int,
    *,
    model_id: str,
    protocol: str,
    run_label: str,
) -> dict[str, Any]:
    model = backend.registry.require_model(model_id)
    provider = backend.registry.provider_for(model_id)
    run_id = f"{run_label}-direct-{index:04d}"
    request = AgentRequest(
        request_id=f"{run_id}:direct:single",
        run_id=run_id,
        graph_revision=0,
        problem=task.question,
        agent=AgentNode(
            "direct",
            model_id,
            DIRECT_CONTRACT,
            role_family="question_answering",
        ),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
        is_output_agent=True,
    )
    response = await backend.runtime.gateway.generate(request)
    execution = execution_record_from_call(AgentCallRecord(request, response))
    evaluation = await evaluate_task(task, response.text)
    return {
        "schema_version": "flowsteer.triviaqa.direct_prediction.v1",
        "task_id": task.task_id,
        "task": task.to_dict(),
        "condition": "direct_local_qwen35_9b",
        "protocol": protocol,
        "model_id": model_id,
        "provider_id": provider.provider_id,
        "provider_model": model.model_name,
        "final_answer": response.text,
        "evaluation": asdict(evaluation),
        "execution": execution.to_dict(),
        "completed_at": _utc_now(),
    }


async def _collect_direct(
    backend: LiveSmokeBackend,
    tasks: Sequence[TaskRecord],
    config: Mapping[str, Any],
    path: Path,
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    bounded = _mapping(config["triviaqa_evaluation"], "triviaqa_evaluation")
    model_id = str(bounded["direct_model_id"])
    protocol = str(bounded["direct_protocol"])
    existing: dict[str, dict[str, Any]] = {}
    task_by_id = {task.task_id: task for task in tasks}
    for value in _read_jsonl(path):
        task = task_by_id.get(value.get("task_id"))
        evaluation = value.get("evaluation")
        if (
            task is not None
            and value.get("model_id") == model_id
            and value.get("protocol") == protocol
            and isinstance(evaluation, Mapping)
            and evaluation.get("valid") is True
            and evaluation.get("evaluator_version")
            == TRIVIAQA_ANSWER_EVALUATOR_VERSION
        ):
            existing[task.task_id] = value

    semaphore = asyncio.Semaphore(int(bounded["concurrency"]))

    async def run(index: int, task: TaskRecord) -> tuple[TaskRecord, Any]:
        async with semaphore:
            try:
                result = await _direct_one(
                    backend,
                    task,
                    index,
                    model_id=model_id,
                    protocol=protocol,
                    run_label=str(config["experiment"]["name"]),
                )
                return task, result
            except BaseException as exc:
                return task, exc

    jobs = [
        asyncio.create_task(run(index, task))
        for index, task in enumerate(tasks)
        if task.task_id not in existing
    ]
    for completed in asyncio.as_completed(jobs):
        task, result = await completed
        if isinstance(result, BaseException):
            failures.append(
                {
                    "task_id": task.task_id,
                    "condition": "direct_local_qwen35_9b",
                    "stage": "generation_or_evaluator",
                    "error": _safe_error(result),
                    "recorded_at": _utc_now(),
                }
            )
        else:
            existing[task.task_id] = result
        _atomic_jsonl(
            path,
            [existing[item.task_id] for item in tasks if item.task_id in existing],
        )
        manifest["direct_progress"] = {"completed": len(existing)}
        _write_json(Path(manifest["manifest_path"]), manifest)
    return existing


def _accepted_answer(record: TaskRecord) -> str:
    payload = record.metadata.get("evaluator_payload", {})
    if isinstance(payload, Mapping):
        answers = payload.get("accepted_answers")
        if isinstance(answers, Sequence) and not isinstance(answers, (str, bytes)):
            for answer in answers:
                if str(answer).strip():
                    return str(answer)
    ground_truth = str(record.ground_truth)
    return ground_truth.split("|", 1)[0].strip()


def _report(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    direct = _aggregate(rows, "direct")
    graph = _aggregate(rows, "agentgraph")
    failures = Counter(str(row["failure_type"]) for row in rows)
    wrong = [row for row in rows if float(row["agentgraph"]["exact_match"]) < 1.0]
    return {
        "schema_version": "flowsteer.triviaqa.round_report.v1",
        "dataset": "TriviaQA",
        "project_split": "validation",
        "sample_count": len(rows),
        "metric_scope": "SkillFlow_Formal_Protocol_10_compatible_answer_EM_token_F1",
        "retrieval_boundary": {
            "implementation": "SkillFlow RetrievalIndex.search/read",
            "mode": "deterministic_question_query_prefetch",
            "same_context_for_direct_and_agentgraph": True,
            "accepted_answers_visible_to_models": False,
            "protocol_10_interactive_search_read_parity": False,
        },
        "direct_local_baseline": direct,
        "agentgraph": graph,
        "agentgraph_minus_direct": {
            "exact_match": graph["strict_exact_match"] - direct["strict_exact_match"],
            "token_f1": graph["strict_token_f1"] - direct["strict_token_f1"],
        },
        "graph_search_diagnostics": aggregate_trajectory_diagnostics(trajectories),
        "failure_types": dict(sorted(failures.items())),
        "wrong_demo_count": len(wrong),
        "typical_wrong_demo_task_ids": [row["task_id"] for row in wrong[:10]],
        "policy_version": str(config["director"]["behavior_policy_version"]),
        "policy_adapter": config["director"].get("behavior_adapter_name"),
        "model_catalog_path": str(config["agent_graph"]["model_catalog_path"]),
        "training_performed": False,
        "skill_injection_performed": bool(config.get("skills", {}).get("enabled", False)),
        "skill_evaluation_mode": (
            "memory_on_active_only"
            if bool(config.get("skills", {}).get("enabled", False))
            else "memory_off"
        ),
        "skill_store_path": config.get("skills", {}).get("store_path"),
        "completed_at": _utc_now(),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    direct = report["direct_local_baseline"]
    graph = report["agentgraph"]
    delta = report["agentgraph_minus_direct"]
    failure_lines = "\n".join(
        f"- `{name}`：{count}" for name, count in report["failure_types"].items()
    ) or "- 无"
    skill_sentence = (
        "本轮只检索通过证据门控的 ACTIVE Skill，并将其作为可拒绝的 prompt prior；"
        "没有 forced intervention 或 Skill 更新。"
        if report.get("skill_injection_performed")
        else "本轮没有注入 Skill。"
    )
    return f"""# TriviaQA 第一轮架构验证

固定项目 validation：**{report['sample_count']}** 题。Direct 与 AgentGraph 使用同一批题、同一 Qwen3.5-9B、同一公开检索观察和同一终局 evaluator。本轮未执行训练、GRPO、反向传播、优化器更新、LoRA 发布、贝叶斯后验更新或 Skill 发布。{skill_sentence}

评测采用 SkillFlow Formal Protocol 10 兼容的答案归一化：对 accepted answers 取最大 token F1，并报告 normalized exact match。检索复用 SkillFlow `RetrievalIndex.search/read`；当前适配采用确定性问题查询预取，不等同于 SkillFlow 的模型驱动多轮 `search/read/complete` 完整协议。

| 条件 | 完成 | evaluator 有效 | 严格 EM | 严格 F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | {direct['completed']} | {direct['evaluator_valid']} | {100 * direct['strict_exact_match']:.2f}% | {100 * direct['strict_token_f1']:.2f}% |
| Progressive AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * graph['strict_exact_match']:.2f}% | {100 * graph['strict_token_f1']:.2f}% |

AgentGraph − Direct：**{100 * delta['exact_match']:+.2f} EM**，**{100 * delta['token_f1']:+.2f} F1**。

## Failure Types

{failure_lines}
"""


async def run_trivia_round(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    prepare_only: bool = False,
    canary_only: bool = False,
) -> Mapping[str, Any]:
    root = Path(project_root).expanduser().resolve()
    resolved_config = Path(config_path).expanduser().resolve()
    config = load_yaml(resolved_config)
    validate_trivia_config(config)
    paths = _paths(config, root)
    selected = _select_tasks(config, root, paths["selected"])
    active_original = selected[:2] if canary_only else selected
    receipts = _prepare_retrieval(active_original, config, paths["retrieval"])
    active = tuple(
        augment_task_with_retrieval(task, receipts[task.task_id])
        for task in active_original
    )
    failures = _read_jsonl(paths["failures"])
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.triviaqa.round_manifest.v1",
        "status": "prepared" if prepare_only else "runtime_preflight",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "manifest_path": str(paths["manifest"]),
        "git_start": _git_state(root),
        "selected_task_ids": [task.task_id for task in selected],
        "active_task_count": len(active),
        "fixed_split": "validation",
        "training_enabled": False,
        "optimizer_updates": 0,
        "retrieval_receipts_completed": len(receipts),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    if prepare_only:
        manifest.update(status="prepared", completed_at=_utc_now())
        _write_json(paths["manifest"], manifest)
        return manifest

    try:
        backend = LiveSmokeBackend.from_config(config, root, evaluation_only=True)
        known_answer = await evaluate_task(
            active[0], f"<answer>{_accepted_answer(active[0])}</answer>"
        )
        if (
            not known_answer.valid
            or known_answer.metrics.get("exact_match") != 1.0
            or known_answer.metrics.get("token_f1") != 1.0
        ):
            raise TriviaRoundError("known-answer TriviaQA evaluator preflight failed")
        _write_json(
            paths["preflight"],
            {
                "evaluator_known_answer": asdict(known_answer),
                "policy_version": config["director"]["behavior_policy_version"],
                "policy_adapter": config["director"].get("behavior_adapter_name"),
                "model_catalog_path": config["agent_graph"]["model_catalog_path"],
                "retrieval_implementation": receipts[active[0].task_id].implementation,
                "completed_at": _utc_now(),
            },
        )
    except Exception as exc:
        manifest.update(
            status="failed_runtime_preflight",
            error=_safe_error(exc),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        raise

    manifest["status"] = "direct_baseline"
    _write_json(paths["manifest"], manifest)
    direct = await _collect_direct(
        backend, active, config, paths["direct"], failures, manifest
    )
    _atomic_jsonl(paths["failures"], failures)

    manifest["status"] = "agentgraph"
    _write_json(paths["manifest"], manifest)
    compatibility_config = dict(config)
    compatibility_config["hotpotqa_evaluation"] = config["triviaqa_evaluation"]
    trajectories = await _collect_graph(
        backend,
        active,
        compatibility_config,
        paths["trajectories"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    rows = _paired_rows(active, direct, trajectories)
    original_by_id = {task.task_id: task for task in active_original}
    for row in rows:
        original = original_by_id[row["task_id"]]
        row["question"] = original.question
        row["model_input"] = next(
            task.question for task in active if task.task_id == original.task_id
        )
        row["retrieval"] = receipts[original.task_id].to_dict()
    _atomic_jsonl(paths["paired"], rows)
    wrong = [row for row in rows if float(row["agentgraph"]["exact_match"]) < 1.0]
    _atomic_jsonl(paths["wrong"], wrong)
    stable_zero = _stable_zero_check(active, direct, trajectories)
    manifest["stable_zero"] = stable_zero
    if canary_only:
        manifest.update(
            status=(
                "stable_zero_confirmed"
                if stable_zero["passed"]
                else "failed_stable_zero"
            ),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        if not stable_zero["passed"]:
            raise TriviaRoundError("TriviaQA canary failed the Stable Zero chain")
        return manifest

    report = _report(rows, config, tuple(trajectories.values()))
    _write_json(paths["report_json"], report)
    paths["report_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_markdown"].write_text(
        _report_markdown(report), encoding="utf-8"
    )
    manifest.update(
        status="completed" if len(failures) == 0 else "completed_with_failures",
        direct_progress={"completed": len(direct)},
        agentgraph_progress={"completed": len(trajectories)},
        metrics={
            "direct": report["direct_local_baseline"],
            "agentgraph": report["agentgraph"],
            "delta": report["agentgraph_minus_direct"],
        },
        failure_type_counts=report["failure_types"],
        git_end=_git_state(root),
        completed_at=_utc_now(),
    )
    _write_json(paths["manifest"], manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/evaluation_triviaqa_round_01.yaml"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = asyncio.run(
            run_trivia_round(
                _resolve(PROJECT_ROOT, args.config),
                prepare_only=bool(args.prepare_only),
                canary_only=bool(args.canary_only),
            )
        )
    except (ConfigurationError, TriviaRoundError, RuntimeError, ValueError) as exc:
        print(f"TriviaQA round failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "active_task_count": manifest["active_task_count"],
                "metrics": manifest.get("metrics"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
