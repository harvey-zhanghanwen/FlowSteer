#!/usr/bin/env python3
"""One bounded, real Action--Observation--Complete compatibility probe per model.

Reuse the deployed Gateway, new optional-tools adapter and SkillFlow calculator.
No HealthBench task, rubric, medical retrieval, training, or model fallback.
The default is prepare-only; --run-probes permits at most two calls per model.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from discover_models import fetch_models_with_receipt, models_url
from probe_model_capabilities import classify_probe_exception, write_capability_receipt
from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, ExecutionPhase
from src.interactive.config_loader import load_model_registry
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_clinical_tools import HEALTHBENCH_CALCULATOR_TOOL_ID, build_healthbench_clinical_tool_registry
from src.interactive.healthbench_tool_adapter import FrozenMedRAGBM25Corpus
from src.interactive.openai_gateway import OpenAICompatibleGateway


async def probe(registry, model_id):
    model = registry.require_model(model_id)
    provider = registry.provider_for(model_id)
    # Reuse the full registration/factory shape without opening the clinical
    # corpus. Only calculator is admitted below; no search/read backend runs.
    unused_corpus = FrozenMedRAGBM25Corpus("probe-unused", "probe-unused", 0, (), (), (), {})
    tools = build_healthbench_clinical_tool_registry(unused_corpus)
    gateway = OpenAICompatibleGateway(timeout_seconds=90, max_retries=0,
                                      default_max_tokens=4096, default_seed=20260906)
    adapter = HealthBenchClinicalReactExecutionAdapter(
        gateway=gateway, tool_registry=tools, max_turns=2, max_tool_calls=1,
        max_action_tokens=4096, enforce_state_conditioned_completion_admission=True,
    )
    request = AgentRequest(
        request_id=f"clinical-react-canary-{model_id}", run_id="clinical-react-canary-v1",
        graph_revision=0, problem="Capability test: calculate 17 * 19 + 23 with the registered calculator.",
        agent=AgentNode("tool_canary", model_id,
                        "Call the calculator once with expression 17 * 19 + 23. "
                        "Then use its Observation to complete with the numerical result as text.",
                        execution_mode="react", allowed_tools=(HEALTHBENCH_CALCULATOR_TOOL_ID,)),
        model=model, provider=provider, phase=ExecutionPhase.SINGLE, is_output_agent=True,
    )
    result = {"model_id": model_id, "provider_id": provider.provider_id,
              "model_name": model.model_name, "thinking_requested": model.metadata.get("chat_template_enable_thinking"),
              "max_model_calls": 2, "fallback_requests": 0}
    try:
        response = await adapter.execute(request)
        trace = response.metadata.get("react_trace", [])
        receipts = response.metadata.get("tool_receipts", [])
        observed = any(
            row.get("tool_id") == HEALTHBENCH_CALCULATOR_TOOL_ID
            and row.get("result", {}).get("value", {}).get("ok") is True
            and "346" in row.get("result", {}).get("value", {}).get("observation", "")
            for row in receipts
        )
        result.update(status="passed" if observed and "346" in response.text else "failed_completion",
                      final_output=response.text, tool_receipts=receipts, react_trace=trace,
                      model_calls=response.metadata.get("model_calls", []))
    except Exception as exc:
        result.update(status="failed", error=classify_probe_exception(
            exc, secret_values=(os.environ.get(provider.api_key_env or "", ""),)),
            react_trace=getattr(exc, "react_trace", ()),
            tool_receipts=getattr(exc, "tool_receipts", ()))
    return result


async def run(args):
    registry = load_model_registry(args.catalog)
    ids = args.model
    for model_id in ids:
        registry.require_model(model_id)
    if not args.run_probes:
        print(json.dumps({"models": ids, "max_total_model_calls": 2 * len(ids), "status": "prepared"}))
        return 0
    output = Path(args.output)
    if output.exists():
        raise ValueError("probe receipt already exists; do not repeat paid calls")
    providers = {registry.provider_for(model_id).provider_id: registry.provider_for(model_id) for model_id in ids}
    discovered = {}
    for provider in providers.values():
        key = os.environ.get(provider.api_key_env or "", "")
        models, _ = fetch_models_with_receipt(models_url(provider.endpoint), key, 30)
        discovered[provider.provider_id] = {row["id"] for row in models}
    for model_id in ids:
        model = registry.require_model(model_id)
        if model.model_name not in discovered[model.provider_id]:
            raise ValueError(f"model is absent from current /v1/models: {model_id}")
    # Serial probes avoid changing the running benchmark's configured concurrency.
    receipt = {"created_at": datetime.now(timezone.utc).isoformat(),
               "catalog": str(args.catalog), "max_total_model_calls": 2 * len(ids),
               "protocol": "optional-clinical-tools.react.calculator.v1", "probes": []}
    for model_id in ids:
        item = await probe(registry, model_id)
        receipt["probes"].append(item)
        write_capability_receipt(output, receipt)
        print(json.dumps({"model_id": model_id, "status": item["status"]}), flush=True)
    return 0 if all(row["status"] == "passed" for row in receipt["probes"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-probes", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
