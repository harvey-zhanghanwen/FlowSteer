# Receipt-backed Failure Demo Reporting Protocol

This document defines the reporting format for later dataset-adaptation
stages. It is an offline evidence/reporting boundary; it does not authorize a
new rollout, Tool call, evaluator call, training step, or architecture change.

## 1. Dataset-specific metric and target

Every report must name the official/reference task target and primary metric
before classifying failures. Do not replace a native metric with a generic
Accuracy, EM, or F1. Examples:

- extractive QA: accepted answer set plus official EM/token F1;
- AIME: canonical integer answer plus Accuracy;
- HealthBench Professional: signed rubric items plus raw and Professional
  length-adjusted aggregate; there is no single reference answer;
- interactive environments: native terminal success/reward receipt; and
- SWE-bench: official harness resolution receipt.

Evaluator-only targets must never be copied into Director/Agent prompts,
Canvas feedback, Tool observations, training inputs, or public Git reports.

## 2. Population and mutually exclusive taxonomy

The report must state the exact population, condition/version, and rule that
selects a Wrong Demo. Primary categories must be mutually exclusive and sum
to the reported Wrong Demo population. Use terminal precedence: a
`max_rounds` or other terminal failure is counted once in the terminal class,
while its earlier Canvas/Director fault remains a subcategory or first
observable failure receipt.

Only use categories applicable to the dataset and saved evidence. Candidate
families include:

- retrieval or Tool execution;
- Director action decoding/parsing;
- Canvas edit, graph validity, relation construction, or topology;
- Agent communication transport/runtime;
- Agent response/reasoning quality when directly supported by a receipt;
- verification when an actual verifier/evaluator boundary exists;
- formatting, extraction, or canonicalization;
- terminal control, including missing `FINISH` and `max_rounds`;
- evaluator failure; and
- provider or collection failure.

For every listed category, report count, percentage of Wrong Demos, and
percentage of the complete fixed population. A supported zero must be written
as `0`; an inapplicable or unobservable category must be written as `N/A`.
Never manufacture a demo for a zero/N/A category. Historical attempts that
were superseded by a successful final receipt are reported separately and do
not enter the final taxonomy.

## 3. Per-category representative demos

Each non-zero category must include at least one deterministic representative
(up to three when distinct subcategories matter). Each complete private demo
contains:

1. task/sample ID, exact condition, model/policy/evaluator versions, and source
   receipt paths;
2. complete task input;
3. the official target: accepted answers, rubric items, environment objective,
   or harness target as appropriate;
4. Direct and AgentGraph final outputs plus native metrics;
5. every Director input, raw response, parsed action, and decoding receipt;
6. every Canvas edit, graph snapshot/revision, and feedback receipt;
7. every Agent identity, model, free-text contract, actual rendered input,
   output, and provider receipt;
8. declared relations and actual upstream/CommunicationEnvelope messages;
9. Output Agent inbox and final response;
10. ReAct StructuredAction–Observation and Tool receipts when they actually
    exist; otherwise an explicit no-Tool/N/A record;
11. terminal and evaluator receipts; and
12. the first observable failure point, affected action/Agent, later receipt
    span, and final result.

ReAct is an execution mode, not an Agent role. Do not invent Thought, Tool
Action, Observation, verifier, or communication receipts that were not saved.

## 4. Causal attribution

Use the established term **first observable failure** for the earliest saved
fault boundary. A temporally earlier rejection does not by itself prove that
it caused a later semantic score loss, especially when the workflow recovered
and reached `FINISH`. Reports must distinguish:

- observed fault receipt;
- supported propagation through explicit artifacts/messages; and
- unresolved upstream cause.

For rubric-only response shortfalls, the evaluator establishes which terminal
criteria were missed. Without direct intermediate evidence, do not attribute
the miss to an invented hidden reasoning or verification step.

## 5. Public and evaluator-private outputs

Tracked public reports may contain aggregate counts, task IDs, scores,
termination, action names, topology, first-error turn, and redacted receipt
summaries. Full conversations, evaluator targets, physician/reference
responses, grader explanations, full candidate outputs, Director prompts,
Agent inputs/outputs/contracts, communication bodies, and provider request
details remain in ignored evaluator-private artifacts.

For HealthBench Professional, the current implementation is:

- tracked summary:
  `reports/healthbench_professional_official_v1/failure_taxonomy_report_zh.md`;
- evaluator-private full demos:
  `artifacts/healthbench_professional_official_v1/evaluation/evaluator_private/failure_taxonomy_private_zh.md`; and
- private manifest:
  `artifacts/healthbench_professional_official_v1/evaluation/evaluator_private/failure_taxonomy_manifest.json`.

The private artifacts are diagnostic outputs only. They must not be used as
Director/Agent input or training data.
