# HotpotQA Round-01 + QA-memory v16 source map

This profile starts from the highest complete, same-128, same-evaluator
HotpotQA AgentGraph condition, `hotpotqa_round_01_stable_zero`.  It does not
start from the later QA-memory v15 architecture profile.

## Frozen Round-01 fields

`config/evaluation_hotpotqa_round01_qa_memory_v16.yaml` keeps the following
sections equal to `config/evaluation_hotpotqa_round_01.yaml`:

- `experiment.seed` and `experiment.prompt_version`;
- the complete `director` mapping, including the Qwen3.5-9B adapter, minimal
  prompt profile, context/action limits, sampling parameters, round budget,
  execute-on-edit behavior, and history window;
- every original `agent_graph` field, including the six scalar Canvas actions,
  ten-Agent bound, free-text contracts, two-bit relations, seeded weighted
  Executor selection, reciprocal-block limit, output uniqueness, output
  reachability, and model catalog;
- the 128 sequential validation tasks, one rollout per task, concurrency four;
- `evaluation`, inactive `grpo`, inactive `policy_sync`, inactive
  `exploration`, inactive `skills`, and `deployment`.

The prepared v16 task projection `(task_id, question, ground_truth)` is exactly
equal, in order, to the saved Round-01 selection.  The aligned v16 records add
source provenance metadata only.

## Necessary QA-memory adaptations

| Adaptation | Source boundary | Reason |
| --- | --- | --- |
| `qa_embedding_retrieval` | SkillFlow-style normalized BGE index plus its bounded `search`/`read` action boundary | Bind the frozen 512 paired training-QA memories to `hotpotqa.qa_memory`; freeze cosine top-2 and Tool budgets; disable Web Search. |
| `required_evidence_tool_id` and `require_evidence_relation` | Existing FlowSteer AgentGraph Tool receipt and relation admission | Require a worker-owned successful search/read receipt to reach the Output Agent through an explicit graph relation before FINISH. |
| `director_feedback_mode: control_plane` | Existing FlowSteer control-plane feedback mode | Keep retrieved QA content out of Director observations; the Director receives execution state and receipt summaries only. |
| versioned experiment/storage paths | Existing HotpotQA evaluation driver | Keep the Round-01 result and QA-memory result independently recoverable. |
| aligned HotpotQA data paths | Existing deterministic 128-held-out/512-train adapter | Use the isolated train-only memory source while preserving the original validation task projection. |
| rollout GPU 0 | Deployment configuration only | Follow the requested inference-device assignment; this does not change the Director or AgentGraph search space. |

The Tool is available only to a worker Agent declared with
`execution_mode=react` and `allowed_tools=[hotpotqa.qa_memory]`.  ReAct remains
an Agent execution policy, not an Agent role or a fixed workflow topology.
When the worker's fully read top-k group is unsupported, the existing Output
path falls back to reasoning over the public task rather than copying a memory
answer.

## Excluded later-profile changes

The v16 profile deliberately does **not** carry over these v15 changes because
they are not required for QA-memory access and would alter the Round-01
architecture:

- model-admissible action-mask v3 constrained sampling;
- `exact_single_answer_tag` terminal protocol;
- `preserve_diagnose_repair_augment` recovery policy;
- `max_agents_per_subgraph` and Formatter requirements.

## Directed verification

- `tests/unit/test_hotpotqa_round01_qa_memory_v16_profile.py` verifies complete
  Director equality, exact original AgentGraph equality plus the three Tool
  boundary fields, inactive training sections, valid configuration, exact
  128-task projection, top-2 QA memory, and disabled Web Search.
- Profile, HotpotQA Tool, memory rebuild, and evaluation adapter suite:
  26 passed.
- AgentGraph Tool routing and Output fallback suite: 35 passed plus 18
  subtests.

These checks are model-free.  They do not constitute a new 128-sample score;
the historical best remains the evidence attached to its original Round-01
condition until the separately versioned v16 evaluation completes.

## Paired-QA memory examples

Every embedding document uses the manifest template
`Question: {paraphrase_question}\nAnswer: {paraphrase_answer_statement}`.  The
question and answer therefore remain paired inside one training-memory record.

| Source train task | Original QA | Indexed memory QA |
| --- | --- | --- |
| `hotpotqa:5a80d30655429938b61421fe` | Q: Are Manhattan West and Singer Building both projects in New York? A: `yes` | Q: Are Manhattan West and the Singer Building both projects in New York? A: The canonical answer is yes. Manhattan West and the Singer Building are both projects located in New York. |
| `hotpotqa:5a7e567b55429949594199a0` | Q: Who is the American internet entrepreneur who founded the company featured on 24 Hours on Craigslist? A: `Craig Newmark` | Q: Which American internet entrepreneur established the organization that was the subject of the 24 Hours on Craigslist program? A: The American internet entrepreneur who founded the company highlighted in 24 Hours on Craigslist is Craig Newmark. |
| `hotpotqa:5a77bd595542995d83181291` | Q: Between two tennis players Kim Clijsters and Mary Pierce, who is older? A: `Mary Pierce` | Q: Which of the two tennis players, Kim Clijsters or Mary Pierce, has the greater age? A: Mary Pierce is older than Kim Clijsters. |

The index contains 512 memories from 512 unique training tasks, with no cycled
record and no task-ID overlap with the frozen 128-item validation partition.
