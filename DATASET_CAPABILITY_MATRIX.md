# Dataset Capability Matrix

Recorded from the current repository configuration, dataset registries,
runtime adapters, evaluator code, and saved reports on 2026-08-19.  The seven
targets are defined jointly by `config/datasets_agentgraph.yaml` and the
dataset keys consumed by `scripts/train_agentgraph_smoke.py`.  Legacy
FlowSteer benchmark entries outside this list are not part of the current
multidataset AgentGraph scope.

| Dataset | Task type | Evaluator and primary metric | Comparable baseline | External tool | ReAct | Coding Agent | Skill type / current status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | Static multi-hop QA | `hotpotqa.official.answer.v1`; normalized EM and answer token F1 | Local Qwen3.5-9B over the same ten passages | No external tool in the closed-context protocol | No | No | Evidence routing, bridge decomposition, verification; runtime retrieval exists, but no HotpotQA Skill is `ACTIVE` |
| TriviaQA | Static factual QA | `triviaqa.official.answer.v1`; maximum normalized EM/F1 over accepted aliases | Local Qwen3.5-9B with the same public retrieval observations | Existing SkillFlow retrieval is deterministic prefetch, not a model-driven Tool loop | Partial: retrieval observations only | No | Retrieval/query/evidence handoff; existing candidates are `CANDIDATE` or `RETIRED`, none `ACTIVE` |
| AIME 2026 | Mathematical reasoning | `skillflow.protocol-v10.static.integer.v1`; strict integer exact match / accuracy | Local Qwen3.5-9B single-call integer submission | None currently; calculator/Python/symbolic execution not wired | No | No | Derivation, independent verification, computation-tool selection; dataset Skills disabled and no validated candidate |
| HealthBench Professional | Open-ended healthcare QA | `openai.simple-evals.healthbench.v1`; public rubric mean raw score | Local Qwen3.5-9B single healthcare response | No Executor tool; GPT-4.1 is evaluator-only | No | No | Clinical evidence coverage and verification; dataset Skills disabled and no paired evidence |
| WebShop | Interactive web environment | `skillflow.ragen_adapter.v2`; official environment return and success | Single Qwen3.5-9B ReAct policy using the same environment and action budget | WebShop benchmark actions only: `search[...]` and `click[...]` | Yes, through the live benchmark environment | No | Search/refinement, attribute comparison, navigation, stopping; runtime Skill interface exists but no `ACTIVE` WebShop Skill |
| ALFWorld | Interactive embodied environment | `skillflow.ragen_adapter.v2`; simulator terminal `won` / success | Single Qwen3.5-9B ReAct policy with the same game and action budget | ALFWorld admissible environment commands | Adapter prepared; live Stable Zero not yet run | No | Subgoal decomposition, state-dependent action selection, failed-action recovery; no evidence-gated Skill |
| SWE-bench Verified | Software engineering / coding | `swebench.harness.v1`; official Docker harness resolved rate | Single Coding Agent under the same repository, tools, and test budget | Repository read/search/edit, shell and test tools are required but not yet wired into AgentRuntime | Required, not yet implemented | Required, not yet implemented | Repository inspection, localization, patch/test/revision; no coding Skill evidence |

## Current evaluation state

| Dataset | Fixed evaluation state | Current saved result | Stable Zero interpretation |
| --- | --- | --- | --- |
| HotpotQA | 128/128 evaluator-valid | Direct EM 72.66%, F1 82.08%; AgentGraph EM 75.00%, F1 84.44% | End-to-end inference/evaluator chain is complete for the frozen project split |
| TriviaQA | 128/128 evaluator-valid | Direct EM 51.56%, F1 57.90%; AgentGraph EM 52.34%, F1 61.80% | Chain is complete, but 12 Director rollouts exhausted `max_rounds`; model-driven retrieval Tool use is not yet proven |
| AIME 2026 | 30/30 evaluator-valid official 2026 tasks | Direct 1/30 (3.33%); AgentGraph 13/30 (43.33%) | Completion and strict integer evaluator chain are complete; computation tools are absent |
| HealthBench Professional | 128/128 evaluator-valid | Direct mean raw score 0.1318; AgentGraph 0.2075 | Public simple-evals-compatible local score only, not the private leaderboard metric |
| WebShop | 126/128 evaluator-valid | Direct success 24.22%; AgentGraph strict success 22.66% | Canary passed, but two operational failures and one terminal failure prevent a complete fixed-128 claim |
| ALFWorld | Dataset/configuration prepared | No Direct or AgentGraph result | `STABLE_ZERO = FAIL`: no live episode, trajectory, or terminal evaluator receipt yet |
| SWE-bench Verified | Dataset/configuration prepared | No official resolved result | `STABLE_ZERO = FAIL`: official Docker harness preflight is blocked and iterative Coding Agent execution is absent |

## Protocol boundaries

- The 128-held-out/512-training views are deterministic project splits.  They
  must not be described as full official leaderboard evaluations, except for
  the separate official 30-task AIME 2026 test view.
- QA results must remain separated into closed-context Direct,
  closed-context AgentGraph, and Tool-enabled AgentGraph protocols.
- WebShop and ALFWorld receive reward only from their benchmark environment.
- SWE-bench receives `resolved` only from the official harness; a generated
  patch, an LLM judgement, or a proxy test is not a resolved instance.
- All seven formal configurations currently have production Skills disabled.
  `ACTIVE Skill = 0`; candidates cannot be presented as deployed capability.

## Immediate capability gaps

1. Add a registry-backed, receipt-bearing Tool execution boundary to the
   existing AgentRuntime without replacing progressive Canvas editing.
2. Replace TriviaQA deterministic prefetch with an explicitly separated,
   model-driven `search/read/complete` Tool protocol when Tool-enabled results
   are reported.
3. Run a fixed ALFWorld canary that proves game identity, admissible actions,
   state transition, terminal `won`, and trajectory persistence.
4. Repair the remaining WebShop operational failures before claiming a
   complete fixed-condition Stable Zero.
5. Implement repository workspace inspection, patching, targeted tests, error
   feedback, and bounded iterative repair for SWE-bench inside AgentGraph.
6. Keep every dataset Skill disabled until randomized paired evidence,
   independent validation, version compatibility, and the evidence gate make
   it `ACTIVE`.
