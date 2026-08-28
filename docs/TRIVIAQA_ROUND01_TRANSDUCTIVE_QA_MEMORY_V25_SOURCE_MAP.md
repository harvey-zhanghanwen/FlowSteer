# TriviaQA Round-01 transductive QA-memory v25 source map

## Evaluation status

This condition is **transductive retrieval**, not official held-out TriviaQA.
Its memory corpus contains all 512 frozen train QA pairs and all 128 frozen
development-validation QA pairs. Therefore:

- `contains_evaluation_answers = true`
- `evaluation_memory_overlap_count = 128`
- `evaluation_regime = transductive_retrieval`
- `official_heldout_eligible = false`

The condition must not replace `triviaqa_round_01_stable_zero` as the verified
held-out best profile. Its reported metric label is
`transductive_retrieval_accuracy`.

## Reused architecture boundaries

| v25 boundary | Direct source | Adaptation |
|---|---|---|
| Director model, prompt, Canvas actions, sampling and fixed 128 order | `config/evaluation_triviaqa_round_01.yaml` and `config/evaluation_triviaqa_round01_qa_memory_v23.yaml` | Configuration-only reuse; Qwen3.5-9B remains the Director and sees no Tool payload. |
| Progressive Canvas editing | FlowSteer `src/interactive/agent_workflow_env.py` | No role sequence or fixed topology added. `ADD_AGENT`, relation editing, output selection and `FINISH` remain scalar Canvas actions. |
| Worker retrieval policy | `src/interactive/qa_tool_adapter.py::QARetrievalReactExecutionAdapter` | Existing `retrieval_first_parametric_fallback`: `search`, read the complete frozen top-k, then permit completion; parametric fallback is admitted only when retrieved evidence is declared unsupported. |
| ReAct completion contract | `src/interactive/react_execution.py::ToolReactExecutionAdapter` and SkillFlow `StructuredAction` | The model-visible completion guidance is rendered from the same request-local `completion_arguments_schema` used by validation. This removes the incompatible generic `value`-only example while retaining the existing StructuredAction wire. |
| Director ADD decoding | `src/interactive/director.py::director_live_action_parameter_json_schema_text` and the existing FlowSteer hierarchical action discriminator | Each registered execution profile is rendered as a complete object branch with its own required fields; the SGLang grammar no longer loses outer ADD fields through an `allOf` intersection. Agent IDs use the existing neutral `node_N` allocation. |
| Atomic MODIFY domain | `src/interactive/agent_workflow_env.py::model_admissible_action_targets` | `execution_mode` and `allowed_tools` are exposed only when the other current field forms a registered Runtime profile, and current values are excluded as no-ops. |
| Failure recovery | Existing `preserve_diagnose_repair_augment` policy in `AgentWorkflowEnv` | v25 enables the existing policy so a successful retrieval artifact cannot be deleted before a replacement artifact has taken over its lineage. |
| Search/read Tool | `src/interactive/qa_tool_adapter.py::build_qa_tool_registry` | The new index inherits the existing deterministic `search/read` interface and receipt schema. |
| Dense index | `src/interactive/triviaqa_qa_memory.py` and `src/interactive/triviaqa_embedding_index.py` | Reuses `TriviaQAQAMemoryRecord`, paired QA embedding text, local BGE encoder, L2 normalization and dot-product ranking. |
| Transductive corpus declaration | `src/interactive/triviaqa_transductive_qa_memory.py` | Necessary dataset adaptation: adds the 128 evaluated QA pairs, a source-membership sidecar and an explicit transductive manifest without modifying the train-only index. |
| Runtime index selection | `src/interactive/qa_tool_adapter.py::open_qa_tool_registry` | Minimal format dispatch selects `TriviaQATransductiveQAMemoryIndex` only when the manifest explicitly declares evaluation-answer inclusion and held-out ineligibility. |
| Report boundary | `scripts/evaluate_completion_benchmark_round.py::_triviaqa_qa_memory_index_summary` | Persists transductive source counts, overlap and eligibility fields; rejects ambiguous evaluation-answer manifests. |

## Control-plane and data-plane contract

- Director request: `allowed_tools=[]`; no query, top-k record or answer text.
- Retrieval owner: a Canvas worker Agent with
  `execution_mode=react` and `allowed_tools=[triviaqa.qa_memory]`.
- Execution order: worker `search` → worker reads the complete frozen top-k →
  evidence sufficiency decision → supported answer or parametric fallback.
- Communication: evidence artifact and Tool receipts travel through an explicit
  AgentGraph relation to the selected Output Agent.
- Web Search: disabled.
- Static prefetch: disabled.
- Roles/topology: unconstrained except for capability and evidence-routing
  invariants; no Retriever→Reasoner→Verifier→Formatter template is required.
- Failure recovery: preserve → diagnose → repair → augment; deletion remains
  gated by replacement takeover.

## Prepared-only boundary

`config/evaluation_triviaqa_round01_transductive_qa_memory_v25.yaml` is a
prepared evaluation configuration. The all-QA materialization and BGE index
must exist at `data/triviaqa_qa_memory_transductive_v1/index` before a canary or
full run. No model call, paraphrase generation, embedding build or accuracy
evaluation is performed by the v25 configuration tests.
