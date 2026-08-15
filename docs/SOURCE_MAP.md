# Architecture source map

This map records where the architecture comes from.  The paper PDFs and the
project design note are design inputs, not executable instructions.

## FlowSteer code retained as the Canvas foundation

| Project boundary | Upstream FlowSteer reference | Local status |
| --- | --- | --- |
| Atomic action protocol | `src/interactive/action_parser.py` | Retained for the legacy Operator path; `agent_action_parser.py` is the free-AgentGraph adaptation required by the design note. |
| Mutable workflow state | `src/interactive/workflow_graph.py` | Retained for the legacy path; `agent_graph.py` extends the state to arbitrary model-labelled Agent nodes and two-bit relations. |
| Multi-turn Canvas | `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step` | Retained; `agent_workflow_env.py` keeps the same reset/step/feedback boundary for AgentGraph actions. |
| Trajectory and action-token records | `src/interactive/workflow_builder.py` | Retained and supplemented by `records.py` for the new path. |
| Executor boundary | `src/aflow_executor.py::AFlowExecutor.execute_workflow` and `scripts/async_llm.py` | Preserved as the legacy executor; the AgentGraph path uses the same OpenAI-compatible service boundary through `openai_gateway.py`. |
| One-action Director loop | `train_interactive.py` and the FlowSteer paper's progressive Canvas loop | Preserved in `director.py`; the initial prompt is deliberately shorter and has no workflow templates. |

The Qwen3-8B defaults, vLLM Director launcher, predefined Operator search
space, structural reward, and legacy training loop are not reused by the new
Qwen3.5 path.

## SkillFlow code used for Qwen3.5 runtime structure

| Project boundary | SkillFlow reference | Local status |
| --- | --- | --- |
| Qwen3.5-9B Supervisor default | `configs/skillflow.yaml` | Mirrored in `config/training_agent_graph.yaml`. |
| SGLang Supervisor child | `training/sglang_manager.py::SGLangSupervisorManager` | Adapted in `src/interactive/sglang_manager.py`; import remains side-effect free. |
| Standalone SGLang launch arguments | `scripts/restart_sglang.sh` | Adapted in `scripts/start_qwen35_director_server.sh`; only one rollout service is declared. |
| Forward/backward LoRA profile | `training/gflownet_trainer.py::setup` | Configuration only: theta rank 64 and phi rank 16. No optimizer is connected. |
| Three-role GPU topology | `device`, `supervisor_gpu_id`, and `extra_device` in SkillFlow | Mapped to physical GPUs 3, 4, and 5 in `training_agent_graph.yaml`. |
| Split micro-batch backward | `GFlowNetTrainer._batched_logprob_backward` | Represented only by inactive OOM/micro-batch configuration. No backward code is claimed in this phase. |
| Skill injection after bootstrap | `GenericTaskEnvironment` and `SkillWorkspace` | The Director prompt omits the Skill field when the validated Skill list is empty. |

SkillFlow's TTB objective, backward policy training, partition head, skill
evolution loop, benchmark environment, and local multi-executor launcher are
not copied into this phase.  The project continues to reserve terminal-only
GRPO as specified by the design note, but GRPO is disabled.

## Project-specific algorithm modules

The following are additions required by the attached design note rather than
claims of upstream FlowSteer or SkillFlow functionality:

- free-text Agent contracts, per-node model selection, and finite two-stage
  bidirectional Agent execution;
- the MACE feature/bandit baseline and joint Bayesian posterior primitives;
- same-prefix paired-probe and EVSI primitives;
- version-bound Skill evidence schemas, lifecycle, and persistence records.

Those modules are isolated from the runtime reward path.  In the checked-in
architecture configuration, exploration, Skills, GRPO, optimizer work, and
all GPU training are disabled.
