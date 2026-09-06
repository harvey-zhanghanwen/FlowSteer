# MBPP+ TTB training source map

## Scope and immutable starting point

This training adaptation starts from
`backup/mbppplus-best-v6-20260906` at
`66e66f02c908d16d097ceb75d65745d90a155d29`. The backup reference is not
modified. MBPP+ is not one of SkillFlow's seven reported IID joint-training
tasks, so this condition is a project dataset adaptation rather than a
bit-exact SkillFlow reproduction.

The project design document specifies Action-Masked One-Pass GRPO. SkillFlow's
formal primary training method is Tempered Trajectory Balance (TTB); GRPO is a
baseline/ablation. This branch selects TTB and keeps GRPO disabled. The losses
are not mixed.

## Direct reuse

| Boundary | Reused implementation |
|---|---|
| Canvas and progressive execution | Existing FlowSteer `AgentWorkflowEnv`, `AgentGraphOrchestrator`, and execute-on-edit runtime |
| Agent execution | Existing `LiveSmokeBackend` runtime with the MBPP public-test ReAct adapter |
| Trajectory receipts | Existing `AgentGraphRolloutCollector` and exact SGLang token/log-prob receipts |
| MBPP training reward | SkillFlow `training/reward.py::code_test_pass_rate`, exposed by the thin `mbpp_training_adapter` |
| TTB learner | SkillFlow `training/gflownet_trainer.py`, `training/flow_metrics.py`, `training/backward_policy.py`, and `training/trajectory.py`, adapted in `src/interactive/ttb_trainer.py` to FlowSteer trajectory receipts |
| LoRA publication | Existing `SGLangPolicyPublisher`, derived from SkillFlow's external SGLang adapter publication boundary |
| Supervisor lifecycle | Existing `SGLangSupervisorManager`, adapted from SkillFlow's Supervisor manager |

The Executor remains frozen. The trainable state is the Director theta LoRA,
the backward-policy phi LoRA, and the separate partition-function parameter Z.

## Formal TTB condition

- 250 optimizer steps;
- 7 questions and 4 on-policy trajectories per question (effective batch 28);
- maximum trajectory horizon `T=12`;
- edge log-probability is averaged only over structured action tokens;
- reasoning tokens are context only;
- `beta=1.0`, `epsilon_min=0.1`, AdamW learning rate `1e-4`, maximum gradient
  norm `3.0`, and KL coefficient `0.01`;
- theta LoRA is rank 64 / alpha 128 over `q_proj`, `k_proj`, `v_proj`, and
  `o_proj`;
- phi LoRA uses the main configuration table's rank 16 / alpha 32 over
  `q_proj` and `v_proj`;
- every tenth step writes the formal interval checkpoint.

The paper's main configuration table does not report LoRA dropout or AdamW
weight decay. This condition uses dropout `0.05` and weight decay `0.01` from
the released SkillFlow trainer and labels both as source-code details rather
than paper hyperparameters.

SkillFlow Appendix L also states phi rank 32. The source is ambiguous; this
condition records both values and explicitly selects the main table's rank 16.

## Project implementation

SkillFlow reports four A100-80GB GPUs and about 70 GB peak memory per GPU, but
does not define a TP/DP layout, ZeRO stage, worker count, sampling parameters,
or GPU role mapping. The following are therefore project implementation, not
paper claims:

- three distinct physical roles (learner, SGLang rollout Supervisor, and
  gradient replica);
- read-only launch gating requiring no existing compute process and at least
  70 GiB free on every selected GPU;
- 28 concurrent jobs inside one rollout batch and no cross-step prefetch;
- two-replica token-cost gradient partitioning;
- a per-step recovery checkpoint;
- pause/drain, candidate load, canary, route switch, and previous-adapter unload;
- discovery of the exact post-switch SGLang `weight_version` from an
  AgentGraph canary followed by a second drained route bind; this exact value
  becomes the next step's behavior-policy admission coordinate;
- W&B as a mandatory commit barrier;
- single-phase unconstrained full-action generation.
- explicit CUDA allocator peaks for the two in-process training replicas; the
  separately managed SGLang Supervisor GPU is reported by the resource
  inventory/W&B system monitor rather than added to those allocator counters.

The last choice is necessary because the local Hugging Face learner currently
recomputes raw-model log-probabilities. It does not implement the identical
JSON-Schema grammar-conditioned normalization used by the best-v6 inference
profile. Constrained best-v6 rollouts therefore cannot be silently admitted to
this raw-logit TTB loss.

## Transaction and admission boundaries

One step is committed only after:

1. the previous theta adapter is the active behavior route;
2. exactly four TTB-eligible trajectories are admitted for each of seven
   distinct questions;
3. terminal public-test rewards are available;
4. theta, phi, and Z each have non-zero gradients and non-zero parameter
   updates;
5. the recovery checkpoint is complete;
6. the new theta adapter is loaded, canaried, and made the active route; and
7. the completed step receipt is successfully logged to online W&B before the
   runner admits the next rollout batch.

The trainer's publication transaction is committed immediately after item 6;
W&B is the runner's next-rollout barrier. If online logging fails, the runner
stops with the published recovery checkpoint intact and does not sample the
next policy batch.

Only then may the next on-policy rollout batch start. Filtered rollouts are
persisted but never optimized. `--stop-after-step 1` executes one real update
and then collects a post-update AgentGraph canary under the newly published
theta route. `--prepare-only` starts no model, API, W&B run, or training.

## Not enabled

MACE, joint Bayesian posterior inference, EVSI, paired interventions, Skill
retrieval/evolution/publication, GRPO, and any large-scale evaluation are not
enabled. Their planned interfaces in the design document are not reported as
implemented or executed by this runner.
