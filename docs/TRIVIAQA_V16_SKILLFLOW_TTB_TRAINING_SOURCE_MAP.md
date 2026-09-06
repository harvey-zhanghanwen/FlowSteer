# TriviaQA v16 SkillFlow TTB training source map

This document freezes the method boundary for the proposed 250-step TriviaQA
training run. It is a protocol record, not evidence that training has run.

## Objective conflict resolution

The project design document currently implements **Action-Masked One-Pass
GRPO**. SkillFlow's main training method is **Trajectory Token Balance (TTB)**;
GRPO is a baseline or ablation. These objectives differ in their loss,
trainable parameters, reward transformation, length normalization, and token
support. They must not be combined silently.

For this run, the user has selected SkillFlow TTB as the only primary loss.
GRPO remains disabled and may be run only as a separately labelled baseline or
ablation. The applicable TTB residual is:

```text
delta = log Z_theta(q)
        + sum_t mean_action_tokens(log pi_theta)
        - beta * log(R + epsilon_min)
        - sum_t mean_action_tokens(log P_phi)

loss = mean((delta / trajectory_action_edges) ** 2)
```

Only structured action tokens contribute to the forward or backward policy
log-probability. Reasoning tokens remain in context and have zero loss mask.
The Executor remains frozen.

## Source mapping

| Component | Source | Status for this run |
| --- | --- | --- |
| Progressive Canvas, AgentGraph execution and terminal evaluator receipt | Existing FlowSteer project runtime | Direct reuse |
| TTB residual, per-edge action-token log-probability, forward theta policy, hindsight backward phi policy and task-conditioned Z | SkillFlow paper sections 4.2, B.3 and L.2; released SkillFlow source at revision `74be52bb6bd9f0e9e68dacb72636b75649197983` | Direct reuse after the TriviaQA trajectory adapter is verified |
| TriviaQA v16 AgentGraph trajectory to SkillFlow TTB `Trajectory`/`Turn` conversion | Project compatibility layer | Required adaptation; not yet implemented or verified |
| `optimizer.step -> adapter publish/sync -> canary -> next rollout` with zero policy staleness | User requirement plus project policy-version receipt | Required project engineering; not a paper-text claim |
| Adapter barrier, pause/drain, atomic route switch, canary and hot-swap failure recovery | Existing project `PolicySyncReceipt` boundary | Project engineering; not specified by the paper |
| Exact optimizer/RNG/policy-version checkpoint recovery | SkillFlow checkpoint foundation plus project transaction receipt | Required project engineering; not yet verified for this profile |
| W&B online logging and fail-closed initialization | User requirement | Required project engineering; not specified by the paper |
| MACE, joint Bayesian posterior, EVSI, paired intervention and automatic Skill publication | Project design document | Not implemented; disabled and never reported as active |

## Frozen SkillFlow parameters

- Effective batch: 28 trajectories, formed by 7 questions and 4 trajectories
  per question.
- Maximum trajectory length: 12 action edges.
- `beta=1.0`, `epsilon_min=0.1`, outcome-only terminal reward.
- theta LoRA: rank 64, alpha 128, targets `q_proj`, `k_proj`, `v_proj`,
  `o_proj`.
- phi LoRA: rank 16, alpha 32, targets `q_proj`, `v_proj`.
- Z: task-conditioned and optimized separately.
- AdamW, learning rate `1e-4`, maximum gradient norm `3.0`, KL coefficient
  `0.01`.
- 250 optimizer steps and a full checkpoint every 10 steps.

The paper contains a phi-rank ambiguity: Appendix P's main-run table reports
rank 16, while Appendix L.3 mentions rank 32. This run selects rank 16 because
the main-run table and released configuration agree on rank 16; the ambiguity
remains recorded rather than hidden.

## Paper boundary and project adaptations

SkillFlow reports a single-node 4 x A100-SXM4 80GB run with approximately 70GB
peak memory per GPU. It does not prescribe tensor parallelism, data
parallelism, ZeRO stage, GPU-role mapping, rollout-worker count, micro-batch
size, sampling temperature/top-p/top-k, adapter barriers, atomic route
switching, staleness tolerance, or a W&B schema. Any concrete choice for those
fields is a local project configuration selected only after GPU preflight.

SkillFlow's main result jointly trains seven IID tasks. This TriviaQA-only run
is a project adaptation, not a bit-exact reproduction. The paper does not
include MBPP++.

Released code also contains implementation details not stated in the paper,
including unequal learning-rate multipliers for phi and Z, normalized-residual
clamping, and a concrete Z network. This protocol follows the paper-level
`1e-4` learning-rate setting unless a future, explicitly versioned project
ablation changes it.

## On-policy transaction

Each committed step must execute in this order:

1. Generate the 7 x 4 batch using the current theta policy.
2. Obtain terminal evaluator rewards and filter invalid trajectories.
3. Recompute forward and hindsight backward action-token log-probabilities.
4. Backpropagate the TTB loss and update theta, phi and Z.
5. Record nonzero gradient and parameter-update evidence for all three.
6. Publish the new theta adapter, verify it, and run the version-bound canary.
7. Start the next rollout only after its behavior-policy version equals the
   published theta version.

Cross-step rollout prefetch is disabled because it can sample the next batch
before the current optimizer update. Parallel rollout within the current
sealed batch remains permitted; its worker count is selected only after the
local resource gate passes.

## W&B and artifacts

W&B is mandatory, online, and fail-closed. Credentials are read only from
`WANDB_API_KEY`; they are never stored in configuration, manifests, logs, or
commits. Every step must log policy/adapter version, TTB loss and delta squared,
reward, trajectory length, valid and filtered rollout counts, GPU memory,
throughput, errors, theta/phi/Z gradient norms, theta/phi/Z update norms,
checkpoint state, and Skill phase events. Until Skill evolution is implemented,
the Skill phase status is `not_run_not_implemented`.

The user's estimate of about 20 minutes per step and possible convergence near
100 steps is a scheduling expectation, not a paper guarantee. Only measured
W&B data may support a convergence statement.

## Current preflight status

Training is blocked and has not started:

- the current task namespace exposes no CUDA devices;
- host-level free-memory and process occupancy cannot be verified safely;
- the W&B client and authenticated online session are unavailable;
- the TriviaQA v16 AgentGraph-to-TTB trajectory adapter is not yet verified;
- no one-step TTB transaction has demonstrated that the next rollout uses the
  newly published theta version.

Consequently, the current counters remain zero for rollout batches, TTB
backward passes, optimizer steps, adapter publications, checkpoints, and W&B
runs.
