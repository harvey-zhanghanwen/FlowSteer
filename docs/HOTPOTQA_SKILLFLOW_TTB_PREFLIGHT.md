# HotpotQA SkillFlow TTB method boundary

The requested SkillFlow-main run has **not started**. Its primary loss must be
Tempered Trajectory Balance (TTB), not the project's Action-Masked One-Pass GRPO.
The latter remains only a separately versioned baseline/ablation. The two
objectives must not be combined or reported under one condition.

The user-provided wording "Trajectory Token Balance" is retained as provenance,
but the paper's formal term is "Tempered Trajectory Balance." SkillFlow TTB
scores only structured action tokens. Reasoning tokens remain
context with a zero loss mask. Each edge uses the mean action-token
log-probability. Current-policy rollouts are on-policy and the Executor is
frozen. The trainable state is theta LoRA, the task-conditioned partition
function Z, and phi LoRA, each with optimizer participation.

The frozen paper settings and their provenance are recorded in
`config/training_hotpotqa_skillflow_ttb.yaml`. In particular, the main table's
phi rank 16 is selected while the appendix rank 32 discrepancy remains
explicit. HotpotQA-only training is a project adaptation; it is not a bit-exact
reproduction of SkillFlow's seven-IID-task joint run.

## Current blockers

1. The AgentGraph trajectory-to-TTB action-token adapter and theta/Z/phi joint
   training path are not implemented in the backed-up FlowSteer version.
2. A real two-step closure has not run. It must prove non-zero gradient and
   parameter updates for theta/Z/phi, successful theta publication/canary, and
   step N+1 receipts produced by theta N+1.
3. This execution environment cannot observe host GPU memory/process ownership
   and has no mapped `/dev/nvidia*`, so no non-conflicting GPU plan is resolved.
4. The selected training environment currently lacks W&B and no online W&B
   credential is configured. Formal training is fail-closed on this condition.

The adapter publication barrier, route switch, zero-staleness policy, W&B
extensions, and eventual GPU role mapping are project engineering additions;
they are not attributed to SkillFlow.pdf.
