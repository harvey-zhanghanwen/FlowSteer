# HotpotQA dual-source training preflight

The requested training run has **not started**. No primary loss is selected.
FlowSteer/the project MD's Action-Masked One-Pass GRPO and SkillFlow's Tempered
Trajectory Balance (TTB) are separate candidate objectives. They must not be
combined, renamed as one another, or reported under one condition.

The paper's formal term is "Tempered Trajectory Balance." SkillFlow TTB scores
only structured action tokens; reasoning is context, each edge uses mean
action-token log-probability, and theta LoRA, phi LoRA, and Z are jointly
optimized. None of those statements applies to the GRPO candidate.

The complete commit-exact mapping and conflicts are recorded in
`docs/FLOWSTEER_SKILLFLOW_TRAINING_SOURCE_MAP.md`. HotpotQA-only training is a
project adaptation; it is not a bit-exact reproduction of SkillFlow's
seven-IID-task joint run.

## Current blockers

1. The primary objective is unresolved. TTB and one-pass GRPO remain disabled.
2. If TTB is selected, its HotpotQA scalar reward (EM versus the release code's
   token F1) and AgentGraph action-token adapter remain unresolved.
3. A real one-step update plus post-update rollout has not run. It must prove
   non-zero gradients/updates for the selected trainable state and a successful
   publication/canary using the new theta version.
4. This execution environment cannot observe host GPU memory/process ownership
   and has no mapped `/dev/nvidia*`, so no non-conflicting GPU plan is resolved.
5. The selected training environment currently lacks W&B and no online W&B
   credential is configured. Formal training is fail-closed on this condition.

The adapter publication barrier, route switch, zero-staleness policy, W&B
extensions, and eventual GPU role mapping are project engineering additions;
they are not attributed to SkillFlow.pdf.
