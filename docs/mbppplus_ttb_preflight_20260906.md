# MBPP+ TTB launch preflight — 2026-09-06

## Status

The earlier candidate-B preflight was `blocked_preflight`. After the complete
FlowSteer/SkillFlow source comparison, the authoritative status is now
`blocked_method_conflict`; optimizer updates: `0`; W&B run: not started; model
and SGLang processes started by this runner: none.

The current method-guard evidence manifest is
`artifacts/mbppplus_ttb_v1/runs/method-gate-20260906-v2/run_manifest.json`.
The artifact directory is intentionally not tracked by Git.

## Ready

- immutable starting point: `backup/mbppplus-best-v6-20260906` at
  `66e66f02c908d16d097ceb75d65745d90a155d29`;
- separate TTB training branch and entry point;
- Qwen3.5-9B model and tokenizer paths;
- MBPP project-adaptation training population: 241 tasks;
- validation population: 62 tasks;
- SkillFlow public-test reward adapter;
- W&B SDK and an existing authenticated W&B credential store (no online run
  was created by prepare-only);
- θ/φ/Z learner, checkpoint, publication acknowledgement, and next-policy
  admission wiring;
- 179 targeted offline tests.

## Blocking launch conditions

- the project has not accepted either Action-Masked One-Pass GRPO or TTB as
  its mutually exclusive primary objective;
- neither candidate has passed the required real Step-1 acceptance run;

- selected GPU 3 has 52,309 MiB free and two existing compute processes;
- selected GPU 4 has 51,023 MiB free and one existing compute process;
- selected GPU 5 has 9,767 MiB free and two existing compute processes.

Every local GPU had an existing compute process at the check time, and no three
devices met the configured exclusive, at-least-71,680-MiB gate. The absent
`WANDB_API_KEY` environment variable is not a blocker because the W&B SDK has
an existing authenticated credential; the live runner will still fail closed
unless `wandb.init(mode="online")` succeeds. The GPU gate is a
project resource-control implementation based on the approximately 70-GB/GPU
peak reported by SkillFlow; the three-role GPU mapping is not claimed to come
from the paper.

## Next admissible launch

If the user accepts candidate B (TTB), its source-mapped implementation is
ready, online W&B authentication succeeds, and three non-conflicting GPUs pass
the gate, the first live invocation must use `--stop-after-step 1`. It must
prove one
complete on-policy rollout → reward → TTB loss → backward → AdamW step →
θ/φ/Z non-zero update → θ LoRA publication → exact post-switch AgentGraph
canary → publication acknowledgement → W&B record. The 250-step invocation is
not admissible until that receipt exists.
