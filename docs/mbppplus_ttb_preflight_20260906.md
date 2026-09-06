# MBPP+ TTB launch preflight — 2026-09-06

## Status

`blocked_preflight`; optimizer updates: `0`; W&B run: not started; model and
SGLang processes started by this runner: none.

The current evidence manifest is
`artifacts/mbppplus_ttb_v1/runs/preflight-20260906-ttb-v3/run_manifest.json`.
The artifact directory is intentionally not tracked by Git.

## Ready

- immutable starting point: `backup/mbppplus-best-v6-20260906` at
  `66e66f02c908d16d097ceb75d65745d90a155d29`;
- separate TTB training branch and entry point;
- Qwen3.5-9B model and tokenizer paths;
- MBPP project-adaptation training population: 241 tasks;
- validation population: 62 tasks;
- SkillFlow public-test reward adapter;
- θ/φ/Z learner, checkpoint, publication acknowledgement, and next-policy
  admission wiring;
- 178 targeted offline tests.

## Blocking launch conditions

- online W&B authentication environment `WANDB_API_KEY` is absent;
- selected GPU 3 has 52,309 MiB free and two existing compute processes;
- selected GPU 4 has 51,023 MiB free and one existing compute process;
- selected GPU 5 has 9,767 MiB free and two existing compute processes.

Every local GPU had an existing compute process at the check time, and no three
devices met the configured exclusive, at-least-71,680-MiB gate. The gate is a
project resource-control implementation based on the approximately 70-GB/GPU
peak reported by SkillFlow; the three-role GPU mapping is not claimed to come
from the paper.

## Next admissible launch

After online W&B authentication and three non-conflicting GPUs pass the gate,
the first live invocation must use `--stop-after-step 1`. It must prove one
complete on-policy rollout → reward → TTB loss → backward → AdamW step →
θ/φ/Z non-zero update → θ LoRA publication → exact post-switch AgentGraph
canary → publication acknowledgement → W&B record. The 250-step invocation is
not admissible until that receipt exists.
