# MBPP+ TTB training-data source map

This materializer is the data-only adaptation for MBPP+ v6 TTB training. It
is directly carried over from the accepted MBPP+ v6 training branch and reuses
the existing AgentGraph task schema and split writer. It does not change the
AgentGraph, Canvas, Agent runtime, rollout, TTB objective, or policy-sync
implementations, and it does not start training.

Local provenance is
`train/mbppplus-v6-skillflow-grpo-300step-20260906` at `41be76a`, limited to
`config/datasets_mbppplus_training_v1.yaml` and
`scripts/prepare_mbppplus_training_data.py`. None of that branch's GRPO
trainer, runner, or training configuration is imported into the TTB branch.

This is a **project adaptation**, not a bit-exact SkillFlow paper split:
SkillFlow's reported main run jointly trains seven IID tasks, and the paper
does not include MBPP+. The intended consumer uses SkillFlow's formal
Tempered Trajectory Balance objective (seven questions times four on-policy
trajectories per optimizer step); this catalog only supplies a disjoint MBPP
question population. It does not add or mix the MD's Action-Masked One-Pass
GRPO loss.

## Sources and selection

| Boundary | Upstream source | Local use |
|---|---|---|
| Training candidates | FlowSteer public `FlowSteer-Dataset/train/train_12k.jsonl` | Select `source=mbpp`; the 2,000 balanced rows contain 374 unique tasks. Identical repetitions are removed by source `meta.task_id` while preserving first-occurrence order. |
| Validation candidates | FlowSteer public `FlowSteer-Dataset/eval/mbpp.jsonl` | Preserve the published order of all 128 candidates before overlap exclusion. |
| Held-out identity population | EvalPlus `MbppPlus-v0.2.0.jsonl` | Use all 378 canonical tasks—not only the fixed 100—to exclude a FlowSteer candidate if its task ID, normalized problem text, or Python entry point overlaps. |
| Formal test population | Existing `mbpp-plus-fixed-100@1` catalog | Remains external and unchanged. The training materializer writes an empty local `test.jsonl` and never opens the fixed-100 files for writing. |
| Training reward | SkillFlow `training/reward.py::code_test_pass_rate` | `metadata.training_evaluator` identifies this directly reused implementation. As in SkillFlow's code-generation reward path, every non-empty line of `extra.test` becomes one `code_test_pass_rate` case. This supplies the terminal reward consumed by TTB; it does not define the TTB loss. |
| Public task schema and atomic split publication | FlowSteer project `scripts/prepare_agentgraph_datasets.py::{_compat_record, SplitWriters}` | Reused unchanged to preserve the unified AgentGraph loader contract and publish train/validation/test/manifest together. |

With the frozen source releases, exclusion leaves 241 training tasks and 62
validation tasks. The materializer validates these counts and fails on source
or overlap drift rather than silently changing the population.

All 378 EvalPlus MBPP+ identities are excluded before train/validation output
is written. Consequently, none of the existing fixed-100 test tasks can enter
either split. The fixed-100 directory remains an external, immutable reference;
the generated local `test.jsonl` is deliberately empty.

## Public and evaluator-private fields

Every public `TaskRecord` keeps `metadata.dataset_key=mbpp_plus`, allowing the
existing v6 MBPP+ Workflow and Tool routing to be reused. Training IDs use
`mbpp-training:<source_task_id>` and validation IDs use
`mbpp-validation:<source_task_id>`. `metadata.training_population` is
`flowsteer_mbpp_train` or `flowsteer_mbpp_validation`.

The public `question` contains the original FlowSteer problem followed by the
complete public test program. The same raw program is recorded as
`metadata.public_test_program`; the reward adapter applies SkillFlow's
non-empty-line split at evaluation time. Public records set both
`ground_truth` and the SkillFlow-compatible `answer` field to `null`.

`evaluator_private.jsonl` is joined by the aligned `task_id` and contains:

- the entry point;
- `test_cases` split into non-empty source lines, plus the unchanged raw test
  program in `test`; and
- the FlowSteer reference solution.

The reference solution and evaluator payload are never copied into public
metadata, Agent input, Agent communication, or trajectory records. The public
assertions are intentionally visible, matching the MBPP code-generation task
protocol and the existing MBPP+ public-test Tool boundary.

This exact upstream line-level reward has a known limitation for a test program
whose setup statement and assertion occupy separate lines: each case executes
in a fresh namespace. The adapter records that behavior instead of silently
changing SkillFlow's reward semantics.

## TTB consumption boundary

- Question sampling, four on-policy trajectories per question, structured
  action-token log-probabilities, backward-policy probabilities, `Z`, LoRA
  updates, publication, and the next-policy rollout are responsibilities of the
  TTB trainer/runner, not this materializer.
- The fixed-100 EvalPlus population is evaluation-only and must never be passed
  to the TTB sampler.
- `training_started=false` and `evaluation_started=false` in the generated
  manifest describe this prepare-only operation; they are not evidence that a
  rollout or optimizer step occurred.
- MACE, Bayesian posterior updates, EVSI, paired intervention, Skill evolution,
  and Skill publication are outside this data adapter and are not claimed here.

## Prepare-only invocation

The default EvalPlus source location is
`data/mbpp_plus/source/MbppPlus-v0.2.0.jsonl`. If an already-downloaded copy is
kept elsewhere, pass it without changing the catalog:

```bash
MBPPPLUS_SOURCE_PATH=/absolute/path/MbppPlus-v0.2.0.jsonl \
python scripts/prepare_mbppplus_training_data.py \
  --catalog config/datasets_mbppplus_training_v1.yaml
```

The command only materializes aligned JSONL and its manifest. It does not run
rollout, reward computation, backward, optimizer update, or evaluation.
