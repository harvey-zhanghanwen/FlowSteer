# HotpotQA formal Step-0 static preflight

## Executive conclusion

This phase improved the existing architecture without starting a new model
service, rollout, paid API request, GRPO pass, backward pass, optimizer step,
LoRA update, or trained-policy publication.  It closes the four static
compatibility gaps identified after architecture-v5:

1. deterministic materialization of a never-updated Qwen3.5-9B Director LoRA;
2. a HotpotQA-train-only immutable task/rollout schedule and exact cursor;
3. adapter plus optimizer-state continuation from formal Step 1 onward; and
4. one pause/drain transaction for adapter validation, Director route switch,
   old-adapter release, rollback, and resume.

The implementation is grounded in the concrete SkillFlow and existing
FlowSteer paths recorded in `docs/SOURCE_MAP.md`.  It does not add a second
workflow framework, a fixed role taxonomy, a required topology, structural
reward, communication reward, or a new Skill path.

The code-level preconditions now pass.  The operational formal Step 0 is still
not instantiated: no initial adapter was written, no live SGLang activation
was attempted, and no formal experiment schedule/cursor was published.  The
current truthful status is therefore:

```text
ARCHITECTURE_RUNTIME_CHAIN_COMPLETE = YES
STATIC_FORMAL_STEP0_PRECONDITIONS_IMPLEMENTED = YES
FORMAL_POLICY_STEP_000000_MATERIALIZED = NO
FORMAL_POLICY_STEP_000000_LIVE_ACTIVATED = NO
FORMAL_TRAINING_SCHEDULE_PUBLISHED = NO
GRPO_OR_WEIGHT_UPDATE_PERFORMED = NO
STEP0_TO_STEPN_LEARNING_VALIDATED = NO
SKILL_EVOLUTION_VALIDATED = NO
READY_FOR_GRPO = NO
READY_FOR_NEXT_DATASET = NO
```

## What changed

| Boundary | Files | Upstream boundary reused | Local necessity | Runtime status |
| --- | --- | --- | --- | --- |
| Untrained formal policy | `src/interactive/hotpot_step0.py`, `scripts/materialize_hotpotqa_step0.py` | SkillFlow deterministic initial-policy builder and bind-before-save checkpoint gate; existing Qwen3.5 PEFT loader | SkillFlow stores forward/backward adapters plus Z; this Director exposes one SGLang `theta` adapter | No-model preflight passed; adapter not materialized |
| Frozen Hotpot training order | `src/interactive/hotpot_training_schedule.py`, `scripts/freeze_hotpot_training_schedule.py` | SkillFlow frozen sequence, ordered provider, exact cursor, and attempt progress | Bind the existing aligned Hotpot train order and task-local rollout ordinals without re-splitting | Real 512-record split resolved in memory; no experiment artifact published |
| Exact optimizer continuation | `src/interactive/smoke_trainer.py`, `scripts/train_agentgraph_smoke.py` | SkillFlow immutable policy+optimizer checkpoint and exact restore identity | Reuse the existing one-`theta` PEFT checkpoint format while requiring the immediately preceding policy/step | Unit tested; no optimizer constructed or stepped in this phase |
| Atomic serve-route transition | `src/interactive/policy_sync.py`, `scripts/train_agentgraph_smoke.py` | SkillFlow Supervisor load → model-list verify → canary → generation switch → old unload, with rollback | The active Director route lives in `SGLangReceiptDirectorClient`, so switch/rollback callbacks must run inside the publisher gate | Unit tested with simulated control plane; no live call made |

The formal-profile flag is opt-in.  Historical smoke behavior remains
unchanged unless `grpo.exact_optimizer_continuation` is explicitly true.
Formal mode requires an explicit behavior adapter, saves optimizer state from
Step 1, and rejects Step 2+ without the immediately preceding optimizer state.

## Initial-policy truth boundary

The existing `theta_smoke_step_000001` had a real cross-dataset smoke update
and is retained only as a warm-start diagnostic policy.  It is not renamed or
silently reclassified as HotpotQA Step 0.

The new materializer defines, but has not written:

```text
policy_version = qwen35-9b-hotpot-step-000000
adapter_name   = theta_hotpot_step_000000
policy_step    = 0
optimizer_updates = 0
```

Its default command only validates configuration.  Model loading and file
creation require the explicit materialization option.  The saved policy
receipt, when later created, must state `training_performed=false` and
`optimizer_updates=0`; activation uses the same truth values and additionally
states `policy_published=false` because loading an existing initial adapter is
not a trained-policy publication.

## Frozen Hotpot-only schedule boundary

The schedule reads the already-aligned `data/agentgraph_v1` records and does
not create another split.  Its invariants are:

- only records whose dataset key is `hotpotqa` and split is `train`;
- ordered train identity fixed before result-dependent execution;
- validation/test IDs rejected;
- each optimizer slot bound to one declared train position and task ID;
- each task's grouped rollout ordinals fixed as `0..K-1`;
- schedule and cursor are write-once;
- a cursor can advance only the exact next step and can be restored exactly.

A no-write real-data preflight resolved 512 HotpotQA training records and a
two-step/two-rollout hypothetical schedule (four rollout coordinates).  That
was an interface check only.  It did not select or publish the future formal
experiment schedule and did not start training.

## Atomic policy transition boundary

The previous runner could finish adapter publication and then enter a second
pause/drain interval to update the Director route.  That left an avoidable
boundary in which publication and routing were not one transaction.  The new
path performs the following under one gate:

```text
pause admission
→ drain admitted rollouts
→ load/verify/canary candidate
→ switch Director policy+adapter route
→ unload prior adapter
→ resume admission
```

On failure after a route switch, it restores the previous route before
removing the candidate.  Receipts distinguish whether route switch/rollback
succeeded and whether the operation represented training publication or only
activation of an already-existing Step-0 adapter.

## Search space and architecture status

The current search space still lets the local Qwen3.5-9B Director choose:

`Agent count × free-text contract × executor model × directed relation ×`
`Output identity × continuation/FINISH`.

The runtime can execute ordinary DAGs, fan-in/fan-out, parallel independent
blocks, and a bounded two-stage reciprocal block.  The free contract can state
objective, expected inputs/dependencies, artifact, and completion condition.
The communication envelope records only facts known by the runtime: source,
target, message type, artifact body, graph revision, and target dependency.
`confidence` and `evidence_refs` were not added because the current Executors
do not produce verified values for them.

The legal topology space must not be confused with learned behavior.  The
latest complete 128-task run (architecture-v3) produced 122 singleton graphs,
six two-node chains, no 3+ node graph, and no heterogeneous multi-model graph
within a workflow.  Its AgentGraph score was 67.97 EM / 80.23 F1 versus the
fixed Direct 72.66 / 82.08.  Architecture-v4's 12-task result was sampling
confounded; architecture-v5 fixed that coordinate defect but deliberately did
not run another model evaluation.  These facts do not justify forcing multiple
Agents, templates, role enums, or topology reward.  They currently support a
Director policy-learning hypothesis, not another hand-authored workflow rule.

The earlier Training-ready Step-0 diagnostic remains separate historical
evidence: development-128 reached 73.44/81.62 and untouched-32 reached
71.88/83.62, but communication masking did not demonstrate causal upstream
use and malformed Output remained.  This phase did not generate a new score,
so it claims no accuracy gain.

## Model and Skill boundaries

- Flow-Director remains local Qwen3.5-9B (`supervisor_theta`).  No API model
  can substitute for it.
- The already canaried eight-model Hotpot executor catalog and exact model-list
  receipts remain in `MODEL_CATALOG_AUDIT.md`; this phase did not discover or
  canary another model and did not alter the frozen catalog.
- MACE, posterior/EVSI, paired probes, and Skill schemas remain isolated future
  method components.  No Skill was discovered, summarized, validated,
  activated, retrieved, or injected.  No MACE/Bayesian/Skill value entered the
  terminal reward.

## Verification

- 54 focused tests covering initial-policy materialization, schedule/cursor,
  policy sync, runner wiring, exact optimizer continuation, and trajectory
  collection: passed.
- Full dependency-light unit/regression suite, excluding the unrelated
  optional pandas dataset-preparation module: 201 passed.
- Ruff on every changed Python file: passed.
- Default initial-policy CLI preflight: passed with `model_load_performed=false`,
  `optimizer_or_backward_performed=false`, `will_write=false`.
- Real aligned Hotpot schedule no-write preflight: 512 train records resolved;
  no artifact written and `training_started=false`.

## Deferred work

The following requires a later explicit training authorization and is not
represented as complete:

1. materialize the untouched `policy_step_000000` adapter;
2. activate and canary it transactionally on the task-owned SGLang service;
3. choose and publish the exact Hotpot-only Step-1…N schedule/cursor;
4. connect that schedule to the formal runner;
5. collect same-task/same-condition grouped rollouts;
6. run action-masked terminal-only GRPO, backward, optimizer update, publish,
   post-update canary, and cursor commit in that order;
7. evaluate every checkpoint on fixed tasks and produce the complete learning,
   workflow, collaboration, routing, and stability curves;
8. only after its evidence gates are wired, evaluate Skill candidate → paired
   evidence → independent validation → ACTIVE → ON/OFF behavior.

`HOTPOTQA_STEP0_TO_STEPN_MULTIAGENT_SKILL_REPORT.md` is intentionally not
created yet: doing so now would imply that the deferred training and Skill
experiments happened.
