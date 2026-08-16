# HotpotQA Architecture v1 — Fixed Development Diagnostic

This is a 14-task architecture-development slice selected before v1 execution.
It contains bridge, comparison, positive-control, negative-complexity-control,
wrong-demo and malformed-terminal cases from the already-used dev128 set. It is
not an untouched score and is not used for GRPO.

## Metrics

| Condition | Valid | EM | F1 |
| --- | ---: | ---: | ---: |
| Local Qwen3.5-9B Direct, v1 seed | 14/14 | 42.86 | 46.43 |
| AgentGraph v1 warm-start diagnostic | 14/14 | 35.71 | 52.86 |
| AgentGraph − Direct | — | -7.14 | +6.43 |

Paired EM outcomes: 4 both correct, 1 AgentGraph-only, 2 Direct-only, and 7
both wrong. The v1 code substantially improved the same slice over the old
AgentGraph outputs (old graph: 14.29 EM / 22.94 F1), but did not beat its
same-run Direct baseline on EM.

The v1 Direct predictions were generated under seed 20260816 before the reuse
bug described below was found. They are now isolated in this version's own
artifact directory. Beginning with v2, the architecture experiment freezes the
original 20260815 Direct baseline through explicit read-only reuse.

## Workflow and routing behavior

| Behavior | v1 observation |
| --- | ---: |
| Singleton graph | 13/14 |
| Two-node chain | 1/14 |
| 3+ nodes | 0/14 |
| Parallel/fan-in/fan-out/reciprocal | 0/14 |
| Exact terminal wrapper | 14/14 |
| Executor nodes | 15 |
| `deepseek-v4-flash` nodes | 14 |
| `deepseek-v4-pro` nodes | 1 |
| Other catalog models | 0 |
| Mean Director turns | 3.64 |
| Total Executor calls | 16 |

Removing the old preferred-model hint did not produce capability-based routing:
the sorted catalog put DeepSeek first, and 14/15 nodes selected its first flash
arm. Multi-model routing is therefore not validated; this is a presentation
order bias, not evidence that DeepSeek is best.

The only multi-Agent graph was a correct comparison task. Its upstream envelope
had the expected source, target, artifact, target dependency and graph revision,
so transport is operational. One edge is insufficient to validate collaboration
or to run a meaningful Normal/Masked aggregate.

## Wrong-demo evidence

- `5ac2a912…`: singleton returned `Hole`; gold is `The Wolfhounds`. The graph
  never separated the two band facts.
- `5ac3e8c6…`: singleton correctly extracted 13 miles but returned a JSON object
  with the second value unknown instead of the requested yes/no span.
- `5ab93287…`: singleton returned the narrow subtype `herbaceous`; gold is
  `plant`.
- `5ab345db…`: singleton mapped Warren Bryant to California rather than Hawaii.
- `5ae27edc…` and `5abee5e2…`: the same unresolved multi-hop disambiguations
  remained wrong.
- `5a8d7341…`, `5a7a0a96…`, and `5a7a5274…` show surface-span losses despite
  semantically close answers. The terminal wrapper is valid, so this is not a
  FINISH-protocol defect.

## Confirmed engineering bug and fix boundary

The v1 config declared `direct_reused_from`, but the evaluation driver did not
consume that field and its destination pointed at the old shared baseline. It
therefore regenerated and temporarily rewrote 14 entries. The newly generated
v1 Direct file was moved into the v1 artifact directory, the original 128-entry
Round-01 file was restored exactly from the pre-run commit, and the runner now
loads a declared reuse source into a distinct destination before deciding that
any paid call is missing. This is a concrete resume/isolation bug, not a method
change.

## Root-cause hypotheses for v2

1. Catalog ordering replaced the old preferred-model bias with a first-entry
   bias. Use a deterministic per-trajectory presentation order; do not choose a
   model for the Director.
2. “Decompose when needed” is too weak. Add a concise dependency-coverage check
   before FINISH without prescribing roles, topology, or Agent count.
3. Four Director turns were invalid, including duplicate actions and outputs
   truncated at the 256-token boundary. Increase only the action budget; keep
   one atomic action per turn.
4. Contracts still bundle all evidence hops into one Output node. Keep the free
   string, but require the Director to declare objective, input/dependency,
   artifact and completion in the text so missing dependencies are visible.

`ARCHITECTURE_V1_READY = NO`

Reason: terminal/transport integrity is fixed, but the fixed diagnostic still
shows 92.86% singleton graphs, one model family in practice, and EM below Direct.
