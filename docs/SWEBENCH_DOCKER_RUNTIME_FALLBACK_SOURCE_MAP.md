# SWE-bench official-Docker repository-runtime fallback source map

Scope: only the two selected instances that share the timed-out
`pydata/xarray` 2022.09 Conda environment (`pydata__xarray-7229` and
`pydata__xarray-7393`).  The other 126 selected instances retain the current
SkillFlow Conda environment plus detached-worktree runtime.

## Direct reuse

- SkillFlow public workspace contract:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/benchmarks/swebench.py`
  (`SWEWorkspaceBackend`, `SWEBenchWorkspaceEnvironment`).
- SkillFlow official workspace implementation:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/swebench_official.py`
  (`_ApptainerSWEWorkspaceBackend`) and
  `swebench_official_worker.py` (`_copy_testbed`, base reset, workspace command
  execution).
- Official SWE-bench instance-image preparation and transient-container lifecycle:
  `/ssd1/iclr/.private/skillflow-resources/SWE-bench-harness-d83/swebench/harness/test_spec/test_spec.py::make_test_spec`,
  `docker_build.py::build_container`, and
  `docker_utils.py::cleanup_container`.
- Existing model-visible Tool semantics and schemas:
  `src/interactive/coding_tools.py::RepositoryToolBackend` and
  `create_swebench_repository_registration`.

## Necessary adaptation

`src/interactive/swebench_docker_workspace.py` copies the official image's
`/testbed` source tree into a task-owned directory, resets it to the public
`base_commit`, bind-mounts that exact tree back at `/testbed` in one persistent
container, and redirects only `bash`/`run_tests` process execution to Docker.
The persistent bind-mounted container is created with the same image, user,
platform and capability arguments used by the official `build_container`; this
is the minimum adaptation required because the official helper has no volume
mount parameter.  Before calling `build_container`, the adapter resolves the
official OCI image index for `test_spec.platform`; the task-scoped rootless
Docker daemon otherwise finishes the pull without materializing an inspectable
local instance-image tag.  The official `setup_logger`/`close_logger` pair is
used so upstream `BuildImageError` retains its expected `log_file` receipt.
Search, view, edit and diff remain the existing `RepositoryToolBackend`
implementation over the same bind-mounted bytes.  Its timeout helper is a
minimal extension of SkillFlow `training/swe_bench_eval.py::_exec_run_with_tolerant_decode`
that also records Docker's `ExitCode`.

The unified Director, Canvas, AgentGraph, `CodingExecutionAdapter`, topology
search space, FINISH behavior, and `OfficialSWEbenchHarness` evaluator are not
changed.  No role or relation is introduced.

## Not validated in this branch

- No Docker daemon, image pull, model/API call, smoke rollout, or official
  evaluation was run while developing this fallback.
- Runtime wiring must use an explicit instance allowlist containing only the
  two IDs above before a live canary.
- The official per-instance images must be locally available or pullable by
  the existing official `build_container` path at live runtime.
