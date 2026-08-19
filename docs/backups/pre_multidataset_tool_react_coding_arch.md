# Pre-multidataset Tool/ReAct/Coding architecture backup

- Recorded on: 2026-08-19 (Asia/Shanghai)
- Source branch: `experiment/joint-qa-progressive-skill-rl-02`
- Source HEAD: `c6c806b5eff689491e8d6ec903f1ecbe52202b32`
- Backup branch: `backup/pre_multidataset_tool_react_coding_arch`

This checkpoint freezes the recoverable source, configuration, tests,
dataset-adapter documentation, and generated reports that existed before the
multidataset Tool/ReAct/Coding architecture work began. Existing user changes
were preserved rather than reset or overwritten.

The repository intentionally ignores `artifacts/`, datasets, model weights,
logs, and local environment files. Those runtime files remain in their
existing local paths but are not part of the Git backup. No credential or
local `.env` file is included.

After the backup commit is created, a standalone Git patch and bundle are
stored outside the repository under `/ssd1/iclr/1/` so this state can be
recovered even without the remote branch.

## Recovery

Restore the Git-backed architecture snapshot by checking out the backup
branch. Runtime artifacts must be restored separately from their preserved
local paths because they are intentionally outside Git.
