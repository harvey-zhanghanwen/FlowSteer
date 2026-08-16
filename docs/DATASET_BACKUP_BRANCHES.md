# Dataset adaptation backup convention

Each dataset-specific architecture adaptation is kept on its own branch so its
code, frozen inputs, run manifests, evaluator receipts, reports, and failure
records can be reviewed without mixing benchmark conditions.

Current HotpotQA branch:

`dataset/hotpotqa-training-ready-step0`

Current Multi-Agent/Skill Step0-to-StepN HotpotQA branch:

`dataset/hotpotqa-multiagent-skill-step0-to-n`

Its exact pre-change backup is:

`backup/hotpotqa_pre_multiagent_skill_step0_v1` at `fc298ae`. The diagnostic
architecture policy is explicitly labelled as the previously updated
cross-dataset smoke warm start; it is not the formal HotpotQA `policy_step_000000`.

Future datasets must use a separate branch only when that dataset's work
actually starts, following:

`dataset/<dataset-key>-training-ready-step0`

Within each branch, use dataset-specific artifact and report directories and
two stage commits where applicable: one frozen architecture/config commit,
then one completed evaluation/report commit.  Do not copy credentials into any
config, artifact, log, commit, or remote URL.
