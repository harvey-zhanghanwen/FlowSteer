# Unified add_subgraph micro-training preflight

## Decision

`CURRENT_ADD_SUBGRAPH_MICRO_TRAINING = NO_GO`

本次只执行了只读前置条件核对，没有启动 SGLang、模型/API、rollout、GRPO、backward、optimizer update 或 LoRA publication。

## Blocking evidence

1. 当前仓库唯一现成的 `add_subgraph` 训练模板是 `config/training_joint_qa_progressive_skill_on_step1.yaml`，其条件为 Skill-on；当前 epoch 7 的 `publication_results.json`、`skills.json`、resolved config、冻结 schedule 和 cursor 尚未物化，Round 7 状态仍为 `prepared_not_executed`。
2. 当前 evidence gate 只有 HotpotQA、TriviaQA 两个 `CANDIDATE` Skill，`ACTIVE Skill=0`。因此现成 Skill-on 路径不具备合法注入条件。
3. 物理 GPU 3 被本任务之外的 VLLM 进程占用约 40.8 GiB；现有配置固定读取 YAML 内的 learner/replica device，不能仅靠环境变量绕过。GPU 4 的任务服务端口 8015 当前关闭。
4. 当前 shell 未配置 10-arm Executor catalog 所需的 provider credential。
5. HotpotQA v6 Step 1–3、旧 Joint Step 1 和 Step 2 attempt 4 的 cursor 均已消费；不能原样重放并把重复 batch 称为新 on-policy 更新。

## Existing positive training-boundary evidence

- `artifacts/joint_qa_micro/step_000001/training_manifest.json`
- `artifacts/joint_qa_micro/step_000002_attempt_04/training_manifest.json`

两份 manifest 各记录一次真实 optimizer update、非零 LoRA 参数变化、adapter/optimizer/policy version 保存、SGLang route switch 与 post-update canary。它们证明旧 joint-QA atomic-action 训练和 LoRA→SGLang 同步边界可执行。

固定 held-out 宏平均从 Step 0 的 EM/F1 `56.25%/68.58%` 变为 Step 2 的 `54.69%/65.31%`，没有正向 learning trend；这些历史 receipts 也不证明当前 Tool/Environment/Coding `add_subgraph` action space 可学习。

## Required next gate

只有同时具备新的版本化 config、冻结 schedule/cursor、合法 Skill condition（或经用户确认的新 Skills-off 实验设计）、一致的 GPU mapping、可用本地 rollout service 与 provider credential 后，才允许执行最多一次 optimizer update 的 bounded smoke run。即使 `max_updates=1`，同组 reward 恒定时也必须保持零更新，不能伪称训练成功。
