# SWE-bench 初版适配：source map 与协议边界

## 1. 本轮范围

本轮只实现 SWE-bench 的 Dataset、Repository Environment、Tool Adapter、Evaluator
与 evaluation 接线，不启动训练、GRPO、MACE、Bayesian posterior、Skill
retrieval/evolution、backward、optimizer update 或 LoRA publication。

统一执行链保持为：

```text
TaskRecord(issue)
  -> Qwen3.5-9B Flow-Director
  -> progressive AgentGraph Canvas edit
  -> execute-on-edit AgentRuntime
  -> Agent repository Tool Action / Observation
  -> task workspace diff
  -> explicit FINISH
  -> official SWE-bench harness
  -> Resolved / Failed EvaluationReceipt
  -> TrajectoryRecord
```

SWE-bench 只增加 task/repository/tool/evaluator adapter，不增加固定
`Coder / Reviewer / Tester` role，也不增加固定 `Coder -> Reviewer -> Tester`
topology。Agent 仍由 `agent_id + model_id + free-text contract` 定义；Director 自主决定
Agent 数量、contract、model、relation、Output Agent 与 `FINISH`。

## 2. 权威来源

### 项目 MD

用户提供的 `FlowSteer_MACE_Bayesian_Skill_Design.md` 是自由 AgentGraph、两比特
relation、有限 reciprocal block、terminal reward、Trajectory receipt 和方法边界的
依据。MD 中的 MACE、Bayesian 和 Skill 闭环属于后续阶段，本轮没有将预留接口写成已
实现能力。

### FlowSteer

以下通用调用链继续直接复用：

- Progressive Canvas `step`：一个 atomic graph edit 被校验、执行，并把真实 feedback
  送入下一次 Director observation；
- `agent_workflow_env.py`：transactional edit、execute-on-edit、revision-local
  `FINISH`；
- `agent_runtime.py`：quotient-DAG 调度、artifact routing、dirty closure、partial
  failure preservation；
- `rollout_collector.py` 与 `records.py`：Canvas turn、Agent execution、Tool receipt、
  evaluator receipt 和 trajectory 持久化。

FlowSteer 论文实验中的 predefined Operator 不作为本项目 SWE-bench 固定 role 或
topology。

### SkillFlow deployed SWE-bench surface

部署源码根目录：

```text
/home/test/SKILLEV/skillflow-bayesian-improve-deploy
```

本轮直接对照：

- `training/environment.py::_setup_swe_repo/cleanup`；
- `_handle_list_files`、`_handle_search_code`、`_handle_view_file`；
- `_handle_str_replace_editor`、`_handle_bash`、`_handle_run_tests`；
- `training/swe_bench_eval.py::_env_name/_env_python` 与
  `environment.py::_handle_bash/_run_tests_in_swe_env` 的 task environment boundary；
- `_generate_workspace_diff`；
- `training/swe_bench_eval.py::evaluate_patch` 与 official harness 路径；
- `training/batch_inference.py` 中上述 Tool 的实际 schema；
- SkillFlow v3 `train_v3.json` / `test_iid_v3.json` 的 `code_generation`
  population。

SkillFlow newer typed surface 的 `read / search / write / test / submit_patch` 属于另一套
versioned action vocabulary，本轮不与 deployed vocabulary 混用。

### SWE-bench official harness

正式判定只来自 pinned `swebench.harness` 的 instance report。Resolved 要求官方
`FAIL_TO_PASS` 与 `PASS_TO_PASS` 规则满足；模型文本、本地 test 输出或 patch 相似度都
不能产生 reward。

## 3. Dataset protocol

`scripts/prepare_swebench_skillflow_v3_dataset.py` 保留 SkillFlow v3 source order，并按
`instance_id` 与官方 SWE-bench Verified 做 evaluator-only join：

| split | source | rows | unique instance_id | repeats |
|---|---|---:|---:|---:|
| train | SkillFlow `train_v3.json`, `task_type=code_generation` | 500 | 372 | 128 |
| validation | none | 0 | 0 | 0 |
| test | SkillFlow `test_iid_v3.json`, `task_type=code_generation` | 128 | 128 | 0 |

train/test instance overlap 为 0。Gold patch、`test_patch`、`FAIL_TO_PASS` 与
`PASS_TO_PASS` 只保留在 TaskRecord/evaluator boundary；Director 和 Agent 的 problem
来源固定为 `TaskRecord.question`。用于 evaluator/resume 的内部 selected-task checkpoint
可保留 gold patch，但公开 paired row 与 Wrong Demo 均写成
`ground_truth=null, ground_truth_role=evaluator_only_redacted`。

当前已验证 smoke condition 的配置入口为：

```text
config/datasets_swebench_skillflow_v3.yaml
config/evaluation_swebench_skillflow_v3_initial_v7.yaml
```

`initial_v1`--`initial_v6` 保留为逐步解除 repository、task environment、official
harness transport 与 workspace-patch publication 阻塞的历史条件，不能与 v7 结果混合。

## 4. 两层动作

Director 只使用统一 Canvas actions：

```text
add_subgraph / modify_agent / delete_agent /
set_relation / set_output / finish
```

Agent 在 `execution_mode=coding` 内使用 task-scoped repository actions：

```text
bash / list_files / search_code / view_file /
str_replace_editor / run_tests
```

这些名称和 schema 来自 SkillFlow deployed source。`edit_file` 与 SkillFlow 私有
`M_exec` 自然语言编辑器不可分离，因此本轮不伪造同名实现；改用同一 deployed source
中的 deterministic `str_replace_editor` 做最小适配。Patch publication 不暴露为
Director action 或模型 Tool，而是在 completion boundary 从当前 worktree 内部读取
workspace diff。

ReAct 是单个 Agent 内部的 `Thought -> Action -> Observation -> ... -> Complete`
execution mode，不是 Agent role，也不是 Director action。

## 5. Source map

| 状态 | 当前文件 | 上游来源 / 语义 |
|---|---|---|
| 直接复用 | `src/interactive/agent_graph.py` | MD 自由 AgentGraph、relation、Output Agent |
| 直接复用 | `src/interactive/director.py` | FlowSteer progressive Canvas；中性 `minimal-neutral.v10` |
| 直接复用 | `src/interactive/agent_workflow_env.py` | atomic edit、execute-on-edit、feedback、FINISH |
| 直接复用 | `src/interactive/agent_runtime.py` | topology 调度、artifact routing、failure preservation |
| 直接复用 | `src/interactive/react_execution.py` | SkillFlow bounded Tool Action–Observation loop |
| 直接复用 | `src/interactive/records.py`, `rollout_collector.py` | trajectory/execution/Tool/evaluator receipts |
| 必要适配 | `scripts/prepare_swebench_skillflow_v3_dataset.py` | SkillFlow v3 population + evaluator-only Verified join |
| 必要适配 | `scripts/prepare_swebench_skillflow_task_environments.py` | SkillFlow `_env_name` + official `make_test_spec` environment/repository scripts；48 个唯一 repo/version 的可恢复物化 |
| 必要适配 | `src/interactive/swe_worktree.py` | SkillFlow detached worktree lifecycle + active-population setup/base/cleanup preflight |
| 必要适配 | `src/interactive/coding_tools.py` | deployed repository handlers + typed ToolRegistry envelope + strict task environment binding |
| 必要适配 | `src/interactive/coding_execution.py` | task workspace diff completion + task-global budget；空 `allowed_tools` 继承当前 task 的 SkillFlow repository ToolRegistry，而不改变 Director action space |
| 必要适配 | `scripts/train_agentgraph_smoke.py::_runtime_for_task` | 每个 task/arm 独立 runtime/worktree；在 cleanup 前物化 authoritative workspace diff |
| 必要适配 | `src/interactive/swebench_adapter.py` | SkillFlow `_env_python`/Conda binding + `evaluate_patch` + fail-closed official preflight；单 ID user namespace 下只规范化 harness tar header owner，不改变 patch、tests 或判分 |
| 必要适配 | `src/interactive/swebench_reporting.py` | receipt-only Tool/failure/Resolved reporting |
| 必要适配 | `scripts/evaluate_completion_benchmark_round.py` | paired Direct vs AgentGraph runner |

本轮没有修改统一 orchestration core 的 Agent 定义、Canvas action vocabulary、relation
semantics 或 topology search space。

## 6. Repository state 与 completion

每个 task、每个 paired arm 都创建独立 detached worktree，固定到相同 public
`repo + base_commit`。同一个 AgentGraph 内所有 Agent 共享该 task worktree；backend
对 Tool call 做序列化，execution receipt 记录真实 Action/Observation。Director 可以
把 repository capability 分配给任意自由 contract，不强制单写入者 role；已有合法
修改与 receipt 按 `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` 保留。

任何模型调用前，正式 runner 对 active population 的每个 task 直接复用同一个
worktree setup/cleanup，记录 expected base commit、observed pinned commit、setup 与
cleanup 状态；任一 task 不可准备即 fail closed。运行时 response metadata 同时记录实际
repository state 与 SkillFlow task environment receipt。

严格 profile 的 `bash` 与 `run_tests` 通过 SkillFlow
`_env_python(repo, version)` 解析的 Conda environment 执行。解析不到环境或 Conda
executable 时不回退到 host Python。Workspace patch 以 SkillFlow `git diff` 为主，并
补入 `bash` 可能创建的非 test untracked 文件；test 文件仍保持上游排除语义。
Conda executable、environment directory 与 persistent setup repository root 已写入
condition config，preflight/runner 不依赖调用者手工 export。

SkillFlow deployed handler 的 `bash -lc` 在当前宿主机会由 login shell 覆盖
`subprocess cwd`，使命令离开 task worktree。本项目在同一 task environment command
prefix 后只做最小兼容性适配为 `bash -c`，保留已验证 worktree cwd；Tool 名称、schema
与 Action/Observation 语义不变。

SkillFlow 原始 episode 是单一 bounded episode，而 progressive Canvas 可能执行多个
Agent。为避免换 Agent 后重置预算，`CodingExecutionAdapter` 增加 task-global 计数：

```text
task_max_turns = 12
task_max_tool_calls = 10
```

Direct 与 AgentGraph 各自拥有独立、同配置的 task adapter，因此使用相同 tasks、
repository snapshot、Tool surface 和 task-global repository episode budget。AgentGraph
额外的 Director Canvas calls 与多 Agent inference 单独记录；两臂不是总计算预算或完整
protocol 等价，差值只作 descriptive architecture comparison。

`completion_policy=workspace_diff` 只接纳当前 worktree 的非空 diff，忽略模型 prose。
本地 `run_tests` 是可选 execution evidence，不是 `FINISH` 必选 role，也不是 Resolved
判定。最终 patch 必须再通过 official harness。

## 7. Evaluator 与报告

唯一正式指标：

```text
Resolved Rate = Resolved / fixed total tasks
```

该正式 aggregate 只在固定 total 的每一项都有 official-evaluator-valid receipt 时发布；
否则同时报告 coverage/failure，并把正式 Resolved Rate 标为不可测，而不是把 evaluator
infrastructure failure 当作 task-level `resolved=0`。

调用链：

```text
task_evaluator._evaluate_swebench
  -> OfficialSWEbenchHarness
  -> SkillFlow training.swe_bench_eval.evaluate_patch
  -> swebench.harness run_instance/get_eval_report
  -> report[instance_id]["resolved"]
```

`swebench_reporting.py` 只读取结构化 receipt，统计 search/view/edit/test/command、invalid
Tool/action、budget exhaustion、local test、provider/executor、patch apply、environment 和
official test failure；不解析模型输出中的 `PASSED` 字符串。Wrong Demo 记录最早可观察
failure layer，并明确 `causal_attribution=false`。

Task-level Wrong Demo 进一步使用 SWE-bench evaluation pipeline、SkillFlow repository
Tool boundary 与 FlowSteer Director/Canvas/runtime boundary 对齐的互斥 primary taxonomy：
collection/receipt、provider、task-scoped repository environment、orchestration、Agent
communication/artifact routing、repository Tool、terminal/budget、workspace patch
protocol/execution、local test validation、workspace patch publication/application、official
target-test、official regression、official unresolved
without F2P/P2P detail、evaluator runtime 与 unclassified structured receipt。每个错误 task
只计入一个 outcome-oriented primary observable category；有效 official terminal receipt
优先于中间 debugging observation，但不据此声称中间失败已经恢复或导致终局。该分类不是
root-cause inference。

报告按 `task_id` 连接 paired row 与完整 trajectory，代表性 demo 保存 Director
input/output/action、Canvas edit/snapshot、Agent request/output、实际 `upstream[]` artifact、
ReAct Action--Observation、Tool receipt、runtime、terminal 和 official evaluator receipt。
`first_observable_failure` 与 causal attribution 严格分开：没有显式 causal receipt 或受控
intervention evidence 时，`first_causal_failure` 与 causal propagation 均为 `null`。SWE-bench
没有 QA answer-string canonicalization；gold patch、test patch、FAIL_TO_PASS 与 PASS_TO_PASS
仍为 evaluator-only redacted fields，不进入公开 Wrong Demo。run-level environment/Docker
preflight blocker 不计入 task-level taxonomy。

当前 SkillFlow `evaluate_patch` 返回的 `details` 是 `resolved/unresolved` 字符串，adapter
尚未持久化 official report 的结构化 F2P/P2P breakdown。因此当前真实 unresolved 只能进入
`official_test_failure_unclassified`；target-test 与 regression 两类保持显式 0，不能从日志
或模型文本推断。若上游未来直接提供结构化 `tests_status`，公开报告只保留 success/failure
count，test IDs 继续 redacted。

SkillFlow official harness 的 infrastructure exception 若带 `diagnostic`，只透传
classification、phase、retryable、test/report presence、container exit/OOM 等结构化
字段；不解析自由文本，也不把 infrastructure failure 转成 `Resolved=false`。

## 8. 已验证状态（2026-08-26，v7）

- 数据与 repository population：500 train / 0 validation / 128 fixed IID test；固定
  test 的 128 个 instance 唯一、与 train 无交叉，并已完成 repository setup/base-state/
  cleanup preflight；
- 当前正式执行范围只是固定 test panel 前 2 题的 Stable Zero canary，不是 128 题完整
  evaluation，也不是 benchmark estimate；
- repository Tool 已在实际 Agent execution 中可用。Direct 共执行 `bash=19、view=1`；
  AgentGraph 共执行 `bash=22、search=2、view=2、str_replace_editor=1`；两臂均未执行
  `run_tests`，AgentGraph 唯一 edit receipt 失败且没有产生 workspace diff；
- AgentGraph evaluator 输入已从 Output Agent prose 改为 task worktree 的真实 diff；没有
  diff 时严格提交 empty patch，不再把自然语言误当 patch；
- official harness archive transport 已在隔离 user namespace 下通过 preflight。适配只把
  tar header 的 uid/gid 规范化为 0/0，SkillFlow `evaluate_patch`、official image、patch
  apply、tests 与 Resolved 判定均保持不变；
- Direct 2/2 和 AgentGraph 2/2 都具有 official-evaluator-valid receipt；两臂均为
  `empty_patch -> unresolved`，因此真实 Resolved Rate 都是 `0/2 = 0.00%`；
- AgentGraph 显式 `FINISH=0/2`、`max_rounds=2/2`。最终自然 topology 都是 parallel：
  一题 8 Agents / 4 reciprocal relations / depth 1，另一题 3 Agents / 1 directed
  relation / depth 2；
- Stable Zero **未通过**。阻塞已从 dataset/repository/evaluator infrastructure 转移为
  可观察的 policy/runtime failure：Direct 耗尽 repository episode budget；AgentGraph
  没有在 20 个 Canvas rounds 内收敛到 patch 与 FINISH；
- 最新定向测试：CodingExecution、SWE-bench adapter、smoke runner 与 completion round
  共 57 项通过；
- training、GRPO、MACE、Bayesian、Skill injection/evolution、backward、optimizer update
  与 LoRA publication 均未运行。

v7 的 `0.00%` 是这 2 个 canary task 的官方 Resolved Rate，可以用于确认端到端记账已经
打通；它不能外推为 128 题 SWE-bench Verified 的准确率。

## 9. 运行入口

数据物化与冻结 selection（不启动模型/API）：

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/prepare_swebench_skillflow_v3_dataset.py \
  --catalog config/datasets_swebench_skillflow_v3.yaml

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_swebench_skillflow_v3_initial_v7.yaml \
  --prepare-only
```

No-model repository/evaluator preflight：

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/preflight_swebench_initial_adapter.py \
  --config config/evaluation_swebench_skillflow_v3_initial_v7.yaml \
  --output reports/swebench_skillflow_v3_initial_v7_preflight.json
```

Task environment plan / materialization（直接复用 SkillFlow environment name 与
official harness scripts）：

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/prepare_swebench_skillflow_task_environments.py \
  --conda-executable /ssd1/iclr/TTT/miniconda3/bin/conda \
  --conda-envs-dir /ssd1/iclr/TTT/miniconda3/envs \
  --jobs 2
```

只有当 128 个 task repository、128 个 SkillFlow task environment、官方 Docker harness
以及现有 provider/Supervisor 配置均可用后，才运行相同冻结 condition 的 smoke
evaluation 与完整 paired evaluation：

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_swebench_skillflow_v3_initial_v7.yaml \
  --canary-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_swebench_skillflow_v3_initial_v7.yaml
```

Artifacts 写入独立目录：

```text
artifacts/swebench_skillflow_v3_initial_v7/
reports/swebench_skillflow_v3_initial_v7_report.{json,md}
```

任何没有 official evaluator receipt 的任务均不得进入 Resolved/Failed 分母。

严格同口径的 best-profile 选择状态记录在：

```text
config/swebench_best_profile_v1.yaml
reports/swebench_best_profile_v1_status.md
```

当前没有 official-evaluator-valid 候选，pointer 保持
`no_evaluator_valid_candidate` / `activation_allowed=false`，不会把 preflight 条件重命名为
best，也不会修改必填 `--config` 的文档化 next-run 入口。
