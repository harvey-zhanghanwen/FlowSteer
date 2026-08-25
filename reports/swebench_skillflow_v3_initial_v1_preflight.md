# SWE-bench 初版适配与 evaluation preflight 报告

日期：2026-08-25  
Condition：`swebench_skillflow_v3_initial_v1`  
正式指标：SWE-bench Verified official **Resolved Rate**

## 结论

Dataset、Repository Environment、Tool Adapter、Evaluator 与统一 evaluation runner 的
初版接线已经完成，且 no-model read-only repository smoke 通过。正式 Direct vs AgentGraph
evaluation 没有开始：固定 128 个 task 的 repository/base-state setup/cleanup 已全部
通过，但目前只有 1/128 个 task 的 SkillFlow Python/Conda environment 可用，且当前
执行用户无 official Docker daemon socket 权限。Runner 在创建模型 runtime 前按
fail-closed 语义停止，没有调用模型、没有生成 patch、没有分配 Resolved/Failed 标签。

因此本轮真实结果是：

| Condition | official-evaluator-valid | Resolved | Resolved Rate |
|---|---:|---:|---:|
| Qwen3.5-9B Direct | 0（未运行） | 不可用 | **不可测** |
| Free AgentGraph | 0（未运行） | 不可用 | **不可测** |

“不可测”不能写成 0%，也不能用本地 test、模型输出或 proxy metric 替代。

## 已完成实现

- SkillFlow v3 `code_generation` population loader 与 official Verified evaluator-only
  join；
- 500 train / 0 validation / 固定 128 IID test，test instance 唯一且与 train 无交叉；
- task/arm 独立 detached worktree 与逐任务 repository/base-state population preflight；
- SkillFlow deployed repository tools：`bash`、`list_files`、`search_code`、
  `view_file`、`str_replace_editor`、`run_tests`；
- task workspace diff 作为唯一 patch publication artifact；
- `bash` 新建的非 test 文件也进入最终 workspace diff，不丢失 untracked artifact；
- `bash` / `run_tests` 严格绑定 SkillFlow `_env_python(repo, version)` 对应的 Conda
  environment；环境缺失时禁止回退到宿主 Python；
- progressive Canvas 多 Agent execution 共享 task-global turn/Tool budget；
- official SkillFlow `evaluate_patch` / SWE-bench harness fail-closed adapter；
- official harness infrastructure exception 的结构化 classification、phase、retryable、
  exit/OOM receipt；
- receipt-only search/view/edit/test/command、invalid action、failure phase、topology 与
  first-observable Wrong Demo reporting；
- 保持自由 `agent_id + model_id + free-text contract`，没有固定
  Coder/Reviewer/Tester role 或 chain。

## 验证结果

定向与兼容性测试：**551 passed，另有 103 个 subtests passed**。覆盖 Dataset、
worktree、Tool schema/backend、ReAct、CodingExecution、Canvas/runtime、统一 runner、
evaluation config、official evaluator adapter 和 SWE-bench reporting。

真实 repository smoke task：`astropy__astropy-14182`。

- detached base state：通过；
- read-only Tool backend：通过（仅 `list_files/view_file`）；
- strict SkillFlow Tool registration：未启用；task-specific environment 未就绪时由
  factory fail closed；
- `list_files`：通过，493 个 source paths；
- `view_file`：通过；
- initial workspace diff：空；
- cleanup：通过；
- model/API calls：0；
- official evaluator call：0。

进一步 environment canary 使用同一个 `astropy__astropy-14182`：

- `swe_astropy_astropy_51` 按 official harness spec 安装完成；
- exact runtime prefix：`conda run -n swe_astropy_astropy_51 --no-capture-output`；
- 独立 detached base worktree cwd：正确；
- `astropy/io/ascii/tests/test_rst.py`：9/9 passed；
- 上游 `bash -lc` 在本机重置 cwd 的兼容性问题已最小适配为 `bash -c`。

Smoke 发现并修复了一个可复现的 Tool Adapter bug：以点开头的合法路径
`.pyinstaller/...` 曾被错误规范化成 `pyinstaller/...`，导致 `list_files` 的结果不能被
`view_file` 回读。修复后用同一个真实 worktree 复测通过。

## Evaluation preflight

固定 selection：128 tasks / 128 unique instance IDs。

Repository availability：

- setup/base-state/cleanup ready：128/128 tasks，10/10 repositories；
- setup/base-state unavailable：0。

Task execution environment：

- SkillFlow `_env_python(repo, version)` ready：1/128；
- unavailable：127/128（剩余 47 个唯一 repo/version environment）；
- strict `bash/run_tests` host fallback：禁用。

Official evaluator：

- SkillFlow evaluator source：已定位；
- SWE-bench Verified dataset：已定位；
- pinned SWE-bench harness checkout：已定位；
- Docker harness runtime：**blocked**；daemon 正常，但当前用户无
  `/var/run/docker.sock` 访问权限；
- proxy metric：未使用；
- task labels assigned：0。

正式 runner 状态：`failed_runtime_preflight`。repository/base-state population blocker
已经解除；当前最早可观察 failure layer 是 **task environment population preflight**，
official Docker harness 权限是另一个独立 blocker。这些都不是某一道 SWE-bench task 的
Wrong Demo，不能归因于 Director、Agent topology 或 patch quality。

## 本轮未运行

- Stable Zero model canary；
- Direct patch generation；
- AgentGraph Canvas rollout；
- official patch apply/tests；
- task-level FINISH/max_rounds、Tool counts、topology distribution 或 Wrong Demo；
- training、GRPO、MACE、Bayesian、Skill evolution、backward、optimizer 或 LoRA。

## 持久化入口

- Source map：`docs/swebench_initial_v1_source_map.md`
- Dataset config：`config/datasets_swebench_skillflow_v3.yaml`
- Evaluation config：`config/evaluation_swebench_skillflow_v3_initial_v1.yaml`
- No-model preflight：`scripts/preflight_swebench_initial_adapter.py`
- Runtime receipt：
  `reports/swebench_skillflow_v3_initial_v1_preflight.json`
- Formal runner manifest：
  `artifacts/swebench_skillflow_v3_initial_v1/run_manifest.json`

在剩余 SkillFlow task environments 与 official Docker harness 全部就绪前，不应启动
128-task paired model evaluation；环境就绪后必须继续使用同一固定
selection、snapshot、Tool surface、task-global repository episode budget 和 official
evaluator。AgentGraph 另有 Director/多 Agent inference，因此 Direct/AgentGraph delta
只能作 descriptive architecture comparison，不能声称总计算预算等价。
