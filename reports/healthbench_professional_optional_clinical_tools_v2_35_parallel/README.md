# v2.35 双 GPU 独立评测

用户授权 GPU5 运行 Direct、GPU6 运行 AgentGraph。本版本仅作必要评测入口
适配和工具顺序 Bug 修复，不采用 v2.36 分类数据库，不改统一 AgentGraph。

| 端 | GPU / SGLang 端口 | 配置文件后缀 | 执行参数 |
| --- | --- | --- | --- |
| Direct | 5 / 8025 | `v2_35_direct_gpu5.yaml` | `--collection-arm direct` |
| AgentGraph | 6 / 8026 | `v2_35_graph_gpu6.yaml` | `--collection-arm agentgraph` |

配置全名前缀为 `config/evaluation_healthbench_professional_optional_clinical_tools_`。
继续使用 `start_qwen35_director_server.sh` 的 SkillFlow SGLang 运行方式。
同一 Qwen3.5-9B 基座、32,768 context、0.82 memory fraction、max-running-requests 8，
保持 thinking、生成参数和每题预算。每端并发 4；总 API 瞬时并发增加，不保证速度翻倍。
525 题顺序、seed、工具条件、模型池、AgentGraph、evaluator 与 v2.35 相同。
配置仅改变存储路径；GPU/端口由各进程环境变量和运行 receipt 显式记录。

## 必要适配与 Bug 修复

- `collection_arm` 直接复用 `_collect_direct`、`_collect_graph`、原 evaluator
  和 exact-resume。默认 `both` 及旧 `direct_only` 语义保持不变。
- 单端不调用另一采集器、不生成 paired delta、不把另一端填成 0 分，
  不宣称 paired Stable Zero。指标复用 `_metrics`、`_aggregate`；未完成时
  strict 指标为 null，completed-only 保留真实有效分母。
- 原 Direct 入口要求配置工具顺序等于已排序的 `ToolRegistry.resource_ids`，
  将相同工具集合误判为不一致。每题在首次模型调用前同步读取语料后失败，
  导致 CPU 忙、检查点为 0。新入口精确核对排序后的成员和数量；receipt
  `tool_resource_ids` 按冻结配置序保存，额外保留 `registered_tool_resource_ids`
  的真实注册序，兼容原严格续跑校验。不增加或删减任何工具。

## 文件与结果边界

Direct 续用原 v2.35 worktree 的已落盘 evaluation 检查点；旧串行进程停止、
锁释放后才启动 Direct-only，防止两进程同时写文件或自动重复 Graph。
Graph 独立目录：
`artifacts/healthbench_professional_optional_clinical_tools_v2_35_graph_gpu6/evaluation/`。
服务及评测启动记录保存在同一 condition 根目录的 `operations/`。

不同服务端口导致展开后的模型目录标识不同，原始 receipt 保留各自端点，
不篡改成同一标识。最终比较须核对模型、生成、工具和 evaluator 条件，再按
同一 Task ID 汇总；核对前不宣称既有 paired-identity 校验通过。本次只启动
两个独立采集进程，不添加自动重复评测循环。

入口定向测试 18 项、原 runner 回归 47 项、工具顺序与精确续跑回归 6 项通过。
所有测试均为本地 mock，不调用模型或 grader；Graph 固定 525 题
prepare-only 通过。旧串行 Direct 未生成已完成预测，不把它称为有效基线。

无训练、LoRA 发布、GRPO、MACE、Bayesian 或 Skill evolution；无新医疗角色、
固定拓扑、提示词或工具变化。该公开集合已用于开发，后续为开发后重评测。
