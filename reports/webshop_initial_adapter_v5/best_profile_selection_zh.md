# WebShop AgentGraph best-profile 选择报告

## 结论

当前版本化 best-profile 指向 `webshop_initial_adapter_v5`。这是现有结果中唯一同时满足以下条件的 AgentGraph 条件：WebShop native validation、固定 `webshop:00500` 至 `webshop:00627` 共 128 条、严格 128 分母、`skillflow.ragen_adapter.v2` 原生 evaluator、128/128 完成且 evaluator-valid、128/128 显式 `FINISH`，并且模型可见上下文不包含 evaluator-private reward。

本轮没有运行模型、WebShop environment、训练或全量评测；指标来自已完整收束的正式 artifacts。

## 已验证指标

| 条件 | 样本 | Average Score | Success | Success Rate | evaluator-valid | FINISH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentGraph v5 | 128 | **50.34 / 100** | 18 | **14.06%** | 128 | 128 |
| Direct local baseline | 128 | 33.87 / 100 | 19 | 14.84% | 128 | N/A |
| AgentGraph − Direct | 128 | **+16.47 points** | -1 | **-0.78 pp** | — | — |

WebShop 的官方主指标是 Average Score；Success Rate 为辅助指标。因此，该条件相对同批 Direct 的主指标提高 16.47 points，但 Success Rate 低 0.78 percentage points。Direct 与 AgentGraph 的模型/编排条件并非完全等价，差值只作为同批本地描述性对照。

## 候选排除

- `webshop_ragen_environment_native_action_v4_stable_zero` 的静态 reward 聚合表面上更高，但 26/128 条 trajectory 的实际 Agent rendered input 含 `Your score` / `Reward Details` 等 evaluator-private terminal reward；按无 evaluator leakage 口径排除。
- `webshop_round_04/evaluation` 使用另一组 task population/split，且 AgentGraph 未完整收束，排除。
- v1 未完整；v2/v3/v4 只是两样本 canary 或 failed Stable Zero；unified condition 为 prepared-only，均排除。

## 当前指针与恢复状态

- 版本化 profile：`config/webshop_best_profile_v1.yaml`
- 当前 profile 指针：`config/webshop_best_profile.yaml`
- 可执行配置：`config/evaluation_webshop_initial_adapter_v5.yaml`
- 实测源码 revision：`b6f9df30d0937a416f36b150122101c5a3c7d0c7`
- 正式 JSON 报告：`reports/webshop_initial_adapter_v5/development_report.json`
- 本地 manifest：`artifacts/webshop_initial_adapter_v5/development/run_manifest.json`

现有 completion runner 没有隐式 default resolver，必须显式传入 v5 config。因此，项目的版本化 next-run 指针已经指向 best-profile，但 runner 不会在省略 `--config` 时自动选择它。原 v5 配置、报告和 artifacts 均未修改或覆盖。

