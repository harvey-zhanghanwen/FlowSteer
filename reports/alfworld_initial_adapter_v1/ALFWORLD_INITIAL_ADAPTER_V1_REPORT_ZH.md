# ALFWorld 初版适配报告

## 1. Architecture Completion Report

ALFWorld Initial Adapter v1 已完成。本版本只在统一架构外接入 Dataset /
Environment Adapter、`alfworld.act(command)` Tool、task-scoped environment
runtime、native evaluator 和 paired evaluation；没有修改统一 AgentGraph 的 Agent
定义、Canvas action space，也没有固定具身任务 workflow。

已由真实 receipt 验证的端到端链路为：

```text
ALFWorld task
  -> Qwen3.5-9B Flow-Director
  -> progressive Canvas edit
  -> current AgentGraph execution
  -> Agent communication artifact
  -> ReAct Agent -> alfworld.act(command)
  -> shared task-scoped world state
  -> native observation / admissible actions
  -> official terminal won / episode score
  -> evaluator receipt / trajectory
  -> FINISH or max_rounds
```

完成并验证：

- SkillFlow protocol-v10 task loader 与 train preflight；
- 完整 official `valid_seen=140`、`valid_unseen=134`；
- 每条 rollout 独立 session，多个 Canvas execution 串行共享同一 world state；
- SkillFlow 20-turn ReAct policy budget 与 TextWorld 50-step simulator cap；
- `alfworld` / `act(command)` public Tool schema；
- native terminal `won` Success Rate 与 episode score；
- Direct / AgentGraph 同 task、environment、action budget、model/tool condition、
  evaluator 的 paired evaluation；
- Canvas、Agent I/O、Agent communication、Tool Action--Observation、terminal、
  evaluator 和 Wrong Demo receipts；
- max-rounds trajectory 的完整 native ledger 选择与 0-model-call deterministic
  replay。

预留但未启用：GRPO、LoRA、backward、optimizer update、MACE、Bayesian
posterior、Skill retrieval/injection/evolution。本轮 optimizer update 为 0。

Stable Zero：train-split canary 中 Direct 与 AgentGraph 均以 4 个 native action
完成 `put a handtowel in garbagecan.`，`won=true`、score=1、显式 `FINISH`，因此
adapter 达到受限 Stable Zero。正式能力以完整 official split 分数为准。

## 2. 实现来源与兼容边界

优先级为项目 MD、SkillFlow 实际源码、官方 ALFWorld、FlowSteer。逐项 source map：
[ALFWORLD_INITIAL_ADAPTER_V1_SOURCE_MAP.md](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/docs/ALFWORLD_INITIAL_ADAPTER_V1_SOURCE_MAP.md)。

| 本地边界 | 来源 | 状态 |
|---|---|---|
| task/split/pinned game identity | SkillFlow protocol v10 + ALFWorld inventory | 必要薄适配 |
| reset/step/observation/admissible actions/state/terminal `won` | SkillFlow official bridge + `AlfredTWEnv` | 直接复用语义 |
| `alfworld.act(command)` | SkillFlow embodied Tool contract | 直接复用接口 |
| progressive Canvas/execute-on-edit/feedback/trajectory/`FINISH` | FlowSteer | 直接复用 |
| free-text Agent contract、自主 Agent/model/relation/Output/topology | 项目 MD + 统一 AgentGraph | core 不变 |
| task-scoped session、evaluator receipt、report adapter | 本项目 | 必要薄适配 |

没有预设 `Navigator`、`Manipulator`、`Planner`、`Verifier`，没有强制 chain、
parallel 或 reciprocal topology。ReAct 是 Agent execution mode，不是 Agent role。

## 3. 完整 official evaluation

Primary metric 是 native environment terminal Success Rate。所有任务均
evaluator-valid；Agent 自述、文本答案和 LLM judge 不参与 reward。

| Official split | Condition | Success / Total | SR | Direct 对照 | AgentGraph - Direct |
|---|---|---:|---:|---:|---:|
| `valid_seen` | `alfworld_valid_seen_unified_architecture_v1` | 46 / 140 | **32.86%** | 43 / 140 = 30.71% | **+2.14 pp** |
| `valid_unseen` | `alfworld_valid_unseen_unified_architecture_v1` | 40 / 134 | **29.85%** | 28 / 134 = 20.90% | **+8.96 pp** |

完成状态：

- `valid_seen`：AgentGraph `140/140 completed`、`140/140 evaluator-valid`，
  native terminal success 46，Director `FINISH=46`、`max_rounds=94`；
- `valid_unseen`：AgentGraph `134/134 completed`、`134/134 evaluator-valid`，
  native terminal success 40，Director `FINISH=38`、`max_rounds=96`；
- unseen 有 2 个 episode 已由 environment 证明 `won=true`，但 Director 未发出
  `FINISH`。最长 task-scoped executor ledger 的 deterministic replay 将其从旧的
  38/134 修正为 40/134；该过程模型调用数为 0；
- unseen 历史 3 次 provider connection failure 已在相同冻结条件下恢复，当前
  unresolved provider/collection failure 为 0。

正式证据：

- [valid_seen JSON](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_seen_report.json)
- [valid_seen Markdown](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_seen_report.md)
- [valid_unseen JSON](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_unseen_report.json)
- [valid_unseen Markdown](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/valid_unseen_report.md)

## 4. Environment telemetry 与自然 topology

| Split / Condition | Policy turns | Native actions | Parse-error turns | Immediate repeats | Native terminal episodes |
|---|---:|---:|---:|---:|---:|
| seen Direct | 2,331 | 2,250 | 81 | 134 | 43 |
| seen AgentGraph | 2,259 | 2,176 | 83 | 114 | 46 |
| unseen Direct | 2,354 | 2,227 | 127 | 204 | 28 |
| unseen AgentGraph | 2,212 | 2,059 | 153 | 178 | 40 |

所有 condition 的 environment `invalid_action_count=0`、
`no_effect_action_count=0`：不在当前 admissible-action domain 的 candidate 被记录为
policy-turn parse error，没有伪装成成功 Tool transition。

最终/evaluated topology：

```text
valid_seen:
  empty 6, single 58, serial_2 31, serial_3_plus 5,
  parallel 18, fan_in 13, mixed 6, reciprocal 3

valid_unseen:
  empty 1, single 45, serial_2 39, serial_3_plus 4,
  parallel 25, fan_in 13, fan_out 2, mixed 4, reciprocal 1
```

Director 真实生成了 chain、parallel、fan-in、fan-out、mixed、reciprocal graph；
adapter 没有把 search space 固定为链式结构。当前结果不能证明复杂 topology
本身有效或无效，因为许多 episode 的决定性错误发生在 environment exploration、
object grounding 或 subgoal sequencing。

## 5. Receipt-causal failure taxonomy

每个 native failure episode 只分配一个 primary cause；`max_rounds` 单独作为
terminal manifestation。完整数量、占比、零计数类别和逐类全链路 demo 见：

[ALFWORLD_FAILURE_TAXONOMY_REPORT_ZH.md](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/reports/alfworld_initial_adapter_v1/ALFWORLD_FAILURE_TAXONOMY_REPORT_ZH.md)。

| Primary failure class | valid_seen（n=94） | valid_unseen（n=94） |
|---|---:|---:|
| Environment exploration/search | 37（39.36%） | 35（37.23%） |
| Object grounding/affordance | 32（34.04%） | 40（42.55%） |
| Subgoal sequencing/action policy | 23（24.47%） | 18（19.15%） |
| Tool/execution-profile | 1（1.06%） | 1（1.06%） |
| Director/Canvas construction | 1（1.06%） | 0 |
| Native parser / communication / runtime / evaluator / provider | 0 | 0 |

Agent communication primary failure 为 0，不表示 graph communication 已被充分
利用：部分任务没有 relation；部分 artifact transport 正常但缺少 environment
evidence；部分 Output pointer 没有接到 stateful Tool artifact。这些属于次生结构
问题，不能在 receipt 无依据时虚构为“消息丢失”。

## 6. 当前最高已验证 AgentGraph profile

严格排除 Direct、canary、小样本、prepared-only、未完成、train-heldout、不同 split
和无效 evaluator 后，每个 official split 只有一个合格的完整 AgentGraph 条件，
因此上述两个 `unified_architecture_v1` 条件分别是各自 split 的当前最高已验证
profile。两个 official split 不互相比较，也不合并为一个官方 SR。

版本边界：

- Dataset protocol：`skillflow.protocol.v10`；
- Evaluator：`skillflow.ragen_adapter.v2`；
- Director prompt：`agentgraph.director.minimal-neutral.v10`；
- Tool/runtime：`skillflow.ragen-alfworld.rollout-session.v1`；
- policy：`qwen35-9b-base-alfworld-initial-v1`；
- model catalog：`config/model_catalog_alfworld_paired_qwen35_v1.yaml`。

版本化 best-profile pointer：
[alfworld_agentgraph_best_v1.yaml](/ssd1/iclr/1/.tmp/FlowSteer-alfworld-initial-v1/config/best_profiles/alfworld_agentgraph_best_v1.yaml)。
它按 `official_split` 指向现有可执行配置及其 manifest/report，不复制 condition、
不改名冒充新实验。当前 runner 仍要求显式传入 `--config`，因此该文件是 canonical
next-run pointer，不是 runner 自动消费的 hidden default。

## 7. 结论与已知边界

ALFWorld 初版适配已完成：只新增 task/environment adapter，没有改变统一
orchestration core；Stable Zero 与两个完整 official split 均有 native receipt。

当前主要能力瓶颈是：

1. environment exploration/search 与 no-progress recovery；
2. object grounding 与 state-dependent affordance selection；
3. transformation、state precondition、count、placement、inspection 的 subgoal
   sequencing；
4. typed `execution_mode`、唯一 stateful Tool owner、`allowed_tools` 与 reciprocal
   block 之间的联合 Canvas action legality 尚未进入结构化 action mask。

本轮没有根据 evaluation error 固定 ALFWorld workflow，没有训练，没有 Skill
注入或 evolution，也没有将 `max_rounds`、Agent 自述或无效 evaluator 伪装成 task
reward。
