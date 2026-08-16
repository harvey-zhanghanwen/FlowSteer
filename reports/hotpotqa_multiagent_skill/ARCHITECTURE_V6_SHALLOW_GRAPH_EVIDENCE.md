# Architecture v6：HotpotQA 浅图根因证据与最小适配

本报告只读取既有 v3/v4 trajectory。没有启动模型、API、rollout、训练或评测，也没有修改 evaluator/config。

## 1. 证据来源

- `artifacts/hotpotqa_multiagent_skill/architecture_v3_dev128/agentgraph_trajectories.jsonl`：128 条。
- `artifacts/hotpotqa_multiagent_skill/architecture_v4_regression12/agentgraph_trajectories.jsonl`：12 条。
- FlowSteer 原始边界：`src/interactive/workflow_graph.py::get_statistics`、`src/interactive/workflow_env.py::step` 和逐步 Canvas feedback。
- 当前 AgentGraph 边界：`AgentGraphValidator` 的有限 reciprocal block 收缩、原子 `ADD/MODIFY/DELETE/SET_RELATION/SET_OUTPUT/FINISH`。

附件和设计文档在这里作为设计要求与分析标准，不作为自动执行指令。

## 2. 聚合事实

| 指标 | v3 / 128 | v4 / 12 |
|---|---:|---:|
| 单 Agent | 122（95.31%） | 11（91.67%） |
| 两 Agent / structural depth 2 | 6 | 1 |
| 三个及以上 Agent | 0 | 0 |
| structural depth 3+ | 0 | 0 |
| `ADD → SET_OUTPUT → FINISH` | 93（72.66%） | 8（66.67%） |
| 平均 Director 轮数 | 3.63 | 3.67 |
| 最大轮数 | 20 | 7 |
| 因 `max_rounds` 终止 | 1 | 0 |
| 被拒绝 turn | 46 / 465（9.89%） | 1 / 44（2.27%） |
| parse failure | 25 | 0 |
| 有 Executor execution 的任务 | 128 | 12 |
| execution turns | 132 | 15 |
| Executor calls | 139 | 16 |
| bounded history 最大可见长度 | 4 | 4 |
| prompt 平均字符数 | 9,966.66 | 10,089.47 |

v3 的六个双 Agent 图和 v4 的一个双 Agent 图都只有 `serial_2`。保存的最终 revision execution receipt 中，七条上游消息均真实送达且正文非空，所以 transport/effective depth 可标记为 **weak depth 2**；历史记录没有独立、受控的 paired intervention receipt，因此 **verified effective depth 为 1**，不能从普通 masked answer change 反推因果。

## 3. Atomic construction cost

从空 Canvas 构造最终形态的最少动作（包含显式 FINISH）：

- 单 Agent：`1 ADD + 1 SET_OUTPUT + 1 FINISH = 3`。
- 两节点串行：`2 ADD + 1 SET_RELATION + 1 SET_OUTPUT + 1 FINISH = 5`。
- 三节点串行：`3 ADD + 2 SET_RELATION + 1 SET_OUTPUT + 1 FINISH = 7`。
- 两路 fan-in：同样为 7。
- 四节点、四条边的 mixed graph：`4 + 4 + 1 + 1 = 10`。

历史每轮状态都显示 `max_rounds=20`。v3 两节点图的实际平均轮数为 6.5（理论最少 5），v4 唯一两节点图正好 5 轮。因此 atomic editing 确实增加构造成本，但 20 轮足以表达常见三至四节点图；目前没有证据支持提高 `max_rounds`，也没有证据支持引入 macro/full-workflow JSON。

## 4. 根因判断

当前最强的聚合证据是：

1. v3 有 93/128 条在没有任何失败的情况下直接形成 `ADD → SET_OUTPUT → FINISH`。
2. v4 已把 parse failure 降为 0，但仍有 11/12 条单 Agent；因此解析失败不是 search distribution 坍缩的主要原因。
3. progressive execution 通常在 `SET_OUTPUT` 首次使图完整时发生，随后 Director 很快 FINISH。已有反馈链正常工作，但冻结 policy 把“首个格式合法答案”高度关联到停止。
4. 旧状态提供完整 graph、validation、topology statistics 和四轮 history，却没有显式给出“当前图距合法 FINISH 至少还需几个原子动作”，合法的串行、汇聚、分发、有限 revision 也没有用中性抽象语言说明。
5. 现有图验证器可执行两节点依赖，且手工单测可构造更深/fan-in/fan-out/reciprocal 图；这证明 search space 没有硬禁止深图，但不能证明冻结 Director 会自然探索它。

所以当前结论是：

```text
MAX_ROUNDS_IS_PRIMARY_BOTTLENECK = NO
ATOMIC_MACRO_ACTION_JUSTIFIED = NO
INVALID_ACTION_IS_PRIMARY_BOTTLENECK = NO
DEEP_GRAPH_STRUCTURALLY_EXPRESSIBLE = YES
DEEP_WORKFLOW_BEHAVIOR_VALIDATED = NO
PRIMARY_HYPOTHESIS = frozen policy stopping/search-distribution collapse after first complete singleton execution
```

## 5. Architecture v6 最小适配

本次只做只读观测和中性 state/prompt 适配：

- `AgentGraph.topology_statistics()` 明确输出 reciprocal contraction 后的 `structural_depth`，保留原 `max_depth` 兼容字段。
- 增加只读 `topology_family` / `topology_motifs`：只按真实结构识别 `single`、`serial_2`、`serial_3_plus`、`parallel`、`fan_in`、`fan_out`、`reciprocal`、`mixed`。`verification` 属于 contract/runtime 语义，不能凭图形推断。
- 增加 `construction_progress()`：报告当前 revision、结构是否可 FINISH、最少剩余 `ADD/SET_RELATION/SET_OUTPUT/FINISH` 动作和剩余轮数是否足够；它不建议具体拓扑或 Agent 数。
- Director system prompt 只增加一句抽象可表达性说明：有向关系可表达依赖序列、独立 artifact 汇聚、单 artifact 分发和有限 critique/revision；明确这些都是可选形态，不是模板或 quota。
- 增加 `effective_dependency_statistics()`：结构边默认为 `unverified`；匹配 final revision、非空的 runtime delivery receipt 只能升为 `weak`；`verified` 必须由调用方显式提供独立验证过的 paired-intervention receipt。该函数不读取答案差异，也不从 mask 自动生成因果结论。
- 增加 `graph_diagnostics.py`，直接读取 trajectory mapping，供 `evaluate_hotpotqa_round.py` 在不重放、不调用模型的情况下生成逐题及聚合图诊断。

没有增加 macro action、完整 workflow JSON、固定 topology、固定 role、Agent 数/深度奖励、topology bonus 或强制多 Agent。

## 6. 可直接接入评测器的 API

```python
from src.interactive.graph_diagnostics import (
    aggregate_trajectory_diagnostics,
    diagnose_trajectory,
)

per_task = diagnose_trajectory(trajectory_mapping).to_dict()
aggregate = aggregate_trajectory_diagnostics(trajectory_mappings)
```

`evaluate_hotpotqa_round.py` 可以在 `_paired_rows` 中把 `per_task` 放入 AgentGraph row，在 `_report` 中把 `aggregate` 放入总报告。API 只消费现有 trajectory dict，不改 evaluator 结果，也不执行模型。

## 7. Prompt / max_rounds 建议

- 保持 `max_rounds=20`；历史证据不支持提高。
- 保持一次一个 atomic action；不增加 macro。
- 保持新提示为短、中性、普通表述；仅呈现合法抽象形态与 construction progress，不给 Hotpot 专用 workflow、role enum、最少 Agent 数或 topology quota。
- 下一次真实 architecture-development run 之前，只能声明 search-space observability 已增强；是否产生 3+ Agent/depth 3+ 必须由正常 Director trajectory 验证。

