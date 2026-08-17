# HotpotQA 串行 DAG 与 fan-in DAG 受控对比 Demo

## 1. 实验边界

本实验固定同一 HotpotQA 样本、同一组 passages、同一 Qwen3.5-9B、同一
temperature/top-p/seed、相同的四个 Agent 和四次模型调用，只改变 dependency
structure 与对应的 Agent contract。实验直接复用 `AgentGraph`、`AgentRuntime`、
`OpenAICompatibleGateway`、独立 `Format Agent` 和
`hotpotqa.official.answer.v1` evaluator；没有调用 Flow-Director，没有训练，也没有
更新 Skill。

这是一条 training split 中已经用于 architecture development 的样本，只用于展示
执行和通信差异，不是 held-out accuracy 证据。

- Task ID：`hotpotqa:5a7e567b55429949594199a0`
- Question：Who is the American internet entrepreneur who founded the company
  featured on 24 Hours on Craigslist?
- Ground Truth：`Craig Newmark`

## 2. 串行 DAG

```text
documentary_evidence → founder_evidence → verification → format
```

- structural depth：4
- maximum width：1
- API calls：4
- Final Answer：`<answer>Craig Newmark</answer>`
- EM / F1：`1.0 / 1.0`
- prompt tokens：6,682
- completion tokens：270
- total tokens：6,952
- wall-clock latency：2,053.49 ms

实际 artifact routing：

1. `documentary_evidence` 从 passages 确定目标实体为 `Craigslist`。
2. `founder_evidence` 接收该 artifact，确定 founder 为 `Craig Newmark`。
3. `verification` 接收 founder artifact，核验 founder、nationality 和 occupation。
4. `format` 接收 verification artifact，输出 `<answer>Craig Newmark</answer>`。

## 3. fan-in DAG

```text
documentary_evidence ─┐
                      ├→ synthesis → format
founder_candidates ───┘
```

- structural depth：3
- maximum width：2
- topology family：`fan_in`
- topology motifs：`parallel`, `fan_in`
- API calls：4
- Final Answer：`<answer>Craig Newmark</answer>`
- EM / F1：`1.0 / 1.0`
- prompt tokens：6,797
- completion tokens：363
- total tokens：7,160
- wall-clock latency：2,130.39 ms

实际 artifact routing：

1. `documentary_evidence` 与 `founder_candidates` 作为两个 root Agents 并发执行。
2. `documentary_evidence` 输出目标实体 `Craigslist` 及 documentary evidence。
3. `founder_candidates` 输出 `Craig Newmark → Craigslist` candidate pair 及 founder
   evidence。
4. `synthesis` 同时接收两个 upstream artifacts，执行 entity matching 并核验
   nationality/occupation constraints。
5. `format` 接收 synthesis artifact，输出 `<answer>Craig Newmark</answer>`。

## 4. 同题结果对比

| Condition | EM | F1 | Depth | Width | API calls | Prompt tokens | Completion tokens | Total tokens | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serial DAG | 1.0 | 1.0 | 4 | 1 | 4 | 6,682 | 270 | 6,952 | 2,053.49 ms |
| fan-in DAG | 1.0 | 1.0 | 3 | 2 | 4 | 6,797 | 363 | 7,160 | 2,130.39 ms |

在这一次执行中，fan-in DAG 相对串行 DAG：

- EM/F1 没有变化；
- structural depth 从 4 降至 3，maximum width 从 1 增至 2；
- total tokens 增加 208（2.99%）；
- completion tokens 增加 93（34.44%）；
- wall-clock latency 增加 76.89 ms（3.74%）。

单次 wall-clock latency 受服务状态与缓存影响，不能作为统计结论。可以确认的是，
fan-in 的两个 root Agents 在 runtime 中并发执行，随后由 `synthesis` 同时消费两条
upstream messages。

## 5. 结论

该问题属于证据链很短、passages 直接给出 founder relation 的 bridge question。串行
DAG 已能正确完成 entity identification、bridge relation 和 verification；fan-in DAG
也正确，但没有带来 terminal metric 增益，并生成了更长的中间 evidence。因此，本
Demo 不支持在这类简单样本上强制采用非链式 topology。是否使用 fan-in、fan-out 或
reciprocal communication，应在需要独立证据分解、comparison、conflicting evidence
或多来源 aggregation 的样本上通过相同条件的 paired comparison 验证。

完整逐 Agent request、upstream artifact、response、token 和 latency receipt 保存在：

`artifacts/hotpotqa_multiagent_skill/topology_comparison_craig_newmark/paired_topology_demo.json`
