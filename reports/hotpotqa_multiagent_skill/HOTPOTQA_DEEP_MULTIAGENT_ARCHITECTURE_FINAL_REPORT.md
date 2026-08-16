# HotpotQA 深层多智能体架构最终报告

日期：2026-08-16

分支：`dataset/hotpotqa-multiagent-skill-step0-to-n`

范围：仅 HotpotQA；未进入其他六个数据集。

## 0. 结论摘要

本轮把架构从“可运行的浅层 AgentGraph”推进到了一个可执行、可记录、可微训练的
Formal Step 0→3 闭环，但**尚不能判定 HotpotQA 多智能体架构适配完成**。

- Executor 池已真实扩大：10 个 evidence-backed、Director 可见且实际用过的模型臂，
  覆盖 Qwen、DeepSeek、GPT、MiniMax、GLM、Kimi 六个家族；Director 始终是本地
  Qwen3.5-9B。
- 原子 Canvas、深图表达、fan-in/fan-out/mixed 的结构能力已接通；正常 dev 中首次
  出现 3-Agent 图。但是实际搜索仍严重坍缩：最终 Step3 有 122/128 singleton，唯一
  structural depth=3 的图未完成执行，其 effective depth 只有 1、状态 unverified。
- 真实异构多模型图已出现，但非常稀少：Step0/1/2/3 分别只有 4/2/2/2 个跨家族图。
- 通信 transport 正常：四步所有指向 Output 的 36 条最终边都产生了非空 inbox
  delivery，方向错配为 0。可是 Step3 的 5 对 Normal/Upstream-Masked 消融中，答案和
  EM/F1 全部不变；因此没有观察到 communication utility。
- 生产 Skill 闭环未完成：Trajectory/Evidence 写入存在，但 paired probe、candidate、
  independent validation、ACTIVE、retrieval 与 gain 均为 0，不能把单测中的合成 Skill
  当作 HotpotQA 生产证据。
- Formal Step1/2/3 各真实执行了一次 `optimizer.step()`，LoRA 更新 L2 均大于 0，
  optimizer continuation、adapter publish、route switch、canary 均成功。但固定 dev128
  曲线不单调，最终与最佳点都没有超过 Local Direct。

固定同题、同 evaluator 的最终结果：

| 条件 | 正确数 | EM | F1 |
| --- | ---: | ---: | ---: |
| Local Qwen3.5-9B Direct | 93/128 | 72.66 | 82.08 |
| Formal Step0 AgentGraph | 86/128 | 67.19 | 78.84 |
| 最佳 AgentGraph（Step2） | 89/128 | 69.53 | 80.18 |
| 最终 AgentGraph（Step3） | 88/128 | 68.75 | 79.77 |

因此按用户要求在 Step3 停止；没有机械继续 Step4–20，也没有启动下一数据集。

## 1. 口径、可比性与 Stable Zero

### 1.1 固定评测口径

- 样本：项目冻结的 128 条 HotpotQA held-out validation，顺序固定。
- 原始来源标签：HotpotQA train；这是项目开发切分，不是官方完整 test benchmark。
- 输入：每题完整十段 context。
- evaluator：`skillflow.training.reward.v1`，128/128 valid；terminal reward 为 token F1，
  同时记录 EM。
- Direct：Step0 的同一 128 条本地 Qwen3.5-9B 结果；Step1–3 直接复用，不重复调用。
- 四步逐题的 Task ID、Question、Ground Truth、Direct answer、Direct EM/F1 完全一致。
- 每步都绑定独立 policy/adapter receipt：

```text
qwen35-9b-hotpot-step-000000 / theta_hotpot_step_000000
→ qwen35-9b-hotpot-step-000001 / theta_hotpot_step_000001
→ qwen35-9b-hotpot-step-000002 / theta_hotpot_step_000002
→ qwen35-9b-hotpot-step-000003 / theta_hotpot_step_000003
```

### 1.2 Stable Zero

Formal Step0 是固定初始化、`optimizer_updates=0` 的真实未训练 adapter。两个预先固定
canary 均完整经过：

```text
Question
→ local Qwen3.5-9B Director
→ atomic Canvas
→ AgentGraph Runtime
→ Output Agent
→ Hotpot evaluator
→ versioned Trajectory/Evidence
```

两题均保存完整 Director turn、seed/token/logprob receipt、Agent execution、Output inbox、
explicit FINISH 和 evaluator receipt，故运行链 `STABLE_ZERO=YES`。这只证明链路稳定，
不等于深层多智能体行为已经有效。另需注意：现存 Step0 preflight 证明 adapter 在
SGLang 中可见且 canary 成功，但不是完整 pause/drain/route-switch activation receipt。

## 2. Model Pool Audit

### 2.1 真实 provider 与纳入边界

- 最新 `/v1/models` receipt：HTTP 200，524 个对象。
- 本轮不是对 524 个对象做全量付费 canary；经 exact ID、text/OpenAI endpoint、当前或
  既有成功 execution/canary 证明并进入目录的模型为 9 个远程 + 1 个本地。
- 新增 canary：`glm-4.5-flash`、`kimi-k2`，各一次，均精确通过
  `<answer>Paris</answer>`；retry=0。
- Gemini：真实列表无 exact ID，未猜 alias。
- Grok：旧条目最近 canary 为 HTTP 429，未纳入。
- embedding、reranker、audio/video/image-only、失效 alias 均未进入 text-QA 池。

| Model | Provider | Family | Evidence | Director-visible | Step0–3 used |
| --- | --- | --- | --- | ---: | ---: |
| `qwen3.5-9b-local` | local SGLang | Qwen | 既有 Hotpot execution | 是 | 是 |
| `qwen3.5-flash` | VectorEngine | Qwen | 既有 execution | 是 | 是 |
| `qwen3.5-plus` | VectorEngine | Qwen | canary pass | 是 | 是 |
| `deepseek-v4-flash` | VectorEngine | DeepSeek | canary pass | 是 | 是 |
| `deepseek-v4-pro` | VectorEngine | DeepSeek | canary pass | 是 | 是 |
| `gpt-4o-mini` | VectorEngine | GPT | 既有 execution | 是 | 是 |
| `MiniMax-M2.5` | VectorEngine | MiniMax | 既有 execution | 是 | 是 |
| `MiniMax-M3` | VectorEngine | MiniMax | canary pass | 是 | 是 |
| `glm-4.5-flash` | VectorEngine | GLM | 本轮 canary pass | 是 | 是 |
| `kimi-k2` | VectorEngine | Kimi | 本轮 canary pass | 是 | 是 |

Director 的真实 rendered state 暴露 exact ID 及中性的 provider/locality/reasoning class/
latency/context/instruction/concise/availability facts；10 个模型的 selection/cheap/fast
weight 均为 1.0。没有题型→模型映射、preferred-model hint 或强制异构规则。

证据：

- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/model_list_receipt.json`
- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/canary_receipt.json`
- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/director_visible_catalog_receipt.json`
- `reports/hotpotqa_multiagent_skill/DIRECTOR_VISIBLE_MODEL_CATALOG_AUDIT.md`

## 3. Graph Depth Evolution

Structural depth 是收缩 finite reciprocal block 后的最长有向依赖路径；effective depth
只把实际 runtime delivery 计为 weak，只有独立 paired intervention 才能升级为 verified。

| Step | Mean / Median / Max structural depth | Depth 1 | Depth 2 | Depth 3 | Depth 4+ | Mean / Max effective depth | Evidence status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Step0 | 1.070 / 1 / 2 | 119 | 9 | 0 | 0 | 1.070 / 2 | 9 weak |
| Step1 | 1.047 / 1 / 2 | 122 | 6 | 0 | 0 | 1.047 / 2 | 6 weak |
| Step2 | 1.117 / 1 / 2 | 113 | 15 | 0 | 0 | 1.117 / 2 | 15 weak |
| Step3 | 1.055 / 1 / 3 | 122 | 5 | 1 | 0 | 1.039 / 2 | 5 weak，1 unverified |

Step3 最终深度条件指标：

| Depth | Tasks | AgentGraph EM / F1 | 同题 Direct EM / F1 |
| --- | ---: | ---: | ---: |
| 1 | 122 | 70.49 / 81.40 | 73.77 / 83.00 |
| 2 | 5 | 40.00 / 56.00 | 40.00 / 56.00 |
| 3 | 1 | 0.00 / 0.00 | 100.00 / 100.00 |
| 4+ | 0 | 不可测 | 不可测 |

Step3 唯一 depth=3 图在 20 turns 内发生 Output 可达性/关系修复循环，最终
`termination_reason=max_rounds`，没有执行 Output；因此不能把 structural depth=3 宣传成
三阶段 reasoning 成功。

### 3.1 原子构造成本

复杂图仍使用 FlowSteer 单原子编辑，没有 macro workflow JSON：

- 1 Agent 的理论最少动作 3，Step3 实际平均 3.30 turns；
- 2 Agent 的理论最少动作 5，Step3 实际平均 5.20 turns；
- 3 Agent 的理论最少动作 7，Step3 唯一样本用了 20 turns，overhead=13。

`max_rounds=20` 足以容纳正常三节点最短构造；坍缩的主因不是硬上限，而是冻结/微更新
policy 偏向首个完整 singleton，以及复杂图关系修复的低稳定性。Step3 的 parse failures
为 13、rejected turns 为 27，均比 Step2 的 5/18 恶化。

## 4. Topology Diversity 与 Agent Count

| Step | Single | Serial-2 | Deep serial | Parallel | Fan-in | Fan-out | Verification | Reciprocal | Mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step0 | 119 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Step1 | 122 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Step2 | 113 | 14 | 0 | 1 个图含 parallel motif | 1 | 0 | 0 | 0 | 0 |
| Step3 | 122 | 5 | 0 | 0 | 0 | 1 个失败图含 motif | 0 | 0 | 1（失败） |

| Step | 1 Agent | 2 Agents | 3 Agents | 4+ Agents |
| --- | ---: | ---: | ---: | ---: |
| Step0 | 119 | 9 | 0 | 0 |
| Step1 | 122 | 6 | 0 | 0 |
| Step2 | 113 | 14 | 1 | 0 |
| Step3 | 122 | 5 | 1 | 0 |

3+ Agent 首次真实出现，Step2 的 3-Agent fan-in 也确实执行了三个节点；但是它把一个
无关的 1977 artifact 与正确的 1984 artifact 一并送到 Output，且 contract 只要求年份，
最终回答 `1984` 而非 Ground Truth `July 5, 1984`。这说明“能生成复杂图”成立，
“复杂图分工正确且有益”不成立。

## 5. Agent Role / Contract

Contract 仍是自由文本，不是固定 Operator/role enum；Director prompt 只要求它能说明
objective、input/dependency、artifact、completion，不注入 Hotpot 模板。

真实 Step2 fan-in：

```text
retriever:     提取无关人物的出生年份（错误分解） → 1977
retriever_woo: 提取 Yeon Woo-jin 出生年份          → 1984
answerer:      消费上游年份，只输出 year           → <answer>1984</answer>
```

优点是三者 artifact/依赖可以区分；失败点是 Director 一开始定义错任务粒度和答案类型，
并建立了无关分支。Step3 的主要系统性问题也不是 contract 字段不存在，而是 contract
没有稳定约束“完整日期/实体/短答案规范化”，以及少数 Output 没有复用已正确收到的实体。

`role_family` 仅作为分析 metadata；没有硬编码角色集合。当前真实运行没有足够证据支持
对 role family 作因果排序。

## 6. Multi-Model Collaboration

| Step | Multi-Agent graphs | 不同 model-ID 图 | 跨 family 图 | 跨 family / Multi-Agent |
| --- | ---: | ---: | ---: | ---: |
| Step0 | 9 | 4 | 4 | 44.4% |
| Step1 | 6 | 3 | 2 | 33.3% |
| Step2 | 15 | 2 | 2 | 13.3% |
| Step3 | 6 | 2 | 2 | 33.3% |

真实跨家族组合包括：

- Step0：Kimi+Qwen、DeepSeek+MiniMax、DeepSeek+Qwen、GPT+Qwen；
- Step1：DeepSeek+GPT、Kimi+Qwen；另有两个不同 Qwen model-ID 的同家族图；
- Step2：2 个 DeepSeek+Kimi，其中一个是 3-Agent fan-in；
- Step3：DeepSeek+Qwen、GPT+MiniMax。

10 个 arm 在每一步都有实际节点/调用，但这只证明 pool 与 renderer 接通。不同 Step 对
同题的 Output route 高频变化，且非随机 routing 未做同图反事实控制，因此不能把粗粒度
模型准确率写成模型排行榜。

## 7. Communication

### 7.1 Transport 与 envelope

真实 Output-directed delivery：Step0/1/2/3 分别为 9/6/16/5，合计 36/36 非空、方向
错配 0。保存字段包括：

```text
source_agent_id
target_agent_id
message_type
artifact/content
graph_revision
request_or_dependency
```

这证明上游信息确实进入下游 prompt，并非只画结构边。实际正常轨迹的 `message_type`
全部是自由文本 `artifact`；evidence/entity/critique/verification 等更细类型没有由模型真实
产出，不能凭空填充。

### 7.2 Step3 Normal vs Upstream-Masked

从 6 个 Multi-Agent 图中选择 5 个有可执行 Output path 的冻结图；另 1 个是 max-round
终止，无法构成配对。诊断跳过 Director、Direct、训练和 publish：

| 指标 | Normal | Upstream Masked | Masked − Normal |
| --- | ---: | ---: | ---: |
| 有效 pairs | 5 | 5 | — |
| EM | 40.00 | 40.00 | 0.00 |
| F1 | 56.00 | 56.00 | 0.00 |
| Raw answer changed | — | — | 0/5 |
| Normal correct → Masked wrong | — | — | 0/5 |

10 个执行臂全部成功、failure=0、`diagnostic_only=true`、`grpo_eligible=false`。因为题目和
context 在 masked 条件仍存在，“无变化”不能证明模型从未读取消息；但它明确表示本轮
**没有观察到上游内容的行为效用**，所以 dependency 只能标 weak/unverified。

证据：`reports/hotpotqa_multiagent_skill/architecture_v6_formal_step3_communication.json`。

## 8. Skill Pipeline

代码层已把以下现有边界串联：

```text
Trajectory
→ Candidate Discovery
→ Structured Candidate
→ Paired Evidence
→ problem-disjoint Validation
→ Evidence Gate
→ Lifecycle
→ ACTIVE
→ rejectable Director retrieval prior
```

并要求 condition/action 能绑定 task context、graph prefix、role、model、relation motif、
graph position、版本、interval/uncertainty、effective problem count 和 failure scope。forced
probe 永久 `grpo_eligible=false`，Skill 不能直接修改 Canvas。

但是本轮**生产数据层没有走通这条链**：

- 四次 dev 的 `probes.jsonl=0`、`posteriors.jsonl=0`；
- Step1/2/3 微训练 evidence 中同样没有 probe/posterior；
- 没有 production CANDIDATE、validated candidate、ACTIVE、retrieval receipt 或 gain；
- 因此不存在可审计的 Skill summary，也不存在 Skill preliminary gain。

结论：Skill 代码/接口存在且有定向测试，不等于生产端到端 ready。

## 9. Architecture Iteration History

| Version | 数据 | 主要问题/修改 | AgentGraph EM/F1 | 说明 |
| --- | --- | --- | ---: | --- |
| v1 | 定向 14 | 初始 Output/continuation/通信诊断 | 35.71 / 52.86 | 与后续全量不可直接比较 |
| v2 | 同类定向 14 | Output 协议与 Agent 通信薄适配 | 50.00 / 61.84 | 与 v1 Direct 口径也有差异，不作训练曲线 |
| v3 | dev128 | 完整通信、terminal、trajectory 接线 | 67.97 / 80.23 | 122 single、6 serial-2 |
| v4 | regression12 | FlowSteer 风格 no-op feedback 修复 | 25.00 / 46.67 | 仅失败子集，不是全量分数 |
| v5 | 静态 | 直接移植 SkillFlow scientific sampling coordinate/seed | 未运行新全量 | 修复 subset/order 改变采样的问题 |
| v6 Step0 | 固定 dev128 | 10-arm pool、construction progress、depth/effective diagnostics、Skill wiring、Formal Step0 | 67.19 / 78.84 | 0 update |
| v6 Step1 | 同一 dev128 | 1 次真实微更新 | 64.84 / 76.93 | 下降 |
| v6 Step2 | 同一 dev128 | exact optimizer continuation，第 2 次更新 | **69.53 / 80.18** | 本轮最佳；1 个 3-Agent fan-in |
| v6 Step3 | 同一 dev128 | exact continuation，第 3 次更新 | 68.75 / 79.77 | 回退；1 个失败 depth-3 mixed |

修改前提交/分支/clean diff 已保存在
`ARCHITECTURE_V6_PRECHANGE_SNAPSHOT.md`，独立备份分支为
`backup/hotpotqa_before_deep_multiagent_arch_v1`。v6 architecture 已单独提交；本报告、
Step1–3 与最终 artifacts 位于本文件所属的后续 HotpotQA 分支提交，不覆盖旧版本。

Git 备份包含代码、配置、报告、完整 AgentGraph trajectories、paired/wrong/communication
结果、训练 behavior/post-update trajectories、group/summary/sync/cursor receipts，以及
Step0–3 的 LoRA adapter。每个约 121 MiB 的 `optimizer_state.pt` 超出普通 Git 单文件边界，
因此只保留在本机 versioned checkpoint 目录，没有伪装成已上传；精确 continuation 已由
训练 summary、cursor、sync receipt 和本地文件保留并验证。

## 10. Micro-Training Curve 与训练真实性

### 10.1 固定 dev 曲线

| Step | Policy | 正确/128 | EM | F1 | 相对前步 EM/F1 | 相对 Direct EM/F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `...000000` | 86 | 67.19 | 78.84 | — | -5.47 / -3.24 |
| 1 | `...000001` | 83 | 64.84 | 76.93 | -2.34 / -1.91 | -7.81 / -5.14 |
| 2 | `...000002` | 89 | 69.53 | 80.18 | +4.69 / +3.25 | -3.13 / -1.89 |
| 3 | `...000003` | 88 | 68.75 | 79.77 | -0.78 / -0.42 | -3.91 / -2.31 |

同题翻转：Step0→1 为 11 错转对/14 对转错；Step1→2 为 16/10；Step2→3 为
12/13。61 题四步恒对、20 题恒错、47 题至少翻转一次。曲线不是稳定学习趋势。

### 10.2 真实 optimizer / adapter 数据

| Update | 训练任务 | Rollout / eligible / trained | Reward mean | Loss | Grad norm | LoRA update L2 | Optimizer | Publish/canary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Step1 | 1 | 8 / 8 / 8 | 0.5513 | -0.062864 | 1.3726 | 0.027059 | fresh，saved | 成功 |
| Step2 | 1 | 8 / 7 / 7 | 0.9375 | -0.047598 | 1.2184 | 0.026665 | restored，saved | 成功 |
| Step3 | 1 | 8 / 8 / 8 | 0.8333 | 0.007014 | 0.9972 | 0.022216 | restored，saved | 成功 |

三步各 1 个 exact/informative/trained group，真实执行 1 次 `optimizer.step()`；每步均
pause/drain、load candidate、canary、route switch、unload old adapter 后才提交 cursor。
Step1/2 的失败尝试在零更新/零 checkpoint/零 publish 边界严格续跑，复用了原 8 条
冻结 behavior rollouts，没有重复初始付费采样。

Provider logprob 数值不进入 loss，也不因超过 0.25 就拒绝 exact group；原始采样 token
ID、executed action mask、route/version/evaluator receipt 和 learner teacher forcing 才是
科学输入。数值漂移仍保存为诊断：Step1/2/3 的 max 为 0.4158/0.2263/0.3383，超过
0.25 的 token 为 2/0/3。provider receipt 的存在、shape、finiteness 仍参与 admission，
不能写成“完全忽略”。

### 10.3 已知训练兼容性缺口

1. Step2 一条轨迹首轮是合法 malformed sampled action，mask=0，之后自然 FINISH 且
   reward=1；当前 all-turn gate 保留 artifact 但排除整轨，只训练 7/8。这与 SkillFlow
   “invalid actions remain data”和 FlowSteer 对 model response 建 mask 的边界不完全一致。
2. SkillFlow 的 Qwen3.5 loader 强制检查 `flash-linear-attention==0.5.2` 与 gated-delta
   kernel；本项目 loader 尚未调用该 enforcement helper。运行环境有依赖，不等于项目
   代码已经证明 FLA 一致性。
3. 本轮每步只有一个不可拆分 group，partition token costs 分别为
   `[100267,0]`、`[92330,0]`、`[112661,0]`；第二 gradient replica 没有实际 group 分片
   负载，因此本轮不能证明多组分批反向扩展性。

这些缺口不否定三次真实权重更新，但禁止把实现描述为 SkillFlow 全量训练架构的直接复刻。

## 11. Cost 与调用分布

下列 latency 是保存 receipt 的累计请求 latency，不是并发后的 wall-clock。

### 11.1 每步 dev128 AgentGraph

| Step | Director turns | Executor calls | API attempts | Retry | Calls/task | Input tokens | Output tokens | 累计 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 438 | 143 | 582 | 1 | 4.55 | 2,049,601 | 97,280 | 1,887.7 s |
| 1 | 423 | 137 | 562 | 2 | 4.39 | 1,968,934 | 104,975 | 1,815.2 s |
| 2 | 463 | 152 | 617 | 2 | 4.82 | 2,172,212 | 104,133 | 1,877.1 s |
| 3 | 448 | 140 | 590 | 2 | 4.61 | 2,090,564 | 130,378 | 2,263.7 s |

Direct baseline 的原始 128 records 含 128 calls、209,485 input、1,361 output、69.2 s；
Step1–3 新增 Direct calls=0。

### 11.2 Step0–3 Executor model 累计

| Model | Calls | Retry | Input tokens | Output tokens | 累计 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3.5-9b-local` | 65 | 0 | 105,616 | 1,175 | 24.3 s |
| `qwen3.5-flash` | 120 | 2 | 212,390 | 296,603 | 3,453.3 s |
| `qwen3.5-plus` | 31 | 2 | 54,406 | 64,618 | 1,554.8 s |
| `deepseek-v4-flash` | 101 | 0 | 160,206 | 2,294 | 629.3 s |
| `deepseek-v4-pro` | 30 | 0 | 49,019 | 240 | 174.7 s |
| `gpt-4o-mini` | 98 | 0 | 153,607 | 1,792 | 219.5 s |
| `MiniMax-M2.5` | 8 | 0 | 11,933 | 1,940 | 59.3 s |
| `MiniMax-M3` | 11 | 0 | 20,044 | 214 | 80.6 s |
| `glm-4.5-flash` | 21 | 2 | 34,334 | 7,321 | 317.8 s |
| `kimi-k2` | 87 | 1 | 158,724 | 1,124 | 475.0 s |

微训练每步包含 8 behavior + 1 post-update canary trajectory，Step1/2/3 分别产生
36/41/42 个 Director+Executor 调用且无 retry。最终通信消融另有 20 个 Executor calls，
36,078 input、11,252 output、218.3 s、retry=0。模型目录本轮外部调用为 1 次 list +
2 次新 canary。

## 12. Wrong Demo 与 Root Cause

| Task ID / 问题 | 现象 | 首个错误点 | 归因 |
| --- | --- | --- | --- |
| `5a8d734...` Milhouse namesake | 证据一直指向 Nixon，输出 `Richard Nixon`/整句而 EM=0 | Output canonicalization 与 Ground Truth 粒度 | Output contract/eval surface |
| `5a832d2...` British intelligence base | inbox 已给 Cheltenham，Output 选 `The Doughnut` | 从正确证据选错实体粒度 | Output selection |
| `5ac1f7f...` brewery regulations | 上游给正确短语，Output 扩写整句；另一步输出中文 | “concise sentence” contract 与语言 route | Contract/routing |
| `5a7a0d4...` actor birth date | 3-Agent fan-in 含无关 1977，contract 只要 year，输出 1984 | Director decomposition/answer type | Architecture policy |
| `5ae7a9c...` steeplechase event | reader 首先误解 aside-from 语义，synthesizer忠实传递 | 上游语义推理 | Model/reasoning |
| `5ae27ed...` film director | 缺失标题时 researcher 自行推断错误电影/导演 | 证据不足下幻觉 | Model/data ambiguity |
| `5abbbd0...` Maurice Hines brother | inbox 有 Gregory Hines，Output 回答 `dancing` | Output 未复用正确实体 | Output/contract |
| `5ac2c35...` Chang Ucchin | 无关 meta Agent、不可达 Output、重复/反向 relation，20 turns | Director graph planning/terminal recovery | Architecture policy |

全量 receipt 未发现污染题序、Ground Truth、Direct comparator、evaluator、adapter 或通信
方向的代码 bug。`failure_type=executor_or_provider_failure` 表示历史中曾发生过错误，可能
已重试成功；不能等同最终 Wrong Demo。当前主要瓶颈分为：

1. **Architecture behavior**：singleton collapse、复杂关系编辑不稳定、答案类型/实体/短
   答案 contract 不稳定、Step3 stopping/action parse 退化；
2. **Policy learning / optimization**：仅 3 个训练任务、24 条 behavior rollout、每步一个
   informative group，方差极大，不足以学出稳定复杂图策略；
3. **Model capability**：仍有 shared semantic reasoning failure；
4. **Surface mismatch**：每步多数 EM-wrong 仍有 F1>0，说明不少错误是近义/过长输出，
   但不能因此更改官方 evaluator。

综合分类：`POLICY_LEARNING_PROBLEM + OPTIMIZATION_LIMIT`，同时保留上述明确架构行为
问题；现有证据不足以靠继续堆规则或 Step4–20 解决。

## 13. Untouched Confirmation

本轮**没有运行**新的 `hotpotqa_final_untouched_confirmation`。原因是最终 Step3 没有达到
Local Direct，且深图、communication、Skill 验收均未通过；此时消费一次性 untouched
slice 不具备实验意义。

```text
FINAL_UNTOUCHED_RUN = NO
FINAL_UNTOUCHED_RESULT = NOT_MEASURED
```

不能把未运行解释成 0 分，也不能声称 final untouched 超过 Direct。

## 14. Source Mapping 与实现归类

详细逐文件映射见 `docs/SOURCE_MAP.md`。关键边界：

| 模块 | 归类 | 上游来源/说明 |
| --- | --- | --- |
| Progressive Canvas、single atomic action、revision/feedback/FINISH | FlowSteer 边界直接复用 + free-AgentGraph 薄适配 | 原版固定 Operator 无法表达自由 model-labelled Agent |
| Qwen3.5/PEFT/SGLang Supervisor | SkillFlow 结构与 publish 事务复用 + single-theta 适配 | FlowSteer 的 Qwen3-8B/vLLM 不兼容 |
| exact sampling/receipt、frozen cursor、optimizer continuation | SkillFlow 直接边界或 dependency-light port | 绑定本地 Hotpot aligned JSONL 与 AgentGraph rollout ordinal |
| action-masked one-pass terminal GRPO | FlowSteer mask 基础 + 用户 MD 项目算法 | 不复用 SkillFlow TTB/backward/Z；无结构/模型/Skill shaping |
| graph depth/topology/effective dependency | FlowSteer statistics 的诊断适配 | 只诊断，不进入 reward |
| MACE/Bayesian/paired probe/EVSI | 用户 MD 项目算法接口 | 本轮未激活 |
| Skill paired evidence gate | SkillFlow store/library/retriever + 用户 MD effect gate | 代码接线存在，生产运行未端到端 |

## 15. 最终判定

```text
MODEL_POOL_EXPANDED = YES

DEEP_GRAPH_SEARCH_SPACE_READY = YES
DEEP_WORKFLOW_BEHAVIOR_VALIDATED = NO

THREE_PLUS_AGENT_WORKFLOWS_OBSERVED = YES
HETEROGENEOUS_MULTI_MODEL_WORKFLOWS_OBSERVED = YES

SERIAL_COLLABORATION_READY = YES
PARALLEL_COLLABORATION_READY = NO
FAN_IN_OUT_READY = NO
VERIFICATION_COLLABORATION_READY = NO
RECIPROCAL_COLLABORATION_READY = NO
COLLABORATION_DIVERSITY_VALIDATED = NO
WORKFLOW_DIVERSITY_READY = NO

COMMUNICATION_TRANSPORT_READY = YES
COMMUNICATION_DEPENDENCY_READY = NO
COMMUNICATION_UTILITY_OBSERVED = NO

SKILL_END_TO_END_READY = NO
SKILL_SUMMARY_VALIDATED = NO
SKILL_PRELIMINARY_GAIN_OBSERVED = NO

MICRO_TRAINING_EXECUTED = YES
LEARNING_TREND_OBSERVED = NO

FINAL_DEV_EM = 68.75
FINAL_DEV_F1 = 79.77
BEST_DEV_STEP = 2
BEST_DEV_EM = 69.53
BEST_DEV_F1 = 80.18

LOCAL_DIRECT_EM = 72.66
LOCAL_DIRECT_F1 = 82.08

FINAL_DEV_BEATS_LOCAL_DIRECT = NO
FINAL_UNTOUCHED_BEATS_LOCAL_DIRECT = NO (NOT MEASURED)

HOTPOTQA_ARCHITECTURE_ADAPTED = NO
READY_FOR_FULL_STEP0_TO_STEPN = NO
READY_FOR_NEXT_DATASET = NO
```

### 建议由用户判断的下一步（未自动执行）

1. 是否将“答案类型/粒度/completion condition”变成可验证的自由 contract 约束，而不是
   Hotpot 题型规则或固定 role enum；
2. malformed sampled action 应按 FlowSteer/SkillFlow 边界训练原 span，还是仅 mask 该 turn
   并保留同轨后续合法 action；
3. 是否先在 train-only、problem-disjoint 的少量图上完成真实 paired communication/Skill
   gate，再决定新 Architecture v7；
4. 是否接入 SkillFlow 的 FLA 0.5.2 gated-delta enforcement 后重新建立可比 Formal Step0；
5. 若扩大微训练样本，应先增加独立 informative groups，而不是继续在当前三个任务上
   追加 Step4–20。

本轮到此停止：没有修改 reward 为深度/节点/边/异构 bonus，没有强制多 Agent、固定
Hotpot workflow、固定模型组合，也没有进入其他数据集。
