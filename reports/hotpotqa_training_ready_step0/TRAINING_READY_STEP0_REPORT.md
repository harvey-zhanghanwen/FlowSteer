# HotpotQA Training-ready Step 0 Report

## 1. Executive Conclusion

本轮没有进行 GRPO、backward、`optimizer.step()`、LoRA 更新/发布、MACE、
Bayesian posterior 拟合、EVSI 或 Skill 注入。运行时只加载了此前已有的
`theta_smoke_step_000001` adapter 做 evaluation-only 推理；三个 manifest 均记录
`training_enabled=false`、`optimizer_updates=0`。

**Stable Zero 仍然成立。** 固定 development-128 和一次性 untouched-32 共
160/160 条 AgentGraph 均完成：本地 Qwen3.5-9B Director、Canvas/AgentGraph、
Executor、实际有向通信、唯一 Output Agent、HotpotQA evaluator 和完整 trajectory；
160/160 显式 `finish`，160/160 evaluator receipt 有效，0 条 collection failure，
0 条 `max_rounds` 终止。最终无模型回归验证为 170 个 unit tests 全部通过，相关
Python 文件 Ruff 检查通过。

**当前没有达到用户定义的 Training-ready Step 0，不允许开始 Step 0 → Step N
GRPO。** 已经形成可冻结、可重放、可比较的 `step_000000 /
training_ready_step0` 工程基线，但仍有三个有直接证据的阻塞项：

1. 独立通信消融的 16 个冻结多 Agent 图中，遮蔽 upstream 后 EM 没有下降，F1
   反而增加 1.11 个百分点；尚无行为证据证明 Output 依赖了 upstream。
2. untouched-32 上 AgentGraph 比同批 Direct 低 6.25 EM / 4.15 F1 个百分点；
   多 Agent/编排价值没有在未见样本上稳定成立。
3. development-128 与 untouched-32 各仍有 1 个 Output 不遵守唯一
   `<answer>...</answer>` 协议；development 的该例中 Canvas 已明确反馈格式错误，
   Director 仍选择 `finish`。

因此本轮到此停止，不进入训练，也不根据 untouched-32 结果继续调 prompt、模型池
或 search space。

## 2. What Changed

所有运行时修改均由 Round-01 保存的真实行为驱动。来源优先级是 SkillFlow 的
Qwen3.5/SGLang/terminal-answer 边界、FlowSteer 的 progressive Canvas/feedback/
trajectory/terminal 边界，以及本项目设计文档要求的最小 AgentGraph 适配。没有新增
固定角色枚举、固定 workflow、Agent 数量奖励、graph complexity 奖励或 Hotpot
答案规则。

| 修改文件 | 修改前 | 修改后 | 真实证据与类型 | Search space / 数据协议 |
| --- | --- | --- | --- | --- |
| `src/interactive/agent_runtime.py`、`src/interactive/openai_gateway.py` | 所有节点共用 task-final wrapper，Round-01 的 15/15 个中间节点都输出 `<answer>` | Runtime 从既有唯一 `output_agent_id` 派生身份；中间节点只产 evidence/fact/partial reasoning/verification，只有 Output 节点产最终 tag | Round-01 “先答后复述”；SkillFlow intermediate observation 与 terminal answer 边界；protocol fix | 不改变 Agent 数量、自由 contract、模型、relation、Output 或 action 空间；不改变 benchmark 输入 |
| `src/interactive/openai_gateway.py` | Director 自由 contract 中的“完整句子/解释”可能覆盖短答案目标 | Output execution protocol 明确优先：一个 concise span、一个 tag、tag 外无解释；上游是证据而非可盲抄答案 | Round-01 entity/yes-no 扩写回归；protocol fix | 不写入具体实体或答案；evaluator 未改 |
| `src/interactive/agent_workflow_env.py`、`src/interactive/director.py` | feedback 没有明确给出 tag 数量/唯一性；已有候选后仍重复给新模型建议 | feedback 增加 tag count、exact-single-tag 和有界 Output inbox；continuation 必须指出 evidence hop/conflict/format/runtime/task mismatch；成功执行后不再无条件提示新模型 | Round-01 已有正确候选后继续修改并改错；FlowSteer execution feedback/explicit finish；protocol fix | 六个 action 与自由 search space 不变；没有 answer-presence 自动 finish |
| `src/interactive/agent_runtime.py`、`src/interactive/records.py`、`src/interactive/rollout_collector.py` | 只能证明消息到达，不能隔离是否使用 | 新增 `normal` / `upstream_masked` condition；只在 rendered prompt 遮蔽跨 Agent 内容，canonical upstream receipt 保持原样；typed diagnostic 永远 `diagnostic_only=true`、`grpo_eligible=false` | 用户要求的 communication utilization 诊断；diagnostic | 标准 benchmark 仍是 Question + 完整十篇 Context + Upstream；不改变训练分布 |
| `scripts/diagnose_hotpotqa_communication.py`、`config/diagnostic_hotpotqa_training_ready_step0_communication.yaml` | 无独立双臂入口 | 对冻结 final graph 做 Normal/Masked 成对 replay，跳过 Director、Direct、trainer 和 policy update | diagnostic-only；复用现有 Runtime/Gateway/Evaluator | 不产生正式 reward，不可进入 GRPO |
| `scripts/evaluate_hotpotqa_round.py`、两份 `config/evaluation_hotpotqa_training_ready_step0*.yaml` | Round-01 runner 固定旧标签和 128 条入口 | 可按 1–128 条严格复评；development-128 复用成功的 Direct；manifest 记录实际运行 GPU/端口/context/memory | 必要 evaluation adapter | evaluator、固定样本、十篇 context、seed 与 catalog 保持；训练全部关闭 |
| `scripts/freeze_hotpotqa_step0_untouched.py`、`data/hotpotqa_training_ready_step0/untouched_validation_32.jsonl` | 无未见确认集 | 复用现有 Hotpot converter/retag 函数，冻结 raw candidate 640–671 | 过拟合检查；dataset adapter | 与 128 validation、512 train 均 0 overlap；没有重切或改变训练集 |
| `tests/unit/*`、`docs/SOURCE_MAP.md`、`docs/ARCHITECTURE_COMPLETION_REPORT.md`、`docs/DATASET_BACKUP_BRANCHES.md` | 缺少上述协议、诊断、来源和分支隔离说明 | 覆盖身份、rendering、condition、record、collector、Director feedback 与 diagnostic ID；登记上游来源和每数据集独立备份约定 | regression tests / documentation | 无运行时方法改动 |

评测机器上配置的 rollout 物理卡 4 已被本任务之外的服务占用；本轮没有停止或修改
这些服务，而把本任务 Qwen3.5-9B evaluation-only SGLang 放到空闲物理卡 0。该最小
资源适配只写入 manifest（端口 8015、context 8192、memory fraction 0.60），没有
改变模型、adapter、prompt、evaluator、catalog 或样本。

## 3. Communication Audit

### 传输与保存

- development-128 最终图：93 个单 Agent、35 个多 Agent；35 个多 Agent 图均为
  单向关系，0 个真实双向 relation。
- 35/35 多 Agent trajectory 的关系方向、实际 upstream 来源和 Output inbox 一致；
  35/35 canonical upstream 内容已保存，未发现消息丢失、方向反转、错误 fan-in 或
  Output 取错节点。
- 新协议下 0/35 中间节点输出 `<answer>`，Round-01 为 15/15。
- 新协议下 0/35 Output 逐字复制完整 upstream，Round-01 为 12/15；仅 1/35 的
  final span 与完整 intermediate 文本完全相同，Round-01 为 14/15。
- 27/35 中间 evidence 文本包含最终 answer span。这表明中间节点通常确实给出了
  相关证据，但不能单凭字符串包含关系证明下游使用了该消息。
- 只有 1/35 intermediate artifact 本身按 Hotpot gold 达到严格 EM；该例没有被
  Output 改错，因此 `correct upstream → wrong Output = 0`。

结论：**消息通道正确，receipt 完整，“所有中间节点先输出 final tag、下游直接复述”
这一协议冲突已经消失；但协作利用率仍未得到证明。**

### 独立 upstream-masked 消融

从 development-128 按冻结顺序选取前 16 个多 Agent final graph。每个图保持相同
Question、十篇 Context、Agent contract、model、relation、Output、seed 和 evaluator，
只在模型看到的 prompt 中将所有跨 Agent 内容替换为 mask；原始 upstream 仍保存在
receipt。该入口调用 Director 0 次、Direct 0 次、optimizer 0 次，32/32 arm 完成，
所有记录均为 diagnostic-only 且 GRPO-ineligible。

| 条件 | 有效/总数 | EM | F1 |
| --- | ---: | ---: | ---: |
| Normal | 16/16 | 56.25 | 81.04 |
| Upstream masked | 16/16 | 56.25 | 82.15 |
| Masked − Normal | — | 0.00 | +1.11 |

只有 1/16 raw/normalized answer 改变，且两臂都不满足 EM；没有
Normal-correct → Masked-wrong。这个结果没有显示 upstream 对最终行为的正向因果
作用。它也不能证明 upstream 永远没用，因为两臂都保留了完整 Question + Context，
Output 可以自行重做问题。

### 双向关系的证据等级

- **实现**：Runtime 保留有限两阶段 reciprocal block 路径。
- **单测**：对应 Runtime/communication tests 通过。
- **真实 benchmark 验证**：未达到；development-128 和 untouched-32 都没有产生
  双向 relation。不能把“实现/单测”表述为“HotpotQA 已验证”。

## 4. Director / Stopping Audit

development-128 的行为统计：

- 480 个 Director turn，平均 3.75/题，范围 3–8。
- 128/128 显式 `finish`；0 `max_rounds`；finish 后 0 次额外执行。
- 128/128 finish 复用同一最新 graph revision 的执行结果；0 stale revision mismatch。
- 7 个 action 被拒绝或解析为 invalid，分布在 7 题：6 个 malformed/非法 relation，
  1 个没有唯一 Output 时的 premature finish。Canvas 全部拒绝，Director 随后恢复，
  7 题最终都完成。
- 在已经得到 execution candidate 后继续 add/modify/delete/model switch 的任务为 0；
  因 continuation 造成的 correct → wrong regression 为 0。Round-01 报告记录的
  4 个额外 Executor call 和至少 1 个 correct → wrong stopping regression 本轮未再出现。
- 共 163 个 Executor call；最终 Output 路由为 local Qwen3.5-9B 45、
  Qwen3.5 Flash 43、GPT-4o-mini 24、MiniMax-M2.5 16。

但 stopping 还不是完全正确：
`hotpotqa:5a7a52745542996c55b2dd4f` 的 Output 返回 JSON 而没有 answer tag，Canvas
feedback 已显示 `answer_tag_count=0`、`exact_single_answer_tag=false`，Director 下一步
仍选择 finish。也就是说“无理由继续”已被压低，但“看到明确格式问题仍过早停止”
仍有 1 个直接反例。

untouched-32 同样为 32/32 explicit finish、0 max-rounds、0 candidate 后继续修改；
119 个 turn，平均 3.72/题，2 个 invalid action 均恢复。

## 5. Output Contract Audit

这里把“concise wrapper compliance”定义为：完整输出恰为一个
`<answer>span</answer>`，且 span 不超过 100 字符和 10 个空白分词。这个长度阈值
只用于诊断，不改变 Hotpot evaluator。

| 指标 | Round-01 development-128 | 新 development-128 | untouched-32 |
| --- | ---: | ---: | ---: |
| 唯一且完整 `<answer>` wrapper | 112/128 | 127/128 | 31/32 |
| malformed/no wrapper | 16/128 | 1/128 | 1/32 |
| 新定义下 overlong span | 未按同一阈值冻结 | 4/128 | 0/32 |
| concise wrapper compliance | 未按同一阈值冻结 | 123/128 | 31/32 |
| yes/no 严格正确 | — | 6/8 | 1/1 |
| correct intermediate → wrong Output | 1 | 0 | 0 |

新协议显著消除了系统性的 intermediate final tag 和大部分 malformed 输出，但没有
做到零违规。development-128 中有 13 个 Graph answer 与 gold 呈规范化包含关系但
不满足 EM；其中 5 个是 Direct 已严格正确、Graph 又把实体/yes-no 扩写后判错的明确
expansion regression。它们不是 evaluator Bug，正式 EM/F1 未被放宽。

代表性证据：

| Task ID | Gold / Direct | Graph | 首个可定位问题 |
| --- | --- | --- | --- |
| `5a7a0693...` | `Arthur's Magazine` | `Arthur's Magazine: 1844; First for Women: 1989` | Output 在合法 tag 内加入无关比较细节 |
| `5ae3918b...` | `no` / `No` | 26-word explanation | yes/no 被解释性句子扩写 |
| `5a736bfa...` | `Glenn Hughes` | `Glenn Hughes is older ...` | 短实体被完整句扩写；也是通信消融唯一改答案例 |
| `5a7a5274...` | `Sir Francis Nethersole` | JSON birth-year object | 自由 contract 要求 JSON，Output protocol 被模型忽略，Director 无视格式 feedback 后 finish |
| `5a82171f...` | `American` | `British` | reasoning/实体链接错误，不是传输或格式问题 |

untouched-32 的 malformed 例 `5a77f88a...` 也返回 fenced JSON 而非 answer tag，说明
该问题不是只存在于已分析的 128 题。

## 6. Paired HotpotQA Result

### 固定 architecture-development 128

Direct 完全复用 Round-01 的 128 个成功 record，没有重复调用。两条件使用相同
Task ID、完整十篇 passage 和相同 normalized EM/token-F1 evaluator。

| 条件 | 有效/总数 | EM 正确 | EM | F1 |
| --- | ---: | ---: | ---: | ---: |
| Local Qwen3.5-9B Direct | 128/128 | 93 | 72.66 | 82.08 |
| Training-ready Step-0 AgentGraph | 128/128 | 94 | 73.44 | 81.62 |
| AgentGraph − Direct | — | +1 | **+0.78** | **−0.46** |

成对类别：both correct 84、Graph-only 10、Direct-only 9、both wrong 25。
AgentGraph 的 34 个 strict-EM Wrong Demo 分为 architecture regression 9、
partial/overlong 14、shared reasoning/model failure candidate 11。另有 1 条两条件均正确
但 Executor 发生重试的 operational classification；最终没有缺题。

相对同一批 Round-01 AgentGraph 75.00 EM / 84.44 F1，新协议结果为 −1.56 EM /
−2.82 F1 个百分点。因此协议行为更干净，不等于任务分数已经提高。

### 一次性 untouched 32

raw candidate 640–671 在运行前冻结，与 development-128 及 Hotpot 512 条训练候选
均无 Task ID overlap；看到结果后没有再改 prompt/search space，也没有重跑。

| 条件 | 有效/总数 | EM 正确 | EM | F1 |
| --- | ---: | ---: | ---: | ---: |
| Local Qwen3.5-9B Direct | 32/32 | 25 | 78.12 | 87.77 |
| AgentGraph | 32/32 | 23 | 71.88 | 83.62 |
| AgentGraph − Direct | — | −2 | **−6.25** | **−4.15** |

成对类别：both correct 21、Graph-only 2、Direct-only 4、both wrong 5。

### 单 Agent / 多 Agent

| 集合与图规模 | 数量 | Direct EM/F1 | Graph EM/F1 | Graph − Direct |
| --- | ---: | ---: | ---: | ---: |
| dev 单 Agent | 93 | 73.12 / 81.78 | 76.34 / 82.80 | +3.23 / +1.02 |
| dev 多 Agent | 35 | 71.43 / 82.86 | 65.71 / 78.48 | −5.71 / −4.38 |
| untouched 单 Agent | 23 | 78.26 / 87.54 | 69.57 / 82.92 | −8.70 / −4.62 |
| untouched 多 Agent | 9 | 77.78 / 88.36 | 77.78 / 85.40 | 0.00 / −2.96 |

这些子集由 Director 选择，不能当作随机因果实验；但它们与 upstream-masked 结果
共同说明，当前没有证据支持“多 Agent 协作已经稳定带来增益”。

用户提供的论文参考 Qwen Direct 60.94/75.70、FlowSteer 89.84/91.20、SkillFlow
92.19/93.95 仅作背景：split、prompt、模型状态、工具和 evaluator protocol 并不保证
相同。当前结论优先依据上述同题 Local Direct paired comparison。

## 7. Skill Boundary Audit

当前代码中已经存在的只是隔离的后续方法原语：

- `skills/schema.py`：condition、action、paired evidence、interval、independent
  validation、version、failure scope 和 candidate/active/suspended/retired 状态。
- `skills/validator.py`、`lifecycle.py`：确定性 evidence gate、延迟激活、版本漂移
  suspension 和 retirement。
- `skills/store.py`、`retrieval.py`：持久化接口，以及只检索 ACTIVE、版本兼容、
  condition 适用 Skill 的原语。
- `exploration/mace.py`、`posterior.py`、`paired_probe.py`：MACE/Bayesian/paired-probe
  的孤立算法原语。

没有实现或接通的是：trajectory → Skill discovery/summary → persisted paired probe →
independent validation → gate → publication → Director retrieval 的真实端到端闭环；
MACE/Bayesian/EVSI 也没有接入本轮 reward 或 rollout。

本轮两份 evaluation config 均为 `skills.enabled=false`、`initial_library=[]`、
`retrieval_top_k=0`。160 条 prompt 中没有 `available_skills`，trajectory 的
`skill_library=none`；validation 记录全部 `grpo_eligible=false`，probe/posterior 文件
为空，没有任何自动 candidate/ACTIVE publication。

因此：**当前没有发生 Skill 注入，也没有任何证据允许声称 Skill 已经学习、变好或
带来 gain。** 将来开启 Skills 前，还需要把“只有通过持久化 paired evidence resolver
和 lifecycle gate 的 ACTIVE record 才能进入 retrieval/prompt”接成不可绕过的唯一
发布路径；当前 Skill-OFF Hotpot 运行不受这个未接线边界影响。

## 8. Training-readiness Checklist

| 检查项 | 结论 | 证据 |
| --- | --- | --- |
| data integrity | PASS | dev 128 个唯一 ID；untouched 32 个唯一 ID；与 dev128/train512 overlap 均为 0 |
| no leakage | PASS | 模型 rendered prompt 无 `ground_truth`/`supporting_facts` 字段；只输入 Question + 完整十篇公开 context |
| runtime | PASS | 160/160 完整链；唯一 Output；合法 final graph；0 collection failure |
| communication transport | PASS | 35/35 dev 多 Agent upstream/inbox/方向/receipt 一致 |
| communication utilization | **FAIL** | 16-pair mask 后 EM 不降、F1 +1.11；未显示行为依赖 |
| Output | **FAIL** | dev 与 untouched 各 1 个 malformed；dev 仍有 5 个明确 expansion regression |
| stopping | **FAIL** | explicit/max-round/stale 均正确，但 1 个可见格式错误仍直接 finish |
| trajectory/observability | PASS | graph、contract、model、relation、Output、finish、upstream、condition、evaluator、reward、request/token/latency/version 均可追踪 |
| evaluator | PASS | 160/160 使用既有 normalized EM + SkillFlow token F1，160/160 valid；未改成宽松 judge |
| train/validation isolation | PASS | validation 全部不可进入 GRPO；untouched 与 train512 overlap=0 |
| frozen Step-0 version | PASS | policy/prompt/tool/evaluator/catalog/split/runtime/seed/source revision 写入 config、trajectory 和 manifest |
| replayability | PASS | final graph、实际 rendered input、raw upstream、model/provider/seed/receipt 和双臂 condition 已保存 |

总判定：工程正确性与可观测性通过，但协议验收 B 和 communication 行为验收未通过。

## 9. Remaining Risks

只保留有本轮证据的问题：

1. Output 在拥有完整 Context 时可以绕过 upstream；当前图的“通信已传递”与“通信有
   行为价值”仍是两件事。
2. untouched-32 的成对回退说明 development-128 上微小的 +0.78 EM 不稳定，不能
   作为启动 RL 的架构收益证据。
3. concise wrapper 仍会被自由 Agent contract 或模型格式服从问题覆盖；Canvas 虽能
   检测，但 Director 不总会修复。
4. development 多 Agent 子集比同题 Direct 低 5.71 EM，且 communication mask 不降分；
   collaboration 目前可能增加调用成本而没有稳定增加信息价值。
5. 7 个非法/被拒 action 均恢复，但说明当前未训练 Director 的 action/schema
   compliance 尚非 100%。
6. 本轮实际使用物理 GPU0 而不是配置中的 GPU4；manifest 已精确记录该适配。未来
   三卡训练前必须重新确认当时资源映射，不能把本轮 inference placement 当成训练拓扑。
7. Skills 的真实生成、paired validation 和不可绕过发布路径尚未端到端接线；在此之前
   不能启用或评估 Skill gain。

## 10. Step 0 → Step N Measurement Plan

本节只定义未来放行后的测量，不启动训练。

### 固定比较边界

- 每个 checkpoint 绑定 policy/adapter、prompt、tool、catalog、evaluator、dataset
  split、runtime、source revision 和 seed policy。
- development-128 继续做同题 paired evaluation；Direct record 固定复用，不重复
  成功调用。
- 固定 16 个 communication diagnostic graph 做 Normal/Masked 双臂，始终
  diagnostic-only、GRPO-ineligible。
- 本次 untouched-32 保持锁定，不用来选 prompt、路由或 checkpoint；若未来需要
  最终 confirmation，应先冻结新的未见集合并只在最终候选上运行一次。
- 每个 rollout/trajectory 记录 behavior policy version；只有 exact receipt、有效
  evaluator、train split 且非 diagnostic/fallback/probe 的记录才可能进入未来 GRPO。

### Performance

每个 Step 记录：EM、F1、相对固定 Direct 的 paired gain、Graph-only、Direct-only、
both wrong，以及 development 与最终未见 confirmation 的分离结果。

### Behavior

每个 Step 记录：explicit finish、max-round failure、illegal action、Director turns、
Agent 数、multi-Agent ratio、relation/depth、model routing、candidate 后 continuation、
stopping regression、malformed/overlong Output、correct upstream → wrong Output、
Normal/Masked answer-change 与 score delta、decomposition/verification 行为。

固定一组 diagnostic trajectories，人工逐项比较：

`Step 0 → Step 1 → ... → Step N`

判断提升来自更好的编排、通信和停止，而不是 evaluator 变化、格式宽松或单纯增加
Agent/call 数。

### Skill（未来真实接入后）

只在完整 evidence gate 与发布路径接通后记录：candidate 数、ACTIVE 数、paired
evidence、interval/uncertainty、independent held-out validation、Skill ON/OFF gain、
suspended/retired、version compatibility 和 failure scope。此前这些指标必须标为
“未实现/未启用”，不得填入推测值。

### 当前阻塞项

1. communication utilization 未得到成对消融支持；
2. untouched-32 AgentGraph 明显低于同题 Direct；
3. Output/Director 对明确格式错误仍不是零失误。

READY_FOR_GRPO = NO
