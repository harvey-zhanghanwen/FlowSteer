# WebShop 初版适配与固定验证报告

## 1. 结论

本轮已在现有统一 AgentGraph 上完成 WebShop 的 Dataset / Environment Adapter、状态化执行闭环、正式 evaluator 接线与固定 128 条 validation 评测。没有运行训练、GRPO、backward、optimizer update、LoRA、MACE、Bayesian posterior update、Skill retrieval 或 Skill evolution。

正式条件为 `webshop_initial_adapter_v5`。AgentGraph 128/128 条 trajectory 均通过 evaluator receipt 校验并显式 `FINISH`，没有 terminal failure、`max_rounds`、collection failure 或 evaluator failure，达到本轮所定义的 Stable Zero 执行闭环。指标仍显示明显的 action-policy 改进空间，因此这里的 Stable Zero 只表示架构和 evaluator 闭环完整，不表示任务性能已收敛。

## 2. 实现依据与 source map

实现优先级遵循 `SkillFlow > FlowSteer > 本项目必要适配`：

- 直接复用 SkillFlow 部署代码中的 `RAGENAdapter.reset/step`、单 episode 可变环境状态、当前 DOM 动态 `available_actions`、原生 `search[...]` / `click[...]` action、10 步 episode budget、simulator terminal reward 与 truncation 语义。
- 直接复用 WebShop evaluator 的 reward：单题 `score ∈ [0,1]`，`Average Score = mean(score) × 100`，`Success Rate = mean(score == 1.0) × 100`。
- 复用 FlowSteer 的 progressive Canvas：Director 每轮执行一次 atomic graph edit，接受的 edit 立即执行，并把公开 execution feedback 作为下一轮 observation；trajectory 保存 graph revision、Director action、模型调用、Agent message、environment action 与 observation。
- 本项目只增加 request-scoped environment binding、public/private observation boundary、唯一 stateful tool owner 的 capability constraint、bounded-episode receipt 持久化和正式 evaluator receipt 校验。

完整文件级来源与兼容性说明见 `docs/webshop_initial_adapter_v5_source_map.md`。

## 3. Architecture Completion Report

### 已完成

执行链已经闭合：

`instruction → Qwen3.5-9B Director → progressive Canvas / AgentGraph → Agent execution → WebShop environment action / observation → Output Agent → FINISH → native evaluator → trajectory`

- Director 的动作域保持为 `ADD_AGENT / MODIFY_AGENT / DELETE_AGENT / SET_RELATION / SET_OUTPUT / FINISH`。
- Agent schema 保持 `agent_id + model_id + free-text contract`，并保留可选 `execution_mode` 与 Tool capability；没有加入固定 Searcher、Reviewer、Buyer 或购物 workflow。
- `webshop.environment` action schema 来自当前环境 observation 的动态 admissible actions，没有静态猜测或 hard-code。
- 每个 rollout 使用独立 WebShop session；同一 stateful episode 只允许一个 Agent 持有写 capability，未实现上游不支持的并发写入。
- raw terminal HTML、reward、hidden target 与 evaluator details 只进入 evaluator/private receipt；模型可见 observation 只含公开页面状态和终止确认。
- 正式 AgentGraph 指标只在合法 `FINISH` 后读取最终 turn 上唯一且 dataset-matched 的 environment receipt；receipt 要求 step 连续、计数一致且 terminal/truncation 互斥。

### 已验证

- v4 两条 smoke test：2/2 evaluator valid、2/2 explicit FINISH。
- v5 固定 validation：128/128 evaluator valid、128/128 explicit FINISH。
- 相关定向测试：317 passed，另有 54 subtests passed；仅出现一个与本任务无关的 Pydantic deprecation warning。
- 128 条正式 trajectory 没有因 provider/runtime 异常成为无效样本；6 次 provider HTTP 429 均在统一 recovery path 中恢复。
- 对 128 条 trajectory 的 Director prompt、Canvas feedback、Agent rendered messages、Agent output 与 public upstream message 进行 evaluator-private marker 核对，`graded_score`、hidden target、raw reward/score 等字段命中数为 0。

### 预留或本轮禁用

- GRPO、LoRA、backward、optimizer update 与 policy synchronization。
- MACE exploration 与 Bayesian posterior fitting。
- Skill retrieval、Skill injection、Skill publication 与 Skill evolution。
- 同一 stateful WebShop session 的多 Agent 并发写入。

### 已知问题

- Director 在 128 条任务上自然选择了 single-agent、0 relation、dependency depth 1；adapter 没有固定该 topology，但本轮没有观察到 multi-agent topology。
- AgentGraph 原生 action 合法率偏低：610 次 action 中 153 次 invalid action。
- 大量 episode 在核验关键属性和 product option 前过早执行 `buy now`。
- Direct 与 AgentGraph 使用相同 samples、environment、10 步 budget 和 evaluator，但 execution model condition 不完全相同，因此两者差值只能作 descriptive comparison，不能解释为纯 orchestration causal effect。

## 4. 固定 128 条正式结果

固定样本为 validation 上顺序选择的 `webshop:00500` 至 `webshop:00627`。没有使用 test Wrong Demo 修改 workflow。

| Condition | Valid / total | Full success | Average Score | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128/128 | 19 | 33.87 / 100 | 14.84% |
| AgentGraph | 128/128 | 18 | 50.34 / 100 | 14.06% |

AgentGraph 相对 Direct：Average Score **+16.47**，Success Rate **-0.78 percentage points**。逐题比较为 AgentGraph 较高 57 条、相同 50 条、较低 21 条。

AgentGraph score 分布：18 条 full success、91 条 partial credit、19 条 zero reward。19 条 zero reward 全部为 10 步 `environment_step_limit`；91 条 partial-credit episode 均到达 terminal purchase。

## 5. Environment 与执行收据

| Condition | Environment actions | State-advancing | Invalid | Terminal episodes | 10-step truncation | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 885 | 871 | 14 | 65 | 63 | 0 |
| AgentGraph | 610 | 457 | 153 | 109 | 19 | 0 |

AgentGraph 比 Direct 更经常完成购买，但 full success 没有提高。证据显示其主要原因不是 terminal semantics，而是购买质量：91 条 partial-credit 中有 72 条只使用三次环境动作，典型路径为 `search → product → buy now`；只有 2 条查看过 `features`。另有 28 个 AgentGraph episode 共出现 153 次 invalid action，其中 15 个最终 truncation、10 个得到 partial credit、3 个仍取得 full success。

执行诊断中记录了 128 次 `EnvironmentExecutionError`：它们发生在 progressive Canvas 已设置 `execution_mode=react`、但尚未通过下一次 atomic `MODIFY_AGENT` 挂载 `webshop.environment` 的中间 revision；全部随后修复并完成正式 evaluator。另有 6 次 MiniMax provider HTTP 429，均切换模型后恢复。这些是 execution feedback 中最早可观察的异常，不是最终低分的充分因果证据。

## 6. 自然 AgentGraph 与模型池

最终 graph 结构分布：

- Agent count：`1` × 128。
- Relation count：`0` × 128。
- Topology：`single` × 128。
- Structural / effective dependency depth：`1` × 128。

最终 Output Agent 的模型分布：

| model_id | Tasks |
|---|---:|
| `gpt-4o-mini` | 73 |
| `deepseek-v4-flash` | 23 |
| `qwen3.5-9b-local` | 11 |
| `MiniMax-M3` | 10 |
| `qwen3.5-flash` | 9 |
| `glm-4.5-flash` | 2 |

因此本轮并非所有 Agent 都使用 Qwen3.5-9B；固定为本地 Qwen3.5-9B 的是 Director。single-agent topology 是 Director 在本次 evaluation-only policy 下的实际选择，并非 WebShop adapter 预设的固定 chain。

## 7. 代表性 Demo

### 正确：`webshop:00508`

Instruction：寻找适合女孩、可水洗、易清洁、无毒且低于 30 美元的 hair chalk。

- Director：`ADD_AGENT → MODIFY_AGENT(react) → MODIFY_AGENT(webshop.environment) → SET_OUTPUT → FINISH`。
- Agent：`gpt-4o-mini`，free-text contract=`shopping_assistant`。
- Environment：`search[washable easy clean non toxic hair chalk for girls under 30.00 dollars] → click[b09kngpkbc] → click[buy now]`。
- 页面证据：商品标题为 “6 Color Hair Chalk for Girls Kids ... Temporary Washable Hair Color Dye ...”，价格 `$8.99`。
- 结果：AgentGraph score `1.0`；Direct 在 10 步内未购买，score `0.0`。

### Action format failure：`webshop:00581`

Instruction：购买 5 pairs、easy to apply、低于 40 美元的 false eyelashes。

- Director：`ADD researcher(gpt-4o-mini) → MODIFY react → MODIFY webshop.environment → SET_OUTPUT → FINISH`。
- Agent 连续输出 `<action>search>false eyelashes easy to apply under 40.00</action>`，而原生语法要求 `search[...]`；10 次均被记录为 `<INVALID>`，environment state 没有推进。
- 结果：`environment_step_limit`，score `0.0`。
- 首个与结果直接相关的可观察 failure layer：`environment_action`，step 0 的 `invalid_native_environment_action`。

### Search / navigation budget exhaustion：`webshop:00516`

Instruction：wine red、75×190cm、低于 130 美元的 beauty-salon folding mattress。

- Director：`ADD webshop_user(glm-4.5-flash) → MODIFY react → MODIFY webshop.environment → SET_OUTPUT → FINISH`。
- AgentGraph：`search → next → next → next → click[b08zxgz9zt] → back → click[search] → search → next → next`，10 步没有购买，score `0.0`。
- Direct：`search → click[b09bzmbwnf] → click[wine red] → click[75*190cm] → click[buy now]`，score `1.0`。
- 与结果直接相关的可观察层是 environment action 的候选选择与 budget exhaustion；保存的早期 capability error 已被修复，不构成该低分的因果结论。

### Product / option mismatch：`webshop:00501`

Instruction：iPhone 13 Pro Max 6.7-inch、heavy-duty、dust-proof、tempered glass、`case+4 protectors`、`redblack`、低于 50 美元。

- AgentGraph 在 provider 429 恢复后使用 `qwen3.5-9b-local`。
- AgentGraph：`search → click[b09m9lzs2m] → click[buy now]`；所购商品仅部分匹配，未选择所需 size/color，score `0.5`。
- Direct：`search → click[b09p572dp9] → click[case+4 protectors] → click[redblack] → click[buy now]`，score `1.0`。
- 结果相关 failure layer：environment action 的错误候选与 option omission；provider failure 已恢复，不能解释最终 partial credit。

### Premature purchase：`webshop:00620`

Instruction：anti-slip water shoes、size 7.5、khaki、低于 40 美元。

- AgentGraph：`search → click[b09t3hrk9l] → click[buy now]`，商品页面实际为 loafers，且未选择 khaki/7.5，score `0.05`。
- Direct 选择目标 water shoes 并设置颜色和尺码，score `1.0`。
- 结果相关 failure layer：product-category mismatch 与 option omission。

## 8. Wrong Demo 汇总与 Root Cause Hypotheses

严格区分“日志中的首个异常”与“有证据支持的结果相关 failure layer”：

1. **Action serialization / parsing**：153 次 invalid action，最明显案例是 `search>` 代替 `search[...]`。这是可直接观察的 execution-interface failure。
2. **Constraint verification / option selection**：91 条 terminal partial credit，且 72 条仅三步完成，说明购买前没有充分核验属性、category、color、size 或其他 options。
3. **Search and navigation budget**：19 条 10-step truncation；部分 trajectory 在重复翻页或重新搜索中耗尽 budget。
4. **Recovered runtime/provider feedback**：128 次 capability-staging feedback 和 6 次 provider 429 均恢复，不能直接归因为最终低分。
5. **Topology observation**：本轮 Director 全部选择 single-agent。该事实表明 base policy 在 WebShop 上尚未产生 multi-agent collaboration，但不能据此断言多 Agent 一定更优；本轮没有训练或 Skill 注入来检验该因果问题。

按照用户约束，本轮没有根据 validation Wrong Demo 写入固定 WebShop workflow、固定角色、目标商品、ground truth 或 goal-aware filter，也没有自动修改 Search Space、MACE、Bayesian 或 Skill 机制。上述内容是后续 architecture / policy experiment 的证据与假设，不是本轮新增规则。

## 9. 复现实验材料

- Condition：`config/evaluation_webshop_initial_adapter_v5.yaml`
- Source map：`docs/webshop_initial_adapter_v5_source_map.md`
- Machine-readable report：`reports/webshop_initial_adapter_v5/development_report.json`
- Generated summary：`reports/webshop_initial_adapter_v5/development_report.md`
- Failure taxonomy 与可复现 Wrong Demos：`reports/webshop_initial_adapter_v5/failure_taxonomy_report_zh.md`
- 完整本地 artifact：`artifacts/webshop_initial_adapter_v5/development/`
  - `selected_tasks.jsonl`
  - `direct_predictions.jsonl`
  - `agentgraph_trajectories.jsonl`
  - `paired_results.jsonl`
  - `wrong_demos.jsonl`
  - `collection_failures.jsonl`
  - `run_manifest.json`

完整 trajectories 和 environment receipts 保存在本地 artifact 目录；大型运行 artifact 不进入 Git。Git 备份只包含可恢复源码、配置、source map、测试和必要报告，不包含 API key。
