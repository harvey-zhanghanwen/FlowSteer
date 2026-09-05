# WebShop：以最高 Average Score 的 v16 为基线，v50 完整评测报告

## 结论

已从 v16 的真实源码树完成最小适配，并实际跑完同一批 **128 题**。
v50 **Average Score 55.37/100，Success Rate 29.69%（38/128）**。
这低于 v16 的 **63.72/100、34.38%（44/128）**，因此 **v50 不提升为最佳版本**。
v16 独立分支保持不变；v50 作为已完整评测但性能回退的候选保留，不能误标为当前最优。

本轮消除了重复搜索被伪装成非法动作的问题，以及公开选项绑定污染原生 evaluator
信息的接口错误；但没有彻底解决重复检索、完整目标核验、详情证据保留和预算规划。
没有训练、backward、optimizer update、LoRA 发布、GRPO、MACE、Bayesian 或 Skill evolution。

## 1. 基线与可复现条件

- 最高主指标基线：`feature/webshop-stepwise-director-v16-20260830`。
- v16 实测源码：`3ca1b3443a97dc10f0bf2da16ad1e37d8b683da9`；附最终报告版本：
  `921486fbbad48adb792219a302a44374399c5018`。
- 本轮分支：`feature/webshop-v16-based-recovery-v50-20260905`。
- 架构提交：`516e7c8`；原生回放接口修复：`c54d573`。
- 条件：`webshop_v16_recovery_v50_128`。
- 固定样本：`webshop:00500..00627`，`development / validation`，不是 525 题或 WebShop 全集。
- 模型、种子、预算沿用 v16：本地 Qwen3.5-9B，非 thinking，seed `20260825`，
  concurrency 1，环境动作 10，Director round 20，Director context 32768，Executor context 8192。
- 评测器：SkillFlow `skillflow.ragen_adapter.v2` 与 WebShop 原生环境 reward。
  Average Score 为 reward 的固定 128 题均值；Success Rate 为 reward=1 的比例。
  不使用 EM/F1、LLM judge、购买样本子集分母或隐藏目标提示。
- Direct 复用 v16 当时实际使用的 128 条记录，不重复推理；不是后来更换条件的 Direct。
- 本轮只改变 task/environment adapter 及公开反馈，保留 Canvas、自由 AgentGraph、
  独立 ReAct execution mode、每动作返回 Director、唯一 Output 和合法 FINISH 边界。
- 在物理 GPU4 / 8016 独立运行；KV 容量 169576 tokens，完整 32768 context，未做输入截断降档。

这些是反复用于架构开发的固定样本，结果不能替代新 held-out test 的泛化结论。
配置与来源见 [source map](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/docs/webshop_v16_recovery_v50_source_map.md)
和 [评测配置](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/config/evaluation_webshop_v16_recovery_v50_128.yaml)。

## 2. 完整结果

| 条件 | 样本数 | Average Score /100 | Success Rate | 满分题数 | evaluator-valid / FINISH |
|---|---:|---:|---:|---:|---:|
| 原 Direct | 128 | 33.87 | 14.84% | 19 | 128 / 不适用 |
| v16 基线 | 128 | **63.72** | **34.38%** | **44** | 128 / 128 |
| v50 本轮 | 128 | 55.37 | 29.69% | 38 | 128 / 128 |

- 相对 v16：Average Score **−8.35 分**，Success Rate **−4.69 个百分点**。
- 相对 Direct：Average Score +21.50 分，Success Rate +14.84 个百分点。
- 相对 v16 的逐题变化：**11 题提高、30 题下降、87 题不变**。
- manifest `status=completed`，Stable Zero 通过，pending evaluator retries=0。
- terminal failure=0，Director max_rounds=0，runtime/provider failure=0。
- 原生动作 751 次，全部真实推进环境；invalid action=0。
- 完成环境购买 98 题；另外 30 题用完 10 步但未购买，仍合法 FINISH 并由官方环境评为 0。
  **AgentGraph FINISH 不等于购物任务成功。**

v16 的 invalid action 为 41 次、未购买预算耗尽为 15 题；v50 分别为 0 次和 30 题。
接口错误减少没有自动带来策略改进。描述性分解中，已购买样本的平均分约为
v16 72.18、v50 72.32，购买成功率约为 38.94% 与 38.78%；整体下降主要伴随购买完成数减少。
这两个条件的购买样本集合并不相同，不能据此断言唯一因果原因，亦不能用上述条件均值替代 128 题主指标。

正式证据：[指标报告](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/reports/webshop_v16_recovery_v50_128/validation_report.json)、
[逐题结果与 v16 差值](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/reports/webshop_v16_recovery_v50_128/per_task_results.jsonl)、
[完整 manifest](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/artifacts/webshop_v16_recovery_v50_128/validation/run_manifest.json)。

## 3. 已完成修复与边界

1. 按 SkillFlow/WebShop 原生语义执行合法重复搜索，照常计预算；重复/零结果历史作为反馈，
   不再生成“扣预算但未执行”的 `<INVALID>`。原始查询标点保留，不套用选项值归一化。
2. 购买前置条件明确只检查可见选项绑定和价格；字段标为 `partial / unverified`，
   不把机械规则的通过冒充完整自然语言目标满足。完整原目标仍传回 Director 与 Agent。
3. 保留 v16 的动作域：它本来就没有“剩一步预算则只准 Buy Now”的过滤。
   本轮所有 98 次实际购买时，动作域至少还有 6 个动作；only-Buy 样本为 **0/98**。
4. 原生公开 radio 的 group/value 进入公开 receipt，避免同名选项绑定错误；
   公开页码与结果数证明下一页为空时才过滤 Next，不强制购物路径或角色。
5. 47 项定向测试、19 个 subtest 通过，包含单动作返回 Director、原目标、预算动作域、
   合法搜索恢复、radio 绑定和原生 evaluator 信息不被修改。

两题 canary 中有一次真实的 `environment_replay_transition_mismatch`：新增公开字段误入原生 `info`。
该接口已修复。失败原事件完整保留，第二题用已有完整 10 步轨迹仅重跑 native evaluator，
无新模型调用、无动作修补、无评分放宽；最终真实 Score 为 0，而非丢弃该失败样本。
随后完整运行复用这两题及全部 Direct，只补剩余 126 条 AgentGraph 轨迹。

## 4. 错误分类：90 条非满分

下表是互斥的可观测行为分类，不把“看过详情”自动等同于“正确理解全部约束”。
详情访问按官方终局实际购买 ASIN 与同 episode 的 candidate history 对应，不能把读过其他商品算作所购商品已取证。

| 类别 | 数量 | 占 90 条非满分 | 典型样本 |
|---|---:|---:|---|
| 未购买，环境预算耗尽 | 30 | 33.33% | 00501 |
| 购买非满分，所购商品未读 Description/Features/Reviews | 50 | 55.56% | 00502 |
| 购买非满分，所购商品已读至少一种详情 | 10 | 11.11% | 00500 |

辅助统计：37/128 个 episode 存在重复查询，首次以外的额外重复为 50 次。
这些查询原生合法，重复不必然错误；例如 Back 后恢复结果页可以合理，但仍消耗动作预算。
本轮 action 格式/非法动作、未恢复 evaluator 故障、provider/runtime failure、Director max_rounds 均为 0，无此类虚构案例。

## 5. 典型真实 Demo

以下三例均为单 Agent、本地 Qwen3.5-9B、ReAct execution mode，因此没有跨 Agent 通信。
每个环境动作的输入含原目标、当前 observation、可用动作、剩余预算和公开历史摘要；
动作结果再返回 Director，由其编辑或 CONTINUE。完整原始输入/输出与 receipt 保存在链接所指 trajectory。

### A. 预算耗尽与规格漂移：webshop:00501，Score 0

目标：iPhone 13 Pro Max 6.7 英寸、heavy duty、dust proof、tempered glass，
`case+4 protectors` 套装、`redblack`，低于 $50。

Director：ADD_AGENT(`buyer_agent`) → SET_OUTPUT → 多轮 CONTINUE → round8 MODIFY_AGENT → FINISH。

环境链路（10 步）：

`search → 打开 B09H4PKRQZ → 选 iPhone13ProMax → 选 red → Features → Back to Search → 相同 search → 打开 B09M9LZS2M → 选 iPhone13ProMax → Features`

- 最早可观察的规格偏离：第 4 步选择 `red`，没有展示与 `redblack` 等价的证据。
- 第 7 步原样重复第 1 步查询；状态与查询历史均已返回 Director，但仍消耗一次检索预算。
- round8 Director 进一步允许缺少 4 个保护片的 close match，放宽原始目标。
- 最后仍可选 Buy Now、Description、Features 等，Agent 选择 Features，随后预算归零。
- 最终无购买；合法 FINISH，官方 `environment_step_limit`，Score 0。

[完整链路和所有输入输出，第2行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/artifacts/webshop_v16_recovery_v50_128/validation/agentgraph_trajectories.jsonl:2)。

### B. 仅标题匹配后购买：webshop:00502，Score 0.6667

目标：oral hygiene dental tools，design=`set of 4`，低于 $30。

Director：ADD_AGENT(`dyer`, contract=`search`) → SET_OUTPUT → CONTINUE → CONTINUE → FINISH。

环境链路：`search[oral hygiene dental tools set of 4] → 打开 B091XPQXP8 → Buy Now`。

Agent 打开后的 observation 是含 “4 Pc / Oral Hygiene / Dental Tools Set” 的商品标题和 $7.99 价格；
随即购买，没有查看 Description/Features，也没有选项操作。
购买决策时还有 8 次动作预算，且详情页动作可用，绝非预算强迫购买。
可确认首个风险点是未经进一步核验便把标题相似度当作满足目标；
仅凭公开轨迹不能确定官方到底扣了哪个属性分，也不能声称查看详情一定满分。
官方终局 Score 2/3、合法 FINISH。

[完整链路和所有输入输出，第3行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/artifacts/webshop_v16_recovery_v50_128/validation/agentgraph_trajectories.jsonl:3)。

### C. 已取证但证据保留与推断不足：webshop:00500，Score 0.6

目标：小边桌，易组装，蓝色涂层钢架、防锈，低于 $70。

Director：ADD_AGENT(`shopper`) → SET_OUTPUT → CONTINUE → round8、9 MODIFY_AGENT → round10 FINISH。

环境链路（9 步）：

`search → 打开 B09GF9SSQN → Features → Prev → Description → Prev → Features（重复）→ Prev → Buy Now`

Features 提到 MDF+steel frame、约30分钟组装；Description 提到 heavy duty powder-coated steel。
现有已读证据并未明确证明全部防锈与颜色—钢架绑定要求。
round9 Director 却把 contract 改为该商品 “meets all task requirements”，要求购买；
当时公开前置条件仍正确标为 partial/unverified，不是 adapter 给出了完整验证结论。

同时确认一个仍存的架构缺口：round9 的实际 Director messages 和执行 Agent 输入不再包含上述详情正文，
只保留商品主页及 inspected_tabs。`_history_text(receipts[-4:])` 摘要记录动作/状态，不含历史 observation 正文。
原目标一直存在，但“知道读过”不等于“关键证据仍可用于判断”。这可能助长过强结论，不能据一题解释全部扣分。
购买前还有2步，动作域不止 Buy Now。最终官方 Score 0.6，合法 FINISH。

[完整链路和所有输入输出，第1行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/artifacts/webshop_v16_recovery_v50_128/validation/agentgraph_trajectories.jsonl:1)。

### D. 成功对照：webshop:00504，Score 1

目标：黑色、hands-free Denon Home 250 wireless speaker，低于 $530。

Director 创建 `shop_agent`、设为 Output，随后逐步 CONTINUE。
环境链路：`search[denon home 250 wireless speaker black] → 打开 B0837JNX9T → 选 home 250 → 选 black → Buy Now`。
5 步完成购买、官方 Score 1、合法 FINISH。
这是该具体目标的成功案例，不说明其他任务可以省略详情核验。

[完整链路，第5行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-recovery-v50/artifacts/webshop_v16_recovery_v50_128/validation/agentgraph_trajectories.jsonl:5)。

## 6. AgentGraph 与下一步判断

| 自然拓扑 | 数量 | Average Score /100 | Success Rate |
|---|---:|---:|---:|
| Single | 126（98.44%） | 55.4550 | 29.3651%（37/126） |
| Serial-2 | 2（1.56%） | 50.0000 | 50%（1/2） |

没有本轮自然形成的非链式图。两个 Serial-2 是 00537（0分）和00577（1分），数量不足以比较优劣。
未训练模型，也未强制多 Agent，不能宣称本轮已经“学会多 Agent 协作”。

下一步应优先评估，而非用固定购物模板掩盖的问题：

1. 薄复用 SkillFlow 的可见 observation 历史，将同商品的已读详情证据持续带给 Director 和 Agent，
   防止只保留访问标记；保持商品身份绑定，不引入隐藏 target。
2. 完整原目标始终高于临时 contract，不能把 redblack 放宽为 red 或把套装放宽为单件。
3. 将预算与证据规划结合，减少无新证据的搜索/详情重复；既不能强制买，也不能只增加核验而不为终局留预算。
4. 单独比较证据保留与提示词调整的效果；本轮同时做了多个适配，尚不能量化每项贡献。

本批源码和模型条件已冻结完成；未在中途修补 policy、挑选成功答案或切换版本。
性能未超过 v16，故保留原基线，不自动启动下一轮或其他数据集。

## 7. 备份与资源收束

- 原 v49 worktree 和备份分支 `backup/webshop-v49-before-best-baseline-v50-20260905` 保留，未破坏性回退。
- v16 原分支未修改；本轮源码、配置、source map、定向测试及报告在独立 v50 分支。
- 完整原始 artifacts 留在本工作树；201MB 轨迹等大型文件不直接塞进 Git。
  完整结果归档至 `/ssd1/iclr/1/backups/flowsteer/webshop-v50-evaluation-20260905.tar.gz`。
  模型权重、WebShop 商品库及外部 SkillFlow 环境依赖继续使用既有路径，不重复打包。
- 可恢复源码 Git bundle：`/ssd1/iclr/1/backups/flowsteer/webshop-v16-based-v50-complete-20260905.bundle`。
- GitHub：本轮 `git push origin HEAD` 因现有非交互认证不可用而失败，**未成功远程备份**；
  没有新建仓库、没有回显或提交凭据。需要恢复原仓库写入认证后推送上述分支。
- 本任务 GPU4/8016 SGLang 已停止；其他服务未动；实际活动子智能体在收尾后为0。
