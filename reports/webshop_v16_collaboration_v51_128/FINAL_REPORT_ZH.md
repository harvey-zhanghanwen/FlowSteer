# WebShop v51：完整 128 题与多 Agent 接线报告

## 结论

本轮已完成 128/128 题：**Average Score 64.770833/100，Success Rate 33.59375%（43/128）**。
相对 v16，Average Score +1.048177 分，Success Rate −0.78125 个百分点（少 1 题满分）。
不能把平均分的小幅提高描述为成功率提高，更不能据单次运行证明统计显著改善。

**本轮自然生成的图全部是单 Agent，跨 Agent relation 为 0。**
已实现并用定向模拟验证协作状态和消息接线，但没有实际多 Agent 购买案例，
不能声称 Director 已学会多 Agent 编排或把本轮得分变化归因于协作。

无训练、backward、optimizer update、LoRA 更新、GRPO、MACE、Bayesian 或 Skill evolution。
所有新推理均为本地 Qwen3.5-9B；Direct 复用 v16 已有记录，无重复 Direct 模型调用。

## 1. 条件与来源

- 分支：`feature/webshop-v16-collaboration-v51-20260905`。
- 实测架构提交：`782617d`，从 v16 源码谱系、v50 已接受接口修复继续薄适配。
- condition：`webshop_v16_collaboration_v51_128`。
- 固定样本：`webshop:00500..00627`，development / validation，128 题；
  不是 525 题，也不是未见过的 held-out test。这个面板多次用于架构开发，不能据此宣称泛化分数。
- 相同原生 WebShop 环境、human goals、全商品库、env_seed=1000、
  `skillflow.ragen_adapter.v2` evaluator，环境动作预算 10。
- Director：本地 Qwen3.5-9B，context 32768、20 rounds、1024 action tokens、
  temperature=1、top_p=1、top_k=-1、seed=20260825、concurrency=1、thinking=false。
- Executor：沿用 v16/v50 的实际目录 `qwen3.5-9b-local`，context 8192、max_tokens 4096，
  环境内层单动作生成预算沿用 512；本轮没有扩充模型池。
- 新提示词版本 `minimal-neutral-scalar-stepwise.v3` 只增加无 Tool Agent
  可分析公开环境状态及经有向边通信的能力说明；不固定角色、Agent 数量或拓扑。
- 运行 GPU4 / 8016。本次未在执行中改变源码、样本、模型、预算、并发度或种子。
- 前两题 smoke 是正式前两条，随后复用并完成另外 126 题；未挑选成功题或重采失败题。
- Direct 复用：
  `/ssd1/iclr/1/.tmp/FlowSteer-webshop-stateful-action-v15/artifacts/webshop_stepwise_director_v16/development/direct_predictions.jsonl`。

来源与不兼容原因详见 [source map](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/docs/webshop_v16_collaboration_v51_source_map.md)；
运行入口见 [配置](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/config/evaluation_webshop_v16_collaboration_v51_128.yaml)。

## 2. 正式结果

| 条件 | 样本数 | Average Score /100 | Success Rate | 满分题数 |
|---|---:|---:|---:|---:|
| v16 原 Direct | 128 | 33.865885 | 14.84375% | 19 |
| v16 AgentGraph | 128 | 63.722656 | 34.375% | 44 |
| v50 AgentGraph | 128 | 55.369792 | 29.6875% | 38 |
| v51 本轮 AgentGraph | 128 | **64.770833** | **33.59375%** | **43** |

v51 对 Direct：Average Score +30.904948 分，Success Rate +18.75 个百分点。
v51 对 v50：Average Score +9.401042 分，Success Rate +3.90625 个百分点。
对 v16 逐题：18 提高、16 下降、94 不变；对 v50：26 提高、9 下降、93 不变。

| 运行指标 | v51 |
|---|---:|
| 完成 / evaluator-valid / 合法 FINISH | 128 / 128 / 128 |
| 终局购买 | 115 |
| 环境 10 步耗尽且未购买 | 13 |
| 原生环境动作 | 594 |
| invalid action | 0 |
| Director max_rounds / terminal failure | 0 / 0 |
| runtime / provider / evaluator 未恢复失败 | 0 / 0 / 0 |
| pending evaluator retries | 0 |
| 重复查询 episode / 首次以外重复次数 | 15 / 20 |
| 单 Agent / 多 Agent | 128 / 0 |
| SET_RELATION | 0 |

AgentGraph FINISH 不等于购买成功；13 个预算耗尽的 episode 合法 FINISH，官方评分为 0，
没有从分母剔除。Average Score 为固定 128 题原生 reward 的均值，Success Rate 为 reward=1 比例，
不使用 EM/F1、LLM judge 或购买样本子集分母。

相对 v50，未购买题由 30 降至 13，重复查询 episode 由 37 降至 15；
但本轮同时调整证据保留和输入表达，不能量化每个改动的独立贡献。
相对 v16 主指标仅小幅提高、满分数反降，v16 仍保留为可恢复对照，不覆盖旧版本。

机器结果：[validation_report.json](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/reports/webshop_v16_collaboration_v51_128/validation_report.json)；
[逐题分数/动作与历史差值](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/reports/webshop_v16_collaboration_v51_128/per_task_results.jsonl)；
[运行 manifest](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/artifacts/webshop_v16_collaboration_v51_128/validation/run_manifest.json)。

## 3. 本轮实现及验证

1. 参照 SkillFlow 的 WebShop Action–Observation 历史，保留当前商品访问中的
   Features/Description/Reviews 原文；经过 < Prev 后仍可使用，不只保存 inspected_tabs。
   换商品/离开当前访问时隔离证据，防止错误的实体—属性绑定。
2. 每个环境动作后向 Director 返回原始目标、动作结果、当前状态、实际可选动作、
   剩余预算、公开历史证据。保留单动作执行并返回的 ReAct 边界。
3. 无 Tool 分析 Agent 可获得公开环境状态；环境 revision 变化时刷新相关分析缓存，
   其 artifact 仍只通过真实图关系传递。单环境写者限制保留，不能并发写同一购物 session。
4. 保存内层动作模型真正收到的 rendered_messages，而不只保存 Runtime 外层请求。
5. 保持自由 contract、模型与关系选择、唯一 Output、合法 FINISH。
   不强制 Searcher/Reviewer/Buyer，也不预设 chain 或非链式模板。

冻结前相关定向单测 162 项通过。评测过程中因用户询问 ADD_AGENT 又做了两个无模型定向检查：
已有 Output 后再 ADD 分析节点和 SET_RELATION 全部接受，分析结论实际进入执行节点输入，
添加分析节点本身不消耗购物动作；使用正式轨迹 live domain 生成的参数 schema 接受合法新 Agent。
这些检查不修改正式轨迹，也不代替真实模型生成的多 Agent 运行。

正文投递抽查固定前 53 条（00500–00552）：15 次详情打开事件，
下一轮 Director 和 Actor 的公开 observation 与环境全文一致，最新实际 Actor 消息也包含全文，
15/15。00531、00545 经过 < Prev 后全文仍保留，模型却重复点击 Features：
这里不能再归因于正文没进请求。

## 4. 是否 ADD_AGENT 有 bug？

### 已排查通过的部分

- 数量上限是 8，不是 1。
- 有 Tool owner 后，第二个 Agent 的必要字段是 id/model/contract，
  默认 execution_mode=reasoning、allowed_tools=[]；不要求第二个环境写者。
- 模型先选择动作类型，再生成该动作参数；本批 ADD_AGENT 总数 128，
  正好每题首次创建一次，没有第二次 ADD 被解析/执行拒绝的实例。
- 从整批实际单节点 live Canvas 可解析到 725 个状态，其中 701 个开放 ADD_AGENT，
  24 个因重复状态—动作反馈只开放 MODIFY_AGENT；这不是所有状态都禁止添加。
- “已有 Output → ADD 分析 Agent → SET_RELATION 分析→执行”定向检查通过。
  分析收到 environment revision=1；执行节点收到来自 analysis 的消息；环境动作数从 1 到 2，
  没有因 ADD 额外搜索。

### 确认的约束缺口与未完成验证

1. **新增 ID 约束遗漏**：ADD_AGENT 参数 schema 未排除 existing_agent_ids；
   本地 schema 验证确认新 ID 和已有 ID 都通过。Runtime 仍会拒绝重复 ID，但生成层没有避免它。
   本批未尝试第二次 ADD，故不能用它解释当前全部单 Agent。
2. **恢复动作域偏窄**：repeated_state_action_count=2 时只允许修改原 Agent，
   到 >=3 才开放 ADD/SET_RELATION 增援。它会延后协作机会，是实际策略限制，
   但也不能解释其余 701 次已开放机会未被使用。
3. 手动 runtime 和 JSON Schema 检查没有验证实际模型会选择 ADD，
   也未运行新增第二 Agent 参数的部署 xgrammar 端到端采样；没有把未测部分写成通过。
4. 新版只让协作状态接线可用，没有训练或已验证的协作 Skill。
   “环境执行入口及 CONTINUE 更容易沿用已有节点”是根因假设，不是已证明唯一原因。

相关来源：[新增参数 schema](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/src/interactive/director.py:2441)、
[恢复动作域](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/src/interactive/agent_workflow_env.py:1172)、
[ADD 默认执行属性](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/src/interactive/agent_workflow_env.py:10910)。
本批冻结版本未在途中修补这些条件；本次询问是诊断，没有自动启动新的条件或追加付费评测。

## 5. 非满分错误分类

共 85 条非满分。以下为互斥的可观察分类，不等同于全部因果已经证明。
详情访问按最终实际购买 ASIN 与同 episode 历史对应，不把其他商品的详情当作所购商品取证。

| 类别 | 数量 | 占85条非满分 | 代表 |
|---|---:|---:|---|
| 未购买，环境预算耗尽 | 13 | 15.29% | 00500、00547 |
| 所购商品未查看详情，购买后非满分 | 71 | 83.53% | 00501 |
| 所购商品查看过详情，购买后仍非满分 | 1 | 1.18% | 00602 |
| 非法动作、格式化/解析、未恢复 provider/evaluator 故障 | 0 | 0% | 无，不虚构 |

大部分部分得分购买没有读所购商品详情，但不能仅凭这个观察断言读详情必满分。
更关键的是约束核验、公开证据利用、候选纠正和预算分配。增加返回内容是必要接口改进，
并非自动保证模型正确执行这些步骤。

## 6. 典型真实 Demo

所有以下案例都是本地 Qwen3.5-9B 单 Agent。Director 在动作后看到新状态再决定
CONTINUE 或编辑，不是 Agent 内部一次连续跑完；没有跨 Agent 通信可供展示。

### A. 已有强匹配证据却离开：00500，Score 0

目标：易组装、小边桌、蓝色涂层钢架、防锈、<$70。
Director：ADD(`director`) → SET_OUTPUT → 多轮 CONTINUE → FINISH，没有协作节点或合同修订。

环境动作：
`search → B08MF23ZPL → blue → Features → Back to Search → 新search → B09KY1WJ97 → Features → Back to Search → 重复search`。

第一候选 $53.99、已选 blue，Features 明确包含 Powder coated steel frame、
不会 rust、少于五分钟组装、不需螺钉工具。完整正文进入 round5 的 Director 与实际 Actor 输入。
首次明显错失是第5个环境动作离开此候选；不是正文丢失。不能反事实保证买它官方必满分，
但已有公开证据足以支持继续核验而非无理由放弃。最终用完10步，无购买，合法 FINISH，0分。

[原始全部输入输出，第1行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/artifacts/webshop_v16_collaboration_v51_128/validation/agentgraph_trajectories.jsonl:1)。

### B. 未核验完整套装/颜色即购买：00501，Score 0.5

目标：iPhone13 Pro Max 6.7、heavy duty/dust proof/tempered glass、case+4 protectors、
redblack、<$50。
Director：ADD(`shop_researcher`) → SET_OUTPUT → CONTINUE×3 → FINISH。
contract 仍有套装和颜色条件，但说 Select best match，没有独立核验节点。

动作：`search → B09M9LZS2M → iphone 13 pro max → Buy Now`。
商品标题 $28.99、Red，选项有 black/camo/red，最终只选择 size=iphone13promax；
未查看 Features/Description，也未建立 redblack、4 protectors 满足的明确证据。
第4步直接购买是可观察的核验缺口；官方得0.5，不能凭公开轨迹擅自指定每项隐藏扣分。

[原始全部输入输出，第2行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/artifacts/webshop_v16_collaboration_v51_128/validation/agentgraph_trajectories.jsonl:2)。

### C. 读到部分属性后放宽匹配：00602，Score 0.75

目标：gray、men's casual sweatpants、machine washable、elastic waist、<$40。
动作：
`search → B09R1R24R9 → Features → Back → 相同search → 相同商品 → Features → < Prev → Buy Now`。
Director round6/7/8 连续 MODIFY 原节点，最后 contract 允许“没有完全匹配则选择预算内替代品”。
Features 明确有 Machine Wash 和 elastic waistband，未证明 gray；
购买时 Features 全文仍保留在公开状态与实际动作输入，不是上游返回截断。
首个风险是缺乏灰色证据仍反复选同一商品，随后 Director 放宽完整匹配要求。官方得0.75。
不能只凭分数断定具体扣分项，但颜色绑定缺少公开支持是可复核的。

[原始全部输入输出，第103行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/artifacts/webshop_v16_collaboration_v51_128/validation/agentgraph_trajectories.jsonl:103)。

### D. 成功候选纠正：00524，Score 1

目标：low carb、sugar free、barbecue marinade、18 ounce、<$60。
动作：
`search → B089GZKQZL → Features → Back → 相同search → B098D1Z5HM → 18 ounce → barbecue → Buy Now`。
第一个商品 Features 仅称 low in sugar，包含 molasses，未证明 sugar free；正文确实进入后续输入。
Director round6 修改原 Agent contract，重申18oz、sugar-free与价格条件，随后打开第二候选。
第二候选标题明确 Sugar Free/Low Carb，$46.99；实际购买选项为18 ounce/barbecue，官方1分。
这是单 Agent 的纠正成功，不是多 Agent 协作成功。

[原始全部输入输出，第25行](/ssd1/iclr/1/.tmp/FlowSteer-webshop-v16-collaboration-v51/artifacts/webshop_v16_collaboration_v51_128/validation/agentgraph_trajectories.jsonl:25)。

## 7. 备份、资源与下一步边界

- 架构提交 `782617d`，本报告和必要指标另做评测阶段提交；v16/v50分支和原始结果保留。
- 源码增量 bundle：
  `/ssd1/iclr/1/backups/flowsteer/webshop-v51-architecture-20260905.bundle`，
  基础完整包为 `webshop-v16-based-v50-complete-20260905.bundle`。
- 评测阶段将另保存 `webshop-v51-complete-20260905.bundle` 增量和
  `webshop-v51-evaluation-20260905.tar.gz` 完整运行档案；恢复增量需先有上述v50完整基础。
- 大型 trajectory、所有实际输入输出、环境 receipts 保留于 artifacts 并归档，不直接塞入 Git。
  模型权重、商品库和外部环境依赖不重复打包。
- GitHub 现有非交互认证此前不可用；远程 push 状态以本轮收尾实际命令为准，
  未成功前不称“已备份到 GitHub”。无凭据写入本报告或提交内容。
- 本任务 GPU4/8016 SGLang 在完整评测完成后已终止，监听端口和PID均已消失；其他服务未动。
  所有本轮有界子任务已完成，不保留冗余监控。
- 不自动训练、不进入其他数据集、不重采旧题、不因分数临时强制多 Agent。
  下一步应先针对上述恢复动作域、新增ID约束和实际采样边界做有界修复/验证，
  不再笼统把单 Agent 归因于模型能力。
