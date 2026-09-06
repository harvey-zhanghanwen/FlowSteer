# HealthBench Professional v2.39：高质量医学来源接入

本次起初为知识源与工具能力接线。用户随后追加：取消分类检索执行配置，
修复既有错误 demo 后重跑同五题。沿用本地 Qwen3.5-9B Director、简洁提示词、
自由 contract 和唯一 Output，不训练。v2.38 的五题结果保留，不冒充新版成绩。

## 1. 实际新增与限制

| 来源 / 工具 | 实现范围 | 尚未具备的能力 |
| --- | --- | --- |
| NCBI Bookshelf / `healthbench-bookshelf.search` | ESearch→ESummary 返回真实 NBK、标题、来源日期；`source.read` 使用官方 Books-OAI 读取许可 XML 正文 | 不是所有免费网页均可取 XML；不批量下载全库、不抓图片/附件，不静默截断大文档 |
| NCI PDQ / `healthbench-pdq.search` | 同一 Bookshelf 客户端的官方 PDQ 集合筛选；可读来源沿用 NBK | 不虚报为完整临床指南库；PDQ 有不同读者版本，应读取实际来源核实，不声称只收录专业版 |
| AHRQ / `healthbench-ahrq.search` | 官方 Bookshelf AHRQ EPC Systematic Reviews 集合（原 Comparative Effectiveness Reviews） | 不代表 AHRQ 所有报告、全部专科指南或 Cochrane 全库 |
| NLM MeSH / `healthbench-terminology.search` | descriptor 标签匹配，按 ID 读取官方 descriptor 和 preferred concept 定义 | 不保证命名试验消歧；未解析完整同义词表；不是临床结局或治疗推荐证据 |
| NIH/HHS HIV Guidelines | 官方来源已确认；当前运行环境目录及实际章节均 HTTP 403 | **未接入、未注册工具**，不以网页阅读工具可见代替仓库客户端可用 |

既有 MedRAG/Textbooks、PubMed、DailyMed、Europe PMC 和 ClinicalTrials.gov
保持可用。Bookshelf、PDQ、AHRQ 存在集合包含关系；同一个 NBK 文献不算三项
独立研究。所有来源均为解题外部资料，不是 HealthBench 官方题库来源清单。

Bookshelf 搜索每次最多 3 条、两次 HTTP；读取一次 OAI 请求，最大 4 MiB，
超过上限明确报错。MeSH 搜索一次 HTTP，读取最多两次。没有自动重试/自动
全库抓取；现有共享工具预算不变。无法取得正文与没有医学证据是不同状态。

## 2. 分类检索有没有提高效果？

**尚无证据证明分类检索本身带来稳定提升，也不能据此判定它完全无效。**

同一组五道低分开发题，使用相同官方 evaluator：

| 已完成版本 | Raw score | Length-adjusted score |
| --- | ---: | ---: |
| v2.35 | 25.36% | 20.91% |
| v2.38 | 35.36% | 38.01% |

Raw 的 +10.00 个百分点全部来自 WATERFALL，其他四题没有改善。两题只有
标题但仍 FINISH，一题拓扑改变后未传入旧证据，一题试验名称消歧错误。
本轮没有实际全文阅读调用，Europe PMC 返回的命中也没有可用摘要正文。
因此工具可以调用，不等于取得有用来源，更不等于证据正确进入最终回答。
版本间还有上下文适配和模型选择变化，不能把全部收益归因于分类检索。

现有 `conversation / medical_references / drug_labels` 是请求级分类索引，
不是三个预先建好的大数据库。把临床查询过早限制在一个来源也可能遗漏证据；
但这个影响没有独立测量，属于需验证的假设。

## 3. 按用户最新要求取消分类执行配置

继续复用已有 execution_profile_allowlist，不新增路由器或固定医学流程：

- 仅保留无工具 reasoning 与一套全 12 项工具 ReAct，不再提供单一来源或
  分类执行 profile。工具本身的来源标识仍保留，便于知道实际查了什么。
- 每个 ReAct Agent 都可以搜索、读正文和查已有证据，Director 不需要先为
  节点限定数据库；不根据某道题、Ground Truth 或 rubric 硬编码来源选择。
- 保留来源类型、集合、标题、日期、NBK/术语 ID、正文可用性和分页信息。
  Bookshelf 纯元数据不会作为正文进入 FTS5；全文和真实词表定义才能供后续
  读取，MeSH 的非临床证据属性随 excerpt/source 保留。
- 不强制增加 Agent、不强制检索、不提高工具额度，也不更改官方评分。
  知识检索依然遵循真实图关系传递，不偷偷向所有节点广播未连边的信息。

本次进一步修复跨阶段历史证据缺口和标题式 completion/FINISH，修复完成与
定向测试已通过：

- 公开任务需要正文而输出只有标题时，返回可恢复错误，由同一 Agent 在
  原 ReAct 预算内继续；合法短答、明确标题/提纲请求不以长度拒绝。
- Canvas 上一版本来源显式交给 reciprocal 原节点，保留正确来源供重核；
  旧答案仍无效，不重放旧 Action history、不重置同阶段预算。peer 的本轮
  草稿仍通过原两阶段屏障传播，自己此前合法取得的来源不丢弃。
- 来源接口/接线 140 项、额外真实客户端 schema/提示词 3 项、正文恢复及
  相关回归 33 项、原 Runtime 与历史证据 58 项，共 234 项定向测试通过
  （分批执行，另含 107 个 subtests）。使用既有
  `/ssd1/iclr/gpf/venvs/skillflow/bin/python`，未安装环境或调用模型。

增加资料源本身不会自动修复这些问题；本版是来源、工具可用性和工程修复的
组合变化，不是分类检索的单因素消融，不能保证真实评分必然提高。

## 4. 验证、运行入口与备份

独立分支：`feature/healthbench-v2.39-guidelines-sources-20260906`；
起点：v2.38 结果提交 `3b42012`。234 项定向测试及 prepare-only 已通过；
新版实测分数 **N/A**，尚不能把配置核验视为正式评分。

2026-09-06 10:13 UTC 的 prepare-only manifest 确认原五个 task ID、
开发评估口径、并发 4、单题 900 秒上限及 GPU6 / 8026；训练关闭。

```bash
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_clinical_reference_sources_v2_39_dev5.yaml \
  --prepare-only
```

上述入口只准备与 v2.38 完全相同的五个开发任务，不调用模型或 grader。
输出目录独立，Direct 与 Graph 的候选工具条件一致；模型、生成设置、样本、
seed 和预算不变。按用户最新授权，修复测试完成后只跑 AgentGraph 同五题，
不重跑 Direct、不扩展 525，不把 prepared 记为 evaluator-valid。

源码复用与 API 差异详见 `docs/source_map.md`、`docs/adaptation_log.md`。
官方来源：

运行后更新：同五题尝试已完成，3题FINISH/有效评分、2题collect超时；完整
五题均分N/A，已完成三题raw33.33%/length-adjusted33.00%。本版未证明整体
提升，不升级为最佳架构。真实结果和残余问题见本目录 `evaluation_report_zh.md`。

[Bookshelf 协议](https://www.ncbi.nlm.nih.gov/books/NBK45615/)、
[Books-OAI](https://www.ncbi.nlm.nih.gov/books/about/oai/)、
[PDQ 集合](https://www.ncbi.nlm.nih.gov/books/NBK82221/)、
[AHRQ 集合](https://www.ncbi.nlm.nih.gov/books/NBK42934/)、
[MeSH API](https://id.nlm.nih.gov/mesh/swagger/ui)、
[NIH HIV 官方目录](https://clinicalinfo.hiv.gov/en/guidelines)。
