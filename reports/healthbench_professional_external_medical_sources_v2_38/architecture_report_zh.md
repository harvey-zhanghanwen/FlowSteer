# HealthBench Professional v2.38：扩充外部医学知识源

状态：新增源的客户端与 Agent 接线完成；只做离线定向验证和五题
prepare-only。没有模型 rollout、grader、训练或新 HealthBench 评分。

## 一、这个 benchmark 是由哪些数据库构建的？

不能把外部医学知识库与评测题库混为一谈。HealthBench Professional 官方
§3.2–3.3 的题目来源是医生测试 ChatGPT for Clinicians 时的真实工作对话和
对抗测试，随后由医生编写、审阅与裁定评分标准。§3.6 的医生回答基线允许
查文献、指南和药品数据库，但这不意味着题目从某几套指定库抽取。
没有依据把 PubMed、MedRAG 或 DailyMed 称作它的官方“出题数据库”。
[官方论文](https://cdn.openai.com/dd128428-0184-4e25-b155-3a7686c7d744/HealthBench-Professional.pdf)。

`rubric_items`、`physician_response` 和 benchmark 题库不进入 Agent 知识源；
只保留原 conversation 输入、外部公开资料和实际图关系传来的工具证据。

## 二、Agent 现在可查询的外部资料

| 来源 | 接入状态 | 实际内容与边界 |
| --- | --- | --- |
| MedRAG/textbooks | 既有，保持 | 本地 125,847 个医学教材片段，既有 SkillFlow BM25；没有重复下载。 |
| NCBI PubMed | 既有，保持 | 在线检索医学文献与摘要；不是所有论文的全文。 |
| NLM DailyMed | 既有，保持 | 药品 SPL 标签，保留版本和日期；不是完整相互作用数据库。 |
| Europe PMC / PMC OA | 本次新增 | `healthbench-literature.search` 查询文献摘要；返回 PMID/PMCID/DOI 和文献类型。仅明确 OA 且有 PMCID 的来源提供全文入口；既有 `healthbench-source.read` 可取可用 JATS 文本并分页。 |
| ClinicalTrials.gov | 本次新增 | `healthbench-trials.search` 查询注册名称、干预、适用人群、分组、结局和时间窗口；`source.read` 读取完整投影及实际存在的发布结果。注册方案、未返回结果、无发布结果明确区分。 |

文献工具参考 [Europe PMC 官方 REST 文档](https://europepmc.org/RestfulWebService)；
注册试验工具参考 [ClinicalTrials.gov 官方数据结构](https://clinicaltrials.gov/data-api/about-api/study-data-structure)。
Europe PMC 与 PubMed 存在收录重叠，不把同一论文从两个来源返回当成两项
独立研究。本版增加可检索范围与全文访问条件，不宣称本地已导入其全部内容。
没有额外接入付费 UpToDate、Embase 或专科指南完整库。

## 三、接线和执行边界

继续使用原有 `Agent → ReAct Tool Action → Observation → artifact → 图关系
通信`。新工具与旧工具共享调用预算，原查询边界、重复请求拦截和来源分页
继续生效。新证据自动写入当前 invocation 的 `medical_references`，已有
`healthbench-knowledge.search` 可以再查，不增加一次 LLM 请求。

原始资料保留在 Tool receipt；有实际关系的下游可见来源证据及 DOI/PMCID、
全文读取入口等元数据。原 conversation 不是外部已证实事实。模型上下文仍
受预算约束，不能把保存完整 receipt 描述成无限上下文。

Director 仍为本地 Qwen3.5-9B，提示词 minimal-neutral.v20 未扩写；自由
contract、自由节点数、单向/有限双向关系和唯一 Output 不变。不设 Doctor、
Researcher、Verifier 固定角色，不强制多 Agent 或某个医疗工作流。

Direct 与 AgentGraph 的新配置均允许同一套 8 项工具。新协议不能直接拿
旧工具条件的 Direct 作为严格配对基线。模型池 thinking/可用性声明沿用
历史记录；未做新模型 canary，Qwen Flash 仍 reasoning-only。

## 四、验证与实际结果

- 外部 schema 探测：Europe PMC 一次（`p53 AND SRC:MED`），
  ClinicalTrials.gov 一次（`asthma`），均使用免费官方 API、非评测原题。
- 客户端离线 fixture：摘要/真实来源 ID、OA 全文投影、注册结果、分页及
  HTTP/解析异常；来源接线检查使用既有 ReAct 执行器和真实 SkillFlow FTS5，
  模型输出用测试替身，不冒充真实模型效果。
- 新配置五题 prepare-only 成功：`status=prepared`、`metrics=null`。
- 定向离线测试：118 项、35 个子项通过；没有模型或网络调用。
- 新版 HealthBench raw/length-adjusted score：**N/A，未评测**。
- 训练、GRPO、LoRA/optimizer、MACE/Bayesian/Skill 更新：均没有发生。

本次不能保证扩大知识源会提高得分。任务含义理解错误、选错查询、把来源
误解为支持结论、以及上下文/超时问题，不会因为新增接口自动消失。

## 五、恢复入口与来源

基线为 v2.37 `df8adc4`；新分支：
`feature/healthbench-v2.38-external-medical-sources-20260906`。

```bash
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_external_medical_sources_v2_38_dev5.yaml \
  --prepare-only
```

配置保留原五题开发回归集，不改样本、seed、并发、模型生成设置与 evaluator。
输出目录独立于 v2.37。运行新评测须明确发起，当前没有后台自动续跑。
复用/必要适配的逐模块说明见 `docs/source_map.md` 和 `docs/adaptation_log.md`。

### 用户随后授权的五题评分

用户明确要求“用最新的来评分5个题试试看”。因此在代码备份后只启动上述
同一组五题的 AgentGraph，不重跑 Direct、不训练、不扩展 525。正式运行结果
单独写入 `evaluation_report_zh.md`；本架构报告中的 N/A 仅是运行前状态。
