# HealthBench 任务产物接口修复：单题真实重跑

## 结论

医学证据格式对中间任务文本的强制限制已解除，且真实运行中 node_1 首次 ReAct COMPLETE
就交付了文本、完整传到 node_2。**但本题总效果下降**：译文不完整，下游用状态说明替代任务内容，
最终只输出标题。不能将本版称为最佳架构或宣称分数提升。

| 指标（同一个开发样本） | deadline_recovery_single | contract_artifact_single |
| --- | ---: | ---: |
| 原生 overall_score | 50.000000% | 0.000000% |
| 原生 overall_score_length_adjusted | 24.704240% | 5.527200% |
| evaluator valid / FINISH | 1/1；1/1 | 1/1；1/1 |
| collection timeout / terminal failure | 0；0 | 0；0 |
| 采集开始至轨迹落盘 | 279.812秒 | 601.809秒 |
| 最终回答字符数 | 10604 | 120 |

原始分下降50个百分点，长度调整分下降19.17704个百分点。
新回答仅120字符，原生长度调整产生正偏移：
`0 - 0.0147 × (120 - 2000) / 500 = 0.055272`。
**5.53%不是正确译文的得分**，更不能解释成5.53%的题目答对；这是该题原生长度校正结果。
本题2条正分标准均未满足；负分标准未触发。完整 rubric 只在 evaluator-private receipt。

## 可恢复版本与边界

- Task：`healthbench-professional:38ed97e78292dfcf0805ac924311fa14`。
- 本次源码：`46a61ae78fdab4f5310ca2654c19eee8c07a813c`。
- 配置：`config/evaluation_healthbench_contract_artifact_single.yaml`。
- Condition：`healthbench_professional_contract_artifact_single`。
- Attempt：`run_attempt_0c597f1790e2790421dfb44d`。
- Evaluator：`openai-simple-evals-healthbench-professional-652c89d@1`；grader `gpt-5.4-2026-03-05`。
- Grader发生1次HTTP500/bad_response_body，原有自动重试成功；总共4次API尝试，
  3项rubric最终均成功，`grader_error=null`。评分阶段约58.89秒，含失败的53.91秒请求。
  这是已恢复的provider故障，不是本题raw=0的原因。
- GPU6的已有本地Qwen3.5-9B Director未更换、未重启；模型池、thinking、配置seed、900秒上限、
  Tool/token预算、语义索引、3条candidate priors和evaluator保持不变。
- 两次都是同题单题开发重跑，没有额外运行Direct、其他题或525题。没有训练、更新权重、
  GRPO/MACE/Bayesian更新或Skill evolution，也没有把本题rubric答案写入实现/提示词。
- Director自主选择了不同的图内执行条件：上次包含DeepSeek检查和reasoning Output；
  本次三个节点均由Director选择本地Qwen、ReAct。不能把所有分差单独归因于接口变更。

## 改了什么，验证到什么程度

1. 新增版本化通信profile `producer_context_contract_artifact_v4`，保留旧v2/v3不变。
   non-Output可交付实际任务文本，或原有结构化医学证据对象。后者仍通过原来的
   provenance/引用span校验；没有把文本自动标为“已验证医学证据”。
2. 原有Tool receipts独立保留和投影。普通任务文本不再因“没有检索文献”被当成非法完成动作。
3. 普通文本投影上限由3600提升到既有产物上限12000字符；消息包最多16000字符，
   全部通信仍共用24000字符预算。不足时明确标记截断；完整内容保存在trajectory。
4. 6项新接口回归测试通过，原有证据适配、临床ReAct、v3投影、知识工具、配置测试通过。
   合成约1万字符文本的一次ReAct完成及全量下游交接已验证；真实本题首节点只产生1185字符，
   因此真实运行只验证了短产物交接，**不能声称完整长译文已在本题成功交付**。
5. 保留上一版时间预算反馈、部分失败轨迹保存、无变化执行缓存与原有FINISH admission。

源码复用及必要适配详见 `docs/source_map.md`、`docs/adaptation_log.md`。

## 清晰的实际执行过程

输入为约一万字符的英文医学表格，用户要求完整翻译成希腊语；不是只翻译标题。

```text
原始完整英文任务（每个节点均可见）
  → node_1：翻译，Qwen3.5-9B / ReAct
       交付1185字符的部分译文，包含混杂语言/乱码
  → node_2：协调/检查，Qwen3.5-9B / ReAct
       实际完整收到1185字符，但只交付106字符状态说明
  → node_3：Output，Qwen3.5-9B / ReAct
       实际完整收到106字符状态说明；没有node_1直连
       首次交付154字符标题，修改后交付120字符标题
  → FINISH → 官方grader：raw=0
```

| Director round | 动作与Canvas | 实际产物/问题 |
| --- | --- | --- |
| 0 | ADD_SUBGRAPH：node_1→node_2，接受，revision3 | node_1一次complete成功；node_2两次解析失败、一次标题拒绝后返回状态说明 |
| 1 | ADD_SUBGRAPH未被接受 | 缺`relations`字段，未修改Graph、未调用Agent |
| 2 | ADD_SUBGRAPH：node_2→node_3，指定Output，接受，revision6 | node_3 contract要求合并node_1部分译文与node_2确认，但关系只接node_2；生成标题 |
| 3 | MODIFY node_3，接受，revision7 | 明确要求完整翻译后，仍经过三次标题拒绝，最终交付另一标题 |
| 4 | FINISH，接受 | 最终120字符标题被原有终局检查放行；评分有效但raw=0 |

node_1实际输出节选（是错误输出，不是推荐译法）：
`Πίνακας 2: Τις επισημες遇到过πτική ...`。
node_2输出：`Translation Quality Assessment Complete - MEDICAL TABLE VERIFICATION REQUIRED BEFORE FINAL OUTPUT DELIVERY`。
最终回答仅一个以 `Πίνακας 2:` 开头的标题，没有请求翻译的表格正文。

### 信息有没有真正传过去

- 从保存的`rendered_messages`解析实际`producer_artifact`：
  node_2收到的文本与node_1全部1185字符逐字一致。
- 两次node_3执行收到的文本均与node_2全部106字符逐字一致。
- 因此本次首个因果失败点是node_1生成不完整且质量低的译文；
  node_2随后没有保留/修复译文，信息进一步丢失。
  这是**产物内容与路由/职责的错误，不是这两跳传输发生字符截断**。
- node_3仍然可见原始英文任务，不能说它完全拿不到问题；缺的是contract点名需要的node_1产物直连。
- 所有11条已记录Agent模型调用均以`finish_reason=stop`返回；没有对应的`length`终止。
  不能把最终只有标题解释成maxtoken截断。

## 错误分类

以下为受影响题目数量/比例，分母仅1，类别可重叠；不是525题上的发生率。

| 类别 | 题数/占比 | 典型可复现证据 |
| --- | --- | --- |
| 任务内容覆盖/翻译质量 | 1/1，100% | node_1部分译文；混杂字符；最终没有表格正文 |
| Agent产物信息丢失 | 1/1，100% | node_2将1185字符输入变成106字符状态说明 |
| Contract–relation不一致 | 1/1，100% | node_3要求node_1产物，但入边仅node_2 |
| Director动作构造/解析 | 1/1，100% | round1缺relations字段，1次动作拒绝 |
| ReAct输出解析 | 1/1，100% | node_2两次parse_error/ValueError |
| 终局完整性检查缺口 | 1/1，100% | 表格翻译任务最终只剩标题仍可FINISH；期间标题拒绝未解决问题 |
| 原医学证据schema强制冲突 | 0/1，0% | node_1不再触发structured_evidence_artifact_requires_evidence |
| 已执行消息的传输截断 | 0/1，0% | 两跳实际可见producer_artifact与各自产物逐字一致 |
| Retrieval/Tool failure | 0/1，0% | 全部Tool calls=0，不声称数据库已改善本题 |
| Timeout/max_rounds/terminal failure | 0/1，0% | 合法FINISH；900秒内返回有效评分 |
| Grader临时provider错误（已恢复） | 1/1，100% | 1次HTTP500返回异常HTML；自动重试成功，总4次尝试 |
| 最终evaluator failure | 0/1，0% | 3项rubric完整返回，grader_error=null；不把model parse error记成provider故障 |

以上每个非零类别对应同一Task ID，完整可复现输入、Graph、动作、Agent I/O、通信、ReAct动作及评分：
`artifacts/healthbench_professional_contract_artifact_single/evaluation/evaluator_private/agentgraph_development_demos.md`。
未对零类别编造案例。公开数值/调用统计在`reports/healthbench_professional_contract_artifact_single/`。

## 后续边界与备份

本次修复解决了指定的类型接口限制，但没有解决语义上的任务完成判定。
下一步值得修复的是：基于公开任务和contract的产物完整性反馈、具体产物依赖与Graph边的一致性，
以及防止状态说明/标题被误当成最终任务交付。不应通过本题标准术语、固定医疗工作流或强制角色链作弊。
本次已经完成一次重跑，到此停止，没有再改条件反复选分。

上一版源码/报告仍在`feature/healthbench-deadline-recovery-20260907`。
本次源码已在评测前推送到`feature/healthbench-contract-artifact-20260907`，
远端`backup`：`https://github.com/harvey-zhanghanwen/FlowSteer.git`。
本报告和公开指标另提交推送同一分支。原始rubric/完整I/O/数据库/模型保留本地，不加入本次Git提交。
