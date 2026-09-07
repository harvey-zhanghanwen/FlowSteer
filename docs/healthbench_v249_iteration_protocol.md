# HealthBench v2.49：恢复合法编辑选择，复用同一候选先验

## 已观察根因与变更

基于v248冻结源69af58220355a0e123d500bcb6f33698d1880dca建立独立worktree；
v248仍运行时不修改其代码/配置/样本。v248已完成的3533…题公开要求翻译三个
计划步骤；node_1只提交87字符前言，真实provider finish_reason=stop，非截断。
原始complete.value、execution、最终答复一致。下一轮完整原问题、Output与
三条候选prior均可见，尚余19轮和7个节点容量，但动作域只有FINISH。

确定配置原因是 `finish_only_when_admissible=true`：现有Env动作域在终局
可接受时直接返回finish-only；本任务semantic_protocol=none，协议有效并不
证明内容已完整。这不是应该再加固定医疗Verifier模板的问题。

v249使用已有开关false，FINISH仍合法，只恢复原本合法的ADD/MODIFY等选择。
不重写Canvas/runtime/模型接口，不把强制多Agent当优化目标。沿用v248同样
三条候选prompt prior，保留ReAct，角色/模型/数量/关系完全由Director选择。

另一真实误拒：fa30…的contract中 `(1)…(2)…(3)`任务编号和`node 1's`被
当作临床事实。本次只在既有literal提取阶段排除完整连续编号及实际存在节点
的空格别名；原contract不变、依赖校验不变、临床数值仍需公开依据。

## 固定条件

与v248相同的新五题3533…、9a16…、fa30…、2014…、cd13…；不再换题。
沿用原模型池、thinking、种子、并发、配置的生成/工具/900秒预算和官方reference
evaluator；profile仍为healthbench.orchestration-candidates.v2.48，不假装
它已经自动学习或发布。无训练、LoRA、MACE、Bayesian、Skill evolution。

## 新确认的预算缺口

v248 2014…题最后成功本地调用input24123+requested8192=32315，距离32768
仅453 tokens；新增大段检索回执后连续出现400。历史失败未保存真实请求/错误
正文，所以不能宣称已精确还原原因。为避免带着确定的预算/可观测性缺口重跑，
v249在真实运行前加入显式opt-in local_agent_context_budget：复用已加载的
同一本地tokenizer和Director既有_context_budget，逐次按真实聊天模板计数，
只将请求输出上限限制在剩余context内。原问题、已有证据、消息和thinking
开关都不截断/改变，保留configured/effective/input计数；无剩余空间发送前
报明确错误。其他API模型不能套本地Qwen tokenizer。

HTTP失败仅记录有界结构化provider error字段，不保存headers/完整HTTP正文。
这是请求兼容和失败诊断，不是提高模型context容量，也不保证所有400都是长度错误。

先做定向测试和prepare-only，待v248收束再启动一次五题，不重复Direct。
5/5有效显式FINISH且原生length-adjusted均分严格超过60%后备份，再进入
525题；否则根据公共错误继续最小迭代。缺失为N/A，原始负分不裁剪。

五题已经是开发批次，不能把迭代后成绩称为独立盲测泛化结果。资料库只包含
真实外部资料/公开对话，evaluator rubric不进入候选建议、模型请求或工具。
