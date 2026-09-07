# HealthBench v2.50：修复真实tokenizer返回类型后重跑同五题

v249源码f4626c5的实际首个Qwen请求中，预检input_tokens=2与服务端3410
不符。直接len(BatchEncoding)返回的是字段数。该批已主动中止，1条terminal、
4条cancelled partial保留；不作为完整评分或已验证有效的预算配置。

v250唯一推理代码修复：直接复用现有rollout_collector._token_ids，把mapping
的input_ids、单批列表或tensor-like容器规范为实际token序列，然后送现有
_context_budget；不另造计数算法、不截断消息、不改变thinking/模型窗口。
以方法内import复用，避免现有collector→gateway的模块循环依赖。

验证必须包含本机已有Qwen3.5 tokenizer实际返回类型的CPU测试（local_files_only），
不能只有返回普通list的模拟。首次真实Qwen返回后对照预检input_tokens与
provider prompt_tokens；若不一致，不继续把错误计数当预算保障。

本轮仍是3533…、9a16…、fa30…、2014…、cd13…五题，所有生成/工具配置与
v249一致；复用v248的三条可拒绝候选prior、自由AgentGraph及ReAct。
正式路径为artifacts/healthbench_professional_candidate_skill_v2_50_dev5，
不得续写v249混合条件。无训练/权重更新/MACE/Bayesian/ACTIVE发布。

五题5/5有效显式FINISH且原生官方length-adjusted均分严格超过60%才进入
525；缺失=N/A，保留负分。开发题不代表独立测试集泛化，也不保证每条医学
陈述均正确。所有实际计分和failure示例留在对应版本报告中。
