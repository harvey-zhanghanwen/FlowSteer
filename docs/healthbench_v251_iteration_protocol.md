# v2.51：本地HTTP400修复的同五题验证

沿用v250真实tokenizer计数、同窗口动态输出预算与有界provider错误receipt；
修复v249引入的非空Canvas节点对象类型错误。AgentGraph.nodes返回AgentNode
tuple，因此只通过既有node.id获取字符串；其余scope检查/运行边界不变。

只改变运行标识和保存路径，不变更五题、模型、seed、并发、thinking、工具、
候选prior、evaluator。无训练、MACE、Bayesian、Skill evolution或525启动。
此前v250已中止并保存原因。定向测试后冻结新源码，再运行这五题。

验证：检查真实请求预检计数与服务端prompt_tokens；对上下文超限在发送前
给出明确错误并保留receipt，不丢弃证据或扩大窗口，不承诺所有400均消失。
报告原生官方raw与length-adjusted，包括负分；缺失=N/A，完整五题与已完成
子集分开列出。本轮HTTP400修复不能当作候选Skill的独立提升证据。
