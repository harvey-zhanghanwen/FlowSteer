# v2.50：上下文计数实测一致，但Canvas节点类型错误使评测中止

冻结源码：ded1585a859371547a654ac479dbeca1c3b64629。
2026-09-07本轮仍使用v248固定新五题，未调整模型/工具/候选prior。

已保存的七个本地Qwen成功请求中，新增input_tokens分别为3891、9246、
13250、12933、13409、4031、11043，与各自服务端prompt_tokens逐一相等。
这是实际请求证据，不仅是模拟测试；但这些请求均未触及上下文边界，不能
据此保证所有长上下文和所有HTTP400均已解决。

四题在已有节点后继续编辑时出现AttributeError：AgentNode没有startswith。
原因是v249操作编号修复误将AgentGraph.nodes当作字符串集合，实际为节点
对象tuple；空图测试未覆盖第二次编辑。该错误与HTTP400独立。
主线SIGINT停止本任务PID17875；四条failed、一条cancelled partial保留，
无完成trajectory，整批官方分数N/A。GPU6模型服务未停止，其他项目未触及。

v251只改为读取node.id，并补非空Canvas上的ADD/MODIFY回归；不改图语义、
医学答案、候选Skill或五题条件。不续写v250目录，不将N/A计作0。
本轮没有训练或权重更新，没有启动525。
