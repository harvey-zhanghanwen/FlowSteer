# HealthBench v2.49：上下文计数兼容问题，已停止，不作为完整评分

首次实际运行源码f4626c5107929dec67d78650ad3a49e381cdfe3c，五题仍为v248新五题。
在首条Qwen真实receipt发现：预检context_budget.input_tokens=2，但provider
prompt_tokens=3410。原因是本地tokenizer返回BatchEncoding/mapping，直接len
得到字段数量，不能得到token数。此前合成测试只用list返回，未覆盖真实类型；
因此不能宣称此次上下文预算预检已经在真实部署验证通过。

主线立即对本任务PID37347发送SIGINT并确认进程退出，未停止GPU6模型服务或
其他项目。已写出的1条terminal trajectory、4条cancelled partial均保留。
首题3533…真实原始分-100%、长度调整-94.83148%；该题发生在错误预算预检条件下，
不把这一单题当整批分数，也不隐藏负分。整轮未完整执行，均分N/A。

后续将单独使用v2.50路径，不往v249目录续写混合条件。修复直接复用现成
rollout_collector._token_ids：该函数已兼容mapping、单批列表及tensor-like
返回；保留此前_context_budget算法、原输入、thinking和工具预算。新增实际
本机tokenizer CPU测试以及返回类型回归，再通过真实服务usage核对计数。

本轮无训练/optimizer/LoRA/MACE/Bayesian/ACTIVE发布。525尚未启动。
v249不是已验证有效版本；源码、未完成运行和中止原因均可恢复。
