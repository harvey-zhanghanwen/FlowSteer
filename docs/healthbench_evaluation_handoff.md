# HealthBench v2.33 → v2.34 监控与自动交接

这是本地运行协调，不是训练或在线修改模型/架构。已完成的架构改动依据低分轨迹离线整合，正式评测前冻结。监控脚本不会根据每题 rubric 自动改写代码、contract 或答案。

配置：`config/healthbench_v233_to_v234_handoff.json`。

## 监控内容

- 每60秒读取旧版manifest与append-only EvidenceStore新增的完整JSONL记录；保存读取位置，避免反复扫描大型trajectory。
- 按已有runner的admission判定汇总有效评分、正式终局失败和待评分数量。
- 本地写入`state.json`及`low_score_queue.jsonl`，仅含task ID、评分与现有receipt诊断；不复制prompt、答案、rubric正文或隐藏thinking。
- `failure_layer`来自既有诊断函数，只代表可观察的失败层，不等于已证明的因果根因。具体案例的因果分析另见v2.34中文报告。

## 交接条件

1. 旧评测进程已经结束且锁可获取。进程退出码0或`completed_with_operational_failures`本身不代表525题完成。
2. 旧版固定525个任务均有有效评分或原协议认可的终局失败；没有pending evaluator。
3. Direct仍是既定459个有效回答与66个strict-zero失败，完整覆盖同一525题；不得重采Direct。
4. 新版使用同序同内容的525题；readiness明确记录定向测试、prepare-only及备份通过；新版尚未启动且锁未占用。
5. 复用原GPU5/8025服务和原模型/Tool/evaluator配置，仅调用已有`evaluate_completion_benchmark_round.py`。

旧版第一次退出仍不完整时，最多调用2次旧runner的精确resume：有效回答不重采，已FINISH但grader无效的记录只补评分。若没有进展、存在未解决的永久403/额度错误或次数耗尽，则写入`blocked`，不自动启动新版、不无限付费重试。

新版仅自动启动一次；新版结束后保存完整性检查结果。若仍不完整，会明确报告，不自行进入另一个架构版本或另一数据集。

## 状态与恢复

运行状态目录：
`artifacts/healthbench_professional_mixed_all_thinking_v2_34_full525_director_evidence_feedback/handoff/`。

- `state.json`：当前phase、旧版进度/均分、续跑次数、新版启动时间和完成状态。
- `low_score_queue.jsonl`：新增失分记录的紧凑索引。
- `readiness.json`：由主线在测试和备份完成后写入，含版本ID和固定任务列表；不是模型输入。
- `old_resume_*.log`、`new_evaluation.log`：仅由协调器实际启动阶段生成。

启动命令：

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/monitor_healthbench_evaluation_handoff.py \
  --config config/healthbench_v233_to_v234_handoff.json
```

协调器有独立`monitor.lock`；不要并行启动多个。旧/新评测锁由子进程继承，避免父进程退出后丢失互斥。`blocked`或已经启动新版后不会静默重入；恢复前需核对原因和实际进程，不要删除旧结果或覆盖已有状态。

正式运行期间不得修改该版本源码或配置。若需要后续方法级变更，另建版本并重新验证，不把候选修复混入已启动条件。
