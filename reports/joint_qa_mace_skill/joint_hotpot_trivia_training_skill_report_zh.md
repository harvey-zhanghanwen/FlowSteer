# HotpotQA + TriviaQA 联合训练与 Skill 验证报告

## 1. 结论

本轮完成了 HotpotQA 与 TriviaQA 的固定验证集 Step 0、Step 1、Step 2 评测，以及 `MACE-style intervention → Bayesian posterior → Skill evidence gate` 链路。两次 LoRA/GRPO optimizer update 都是真实权重更新，但没有提高双数据集 held-out 宏平均；独立配对验证中的候选 Skill 也没有产生正向效应，因此 Skill evidence gate 正确地阻止了发布。

- 固定 benchmark 上表现最好的是 **Step 0 zero LoRA**：宏平均 EM `56.25`，F1 `68.58`。
- 两次累计联合训练后的 **Step 2**：宏平均 EM `54.69`，F1 `65.31`。
- HotpotQA Skill 候选配对效应：EM `-5.00` 个百分点，F1 `-0.71` 个百分点。
- TriviaQA Skill 候选配对效应：EM `-5.00` 个百分点，F1 `-4.96` 个百分点。
- 最终 `ACTIVE Skill = 0`。因此“联合训练 + 通过验证的 Skill”可发布策略仍为 Step 2 本身；没有把被拒绝的候选 Skill 强行注入正式 benchmark，也没有伪造 Skill 增益。

## 2. 评测协议

训练曲线与 Skill 因果验证使用不同的数据块：

| 用途 | HotpotQA | TriviaQA | 是否进入 GRPO |
|---|---:|---:|---|
| 固定 benchmark | `validation[0:32]`，32 题 | `validation[0:32]`，32 题 | 否 |
| Skill discovery | train split，3 题 | train split，3 题 | 否 |
| Skill independent confirmation | `validation[32:52]`，20 题 | `validation[32:52]`，20 题 | 否 |

- HotpotQA evaluator：`hotpotqa.official.answer.v1`
- TriviaQA evaluator：`triviaqa.official.answer.v1`
- EM 使用规范化 Exact Match；F1 使用官方答案 token F1。
- 固定 benchmark 的每个 policy step 使用相同 task ID、evaluator、Director、Model Catalog、采样配置与终局协议。
- forced probe 的 `grpo_eligible=false`，不进入训练奖励，也不进入正式 benchmark。
- Skill confirmation 从共享的 empty Canvas snapshot 分叉，在同题、同 policy version、同 evaluator、同采样坐标下比较 incumbent 与 candidate 的完整 trajectory total effect。

## 3. HotpotQA Step 0–2

| Policy | 有效样本 | EM | F1 | 相对 Step 0 EM | 相对 Step 0 F1 |
|---|---:|---:|---:|---:|---:|
| Direct Local Baseline | 32/32 | 68.75 | 78.18 | — | — |
| Step 0：zero LoRA matched control | 32/32 | **71.88** | **82.81** | 0.00 | 0.00 |
| Step 1：1 次联合 GRPO update | 32/32 | 68.75 | 79.69 | -3.13 | -3.13 |
| Step 2：2 次累计联合 GRPO update | 32/32 | 65.63 | 77.50 | -6.25 | -5.31 |

HotpotQA 在两次更新后连续退化；Step 2 比 Direct Local Baseline 低 `3.13` 个 EM 百分点、`0.68` 个 F1 百分点。

## 4. TriviaQA Step 0–2

| Policy | 有效样本 | EM | F1 | 相对 Step 0 EM | 相对 Step 0 F1 |
|---|---:|---:|---:|---:|---:|
| Direct Local Baseline | 32/32 | **50.00** | **57.29** | — | — |
| Step 0：zero LoRA matched control | 32/32 | 40.63 | **54.36** | 0.00 | 0.00 |
| Step 1：1 次联合 GRPO update | 32/32 | 43.75 | 53.13 | +3.13 | -1.23 |
| Step 2：2 次累计联合 GRPO update | 32/32 | 43.75 | 53.13 | +3.13 | -1.23 |

TriviaQA 的 Step 1 只增加 1 个 EM 正确样本，但 token F1 下降；Step 2 与 Step 1 完全持平。Step 2 仍比 Direct Local Baseline 低 `6.25` 个 EM 百分点、`4.17` 个 F1 百分点。

## 5. 双数据集联合曲线

| Policy | HotpotQA EM/F1 | TriviaQA EM/F1 | 宏平均 EM | 宏平均 F1 |
|---|---:|---:|---:|---:|
| Direct Local Baseline | 68.75 / 78.18 | 50.00 / 57.29 | 59.38 | 67.74 |
| Step 0 | 71.88 / 82.81 | 40.63 / 54.36 | **56.25** | **68.58** |
| Step 1 | 68.75 / 79.69 | 43.75 / 53.13 | 56.25 | 66.41 |
| Step 2 | 65.63 / 77.50 | 43.75 / 53.13 | 54.69 | 65.31 |

在 AgentGraph policy 内部比较，Step 2 相对 Step 0 的宏平均变化为 EM `-1.56`、F1 `-3.27` 个百分点。相对 Direct Local Baseline，Step 2 的宏平均变化为 EM `-4.69`、F1 `-2.42` 个百分点。

## 6. 真实 LoRA 更新证据

| Update | Informative group | 实际训练 trajectory | Loss | Gradient norm | Trainable update L2 | Optimizer step | Adapter sync/canary |
|---|---|---:|---:|---:|---:|---:|---|
| Step 1 | TriviaQA 1 group | 7 | -0.042232 | 0.649876 | 0.027050 | 1 | 成功 / 2 个数据集均通过 |
| Step 2 | HotpotQA 1 group | 7 | -0.008829 | 0.719158 | 0.025979 | 1 | 成功 / 2 个数据集均通过 |

Step 1 和 Step 2 都执行了 `optimizer.step()`，并把新 LoRA adapter 同步到 SGLang Supervisor。Step 2 恢复了 Step 1 的 optimizer state。OOM backoff 次数均为 0。零方差的 exact GRPO group 按 action mask 规则排除，没有伪造梯度。

## 7. MACE、Bayesian posterior 与 Skill evidence gate

### 7.1 Discovery 与 posterior

- 预注册候选：`independent_evidence_fan_in`、`answer_span_verification`。
- discovery 共 6 个 problem-level paired interventions，每个数据集 3 个。
- 使用 joint feature map 的 candidate main effect 与 candidate × dataset interaction，并以 Bayesian linear posterior 更新效应估计。
- posterior UCB 在两个数据集上都选择 `independent_evidence_fan_in` 进入独立确认。
- discovery、natural candidate、confirmation 共保存 94 条 evidence trajectory；这些 trajectory 不属于 GRPO 训练数据。

### 7.2 Skill 候选的独立配对效应

以下结果来自 `validation[32:52]`，不能与前述固定 benchmark 的 `validation[0:32]` 混算：

| 数据集 | 配对数 | Incumbent EM/F1 | Candidate EM/F1 | EM 效应 | F1 效应 | F1 正/零/负 | Bootstrap F1 区间 | Harm probability | Gate / 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HotpotQA | 20 | 70.00 / 83.17 | 65.00 / 82.45 | -5.00 | -0.71 | 1 / 16 / 3 | [-11.67, 12.62] | 0.5578 | rejected / `CANDIDATE` |
| TriviaQA | 20 | 65.00 / 71.67 | 60.00 / 66.70 | -5.00 | -4.96 | 1 / 17 / 2 | [-14.24, 1.05] | 0.8721 | rejected / `RETIRED` |
| 宏平均 | 40 | 67.50 / 77.42 | 62.50 / 74.58 | -5.00 | -2.84 | — | — | — | no publication |

两项都没有达到预注册 practical effect lower bound，negative-transfer probability 也超过 gate 上限；因此没有 activation receipt，也没有 `ACTIVE` Skill。

## 8. 联合训练与 Skill gate 后的结果

最终 policy/adapter：

- policy version：`qwen35-9b-jointqa-step-000002`
- adapter：`theta_jointqa_step_000002`
- active Skill 数：`0`

由于候选 Skill 未通过独立验证，安全发布语义是回退到未注入 Skill 的 Step 2 policy。其固定 benchmark 结果为：

| 数据集 | 有效样本 | EM | F1 |
|---|---:|---:|---:|
| HotpotQA | 32/32 | 65.63 | 77.50 |
| TriviaQA | 32/32 | 43.75 | 53.13 |
| 两数据集宏平均 | 64 | 54.69 | 65.31 |

这里是已验证的 Step 2 benchmark 结果复用，不是一次“强制 Skill-on”新评测。若强行把被拒绝的候选当成 Skill 注入，既违反 evidence gate，也会把 confirmation block 的负效应错误解释为可发布收益。

## 9. 结果解释

1. 本轮验证了训练闭环和 Skill evidence gate 的工程正确性，但没有验证出精度收益。
2. 两次 optimizer update 都只来自一个 informative group，估计方差很高；在 32 题固定验证集上，一题对应 3.125 个 EM 百分点。
3. Step 0 TriviaQA 已低于 Direct，且此时 LoRA 为 zero-initialized，因此不能把初始差距归因于 HotpotQA LoRA 过拟合。
4. `independent_evidence_fan_in` 对部分题有效，但平均效应为负且存在 negative transfer；不应进入 Skill Library 的 active retrieval path。
5. 下一轮若继续，应优先扩大独立 informative groups、修复 TriviaQA retrieval recall 与 terminal answer specificity，再重新进行预注册配对验证；不应基于本轮结果扩大训练规模。

## 10. 结果文件

- 训练曲线：`reports/joint_qa_curve/final/joint_qa_curve.json`
- 训练曲线 CSV：`reports/joint_qa_curve/final/joint_qa_curve.csv`
- 本报告结构化摘要：`reports/joint_qa_mace_skill/joint_qa_step_and_skill_summary.json`
- 本报告 CSV：`reports/joint_qa_mace_skill/joint_qa_step_and_skill_summary.csv`
- Skill publication results：`artifacts/joint_qa_mace_skill/publication_results.json`
- 配对 observations：`artifacts/joint_qa_mace_skill/paired_observations.jsonl`
- Skill records：`artifacts/joint_qa_mace_skill/skills.json`
- 完整本地 evidence store：`artifacts/joint_qa_mace_skill/evidence/`
