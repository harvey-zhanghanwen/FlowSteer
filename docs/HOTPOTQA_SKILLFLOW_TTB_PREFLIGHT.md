# HotpotQA MD training preflight

`MD_FULL_COMPLIANCE_20260906_V2` 已确定任务学习主算法：terminal-only、same-problem/same-condition、Action-Masked One-Pass GRPO。SkillFlow Tempered Trajectory Balance、φ-LoRA、partition function Z 和 TTB residual 全部禁用；本文件保留原名称仅用于说明旧 preflight 已退役，不是可启动的 TTB 配置。

当前训练未开始：optimizer steps 为 0，model service、GPU task 与 W&B run 均未启动。

完整 compliance matrix 与 commit-exact source map：

- `docs/MD_FULL_COMPLIANCE_20260906_V2.md`
- `docs/FLOWSTEER_SKILLFLOW_TRAINING_SOURCE_MAP.md`

## 当前门控

1. 当前处于 Phase 0 数据可信性，尚未以 fresh HotpotQA trajectory 完成 terminal reward lineage 与 snapshot replay 验收。
2. Phase 1–4 只有部分 primitive/schema/unit test；没有真实阶段实验与晋级证据。
3. 当前 θ checkpoint 只覆盖 adapter 与 AdamW state；scheduler、Python/NumPy/PyTorch/CUDA RNG 和完整训练元数据的保存/恢复尚未完成。
4. W&B 已固定绑定 `zhanghanwen6660909-dut/flowsteer-hotpotqa` 与 online 模式，但尚未创建 run；per-step validation 和 recoverable checkpoint artifact 接线未完成，因此没有 run URL。
5. 尚未执行不冲突 GPU resource admission。
6. 一个真实 GRPO optimizer step 及其 checkpoint→publish→route switch→canary→new-policy rollout 尚未运行。

因此当前 `real_step_authorized=false` 且 `long_training_authorized=false`。只有 Phase 0 与单步闭环依次通过，才允许进入后续阶段或 250–300 steps。
