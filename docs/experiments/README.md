# OmniBob 实验记录

这里是已经结束的实验、版本验收、practice 复盘和交接记录。正文保持当时的内容，只改了指向其他文档的相对链接。

现行说明在上一级目录：

- [本机安装、训练与运行](../OMNIBOB_TRAINING.md)
- [Windows 长时间训练](../OMNIBOB_WINDOWS_TRAINING.md)
- [本地 v0.3 发行](../OMNIBOB_LOCAL_RELEASE.md)
- [只读实时建议](../OMNIBOB_REAL_CLIENT_DESIGN.md)
- [实验配方](../OMNIBOB_EXPERIMENT_RECIPES.md)
- [rizling.top 接口核验](../OMNIBOB_ARENA.md)
- [v0.2 复现范围与主动简化](../OMNIBOB_REPRODUCTION_V02.md)

记录里的命令和文件路径是当时的写法。紧凑证据在 [docs/evidence](../evidence/)，那里的文件带 SHA256SUMS，保留原路径，没有随本目录移动。逐局数据和检查点在本机 `runs/`，不在 Git 里。

“代码”一列是 `omnibob/` 下只为该实验写的模块。这些模块不在训练和推理的 import 路径上，清单见 `omnibob/__init__.py`。

## 一、版本记录

| 日期 | 记录 | 代码 | 证据 |
|---|---|---|---|
| 2026-10-28 | [OmniBob v0.1：本机可运行基线](OMNIBOB.md) | | [omnibob-v0.1](../evidence/omnibob-v0.1/) |
| 2026-10-28 | [OmniBob 复现规格 v0.1](OMNIBOB_REPRODUCTION.md) | | |
| 2026-10-28 | [OmniBob v0.1 本机验证记录](OMNIBOB_VALIDATION.md) | | [omnibob-v0.1](../evidence/omnibob-v0.1/) |
| 2026-10-28 | [OmniBob v0.2 夜间探索与本地验收](OMNIBOB_EXPLORATION.md) | | [omnibob-v0.2](../evidence/omnibob-v0.2/) |
| 2026-10-28 | [首场 practice 复盘与当前优先级](OMNIBOB_PRACTICE_REVIEW.md) | practice_review | [omnibob-practice-followup](../evidence/omnibob-practice-followup/) |
| 2026-11-01 | [v0.3 单场 practice：完成与复盘](OMNIBOB_V03_PRACTICE.md) | | |

## 二、训练实验

| 日期 | 记录 | 代码 | 证据 |
|---|---|---|---|
| 2026-10-28 | [OmniBob 持续迭代与续跑入口](OMNIBOB_ITERATION.md) | | |
| 2026-10-28 | [冻结 v0.2 后的有界 PPO 续训](OMNIBOB_PPO_CONTINUATION.md) | continuation_report | [omnibob-ppo-continuation](../evidence/omnibob-ppo-continuation/) |
| 2026-10-28 | [训练与部署动作掩码对齐实验](OMNIBOB_TRAINING_ALIGNMENT.md) | guarded_report, guard_audit | [omnibob-training-alignment](../evidence/omnibob-training-alignment/) |
| 2026-10-28 | [冻结学习对手的单变量实验](OMNIBOB_FROZEN_OPPONENTS.md) | opponent_report | [omnibob-frozen-opponents](../evidence/omnibob-frozen-opponents/) |
| 2026-10-28 | [固定续训预算：50 更新对 200 更新](OMNIBOB_PPO_BUDGET.md) | budget_report | [omnibob-ppo-budget](../evidence/omnibob-ppo-budget/) |
| 2026-10-28 | [三份 200 更新续训模型的等权平均](OMNIBOB_AVERAGE200.md) | | |
| 2026-10-28 | [PPO 采样窗口：固定决策预算下的 128 步与 512 步](OMNIBOB_HORIZON.md) | bounded_report | [omnibob-horizon](../evidence/omnibob-horizon/) |
| 2026-10-29 | [固定决策预算：8 环境对 32 环境](OMNIBOB_ENV_BATCH.md) | parallel_report | |
| 2026-10-29 | [固定 32 环境：25 对 50 更新预算对照](OMNIBOB_BUDGET32.md) | | |

## 三、推理和控制器实验

| 日期 | 记录 | 代码 | 证据 |
|---|---|---|---|
| 2026-10-28 | [打出、取消、移位：本地单变量实验](OMNIBOB_COMPOUND_CYCLE.md) | cancel_memory, reorder_exit, experiment_reorder, compound_report | [omnibob-compound-cycle](../evidence/omnibob-compound-cycle/) |
| 2026-10-28 | [结束回合前腾位：独立本地实验](OMNIBOB_END_TRADE.md) | end_trade, economy_audit, end_trade_report | |
| 2026-10-28 | [固定移位预算的独立复核](OMNIBOB_MOVE_BUDGET.md) | bounded_report | |
| 2026-10-29 | [固定 v0.2 权重：argmax 与温度 1 采样](OMNIBOB_INFERENCE_SAMPLING.md) | | |
| 2026-11-01 | [满场满手：出售、打出、继续经营](OMNIBOB_CAPACITY_TRADE.md) | capacity_trade | |
| 2026-11-01 | [腾位出售对象：最低数值评分单变量对照](OMNIBOB_CAPACITY_SELECTOR.md) | capacity_selector | |

## 四、诊断和调研

| 日期 | 记录 | 代码 |
|---|---|---|
| 2026-10-29 | [OmniBob 学习瓶颈诊断](OMNIBOB_LEARNING_DIAGNOSIS.md) | learning_diagnosis, learning_microdiagnosis, learning_signal |
| 2026-11-05 | [定向权重编辑：对 OmniBob 的适用判断](OMNIBOB_WEIGHT_EDITING_RESEARCH.md) | |

## 五、交接和审核

| 日期 | 记录 |
|---|---|
| 2026-10-29 | [OmniBob 本地迭代交接](OMNIBOB_AUTONOMOUS_20261029.md) |
| 2026-10-29 | [Claude 零上下文接手：审核与建议提示词](OMNIBOB_CLAUDE_REVIEW_PROMPTS_20261029.md) |
| 2026-11-01 | [Claude 接手入口：OmniBob 现状交叉评估与下一步选择](CLAUDE_NEXT_STEP_REVIEW_20261101.md) |
