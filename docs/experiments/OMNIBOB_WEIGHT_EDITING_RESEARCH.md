# 定向权重编辑：对 OmniBob 的适用判断

2026-11-05；仅查阅原始公开记录，未运行其代码、下载模型、编辑权重或启动训练。

原始材料：[Qwen 实验](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)、[Gemma 对照](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/gemma_followup/README.md)、[Kinetix 策略修复](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md)。下列为作者报告，未独立复现实验。

Qwen3.5-0.8B 的 MATH-500 从7.9%升至48.1%，涉及两个MLP层的131072个权重。编辑过程使用练习题、Python及50批前向调用，不是只看权重一次猜中。该轮耗时20.3小时，包含4.7小时服务等待；未据此建立与同预算优化器的公平速度结论。主要改善是减少推理循环；Gemma对照没有超过约1个百分点的提升。

与我们更接近的是Kinetix案例：将明确的倒车助跑控制逻辑写进局部网络，作者报告1024测试回合全部成功。这支持研究特定故障的定向修复，不能外推为战棋总体强度保证。

当前实机 autoplay 是可见信息规则策略，不加载冻结权重。因此首先修复观察、动作和技能覆盖；改v0.3权重不会直接改变该运行器。

后续建议只做一个有界实验：先从冻结策略的可见状态和动作概率定位可重复的经营故障，若排除编码、合法动作掩码和执行错误，再制作单一小范围权重补丁。保留原模型，用新开发种子筛选、一次未参与选模的验收比较实际名次，覆盖对其他场景的退化。记录LLM、前向评测和人工分析全部成本；不得把回放纠错率当作名次提升。没有明确故障就不盲改权重。本文件是建议，尚未冻结实验参数或启动实验。
