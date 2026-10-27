# 可复制的本地实验配方

所有命令在仓库根目录运行。以下训练是有界的实验配方，不保证复现相同数值，更不保证继续训练一定变强。
先按[安装说明](OMNIBOB_TRAINING.md)建立本机环境；每次输出到新目录。原始夜间配置和全部失败记录
保存在 `runs/exploration-20261028/`。本页不会自动启动任务。

## 从示教起步

```bash
.venv/bin/python -m omnibob.imitation collect --rules configs/arena-v2.json --games 128 --seed 10000000 --max-seconds 120 --output runs/repro-teacher128
.venv/bin/python -m omnibob.imitation fit --dataset runs/repro-teacher128 --epochs 8 --device mps --max-seconds 600 --output runs/repro-bc128
.venv/bin/python -m omnibob.train --initialize runs/repro-bc128/best.pt --config configs/ppo-warm-v02.json --rules configs/arena-v2.json --updates 100 --device mps --max-seconds 900 --output runs/repro-warm128
.venv/bin/python -m omnibob.train --initialize runs/repro-warm128/update-000100.pt --config configs/ppo-efficiency-0.0003.json --rules configs/arena-v2.json --updates 200 --device mps --max-seconds 1500 --output runs/repro-efficiency
```

这对应最初128局示教的主线。早期源码只有逐文件摘要，部分没有完整历史覆盖包；上述命令使用
当前代码重建流程，不是宣称逐位复现早期权重。MPS不同版本也不保证逐位一致。

## 冻结起点后的混合对手续训

夜间实际起点是 `efficiency-s30000000-cost0.0003/update-000200.pt`；其他分支也从相同权重开始。
`configs/ppo-mixed-v02.json` 保存seed43000000、32环境、128步、学习率0.0001、参考KL0.03，
混合上游/冻结启发式对手、打乱行动顺序、本玩家名次确定后结束训练轨迹。先用3次更新验证本机：

```bash
.venv/bin/python -m omnibob.train --initialize runs/exploration-20261028/efficiency-s30000000-cost0.0003/update-000200.pt --config configs/ppo-mixed-v02.json --rules configs/arena-v2.json --updates 3 --device mps --max-seconds 90 --output runs/repro-mixed-probe
```

确认性能后，新建输出目录，把 `--updates` 改为300、`--max-seconds` 改为2400。
如要重做三种训练随机种子，分别复制JSON并把 `seed` 设为41000000/42000000/43000000。
JSON覆盖同名CLI默认项，所以不能只在已含seed的配置旁另加 `--seed`。
固定150/300检查点评测；不要只展示最优检查点或最优随机种子。

```bash
.venv/bin/python -m omnibob evaluate --policies ppo omnibob scripted --checkpoint runs/repro-mixed-probe/latest.pt --repeat-limit 1 --move-budget 0 --games 128 --split dev --seed 1100 --output runs/repro-mixed-scripted
.venv/bin/python -m omnibob evaluate --policies ppo omnibob scripted --checkpoint runs/repro-mixed-probe/latest.pt --repeat-limit 1 --move-budget 0 --games 128 --split dev --seed 4000 --opponents heuristic --scheduler shuffled_rounds --output runs/repro-mixed-heuristic
```

使用同一权重另跑 `--repeat-limit 0 --move-budget 0`，区分学习权重和推理控制器的收益。
开发集多次使用后已参与选模；后续模型需预先留出新的最终种子块，不能把本轮final再当未见验收。

## 另一个起点与阴性对照

512局示教、16epochs，再用 `configs/ppo-bc-anchor-v02.json` 从其权重初始化300次PPO，
仍是单独的实验分支。它的预训练数据、训练历史不同，不能把与128局主线的差异归因为单个因素。
夜间seed40在第92次更新碰到引擎截断，停止日志和从安全检查点恢复的过程均保留；44/45是独立续训种子。
如显式采用 `--max-truncated-episodes 10`，必须报告截断与丢弃决策数，不能删掉失败记录。

从零PPO的小规模三种子实验没有学会有效经营，也是本轮结果。公开Bench的动作一致率和训练回报
不替代完整对局成绩。冻结历史模型作为对手的实验同样保留，没有把它称为真人校准的对手库。
