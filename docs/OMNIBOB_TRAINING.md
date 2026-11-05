# OmniBob v0.2：本机训练与竞技协议准备

当前检出以 `fb61201` 为基础，应用 rizling.top 公开的 observe2 补丁，TavernGym **1.1.0**。
v0.1 的策略、观测类及合同继续冻结；[v0.1 使用说明](OMNIBOB.md)记录迁移前的环境。
这里的模型和训练代码是独立实现，不是作者发布的 TavernAgent。

## 安装与快速使用

```bash
cd /Users/ostrovsky/Projects/Zen/omnibob
bash tools/setup-omnibob.sh --mps
.venv/bin/python -m omnibob check
.venv/bin/python -m omnibob play --policy omnibob --config configs/arena-v2.json --split train --seed 42 --output runs/my-v02-game
.venv/bin/python -m omnibob replay runs/my-v02-game
.venv/bin/python -m omnibob evaluate --policies random scripted omnibob --config configs/arena-v2.json --games 64 --split dev --seed 2000 --output runs/my-v02-baselines
```

每个输出目录必须是新的。环境只装在 `.venv` 和 `.local-tools`。CPU 模拟器不需要 torch；
训练、学习策略及其测试需要 `requirements-omnibob-mps.lock`。MPS 不可用时明确改用 `--device cpu`。
没有云资源或付费 API。不要复制 Linux 二进制到 Mac。

完整对局默认保留动作轨迹。可生成一个独立 HTML 复盘文件，用浏览器打开，逐步查看当时己方
战场、手牌、商店、选项、动作和模型理由；生成前会重放验证观测摘要与最终结果：

```bash
.venv/bin/python -m omnibob.review runs/my-v02-game --output runs/my-v02-review.html
open runs/my-v02-review.html
```

批量对局需在 evaluate 时加 `--trace`。复盘页无外部资源和网络请求，只导出正常玩家视图；
最终名次是结果展示，不会回灌给策略。未完成、截断或损坏的轨迹会拒绝生成通过验证的复盘。

## 可复跑的三个阶段

先收集**训练域**内冻结启发式的完整对局，再模仿；每五局中的一局保留为整局验证，
不将相邻决策随机拆入训练/验证。动作一致率只能诊断模仿，不能代替对局成绩。

```bash
.venv/bin/python -m omnibob.imitation collect --rules configs/arena-v2.json --games 128 --seed 10000000 --output runs/my-teacher --max-seconds 120
.venv/bin/python -m omnibob.imitation fit --dataset runs/my-teacher --output runs/my-bc --epochs 8 --device mps --max-seconds 600
.venv/bin/python -m omnibob evaluate --policies bc omnibob scripted --checkpoint runs/my-bc/best.pt --games 64 --split dev --seed 2100 --output runs/my-bc-eval
```

随后从权重初始化 PPO；新建优化器、环境和采样 RNG。以下是短实验，不保证比启发式强：

```bash
.venv/bin/python -m omnibob.train --initialize runs/my-bc/best.pt --config configs/ppo-warm-v02.json --rules configs/arena-v2.json --updates 15 --device mps --max-seconds 180 --output runs/my-ppo
.venv/bin/python -m omnibob evaluate --policies ppo omnibob scripted --checkpoint runs/my-ppo/latest.pt --games 64 --split dev --seed 2100 --output runs/my-ppo-eval
.venv/bin/python -m omnibob play --policy ppo --checkpoint runs/my-ppo/latest.pt --split dev --seed 2101 --output runs/my-ppo-replay
.venv/bin/python -m omnibob replay runs/my-ppo-replay
```

从零 PPO 则省略 `--initialize`。`--resume runs/my-ppo/latest.pt` 延续优化器、环境存档、RNG、
更新计数和保存的训练配置；`--updates` 是额外更新数。不要用 `--resume` 加载模仿权重。
CPU 更新边界恢复已测试与连续训练逐参数相同；MPS 不承诺跨设备或跨版本逐比特相同。

不同运行之间可计算完整八席种子块上的配对差值；负值表示候选平均名次更好：

```bash
.venv/bin/python -m omnibob.compare runs/my-ppo-eval runs/my-bc-eval --baseline-policy bc --output runs/my-paired-comparison.json
```

比较器拒绝环境、规则、split、调度或对手不同的结果；使用启发式对手时还检查其源码摘要。一块有失败/截断就排除整个八席块并报告数量。
开发集上的重复选模仍有偏差，bootstrap 区间不是最终验收。
单策略汇总的 `mean_placement` 使用全部正常完赛局；若有不完整八席块，置信区间对应另列的
`complete_block_mean_placement` 和 `complete_block_metric_denominator`。两者分母不同，不能把
完整块区间误当成全部完赛局均值的区间。没有失败时两者一致。

训练支持 `--deadline UNIX时间戳`、`--max-seconds`、`--max-rss-gib`；Ctrl+C / SIGTERM 在下一检查点
请求安全退出。每五次更新留档，最新权重原子替换。硬杀进程、系统掉电不能保证最近一步保存。
限时在 Python/原生调用之间检查，不能抢占挂起的内核。部分 rollout 中断时保留环境和权重，
但不声称恢复被丢弃的 rollout 或更新到一半的 minibatch。

原生模拟器可能出现第100回合强制结束的对局。默认训练遇到这种截断仍停止；可显式使用
`--max-truncated-episodes 10` 给本次运行一个有界容错预算。截断局单独写入日志、不给名次奖励，
重置该环境，并从当前rollout中排除这段轨迹的全部剩余片段；其他环境、此前已完整结束的轨迹
及重置后的新轨迹仍可训练。超过预算则停止。更早更新中已经使用价值估计的部分无法追溯撤销，
因此必须报告截断次数和丢弃决策数；这不是消除了模拟器截断偏差。完整评测始终排除截断局，
配对统计排除受影响的整个八席种子块。失败运行的latest.pt不能部署或恢复，选择此前安全检查点。

评测也处理Ctrl+C / SIGTERM，尽可能写出已完成、当前中断和未开始的数量；强制kill或进程崩溃
只能依靠已刷新的逐局日志恢复证据，不能宣称正常结束。

## 实现边界

- 393,552 参数，宽 64、两层、四头实体 Transformer；49 个实体 token。
  数值特征、卡牌 ID/静态属性、可见附着效果、自己的计数器和对手公开/上次见到的信息。
  使用完整有语义的 observe2 扩展；预留的零 conditioning 槽位不作伪造输入。
- 245 动作分组 pointer head，采样和推理均屏蔽非法动作；学习策略只收到只读玩家观测。
- 默认自定目标为终局 `1 - 2*(名次-1)/7`。`TrainConfig.decision_cost` 默认 0；效率对照实验
  另测试每次决策扣 0.0003，属于明确改变的 OmniBob 目标。PPO、GAE、KL 提前停止、梯度裁剪。
  这些与论文的完整奖励/课程方案不同。
  另提供默认关闭的 `reference_kl_coef`：使用 `--initialize` 时冻结初始策略，增加
  `KL(初始策略 || 当前策略)` 正则，并把参考权重存入训练检查点供恢复。它与单次 PPO 更新的
  `target_kl` 提前停止不同；系数和具体实验必须另行报告。
- 训练明确生成 `[0, 10^9)` 种子；dev 为 `[10^9, 2*10^9)`，final 为 `[2*10^9, 3*10^9)`。
  训练和评测默认使用轮流逐动作的七个上游脚本对手。也可显式选固定启发式/混合对手与打乱调度，
  每种设置分别统计。原生 BatchEnv 仍只支持 observe v1，
  因此 v2 训练使用独立 VectorGames；不声称具有论文的真实对局校准对手库。
- 保存引擎/动作/卡表/观测合同、规则、模型结构、训练配置和源码摘要。checkpoint 配置与评测
  显式配置不一致会拒绝；默认评测学习策略时使用 checkpoint 内的规则。
- 上游游戏逻辑保持不变；公开补丁增加可见观测和 Python 绑定，并升级包版本。
  `configs/arena-v2.json` 明确每回合 100 操作；原始默认是 200，不能混合比较。

## 对手与行动顺序的压力测试

```bash
.venv/bin/python -m omnibob evaluate --policies ppo omnibob scripted --checkpoint runs/my-ppo/latest.pt --opponents heuristic --scheduler shuffled_rounds --games 64 --split dev --seed 4000 --output runs/my-stronger-opponents
```

`--opponents` 支持 `scripted`、`heuristic`（七个冻结 tempo-v0.1）、`mixed`（按 seed+seat 奇偶选择两者）。
`--scheduler shuffled_rounds` 使用与引擎独立的 RNG，每轮打乱准备行动席位的顺序，仍保持可重放。
训练也支持这两个参数；恢复会检查配置与已保存的各对手状态、调度 RNG。它们是合成压力测试，
不能替代真实对局校准，也不能证明竞技场的异步网络调度已验证。

本地评测还支持 `--opponents learned --opponent-checkpoint runs/my-policy-package/policy.pt`，
用七个冻结学习策略作压力测试。每席独立控制器状态、共享只读模型；候选和对手的规则必须一致，
对手权重摘要纳入配对检查。这不是论文的校准对手库。
完整八席同模型对局应为平均名次 4.5，可用于检查座位和状态隔离。

`--opponents mixed_learned --opponent-checkpoint ...` 则按 `(seed+seat)%3` 固定分配上游机器人、
冻结启发式和冻结历史模型。每个八席种子块轮换受控席位；对手分配可复现，不读取未来随机流。
这提供一个小型异质合成课程，仍不包含真实对局校准。训练与评测都支持该设置，恢复同时保存
各席启发式历史和学习策略控制器历史；同一份权重只读共享。

训练也可指定 `--opponents learned --opponent-checkpoint ...`。冻结对手始终在CPU推理，
每席只看自己的正常观测；训练器保存控制器历史和对手权重摘要，恢复时拒绝变更的对手。
模型权重在同一进程内共享，各席历史不共享。与更新中的学习者没有梯度连接。
本机初测8环境、128步、单工作线程更适合这个较慢的对手；4工作线程的短探针反而更慢。
建议先跑3次更新验证速度和内存，再增加预算：

```bash
.venv/bin/python -m omnibob.train --initialize runs/my-ppo/latest.pt --opponents learned --opponent-checkpoint runs/my-policy-package/policy.pt --envs 8 --horizon 128 --cpu-threads 1 --updates 3 --rules configs/arena-v2.json --device cpu --max-seconds 180 --output runs/my-frozen-opponent-probe
```

使用JSON训练配置时，其同名字段优先于CLI默认项；具体夜间实验使用 `cpu_threads: 1`，
不是把增加工作线程视为必然加速。冻结历史模型对手仅是本地合成课程，尚无真人对局校准。

训练配置可选 `terminate_on_elimination: true`：当本玩家名次已经由引擎确定时结束该训练轨迹。
测试验证其可见决策轨迹和最终自身名次与继续打完整场一致。此时未知的其他玩家名次仍为 0，
`completed_episodes_this_run` 与 `completed_games_this_run` 分开报告；评测始终打完整场并排除截断。

可在本地推理/评测加 `--move-budget 7` 测试站位控制器。它根据已经观察到的己方排列阻止循环，
每回合最多建议七次移动，不模拟随机结果。原始训练权重默认关闭，只有学习策略支持；CLI 覆盖不会修改 checkpoint。
这是显式规则与学习模型的组合，版本和运行参数会记录，不能把组合收益全部归为 PPO 收益。

另可加 `--repeat-limit 1` 限制“相同可见状态下重复同一动作”，用于诊断打出手牌后又取消等循环。
只忽略observe2中已核验的己方本回合操作计数，其他特征、卡牌和合法mask全部参与状态摘要。
这依然是启发式：有限观测相同不代表完整游戏状态相同，可能限制有用操作，因此需要单独消融；
所有候选被过滤时保留原合法mask。每回合/每场清空历史，默认关闭，可通过 `--repeat-limit 0` 禁用。

## 导出可部署权重

```bash
.venv/bin/python -m omnibob.export runs/my-ppo/latest.pt --output runs/my-policy-package
.venv/bin/python -m omnibob play --policy ppo --checkpoint runs/my-policy-package/policy.pt --split dev --seed 2101 --output runs/my-exported-game
# 若要把站位控制器一起冻结到部署定义：
.venv/bin/python -m omnibob.export runs/my-ppo/latest.pt --move-budget 7 --output runs/my-guarded-package
# 可选再保存重复动作控制器，须单独评测确认：
.venv/bin/python -m omnibob.export runs/my-ppo/latest.pt --move-budget 7 --repeat-limit 1 --output runs/my-repeat-guarded-package
```

部署包只有权重、模型结构、环境/规则合同及来源摘要，移除优化器、训练模拟器存档和 RNG。
它可以推理或 `--initialize`，不能用于精确 `--resume`。BC 模型导出后仍使用 `--policy bc`。
带控制器的部署包记录预算、版本和实现摘要，play/evaluate/竞技适配器自动使用同一控制器；
显式 `--move-budget 0` 可做关闭控制器的本地消融。每场竞技连接上下文独立保存控制状态，
共享只读模型权重。PPO 权重初始化不继承推理控制器，训练配置和部署运行定义分别记录。
加载时检查模型实现 SHA256；早期实验权重缺少内嵌摘要时需保留原目录的 `manifest.json`，
通过导出命令可补齐部署元数据。修改模型代码后要显式迁移，不要绕过实现检查。

后续实验的 manifest 同时记录 `source_archive`，指向 `runs/source-snapshots/` 下的源码覆盖包。
先检出记录的 Git 基点，再恢复该包并本机安装依赖；包不包含凭据、虚拟环境或原生二进制。

## 可选的等权重融合实验

`omnibob.average` 对同一初始化检查点的多个PPO续训权重做等比例参数平均，导出后仍是同样规模
的单个网络。它检查模型、环境、规则、初始化摘要和数值，不继承各父模型的推理控制器。
独立随机初始化的隐藏单元可能不对齐，因此工具拒绝不同初始化；相同初始化也不保证融合有效。

```bash
.venv/bin/python -m omnibob.average runs/seed1/update-000150.pt runs/seed2/update-000150.pt runs/seed3/update-000150.pt --output runs/my-average
.venv/bin/python -m omnibob evaluate --policies ppo omnibob --checkpoint runs/my-average/policy.pt --repeat-limit 1 --games 128 --split dev --seed 6500 --output runs/my-average-dev
```

融合没有新增优化器更新或轨迹，因此自身 `steps/update` 为0，策略名称明确标为uniform-average。
`weight_averaging`保存每个父权重、配置、来源摘要和等权比例；父阶段步数有共同祖先，不能简单相加
成一条训练轨迹。融合收益需要单独报告；仅在开发集选择，冻结后再做最终验收，不视为论文方法。

## 竞技准备

```bash
.venv/bin/python -m omnibob.arena --output runs/my-arena-preflight.json
# 学习权重还要核对训练时的规则：
.venv/bin/python -m omnibob.arena --checkpoint runs/my-ppo/latest.pt --output runs/my-model-preflight.json
```

该命令仅匿名 GET 公开 manifest，不读凭据、不入队。模型入口为 `omnibob.arena:act`，接收官方
连接器的 `(observation, context)`；默认用冻结启发式，`OMNIBOB_CHECKPOINT` 可选择本地权重。
`OMNIBOB_JOURNAL_DIR` 可记录提供给己方的观测、建议动作和理由。最终名次由连接器结果另行保存。
尚未联网对局；不能将兼容检查或本地训练视为参赛验证。首次真实连接前还需用户授权和本地 key。

## 人工执行的真实游戏记录

已提供本地日志工具；**尚未实现真实客户端自动观测编码或验证真实游戏实力**。
每步先保存己方可见截图/文本，再记录建议，执行后追加实际动作与偏离原因；最后记录名次。
版本、建议和结果均为用户/助手报告，日志不能独立证明其真实性。只录入己方正常可见信息。

```bash
.venv/bin/python -m omnibob.manual init runs/my-real-game --game-version '填写实际客户端版本' --rules '填写模式与规则' --policy '填写策略/模型版本'
.venv/bin/python -m omnibob.manual suggest runs/my-real-game --turn 1 --evidence /absolute/path/to/own-visible-screenshot.png --advice '购买左侧随从' --reason '根据当前可见场面保住血量'
.venv/bin/python -m omnibob.manual execute runs/my-real-game --decision 1 --actual '购买左侧随从'
# 若没有照建议执行，增加 --deviation '具体原因'
.venv/bin/python -m omnibob.manual finish runs/my-real-game --placement 4 --notes '实际结果与执行异常'
.venv/bin/python -m omnibob.manual report runs/my-real-game
```

截图/文本会复制进会话目录并记录 SHA256；建议、执行、结果分别带时间戳，缺失执行记录不会被当作已执行。
上述命令只是用法示例，本轮没有真实游戏对局。日志保留在本机，没有上传或自动控制游戏。

## TavernBench 的有限诊断

```bash
.venv/bin/python -m omnibob.diagnose_bench --output runs/my-bench-heuristic
.venv/bin/python -m omnibob.diagnose_bench --checkpoint runs/my-bc/best.pt --output runs/my-bench-model
```

仅支持公开提供动作到答案映射的 D2、D4、N1、N2，共 **80/368** 题。按题目列出的合法候选限制
动作，再用己方可见观测选取；驱动器不推进存档，不模拟战斗，不读取未来随机结果，模型不读取答案。
全部回答写完后才载入标签评分。输出逐场景结果、错误和任务规则差异；不输出完整 TavernBench 总分。
旧任务配置与竞技训练配置不同，所以这是带分布差异的公开诊断。不要据此调参再宣称独立泛化。

## 常见问题

`incompatible TavernGym`：先检查实际加载的 `.venv` 与原生构建，按安装命令本机重新编译。
不要改合同哈希掩盖不兼容。公开接口改变需重新审阅，而不是截断或补零。

HTTP 403：站点对默认 Python urllib User-Agent 曾返回 403；预检使用明确客户端标识后 GET 成功。
HEAD 返回 405 不代表 GET 不可用。网络错误要记录，不能据缓存宣称当前在线兼容。

训练损失下降但名次很差：均匀合法动作包含大量无效经营与站位动作；稀疏终局奖励很难从零探索。
查看完整对局、每局操作数及冻结基线成绩。模仿分数、训练集回报和开发名次都不等于最终泛化证据。

更多有界训练命令、三种子配方和复现边界见[实验配方](OMNIBOB_EXPERIMENT_RECIPES.md)；
论文缺失材料与本机主动简化见[v0.2复现范围](OMNIBOB_REPRODUCTION_V02.md)。

对多次恢复的同源分支，可显式传入 `omnibob.average --initialization <共同起点>`，逐级验证所有父文件和SHA256，不改写立即父记录。详见 [200更新平均实验](OMNIBOB_AVERAGE200.md)。
