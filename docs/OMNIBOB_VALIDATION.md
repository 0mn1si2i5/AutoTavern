# OmniBob v0.1 本机验证记录

这是迁移前的历史记录；当前v0.2训练与最终本地验收见[夜间探索报告](OMNIBOB_EXPLORATION.md)。

日期：2026-10-28（Asia/Singapore）。**本轮可运行基线与评测闭环已交付；训练、竞技场参赛、真实游戏未完成。**

## 环境与安装

- Apple M3 Pro：12 核 CPU（6P+6E），18 核 GPU，18GB 统一内存；不是用户预估的 14 核 CPU。
- macOS 27.0.1，arm64；起始可用磁盘约 244GiB。
- Python 3.12.2 / NumPy 2.5.4 / PyTorch 2.14.1 / Rust 1.98.1 / maturin 1.15.0。
- 全新 clone 用户 fork，HEAD `fb61201`；本机 release 构建指纹 `c807e9aa1a6746ee`，与当前源文件匹配。
- `.venv` 约 652MB，项目内 Rust 工具链约 502MB，构建产物约 334MB；未复制云环境或二进制。
- `bash tools/setup-omnibob.sh --mps` 已复跑成功；上游引擎、Python 接口、Cargo.lock、pyproject 及 Rust 版本约束未修改。

安装首次失败：rustup 安装与 maturin 自动工具链检查同时下载，出现 `.partial` 文件重命名失败。
串行完成相同版本工具链安装后构建成功；保留 [初次失败日志](evidence/omnibob-v0.1/initial-build-failure.txt)。
初始 OmniBob smoke 也发现 card_info 不接受空索引 0、默认配置省略 experimental_mode 两个接口假设问题；
均在评测前修正，初次日志仍在本地 `runs/setup/`，未作为成功样本。

## 行为与集成检查

| 检查 | 本机结果 |
| --- | --- |
| 上游 Python 测试，改动前 | 46 passed |
| 全部 Python 测试，含 OmniBob | 54 passed；安装脚本再次验证亦通过 |
| Rust `cargo test --release -p taverngym --lib` | 15 passed；未声称跑过全部 Rust integration/content suite |
| 三策略×八座位完整游戏及动作重放测试 | 24 局完整结束，最终名次与玩家观测哈希一致 |
| 信息边界 | 改变对手隐藏手牌/未见阵容及未来 RNG，候选当前观测和 OmniBob 决策不变 |
| 失败边界 | 非法动作、driver 限步、KeyboardInterrupt、引擎回合上限分别记录；截断不入主成绩 |
| 版本边界 | 同尺寸但不同词表被拒绝；observe2 不能冒充 v1 |
| 用户命令路径 | `play --seed 27 --seat 3` 完整运行；`replay` 验证 1 局和所有候选观测哈希 |
| TavernBench | 本机重评分已发布答案成功；没有调用模型 API，没有生成 OmniBob Bench 成绩 |
| 文件检查 | `git diff --check`、shell 语法检查通过；上游核心目录无差异 |

证据：[Python 日志](evidence/omnibob-v0.1/python-tests.txt)、[Rust 日志](evidence/omnibob-v0.1/rust-lib-tests.txt)、
[已发布答案重评分](evidence/omnibob-v0.1/released-answer-scoring.txt)。
测试不是对全部卡牌规则、真实客户端或竞技场的验收。

## 冻结基线结果

每局一席候选对七个上游脚本机器人。同一种子在八席轮换，三种策略共享相同种子/座位计划。
策略在两批之间**没有调参**；[evidence.json](evidence/omnibob-v0.1/evidence.json) 核验两批
`policies.py`、`observation.py`、合同文件的 SHA256 一致。
后续版本请新建策略版本文件，保留 v0.1 实现和基线记录，不能覆盖后继续沿用同一个版本名。

第一批每种 64 局（8 个开发种子块）：OmniBob 3.2344 / 20.3125% / 75%，脚本 4.5 / 12.5% / 50%，
随机 7.9531 / 0% / 0%。这批只用于功能观测；没有据此修改策略。

第二批每种 256 局（32 个新开发种子块，绝对种子 1000000100–1000000131）：

| 策略 | 正常完成 | 平均名次 ↓ | 吃鸡率 | 前四率 | 名次均值 95% 种子块 bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 随机合法 | 256/256 | 7.9609 | 0% | 0% | [7.9375, 7.9844] |
| 上游脚本 | 256/256 | 4.5000 | 12.50% | 50.00% | [4.5000, 4.5000] |
| OmniBob tempo-v0.1 | 256/256 | 3.4414 | 15.23% | 73.05% | [3.2461, 3.6329] |

两批共 960 局全部完成，失败/中断/截断/非法动作均为 0。第二批墙钟总耗时 7.19 秒。
OmniBob 49,660 次候选决策，平均 0.0321ms、P95 0.0672ms，策略对应完整对局累计 3.59 秒；
观测构造与对手开销单独记录。测试/安装活动可能影响墙钟延迟，这不是专用空载机器的性能承诺。

上游脚本行的区间退化为 4.5，是相同全机器人大厅轮换观测全部八席的算术结果，
不意味着真实对局没有不确定性。32 个独立种子块仍有限，结果只支持“当前固定脚本对手中表现较好”，
不代表论文成绩、强 Agent 对抗水平、竞技排名或真人强度。**final 种子域完全未运行。**

原始记录与配置：
[64 局/策略汇总](evidence/omnibob-v0.1/v0.1-dev-64/summary.json)、
[256 局/策略汇总](evidence/omnibob-v0.1/v0.1-dev-256/summary.json)、
[逐局记录](evidence/omnibob-v0.1/v0.1-dev-256/games.jsonl)、
[源码/环境/配置清单](evidence/omnibob-v0.1/v0.1-dev-256/manifest.json)。

## CPU / MPS 实测

[最终探针 JSON](evidence/omnibob-v0.1/performance.json) 含设备、依赖、配置与源码指纹。
原生 CPU 连续 128 局机器人游戏：0.417 秒，约 **307 局/秒**。
首次探针为 349 局/秒，说明短样本墙钟存在波动；未据此声称超过论文。

BatchEnv（128 步，固定选择首个合法动作，不是策略训练吞吐）：

| 并行游戏 / Rust 线程 | ready 决策/秒 | 测量结束 RSS |
| --- | ---: | ---: |
| 1 / 1 | 2996 | 87 MiB |
| 8 / 1 | 3148 | 90 MiB |
| 8 / 2 | 3739 | 92 MiB |
| 32 / 4 | 7129 | 96 MiB |

MPS built/available 均为 true，未启用 `PYTORCH_ENABLE_MPS_FALLBACK`。
实体网络探针 346,678 参数，49 token、d=64、2 层、4 头、FF=128：

| batch | CPU 更新耗时 | MPS 更新耗时 |
| --- | ---: | ---: |
| 8 | 6.73ms | 13.80ms |
| 32 | 13.12ms | 13.76ms |
| 128 | 37.79ms | 16.77ms |

每种配置两次热身、八次计时，均完成前向/反向/Adam 更新且 loss/梯度有限。
同参数 CPU/MPS 合法 logits 最大绝对差约 `3.35e-7`。
进程累计峰值 RSS 约 **487MiB**（510,869,504 bytes）；Apple GPU 和 CPU 共享内存，
MPS driver allocation 与 RSS 不宜简单相加。这里未分配大 rollout buffer，不能视为完整 PPO 内存上限。

结论：模拟端先用 32 环境 / 4 Rust 线程；小批量推理允许 CPU，batch 128 更新优先测 MPS。
没有必要为 v0.1 购买云资源，也没有理由直接选论文 18M 模型或大规模并行。

## 下一阶段最值得做的一个实验

**小型实体 PPO 能否在相同脚本对手下学到超过随机、接近冻结启发式的策略？**

建议一个有界实验：d=64、2 层、4 头，增加实体类型/空槽 mask；32 个环境、4 线程，
每次收集 128 步（4096 决策），minibatch=128，2 epochs。先 CPU 完成一个真实 rollout→GAE→PPO
更新，再验证 MPS；从 3 个训练种子各 15 次更新开始（合计约 184k 决策），总墙钟上限 10 分钟、
RSS 软上限 4GiB，超限保存诊断并停止。此处是实验建议，本轮没有启动。

首轮目标明确采用 OmniBob 自定的线性终局名次效用 `1 - 2*(placement-1)/7`，
γ=1、GAE λ=0.95、clip=0.2；它是管线验证目标，**不是论文最终训练目标复现**。
完整记录参数、动作 mask、终止/截断、梯度、KL、吞吐和 checkpoint 合同；非法动作应为零。

训练仅用 train 域。开发选模使用尚未用过的 dev 偏移（如 1000 起），同时重跑冻结随机、脚本和
OmniBob v0.1；候选冻结后才首次使用 final 偏移 0 起，每种至少 256 局的八席配对比较。
不达标就保留失败曲线；如果只会利用脚本弱点，则下一项研究是对手多样性/真实对局校准，
不能把简单自我对弈包装成论文的对手库。

## 剩余范围

- 竞技场已核实公开协议，但当前环境版本不兼容；补丁审阅、迁移、连接器、认证练习和真实竞技未做。
- 实际 Hearthstone 客户端信息适配、人工执行记录与真实游戏强度未验证。
- 启发式没有完整卡牌语义/目标规划；训练代码、训练权重、校准对手库、论文级模型均未完成。
- 额外云采购、付费 API、上传凭据/权重、长训练、Git 提交/推送、参赛提交：均未执行。
