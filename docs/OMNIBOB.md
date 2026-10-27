# OmniBob v0.1：本机可运行基线

> 这是迁移前 v0.1 的记录。当前检出已迁移 TavernGym 1.1.0 / observe2；安装、训练和竞技准备请先读
> [v0.2 使用说明](OMNIBOB_TRAINING.md)。下文关于未实现训练、未修改上游接口的描述仅适用于 v0.1。

目标是先得到可复跑的完整对局与比较结果。v0.1 是独立启发式策略，**没有训练权重**。
上游引擎保持 `fb61201a557ddd36a411cf44aac51ce4d9b606ce`、TavernGym 1.0.0。
运行入口在仓库根目录的 `omnibob/`，不更改上游 `python/taverngym`、Rust 引擎及版本约束。

## 安装

已验证环境：macOS 27.0.1、arm64、Python 3.12.2、Rust 1.98.1。
本机为 M3 Pro，12 核 CPU（6P+6E）、18 核 GPU、18GB 统一内存。

```bash
cd /Users/ostrovsky/Projects/Zen/omnibob
bash tools/setup-omnibob.sh --mps
```

需要已有 `uv`、Xcode Command Line Tools、网络用于下载公开免费软件包。
不带 `--mps` 可只安装 CPU 模拟器和测试依赖。虚拟环境为 `.venv/`；Rust 安装到
`.local-tools/`，不会修改 shell 配置或全局 Rust。安装脚本不删除现有环境、不升级上游约束。
两份 `requirements-omnibob*.lock` 固定本次实际安装的版本；上游 `Cargo.lock` 原样使用。

## 直接运行

一局 OmniBob 对七个脚本机器人；可指定训练/开发/最终种子域、域内偏移和座位：

```bash
.venv/bin/python -m omnibob play --policy omnibob --split train --seed 1 --seat 0 --output runs/my-first-game
.venv/bin/python -m omnibob replay runs/my-first-game
```

`play` 默认记录全局动作序列，包含 OmniBob 的理由、每次决策延迟和玩家可见观测哈希。
`replay` 重建游戏、重放每个动作并检查观测哈希与最终八席名次，不重新调用策略。
它验证可复现性，不证明策略有优势。记录只在本机保存。

三种策略公平比较，每种 64 局，共 192 局：

```bash
.venv/bin/python -m omnibob evaluate --policies random scripted omnibob --games 64 --split dev --seed 0 --output runs/my-dev-64
```

`--games` 是**每种策略的局数**，必须是 8 的倍数。同一引擎种子依次让候选策略坐 0–7 席，
其余七席使用同一上游机器人；不同策略共享种子和座位，并交错运行。
引擎按座位递增逐次执行动作，策略改变后共享卡池与随机路径可能分叉；配对不保证后续商店相同。
`scripted` 自比较的八席均值应为 4.5、第一率 12.5%、前四率 50%，可作为驱动器检查。
这是本地固定调度，不能等同竞技场网络异步到达顺序。

输出目录必须不存在，避免覆盖证据。需要逐动作记录时加 `--trace`；这会增加 I/O 和延迟。
默认每局最多 100,000 动作、30 秒，`--max-steps` / `--timeout` 可调；时间限在原生调用之间检查，
无法抢占卡死的原生调用。Ctrl+C 会保留已完成记录、当前中断和未开始数量。
遇到原生进程崩溃/强制杀进程时，用 `events.jsonl` 的 started/finished 差集辨认未完成局；
未写出的结果不得记为胜负。本版本不自动恢复硬崩溃任务。

支持 `--config path.json` 提供 `RulesConfig` 覆盖项，运行时写出与默认配置合并后的完整配置和哈希。
v0.1 限定八人、七格棋盘、非 experimental 规则。不要混合不同配置的成绩。

## 输出解释

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | Git commit/脏状态、逐文件源码 SHA256、引擎编译指纹、卡表/词表/动作/布局哈希、配置、策略版本、完整种子座位计划、环境版本 |
| `games.jsonl` | 逐局状态、八席名次、候选名次、步数、非法动作、决策/观测延迟、耗时、截断和失败原因 |
| `summary.json` | 平均名次、吃鸡率、前四率、分母、完成/失败/中断/截断/未开始数、延迟分位数、运行总耗时 |
| `events.jsonl` | 每局开始/结束事件，便于辨认异常退出 |
| `decisions.jsonl` | 可选动作日志，`play` 默认开启 |

决策延迟只计算 `act`；观测生成另记，整局耗时包含所有对手和驱动开销。
引擎战斗截断或回合上限结束均标为 `truncated`，保留名次但不纳入主成绩。
非法动作立即失败，不偷偷替换成合法动作。失败会影响完成率，不能只展示幸存的对局成绩。
CLI 遇到任何未正常完成局返回 2，完整完成返回 0。

置信区间对**完整八席种子块**做 bootstrap（2000 次）；座位不是八个独立种子。
至少两块才计算区间。64 局只有八块，仍是小样本工程验证，不能宣称统计显著提升或竞技强度。
策略比较遇到缺失或截断时，应先调查失败，再在相同完整种子块上比较。

## 策略边界

- `random`：在当前全部合法动作中均匀采样，使用与引擎分离的局内 RNG。
- `scripted`：原样调用上游 `Game.bot_action`，明确为比较对象；不是训练成果。
  该适配器有原生 chooser 回调，其余策略没有游戏句柄。上游 bot 只检查自己的区域和合法性；
  选择部分选项时借用引擎随机数，Python binding 事后恢复随机状态。它不具备外部纯观测部署接口。
- `omnibob`：自有数值启发式。优先打出手牌、磁力合体、替换弱随从、按经济曲线升级、
  结合身材/关键词/对子/种族选购，有限刷新、技能/赐福/激活与攻击力站位。
  每步给出理由。英雄仅按可见护甲估值，复杂法术目标、饰品、延迟效果没有完整语义规划。

`Policy.act(Observation) -> Decision(action, reason)`；Observation 只有只读 features/cards/mask。
策略使用 `Game.observe`，**不使用 `Game.view` 调试视图**（含其他玩家私有调试字段）、
`state_json`、`debug`、未来随机数或未见阵容。测试中的状态编辑只用于构造信息隔离反例。
v0.1 不做模拟搜索；未来搜索必须重新采样未知信息与未来随机流，不能直接利用保存的确定随机状态。

`omnibob/contract-v1.json` 冻结当前引擎/编码/词表。`python -m omnibob check` 和每次评测启动
均检查它及实际加载的原生二进制。接口、卡表或引擎变化必须评审后迁移；禁止靠补零或自动更新
合同文件假装兼容。尚无模型 checkpoint；将来 checkpoint 也必须绑定该合同和配置。

## 种子使用纪律

| split | 引擎种子范围（左闭右开） | 用途 |
| --- | --- | --- |
| train | `[0, 1000000000)` | 测试、性能探针、训练与调试 |
| dev | `[1000000000, 2000000000)` | 选模、规则迭代和开发比较 |
| final | `[2000000000, 3000000000)` | 冻结策略后的独立验收，不用于选模 |

`--seed` 是域内偏移，不是绝对引擎种子；每八局增加一。分界检查禁止越界。
本轮只运行 train/dev，**final 未运行**。final 一旦被用于修改策略，该批种子应退役，另分配未用偏移。
种子划分是研究纪律，不是隐瞒公开种子的安全机制。
原始基线快照及代码哈希见 `docs/evidence/omnibob-v0.1/`；下一次比较保留冻结版本，再跑候选。
TavernBench 只作诊断；公开答案重评分不是 OmniBob 成绩，公开题调参后的分数不是独立泛化验证。

## 性能与测试

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m omnibob benchmark --output runs/my-machine-probe.json
# 无需 torch 的模拟器测量：
.venv/bin/python -m omnibob benchmark --no-torch --output runs/my-cpu-probe.json
# 如需使用项目内 Rust：
source tools/omnibob-env.sh
cargo test --release -p taverngym --lib
```

探针限 128 个原生完整 bot 游戏、四组 128 步 BatchEnv，及 CPU/MPS 各三种小批量的十次更新
（两次热身、八次计时）。模型仅 49 token、d=64、两层、四头；没有训练轨迹、GAE 或 checkpoint。
真实观测配合合成优势/价值目标，仅验证算子、有限梯度与性能，不能作为训练收益。
MPS 不可用时仍跑 CPU；没有自动启用 MPS 算子 CPU fallback。结果包含错误和设备状态。
实测与下一步实验见 [本机验收记录](OMNIBOB_VALIDATION.md)。

## 常见问题

- **Rust 下载 partial 文件缺失**：不要并行启动 rustup 与首次 pip/maturin 构建。先让
  `rustup toolchain install 1.98.1 --profile minimal` 完成，再重跑安装。此次失败日志在 `runs/setup/`。
- **ImportError / 原生二进制过期**：在本目录重新执行安装脚本；不拷贝云 `.venv`、Linux wheel 或 `.so`。
- **不在根目录无法 import omnibob**：先 `cd` 到本仓库，使用 `.venv/bin/python -m omnibob ...`。
- **输出目录已存在**：换一个新名字，已有结果不自动覆盖。
- **MPS 不可用或报算子错误**：保留 JSON 中的错误，使用 CPU；先验证同模型 CPU 路径，不盲目换依赖。
- **比赛结果没有均值**：查看 completed 分母和失败/截断详情；不是自动当作零分或第一名。
- **竞技场维度不同**：当前站点是 1.1.0/observe2，详见 [协议核验](OMNIBOB_ARENA.md)，本版本不直接参赛。

额外资源采购和付费 API 使用为零；安装仅下载公开工具链/软件包。本轮不推送、不注册、不提交比赛。
