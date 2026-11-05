# OmniBob 本地 v0.3

v0.3 是正式确认通过的固定 env32 候选的推理发行，权重逐字节保留，没有新训练。
默认使用 CPU 单线程、合法动作 argmax、RepeatGuard1、MoveGuard0 和 shuffled_rounds 调度。
本地发行版本由 `deployment.json` 标识，与历史研究模块的包版本独立。

## 运行和回退

在当前已安装的工作区执行，所有对局输出目录必须尚不存在：

```sh
cd /Users/ostrovsky/Projects/Zen/omnibob
.venv/bin/python -m omnibob.deployment status
.venv/bin/python -m omnibob play --seed 610010 --output runs/my-v03-game
.venv/bin/python -m omnibob replay runs/my-v03-game
.venv/bin/python -m omnibob.review runs/my-v03-game --output runs/my-v03-game.html
```

回退和重新启用都只原子切换本地指针，不改任何权重：

```sh
.venv/bin/python -m omnibob.deployment use v0.2
.venv/bin/python -m omnibob.deployment status
# 重新使用 v0.3
.venv/bin/python -m omnibob.deployment use v0.3
```

`play` 和 `evaluate` 不传策略时使用当前发行。显式 `--policy omnibob` 仍为启发式基线；
`--policy ppo --checkpoint 路径` 保留研究用法。默认发行不接受 checkpoint 或控制器覆盖，
需研究消融时显式选 `ppo`。未安装本地发行、权重或兼容检查失败会报错，不默默回退启发式。
`runs/releases/current.json` 是活动版本指针，`catalog.json` 登记各版本清单及其校验和。

```sh
# 八席运行；用于本地检查，不作为新验收
.venv/bin/python -m omnibob evaluate --games 8 --seed 610011 --output runs/my-v03-eight-seats
```

每次默认运行保留 `deployment.json`、`manifest.json` 和逐局结果，可核对实际权重、控制器、
调度和环境。切换前会验证权重、推理格式、模型实现、规则、卡表、观察/动作编码和运行配置。
失败的切换不会覆盖之前的活动指针。

## 离线适配

```sh
.venv/bin/python -m omnibob.offline_check \
  --checkpoint runs/releases/omnibob-v0.3-20261101/policy/policy.pt \
  --recording runs/practice-20261028-103720/policy/decisions.jsonl \
  --output runs/my-v03-offline-check
```

该命令在本地重放己方可见观察，核对直接推理和协议适配器动作、合法性及会话状态隔离。
它不访问服务器、不读 `.env`，也不复现候选的反事实名次。竞技连接器仍要求显式指定模型，
本地激活不会启动连接器、排队或发送动作。

## 强度证据和限制

正式确认：384个新种子块，四场景共21,888局，21,887局自然结束，1局自然封顶按候选第8悲观计分，
零技术失败。四场景等权候选减基线平均名次差为 **−0.09920247**，95%区间
**[−0.13427734, −0.06412557]**，综合上界小于0，所有场景上界小于+0.15，满足预设门槛。
v02_field 按冻结协议使用八席对称参考值4.5，单席见证均值不是该参考值的替代。

此前8对32环境的双重验收未通过，原结果不变。本次证明固定候选在新种子本地套件上改善，
不能据此归因于32环境本身，也不是线上竞技或真实客户端验证。没有校准的真实对手库，
不宣称复现论文成绩。重复阅读公开评测也不等于独立泛化证据。

正式原始证据：`runs/formal-confirmation-v2-20261029T070700Z/`。
发行：`runs/releases/omnibob-v0.3-20261101/`；本次验证：`runs/local-upgrade-v03-20261101/`。
v0.2 原目录、完整包和历史失败结果均保留。发行包含推理权重、紧凑证据、源代码覆盖快照和校验和；
只承诺当前本机安装验证，不包含虚拟环境、凭据或原生二进制，也不声称已在新机器安装通过。
