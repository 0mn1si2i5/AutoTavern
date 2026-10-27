# rizling.top：已核验的接口与接入边界

核验日期：2026-10-28（Asia/Singapore）。公开资料来自
[Agent 说明](https://rizling.top/agent.md)、[协议](https://rizling.top/protocol)、
[manifest](https://rizling.top/api/v1/manifest) 和 [官方 Python 连接器](https://rizling.top/downloads/client.py)。
这是日期快照；未来真正连接前需再次预检。

| 项目 | 本轮核验的公开契约 |
| --- | --- |
| 游戏版本 | `36.6.3.253932` |
| 规则 ID | `tg36-bc45cfc55a0e5cd6c14c` |
| 协议 | v2，API 路径 `/api/v1` |
| 环境 | TavernGym 1.1.0，以 `fb61201` 为基础的公开 observe2 补丁 |
| 引擎指纹 | `6d8712c93022ba11` |
| 观测 | float32[4017]、int32[447]、uint8[245]；little endian / base64 |
| 补丁 SHA256 | `1e48a071cad63c284debbf917cb012595a405a4ef5ab66180eb2918e205aaac0` |
| 站点卡表摘要 | `c4c42e6ce5060e1b79fbcd5f4edbf6d17495f0f52690072481d690610f7ef6d7` |
| 策略入口 | 本地 `act(observation, context)`，只向服务发送动作 |
| 时控 | 选将10秒；每席每回合最多100次合法操作；共享回合墙钟 |
| 名义回合时长 | 60/75/90/105/120秒，后期有加时；所有外部AI结束可提前推进 |
| 连接 | HTTPS / WSS，Bearer key；10秒心跳，重连宽限45秒 |
| 并发 | 每账号一席、一场并发 |
| 赛制 | practice不计分；arena可机器人补位；standard等待8个账号 |
| 本地资源限制 | 公开文档未给出模型参数量、本地GPU或内存配额 |

本轮已在隔离检出审阅并本机编译公开补丁，通过测试后迁移主工作区。补丁增加可见观测、
玩家视图和绑定，未修改游戏规则实现。`configs/arena-v2.json` 明确100次操作上限；
原生默认规则是200次，不能混用。完整本地合同在 `omnibob/contract-arena-v2.json`，
公开manifest快照在 `configs/arena-manifest-v2.json`。

## 现在可以验证什么

```bash
cd /Users/ostrovsky/Projects/Zen/omnibob
.venv/bin/python -m omnibob.arena --output runs/my-arena-preflight.json
.venv/bin/python -m omnibob.arena --checkpoint runs/my-policy-package/policy.pt --output runs/my-model-preflight.json
```

只匿名读取公开manifest，核对引擎、观测、动作、卡表和模型规则，不读取key、不注册、不入队。
默认urllib User-Agent曾被返回403；显式客户端标识的GET成功。HEAD返回405不能替代GET结论。

本机还实际跑通了已审阅官方连接器的匿名检查路径。连接器存放在子目录时，临时设置
`PYTHONPATH`为仓库根目录，保证能导入独立的OmniBob模块；不修改全局环境。以下仍只做检查：

```bash
env -u ARENA_API_KEY PYTHONPATH="$PWD" OMNIBOB_CHECKPOINT="$PWD/runs/my-policy-package/policy.pt" \
  .venv/bin/python runs/exploration-20261028/vendor/arena_client.py \
  --url https://rizling.top --policy omnibob.arena:act --check \
  --expected-ruleset tg36-bc45cfc55a0e5cd6c14c \
  --catalog-sha256 c4c42e6ce5060e1b79fbcd5f4edbf6d17495f0f52690072481d690610f7ef6d7
```

这里先按训练说明导出`runs/my-policy-package`；本轮检查结果为`compatible: true, queued: false`。
该结果只证明公开契约检查和本地模块加载，不包含认证、动作提交或完赛验证。

策略入口 `omnibob.arena:act` 已实现；`OMNIBOB_CHECKPOINT` 指向本地部署权重，
无权重时使用冻结tempo-v0.1。带站位控制器的权重按部署定义加载，每场context独立保存历史。
`OMNIBOB_JOURNAL_DIR` 可记录服务提供给己方的视图和建议，不读取他人隐藏状态。

官方连接器观测解码 → 本地适配器 → 真实本地模拟器的整局联调已完成：8席各跑直接调用和
连接器解码两种路径，共16场，名次、步数和己方决策数一致。它没有使用网络或模拟外部服务的
认证流程，不能证明真实网络重连、心跳、共享计时或比赛结果上传已经工作。

另已用只监听`127.0.0.1`的本地HTTP测试服务，驱动未修改的官方连接器与真实TavernGym整局。
正常连接、一次状态503加一次“已接受动作但丢失响应”两种条件下，224个己方动作、可见观测
摘要和终局结果一致；重试保持相同请求内容，动作只执行一次。策略被测试性暂停10.2秒时，
默认10秒心跳正常发送，测试结束服务均关闭。这里没有真实账号或远端入队，也没有验证
实际服务器的认证、共享时钟、断线宽限和计分行为；本地结果不能作为竞技场成绩。

## 首次真实比赛前仍需完成

用户授权实际对局并在本机提供key后，再核对最新manifest。调用官方连接器时必须显式传入
已批准的 `--expected-ruleset` 和 `--catalog-sha256`，避免远端规则更新后继续使用旧模型。
不要仅依赖连接器默认的引擎/卡表检查。先做一场有完整日志的不计分practice，复核动作、
网络重试、时控和完赛结果，再决定是否参加计分模式。

协议使用decision_id、seat_revision和request_id；同一动作的网络重试必须复用请求payload，
过期或非法回应后重新读取状态。不同AI按请求到达顺序影响共享卡池；本地轮换/打乱顺序只是
压力测试，不能模拟完整网络时序。控制器记录的是建议次数，远端拒绝后可能保守地少做一次移动。

本轮没有注册、获取或上传凭据，没有入队、练习或计分比赛，也未上传模型。真实客户端的
人工执行流程见 [训练与使用说明](OMNIBOB_TRAINING.md)，本轮没有真实游戏成绩。

## 公开比赛资料能支持的分析

再次匿名GET核验了公开 `/api/v1/matches`：2026-10-28凌晨读取的一页30条均为已结束对局，
提供模式、人数、规则、名次和最终阵容可用性。它们是其他参赛者的公开记录，不是OmniBob成绩。
[接入说明](https://rizling.top/agent.md)明确没有中间动作/观测回放，最终阵容也不含手牌、商店或种子。
因此不能凭这些资料重建对手完整训练轨迹。当前仅记录接口能力与缺口，未用于训练或选模；
后续真实参赛仍需保存本方收到的可见观测、决策、网络执行情况和终局结果。

## 关联决策、动作确认和终局结果

`omnibob.connector_audit`在本地包装已审阅连接器，不修改其认证、重试或比赛逻辑。
它按`decision_id`/`seat_revision`/`request_id`记录己方可见观测、动作请求和最终回应，
并保留终局结果、时间和源码/模型摘要；心跳并发写入也使用同一顺序日志。
不记录请求头、Agent Key、URL查询内容或任意错误消息。输出目录必须新建。

下列命令已经以匿名检查路径验证；仍不会注册或入队：

```bash
env -u ARENA_API_KEY OMNIBOB_CHECKPOINT="$PWD/runs/my-policy-package/policy.pt" \
  .venv/bin/python -m omnibob.connector_audit \
  --client runs/exploration-20261028/vendor/arena_client.py --output runs/my-connector-check \
  -- --url https://rizling.top --policy omnibob.arena:act --check \
  --expected-ruleset tg36-bc45cfc55a0e5cd6c14c \
  --catalog-sha256 c4c42e6ce5060e1b79fbcd5f4edbf6d17495f0f52690072481d690610f7ef6d7
```

真实比赛获授权后，可用相同包装转交官方比赛参数，另设置`OMNIBOB_JOURNAL_DIR`保存策略理由。
`requests.jsonl`记录**逻辑请求及其最终返回**，不逐个记录连接器内部的HTTP重试；失败异常只保存
类型，未完成操作保留为缺少返回。网络确认是服务器报告的执行证据，并非对游戏正确性的独立证明。
官方连接器的SHA256必须匹配本轮已审阅版本；换版本需重新审阅，不能直接跳过检查。

随本地v0.2交付包还提供独立离线`audit-session.py`，消费上述日志并关联可见状态、合法mask、
动作身份、服务器报告的接受/拒绝和终局。它把未知执行结果、缺失事件和匿名预检明确区分，
不把“请求已发出”当成“动作已执行”。已用所选模型的真实本地HTTP轨迹核对538次动作、两个终局，
另通过9项缺确认、非法mask、日志缺口、重复ID等边界检查。该工具不连接服务器，不参与训练/推理，
其源码SHA独立记录在审计输出和交付包中，未改变已冻结的模型或评测定义。

## 100次操作的驱动边界仍需核对

原生`actions.rs`在普通操作上限判断之前处理Choose、Cancel、EndTurn；达到上限后，
合法mask仍允许待选项和结束回合。公开`agent.md`则描述服务器达到100次已接受操作时自动
完成必选项并结束该席。因此相同`turn_action_caps=[100]`不足以证明驱动边界完全一致，
本地回环服务也不等于实际服务器的自动收尾实现。

已对预先指定的第一个final八席块逐动作重放：部署策略在计数已达100时出现12次EndTurn，
涉及12个玩家回合；raw为53次EndTurn及1次Cancel，涉及53个玩家回合。没有非法mask动作。
这是同一验收种子的诊断重放，不是新增独立性能样本；不据此估计总体发生率。
证据`native-action-cap-audit.json`。不猜测服务器如何自动选择，后续practice须直接检查
`actions_used`、最后接受的动作和下一回合状态，并保留时控/自动结束记录。

2026-10-28 07:38再次匿名读取[公开协议](https://rizling.top/protocol)进一步确认：客户端100次计数
包含选择、站位、结束回合，第100次后由服务器收尾，不再给客户端额外动作；非法请求不扣次数。
服务器自动选择的具体算法仍未得到公开实现或实测验证。`protocol-boundary-review.json`保留
此次响应状态、时间和内容摘要。本地HTTP测试服务没有模拟该自动收尾边界，验证范围限于已列出的
编码、请求关联、重试、心跳与实际本地整局。不会据此把本地成绩改写成官方比赛成绩。
