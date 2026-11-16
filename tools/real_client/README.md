# OmniBob 实时建议面板

本地冻结模型给建议，人操作游戏。工具只读已有 Power.log，显示当前玩家可见的阵容、手牌、商店、资源和选择。按 **Ctrl + Option + Space** 刷新建议；不会向游戏发送快捷键、点击、拖动或匹配请求。无需保持聊天连接。

## 启动

双击 `tools/real_client/advisor.command`，或在仓库根目录执行：

```sh
.venv/bin/python -m tools.real_client.advisor live --panel
```

首次启动编译本机 Swift 浮动面板，其后使用校验过的缓存。打开炉石后自动发现 `/Applications/Hearthstone/Logs/Hearthstone_*/Power.log`，不激活或控制炉石。未发现日志或客户端未运行时显示等待，不从旧日志继续给建议。

1. 在游戏中进入可操作局面。
2. 按 Ctrl + Option + Space，或点击面板“刷新建议”。
3. 核对面板中的场面，人工执行所示卡牌、位置和目标。
4. 操作后再按快捷键。状态变化会撤下旧建议；没有变化时刷新保留同一建议。
5. 可以在面板标注自己的不同选择及原因。标注不训练模型。

浮动面板不抢游戏焦点。全局快捷键冲突时会提示；在面板内也可以按 Cmd + Enter。关闭面板、终端 Ctrl-C 或创建会话目录内的 `STOP` 都会停止工具。没有自动下一局。

省略 `--panel` 可用浏览器打开终端给出的本机 URL；浏览器模式的键盘快捷键仅在页面获焦时有效。可用 `--log /absolute/path/Power.log` 指定日志、`--catalog /absolute/path/catalog.sqlite` 指定已有卡表。

## 用示范局逐步查看

```sh
.venv/bin/python -m tools.real_client.advisor replay \
  --log runs/raw-client-logs-20261106/Hearthstone_2026_11_05_23_31_31/Power.log \
  --panel
```

“下一步”读取下一次记录动作**之前**的字节前缀，然后生成建议。“上一步”可重看。第一个边界可能仍在等待选英雄动画，不会用服务器提前提供而尚未展示的选项猜测。

导出整份示范的逐操作建议，不运行模拟对局、不计算策略强度：

```sh
.venv/bin/python -m tools.real_client.advisor replay \
  --log runs/raw-client-logs-20261106/Hearthstone_2026_11_05_23_31_31/Power.log \
  --export --output runs/advisor-demo-new
```

输出目录必须不存在。每次会话保存 `SESSION.json`、`MODEL.json`、`STATE.json`、`events.jsonl`，每次刷新保存可见状态、输入数组、缺失字段、建议或失败原因。推理期间状态变化会保存 `DISCARDED.json` 并撤销旧建议。观察到的客户端动作与回执、人工标注分开记录，刷新本身不算执行成功。

## 输入与模型边界

状态来源改为已展示的实体区域和动态标签；旧选项文字中的区域注释不再决定卡牌位置。已修复后期误报空场、选择提前泄露、关键词撤销、血量归一化、英雄选择编码、消费计数混淆和发动/目标等映射。

**当前是冻结模型的实验性实机迁移建议，不是全部输入等价或强度验收。** 对手历史、部分附加效果和引擎计数仍不能完整恢复。未知数值和卡牌槽位逐项记录，仍以零占位输入没有缺失通道的旧模型；面板持续提示这一限制。未知卡牌、关键状态缺失或卡表与日志 build 不匹配时不输出建议。自动下载新卡表不等于模型学会新机制。

目标/分支建议只使用操作前可见状态，不预测战吼后的新局面；实际出现新选择时应再次刷新。面板不会把模型概率解释为“建议正确率”，也不会编造策略理由。

使用默认冻结发行，保持 `runs/releases/current.json` 不变；可用 `--release v0.2` 或 `--release v0.3` 为单次会话选择已保存发行，不改指针、不训练。

## 旧入口和回退

自动代打、单次点击、拖动工具及其启动器已移出当前源码路径。旧源码（含此前未提交修改）、原始记录、冻结权重均保留。

本次迁移备份：`runs/advisor-migration-20261112T072104Z/SOURCE_BEFORE.tar.gz`。`BEFORE.json` 是源码、冻结发行、原始日志的哈希清单；`RETIRED.json` 列出移除路径。需要检查旧实现时解包到一个新目录，不覆盖当前工作区：

```sh
mkdir -p runs/advisor-rollback-inspection
tar -xzf runs/advisor-migration-20261112T072104Z/SOURCE_BEFORE.tar.gz \
  -C runs/advisor-rollback-inspection
```

旧运行记录中的自动操作源码仅作追溯和显式回退，不属于当前建议器依赖。

## 只读辅助工具

原有 `cli.py doctor/capture/inspect` 单窗口采集与 OCR、`catalog.py` 本地卡表、`data_sync.py` 公开数据同步和 `offline_replay` 账本保留。建议面板不依赖截图、不移动鼠标、不调用付费 API、不读取 `.env`。

若无日志，先检查客户端现有日志是否开启；本工具不替你修改游戏配置。若新版本卡牌不受冻结词表支持，面板会保留局面和具体原因，让人自行判断，不以简单规则冒充模型。
