# OmniBob 真实客户端只读探针

这是实机开发的第一个可运行入口：检查已安装客户端和权限、仅截取炉石窗口、用Apple Vision本地OCR生成带框查看页，并输出带证据的大厅阶段判断。它还不能生成ObservationV2、推理游戏动作或操作鼠标。

不安装Python依赖，不修改系统配置。首次运行通过本机Swift编译器构建原生工具，缓存和所有记录在被Git忽略的`runs/`目录。使用macOS14及以上的ScreenCaptureKit与Vision公开API；当前实测主机为macOS27.2/arm64。

## 命令

从仓库根目录执行：

```sh
cd /Users/ostrovsky/Projects/Zen/omnibob
.venv/bin/python tools/real_client/cli.py doctor
```

打开炉石并停在待检查页面。无需切换前台或移动窗口；不会接管鼠标键盘、点击按钮或开始匹配：

```sh
.venv/bin/python tools/real_client/cli.py capture \
  --frames 3 --seconds 60 --interval 2 --delay 0 \
  --output runs/real-client-capture-001
```

如果有多个炉石窗口，从`doctor`输出选择`--window-id`。只接受真实炉石bundle的可见窗口，使用独立窗口捕获；不激活窗口、不移动鼠标，不回退到全桌面截图。输出目录必须不存在，防止覆盖。

对已有PNG运行离线OCR，不需要屏幕录制权限：

```sh
.venv/bin/python tools/real_client/cli.py inspect /absolute/path/frame.png \
  --timeout 60 --output runs/real-client-inspect-001
```

产物：`SESSION.json`、`frame-*.png/json`、`RESULT.json`和`index.html`。打开`index.html`查看截图与OCR框；置信度仅代表OCR模型的文字置信度，不等于游戏状态正确率。

## 停止与失败

- 每次最多120帧、60秒，默认1帧；时间预算包括切回窗口的等待，不包括首次编译。
- `Ctrl-C`、SIGTERM或在输出目录创建`STOP`阻止下一帧。正在进行的原生只读调用有超时上限，不会发送操作事件。
- 没有窗口、权限缺失、OCR故障或超时会留下失败报告并退出，不自动重试。
- 需要屏幕录制权限时，在macOS系统设置的隐私与安全性中允许启动本命令的应用，再使用新输出目录重试。工具不会自行授予权限。
- 本机首轮Vision默认OCR和首次CPU配置调用均超过20秒；堆栈显示模型初始化/编译。本机一次60秒有界初始化在42.44秒完成，之后同一测试图约0.49秒。当前显式请求CPU计算设备，但这不代表已测得所有系统模型编译都只用CPU。原始失败和堆栈保存在`runs/real-client-validation-20261105/`。
- 离线OCR默认允许最多60秒冷启动；实时采集仍受整次最多60秒的预算约束。截图与OCR分别执行，识别失败时保留原图和失败原因。

## 本次真实验证

7项边界测试通过；真实命令的STOP验证在采集前停止，0帧、0操作。已捕获到真实炉石菜单的系统缩略图（258×266像素），约0.72秒完成单帧截图和OCR，但文字不可用，报告标记`low_resolution_frames: 1`。这证明单窗口管线可运行，不代表完整游戏画面识别通过。

使用时让炉石保持展开可见即可，不需要抢占鼠标键盘。工具不会自行展开、移动或激活窗口。若系统只提供缩略图，原图仍会留档并标记低分辨率；`completed`仅表示采集流程完成。完整记录见`runs/real-client-validation-20261105/VALIDATION.json`，原失败均保留。

## 真实缺口

本机炉石build37.0.255790，模型规则36.6.3.253932；`doctor`如实报告版本不匹配，即使以后build相同也不会仅凭版本号认定语义兼容。具体输入缺口见[INPUT_COVERAGE.md](INPUT_COVERAGE.md)。

官方客户端实战、完整状态识别、合法动作重建、鼠标执行、当前版本卡表适配都未验收。OCR测试图只能证明文字管线，不能冒充真实对局识别。

API依据：[ScreenCaptureKit单窗口截图](https://developer.apple.com/documentation/ScreenCaptureKit/SCScreenshotManager/captureImage%28contentFilter%3Aconfiguration%3AcompletionHandler%3A%29)、[Vision文字识别](https://developer.apple.com/documentation/vision/vnrecognizetextrequest)。

## 大厅识别（2026-11-05增量）

在同一张真实中文大厅截图上修正OCR语言配置：中文优先并启用自动语言检测，原先误读为拉丁字母的“酒馆战棋、商店、收藏、开始”已正确识别。原图与旧识别均保留。新中文模型首次调用约22秒，后续3帧实际截图及识别分别约1.41、0.87、0.89秒。

现有`capture`和`inspect`命令自动生成`interpretation`：阶段、引用的OCR行号、各菜单文字位置，以及明确为空的局内字段。大厅判断需要顶部标题及三个菜单标签同时满足条件；低分辨率、关键标签缺失/歧义或低置信度返回`unknown`。金币等字段不会从大厅账号货币推断。

连续3张真实新画面均判断为`battlegrounds_lobby`，14项边界测试通过。仅验证当前中文大厅布局；其他语言和布局尚未实机验证。文字框不是已验证的按钮点击区域，当前没有执行器。记录：`runs/real-client-validation-20261105/LOBBY_VALIDATION.json`；查看页：`runs/real-client-validation-20261105/lobby-live/index.html`。
