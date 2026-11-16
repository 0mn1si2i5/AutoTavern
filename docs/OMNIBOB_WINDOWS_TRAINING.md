# OmniBob 在 Windows 上长时间训练

这份说明写给在 Windows 电脑上执行训练的人，包括只负责执行的 Codex。训练从 v0.3 发行权重开始，沿用 v0.4 的训练参数，目标 19,000 次更新。所有命令都在仓库根目录执行，脚本在 `tools\windows\`。

## 一、这次训练的内容

- 起点：`runs\releases\omnibob-v0.3-20261101\policy\policy.pt`（v0.3 发行权重）。
- 训练参数：`configs\long-stage-v03.json`。与 v0.4 的三个训练种子相同：32 局并行，每次更新 512 步，学习率 3e-5，参考 KL 系数 0.03，混合对手，按轮打乱行动顺序。只换了种子（161000000），PyTorch 线程数设为 2。
- 规则：`configs\arena-v2.json`，与 v0.3 训练时相同。
- 长度：19,000 次更新，每次 16,384 个决策，共约 3.1 亿个决策。
- 速度参考：Mac M3 Pro 上每次更新 35 到 40 秒，按这个速度要 8 到 9 天。Windows 上的实际速度和剩余天数看状态输出。
- CPU 占用：一个训练进程大约只用 1 个核。对局模拟大部分是 Python 代码，多开对局线程不会更快（Mac 上实测 4 个线程与 1 个线程速度相同、结果相同），所以配置里保持 1 个。
- 这次不改模型结构、规则和奖励，只把训练时间拉长。参考 KL 约束和较低的学习率可能限制长训练的收益，这次先不动，让训练长度是唯一变化的条件。

## 二、安装（只做一次）

1. 安装 Git、uv（`winget install astral-sh.uv`）和 Visual Studio 2022 Build Tools，后者勾选“使用 C++ 的桌面开发”。Rust 编译 TavernGym 扩展需要它。
2. 取代码，放在不含中文和空格的路径下，例如 `D:\omnibob`：

   ```bat
   git clone https://github.com/0mn1si2i5/AutoTavern.git D:\omnibob
   cd /d D:\omnibob
   git checkout omnibob-v0.2-local
   ```

   仓库的 `.gitattributes` 规定源码用 LF 换行，`.cmd` 用 CRLF，与 `core.autocrlf` 设置无关。源码换行不对会导致引擎指纹和模型哈希对不上，训练会拒绝启动。

3. 运行 `tools\windows\setup.cmd`。它把 Python 3.12、Rust 1.98.1 和依赖都装在仓库里的 `.venv` 和 `.local-tools`，然后编译扩展（几分钟），最后运行 `omnibob check`，输出一段 JSON 即为成功。
4. 放入 v0.3 权重包。权重不在 git 里，从 Mac 拷贝 `runs/releases/omnibob-v0.3-20261101.tar.gz`（6.8 MB）到 Windows 仓库的 `runs\releases\`，核对哈希后解压：

   ```bat
   mkdir runs\releases
   certutil -hashfile runs\releases\omnibob-v0.3-20261101.tar.gz SHA256
   tar -xzf runs\releases\omnibob-v0.3-20261101.tar.gz -C runs\releases
   ```

   哈希应为 `305763ab2608805ef3ebbb1b456e7edb93399e56ab9c518021427d8d9b5eafe8`。解压后应有 `runs\releases\omnibob-v0.3-20261101\policy\policy.pt`。

5. 运行 `tools\windows\check.cmd`。这是训练、续跑、暂停、低功耗和运行锁的测试，应全部通过；其中模拟断电的一项只在 macOS 和 Linux 上运行，在 Windows 上显示为跳过。

## 三、开始训练

1. 运行 `tools\windows\train-init.cmd`，只做一次。它建立 `runs\long-stage-v03`，复制训练参数和规则，记下起点权重的 sha256，并检查权重与规则、模型结构是否匹配。目录已存在时会拒绝。
2. 运行 `tools\windows\train-run.cmd`。窗口开着训练就在进行。进程是“低于正常”优先级，前台程序要用 CPU 时系统先给前台。
3. 窗口只在每段开始和结束时打印一行。每次更新的输出在 `runs\long-stage-v03\segments\0001.log`。

## 四、日常操作

都可以在训练运行时双击，几秒内生效：

- `train-status.cmd`：运行状态、进度、最近 20 次更新的速度、最近 200 局平均名次、按当前速度的剩余天数、最新检查点保存于几分钟前。
- `train-low.cmd`：低功耗。PyTorch 只用 1 个线程，并且计算一段就休息一段，默认只有 35% 的时间在计算，CPU 占用约为正常时的三分之一，训练相应变慢。
- `train-normal.cmd`：回到正常速度。
- `train-pause.cmd`：先保存检查点，然后停住，不占 CPU。进程和窗口还在。用 `train-normal.cmd` 或 `train-low.cmd` 恢复。
- `train-stop.cmd`：保存检查点后退出。之后再运行 `train-run.cmd` 即从断点继续。

需要调整时在命令行执行，例如把低功耗时的计算比例改为 20%，或把正常模式的 PyTorch 线程数改为 4：

```bat
.venv\Scripts\python.exe -m omnibob.campaign mode runs\long-stage-v03 low --low-duty 0.2
.venv\Scripts\python.exe -m omnibob.campaign mode runs\long-stage-v03 normal --threads 4
```

计算比例可设 0.05 到 1。也可以用 `train-run.cmd --priority idle` 启动，让训练只用系统空闲的 CPU。

一般用法：玩游戏或做重活时 pause 或 low，不用电脑时 normal。

## 五、中断和续跑

- 检查点：`latest.pt` 每 10 次更新覆盖一次（约 6 分钟），每 250 次更新另存一份 `update-NNNNNN.pt`。每份约 17 MB，全程另存约 76 份，共 1.3 GB 左右。
- 在训练窗口按 Ctrl+C：训练进程保存检查点后退出。cmd 可能接着问“终止批处理操作吗(Y/N)”，选哪个都可以。
- 直接关窗口、重启、断电、蓝屏：来不及保存，最多损失最近 10 次更新。重新运行 `train-run.cmd` 即从最新检查点继续，并新开一段（`segments\0002`）。
- 每段最长 24 小时，到时自动保存并开始下一段，不需要人处理。
- 训练运行时 Windows 不会因为空闲进入睡眠，显示器仍会按设置关闭。手动让电脑睡眠只会让训练暂停，唤醒后继续。pause 模式下也不会自动睡眠；想让电脑睡眠，先 stop。
- 同一个训练目录同时只能有一个 `train-run`，第二个会提示后退出。
- 进程内存超过 4 GiB 会保存并开新段（Mac 上实测峰值约 1.7 GB）。
- 训练报错、某段没写出 `summary.json` 就退出、或连续 3 段没有进展时，`train-run` 停下并打印日志位置。这时不要删除或改动任何文件，把 `train-status.cmd` 的输出和最后一段日志的末尾发回来。
- 训练完成前不要更新代码（不要 `git pull` 或切换分支）。续跑时会核对引擎指纹和模型实现哈希，代码变了会拒绝续跑。

## 六、Codex 在这台机器上的职责

只执行本文的步骤，不改代码和参数：

- 不修改 `omnibob\`、`crates\`、`configs\`、`tests\`、`tools\` 下的文件，不改训练参数，不重新 init 已有的训练目录。
- 不删除、不移动 `runs\long-stage-v03` 里的任何文件，包括旧的检查点和日志。
- 安装失败、测试失败、训练报错时停下，报告执行的命令、完整输出和状态输出，不自行修补。
- 不运行评测，不参赛，不连接游戏客户端，不读取 `.env`。这些需要另行授权。
- 切换 normal、low、pause、stop 只按用户的要求做。

## 七、训练完成以后

- 完成时 `runs\long-stage-v03\DONE.json` 记录最终检查点路径和 sha256，`train-run` 打印“训练已完成”。之后再运行也只是打印这一句。
- 训练中显示的平均名次是对训练分布内对手的成绩，不能用来判断是否比 v0.3 强。
- 事先定好的判断方法：在 Mac 上把最终检查点与 v0.3 做配对评测，使用 dev 和 final 种子，并加入训练时没有见过的对手；平均名次至少好 0.10，才算这次长训练有效。评测需要另行授权，不在本次执行范围内。
- 拷回 Mac 的文件：`DONE.json` 指向的检查点，`campaign.json`、`events.jsonl`，以及每段的 `manifest.json`、`summary.json`、`metrics.jsonl`。

## 八、WSL2（备选，不推荐）

也可以在 WSL2 的 Ubuntu 里按 Linux 方式运行（`bash tools/setup-omnibob.sh`，命令改为 `.venv/bin/python -m omnibob.campaign ...`）。区别：

- 低优先级只在 WSL 虚拟机内部有效，不会让位给 Windows 上的程序。要限制占用，需要在 `%UserProfile%\.wslconfig` 里设置 `processors=` 和 `memory=`，并且重启 WSL 才生效。
- 仓库要放在 WSL 自己的文件系统里（例如 `~/omnibob`），不要放在 `/mnt/c`，后者读写很慢。
- 关闭所有 WSL 窗口后，虚拟机可能被自动关掉，训练随之中断。

所以默认用 Windows 原生方式。
