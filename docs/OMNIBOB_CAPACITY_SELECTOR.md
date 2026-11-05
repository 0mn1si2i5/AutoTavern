# 腾位出售对象：最低数值评分单变量对照

默认保持 v0.3，冻结 v0.2 与旧动作链不变。不训练、不参赛、不付费、不自动升级。

## 诊断与候选

只分析上一轮开发集的 483 条出售链，没有按上一轮验收题目选择规则。236 次出售选择的可见数值评分高于场上最低值；这些案例的评分差中位数为 38.55。该评分来自既有 `OmniBobPolicy.minion_value`，包括攻血、等级与若干关键词；不衡量成长引擎、资源生成或体系协同，因此不能把“不是最低分”直接判为错误。

候选仅改变腾位链的出售选择：将合法出售集合缩小至该既有评分的最小值；同分时仍由冻结网络选择。触发条件、每回合一次、剩余操作数限制、打出选择、站位、后续经营、RepeatGuard、权重与游戏规则全部沿用旧动作链。旧文件 `capacity_trade.py` 不修改，新文件为 `capacity_selector.py`。

## 冻结方案

实际目录：`runs/capacity-selector-v2-20261101/`。

- 三组：baseline＝冻结 v0.3；control＝原动作链；candidate＝只换出售评分选择器。
- 四场景：scripted、heuristic、mixed_learned、learned。学习对手固定 v0.3。
- 开发：每组/场景 64 局，共 768，种子 1000600000–1000600007，全部八席。
- 条件验收：每组/场景 256 局，共 3072，种子 2000600000–2000600031，全部八席。
- 开发双门槛：候选对 v0.3 综合名次差 ≤ −0.05，对旧动作链综合差 < 0；两项比较的各场景差均 < +0.15。
- 验收双门槛：候选对 v0.3 综合差 ≤ −0.10，对旧动作链综合差 < 0；两项比较均要求综合 95% 区间上界 < 0，各场景区间上界 < +0.15。
- 八席种子块配对 bootstrap 10000 次，固定种子 20261102，四场景等权、共享重采样索引。
- 缺局、重复派发、技术失败、非法动作、战斗截断使阶段无效。每组/场景自然封顶至多 1%，按 [1,8] 悲观界处理。
- 最多 2 个 CPU 评测进程，每进程 1 线程、RSS ≤ 3 GiB，单局 45 秒；整轮最多 75 分钟，提前 180 秒停止新局留出审计时间。第一批每组 16 局共 48 局计入开发，不重跑。

开发只有每场景 8 个种子块，是资源有界的筛选，不作为独立强度结论。只有完整验收双门槛通过，才允许讨论后续晋级；本轮不自动切默认。

## 启动前失败

旧准备目录 `runs/capacity-selector-20261101/` 保留。种子扫描器误把历史训练 `split=train, seed_offset=600000` 当成本轮开发完整种子 `1000600000` 冲突。冻结在 PROTOCOL 创建之前失败，随后 preflight 在 set-e 下退出，没有启动对局。

最初失败诊断误称实际种子重叠，已另写 `PREPARATION_DIAGNOSIS.json` 纠正，保留原记录。在 v2 新目录改为完整种子与带 split 的偏移核对，候选与预留的开发/验收种子均未变化。

## 本地检查

```sh
cd /Users/ostrovsky/Projects/Zen/omnibob
.venv/bin/python -m pytest -q tests/test_capacity_selector.py tests/test_capacity_trade.py tests/test_end_trade.py tests/test_deployment.py
.venv/bin/python -m omnibob.deployment status
```

12 项检查通过；报告门槛另有 8 项边界检查。实验的来源、协议与脚本分别冻结在 SOURCE_IDENTITY.json、PROTOCOL.json、CODE_FREEZE.json。已有 STARTED.json 后不要再次运行 run.py；完整动作轨迹和干预可见观测保存在逐局 gzip 中。

## 结果

dev 阶段未通过双重门槛；对baseline差-0.027344，95%区间[-0.062500,+0.011719]；对control差-0.031250，95%区间[-0.054688,-0.007812]。768 局完整留档并独立复算。运行器已退出，新目录 STOP 已写入；默认 v0.3、冻结 v0.2、旧动作链和旧实验记录均保留。详情见本轮 REPORT.md / INDEPENDENT_VERIFICATION.json。
