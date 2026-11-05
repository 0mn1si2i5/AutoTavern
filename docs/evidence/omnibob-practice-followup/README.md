# 首场 practice 后本地实验的证据索引

- `PROTOCOL.json`：开发结果前写下的样本、候选及晋级标准。
- `HOLDOUT_FREEZE.json`：新验收种子启动前的候选源码摘要和开发结果。
- `RESULTS.json`：1,792 局正式对照的指标、逐局文件摘要和源码覆盖包指针；阴性结论保留。
- `practice-analysis.json`：原比赛 482 次策略重放、逐回合资源、移位循环和首次候选分歧。
- `integrity.json`：原发行包、比赛记录与冻结候选源码的完整性核对。
- `tests-final.txt`：本轮 105 项 Python 测试结果。
- `final-source.json`：本轮最终源码覆盖包、Git 基点、环境和卡表/动作/观测指纹。

逐局原始数据和独立超时失败探针保留在 `runs/practice-followup-20261028/`；未删除失败。
原 practice 位于 `runs/practice-20261028-103720/`，未修改。
旧独立审计器的真实响应兼容缺口由该目录中的 `audit-live.py`/`audit-live.json` 记录；
本轮只修正当前结构化日志的终局元数据保留，没有改写冻结审计器。

没有读取凭据、访问比赛队列、重新参赛、启动训练或调用付费服务。
