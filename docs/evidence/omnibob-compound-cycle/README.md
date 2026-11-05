# 复合循环本地实验的证据

- `PROTOCOL.json` / `POLICY_FREEZE.json`：正式开发前的候选、唯一变化与名次晋级标准。
- `seed-audit.json` / `HOLDOUT_FREEZE.json`：新种子检查、开发门槛和验收前冻结记录。
- `DEV_DECISION.json` / `FINAL_DECISION.json`：两个对手场景的分项指标和完整种子块统计；最终不晋级。
- `RESULTS.json`：4,608 局结果、逐局日志摘要与源码覆盖包指针。
- `reproduction-comparison.json` / `dev-regression-diagnostics.json`：原循环反例与退步案例；均不是独立验收证据。
- `integrity.json`：发行包、practice、旧控制器与新候选的完整性。
- `tests-final.txt` / `tests-final-focused.txt`：113 项全量与 8 项相关复验通过。
- `test-before.txt`：模块尚未实现时的失败测试记录。
- `final-source.json` / `resource-snapshot.json`：源码身份和运行资源记录。

原始逐局日志在 `runs/compound-cycle-20261028/`；没有调用网络接口或启动训练。
继续使用冻结 v0.2；新规则仅通过显式本地实验入口调用。
