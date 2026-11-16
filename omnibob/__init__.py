"""OmniBob: local, player-visible Battlegrounds research baselines.

The package is flat. Its modules fall into three groups.

Training and inference: train, campaign, training_env, run_control, compat, model,
benchmark, evaluate, learned, move_guard, observation, observation_v2, opponents,
policies, repeat_guard, runtime_state, training_repeat, plus the CLI (__main__) and
deployment. Resuming a run checks the SHA-256 of model.py, of policies.py for learned
opponents, and the engine contract.

Tools used by the current guides: export, average, imitation, offline_check, review,
compare, arena, connector_audit, manual, diagnose_bench.

Finished single experiments, imported by neither group above; their records are
indexed in docs/experiments/README.md: bounded_report, parallel_report, budget_report,
continuation_report, guarded_report, opponent_report, compound_report,
end_trade_report, guard_audit, economy_audit, learning_diagnosis,
learning_microdiagnosis, learning_signal, practice_review, cancel_memory,
reorder_exit, experiment_reorder, end_trade, capacity_trade, capacity_selector.
Several of them compare their own SHA-256 with frozen protocols, so they are not moved.
"""

__version__ = "0.2.0"
