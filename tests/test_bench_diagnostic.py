import json

from omnibob.diagnose_bench import diagnose


def test_real_benchmark_direct_action_coverage_is_explicit(tmp_path):
    summary = diagnose(tmp_path/"diagnostic")
    assert summary["attempted"] == 80 and summary["unsupported"] == 288
    assert summary["errors"] == 0
    assert set(summary["scenarios"]) == {"D2", "D4", "N1", "N2"}
    answers = [json.loads(line) for line in (tmp_path/"diagnostic/answers.jsonl").read_text().splitlines()]
    assert len({r["id"] for r in answers}) == 80
    assert all(r["error"] is None and 0 <= r["action"] < 245 for r in answers)
