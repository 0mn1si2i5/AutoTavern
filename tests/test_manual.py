import pytest

from omnibob.manual import events, initialize, record, report


def test_manual_journal_preserves_visible_evidence_and_execution_gaps(tmp_path):
    source = tmp_path/"visible.txt"
    source.write_text("my board, my shop; no opponent hidden state")
    directory = tmp_path/"game"
    initialize(directory, "test-only", "local demonstration", "example advice")
    record(directory, "suggestion", turn=1, evidence=source, advice="buy left minion", reason="tempo")
    source.write_text("changed")
    assert "my board" in (directory/"visible-00001.txt").read_text()
    with pytest.raises(ValueError, match="why actual"):
        record(directory, "execution", decision=1, actual="refresh", deviation="")
    record(directory, "execution", decision=1, actual="refresh", deviation="human chose differently")
    with pytest.raises(ValueError, match="unexecuted"):
        record(directory, "execution", decision=1, actual="refresh", deviation="duplicate")
    record(directory, "suggestion", turn=2, evidence=source, advice="end turn", reason="no gold")
    record(directory, "result", placement=4, notes="test only")
    summary = report(directory)
    assert summary["missing_execution_records"] == [2]
    assert summary["deviations"] == 1 and summary["result"]["placement"] == 4
    assert len(events(directory)) == 4
    with pytest.raises(ValueError, match="finished"):
        record(directory, "result", placement=1, notes="cannot rewrite")
