import json

import pytest

from omnibob.evaluate import evaluate
from omnibob.review import build_review


def test_review_reconstructs_real_owner_views_and_rejects_bad_trace(tmp_path):
    run = tmp_path/'game'
    evaluate(run, ['omnibob'], split='dev', seed=6199, games=1, seat=0, trace=True)
    trace = run/'decisions.jsonl'
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    candidate = next(row for row in events if row['visible_sha256'])
    candidate['reason'] = '</script><script>unsafe()</script>'
    trace.write_text(''.join(json.dumps(row)+'\n' for row in events))
    output = tmp_path/'review.html'
    result = build_review(run, output)
    assert result['verified_games'] == 1 and result['visible_decisions'] > 20
    html = output.read_text()
    assert candidate['reason'] not in html
    raw = html.split('<script id="payload" type="application/json">', 1)[1].split('</script>', 1)[0]
    payload = json.loads(raw)
    for frame in payload['frames']:
        assert frame['view']['seat'] == 0
        for opponent in frame['view']['opponents']:
            assert not {'hand','shop','board','gold','secrets'} & opponent.keys()
    candidate['visible_sha256'] = '0'*64
    trace.write_text(''.join(json.dumps(row)+'\n' for row in events))
    with pytest.raises(ValueError, match='observation mismatch'):
        build_review(run, tmp_path/'bad.html')
    assert not (tmp_path/'bad.html').exists()
