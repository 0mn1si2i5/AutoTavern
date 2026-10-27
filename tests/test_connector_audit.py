import json
from types import SimpleNamespace

import pytest

from omnibob.connector_audit import instrument, load_reviewed_connector


def test_audit_links_observation_action_ack_and_result_without_credentials(tmp_path):
    secret = 'test-only-agent-secret'
    class Connector:
        def __init__(self, key): self.key = key
        def request(self, method, path, body=None, authenticated=True):
            assert self.key == secret
            if path.startswith('/api/v1/agent/state'):
                return dict(type='decision', decision_id='d1', seat_revision=9, observation={'features_b64':'AAAA'},
                            player_view={'me':{'gold':3}}, authorization=secret)
            return dict(ok=True, seat_revision=10, message=secret, debug_headers={'Authorization':secret})
    module = instrument(SimpleNamespace(ArenaClient=Connector), tmp_path/'audit')
    client = module.ArenaClient(secret)
    observed = client.request('GET','/api/v1/agent/state?after='+secret)
    assert observed['authorization'] == secret  # Transport behavior is preserved; journal is allowlisted.
    client.request('POST','/api/v1/agent/actions',dict(decision_id='d1',seat_revision=9,action=8,request_id='r1',key=secret))
    text = (tmp_path/'audit/requests.jsonl').read_text()
    assert secret not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[1]['operation'] == rows[0]['sequence']
    assert rows[1]['response']['decision_id'] == rows[2]['body']['decision_id']
    assert rows[3]['operation'] == rows[2]['sequence'] and rows[3]['response']['ok']
    assert rows[1]['response']['player_view']['me']['gold'] == 3
    with pytest.raises(FileExistsError): instrument(SimpleNamespace(ArenaClient=Connector),tmp_path/'audit')


def test_audit_preserves_failure_without_echoing_exception_secrets(tmp_path):
    class Connector:
        def request(self,*args): raise ConnectionError('sensitive test token')
    module = instrument(SimpleNamespace(ArenaClient=Connector),tmp_path/'audit')
    with pytest.raises(ConnectionError):module.ArenaClient().request('GET','/api/v1/agent/state')
    text=(tmp_path/'audit/requests.jsonl').read_text()
    assert 'sensitive test token' not in text and 'ConnectionError' in text
    unreviewed=tmp_path/'client.py';unreviewed.write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError,match='checksum'):load_reviewed_connector(unreviewed)
