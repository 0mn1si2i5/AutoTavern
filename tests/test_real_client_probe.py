"""Read-only probe boundaries; real native OCR is exercised separately on macOS."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools/real_client/cli.py'
spec = importlib.util.spec_from_file_location('real_client_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_client_build_does_not_establish_semantic_compatibility():
    env = {'game_version': '36.6.3.253932'}
    assert probe.compatibility('37.0.255790', env)['status'] == 'version_mismatch'
    same = probe.compatibility(env['game_version'], env)
    assert same['status'] == 'build_matches_semantics_unverified'
    assert not same['inference_ready']
    assert probe.compatibility(None, env)['status'] == 'version_unknown'


def test_archive_write_refuses_overwrite(tmp_path):
    p = tmp_path / 'original.json'
    probe.write_new(p, {'original': True})
    original = p.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_new(p, {'original': False})
    assert p.read_bytes() == original


def test_report_escapes_ocr_text_and_preserves_boxes(tmp_path):
    row = {'frame': 0, 'image_file': 'frame-0000.png', 'text': [
        {'text': '<script>alert(1)</script>', 'confidence': .9,
         'box': {'x': .1, 'y': .2, 'width': .3, 'height': .4}}]}
    probe.render_report(tmp_path, [row], {'status': 'completed'})
    value = (tmp_path / 'index.html').read_text()
    assert '<script>' not in value and '&lt;script&gt;' in value
    assert 'x="100.00" y="200.00" width="300.00" height="400.00"' in value


def test_bad_capture_budget_rejected_before_native_build(monkeypatch):
    monkeypatch.setattr('sys.argv', ['cli.py', 'capture', '--output', '/not-used', '--frames', '0'])
    def no_build():
        pytest.fail('must reject invalid budget before building or accessing screen')
    monkeypatch.setattr(probe, 'build', no_build)
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2


def test_native_failure_is_archived_with_zero_actions(tmp_path, monkeypatch):
    output = tmp_path / 'session'
    monkeypatch.setattr('sys.argv', ['cli.py', 'capture', '--output', str(output), '--delay', '0'])
    monkeypatch.setattr(probe, 'build', lambda: tmp_path/'native-boundary')
    monkeypatch.setattr(probe, 'metadata', lambda _: {'readonly': True})
    monkeypatch.setattr(probe, 'native', lambda *a, **kw: {'screen_recording_permission': True, 'windows': []})
    probe.STOPPED = False
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    result = json.loads((output/'RESULT.json').read_text())
    assert result['status'] == 'failed' and result['actions_sent'] == result['frames'] == 0
    assert 'found0' in result['error']
    assert (output/'index.html').is_file()


def test_ocr_timeout_preserves_captured_pixels(tmp_path, monkeypatch):
    output = tmp_path / 'session'
    monkeypatch.setattr('sys.argv', ['cli.py', 'capture', '--output', str(output), '--delay', '0'])
    monkeypatch.setattr(probe, 'build', lambda: tmp_path/'native-boundary')
    monkeypatch.setattr(probe, 'metadata', lambda _: {'readonly': True})
    def native(_binary, command, *args, **_kwargs):
        if command == 'doctor':
            return {'screen_recording_permission': True, 'windows': [{'window_id': 42}]}
        if command == 'capture':
            Path(args[1]).write_bytes(b'captured bytes')
            return {'text': [], 'width': 100, 'height': 100}
        raise RuntimeError('OCR service timeout')
    monkeypatch.setattr(probe, 'native', native)
    probe.STOPPED = False
    with pytest.raises(SystemExit):
        probe.main()
    assert (output/'frame-0000.png').read_bytes() == b'captured bytes'
    row = json.loads((output/'frame-0000.json').read_text())
    assert row['ocr_status'] == 'failed' and 'timeout' in row['ocr_error']
    assert row['observation_quality'] == 'low_resolution'
    result = json.loads((output/'RESULT.json').read_text())
    assert result['status'] == 'failed' and result['frames'] == 1 and result['actions_sent'] == 0


def test_archived_thumbnail_is_not_treated_as_readable(tmp_path):
    image = tmp_path/'frame-0000.png'
    image.write_bytes(b'pixels')
    row = probe.archive_frame(tmp_path, {'width': 258, 'height': 266, 'text': []}, 0, probe.time.monotonic())
    assert row['observation_quality'] == 'low_resolution'


def lobby_ocr():
    # Independently specified labeled layout: normalized text centers, not pixels.
    items = [('酒馆战棋', .4, .06, .5), ('商店', .23, .65, 1),
             ('收藏', .52, .65, 1), ('开始', .8, .8, 1), ('2355', .9, .97, 1)]
    return {'width': 1920, 'height': 1080, 'ocr_status': 'completed', 'text': [
        {'text': text, 'confidence': confidence,
         'box': {'x': x-.02, 'y': y-.01, 'width': .04, 'height': .02}}
        for text, x, y, confidence in items]}


def test_lobby_fields_do_not_confuse_account_currency_with_match_gold():
    result = probe.interpret_frame(lobby_ocr())
    assert result['phase'] == 'battlegrounds_lobby'
    assert result['game_fields']['gold'] is None
    assert result['game_fields_reason'] == 'not_applicable_in_lobby'
    assert result['ui_labels']['start'][0]['text'] == '开始'
    assert not result['inference_ready']


@pytest.mark.parametrize('defect', ['missing_title', 'low_confidence', 'body_title', 'duplicate_start', 'thumbnail', 'ocr_failed'])
def test_unreliable_or_ambiguous_frame_is_unknown(defect):
    data = lobby_ocr()
    if defect == 'missing_title': data['text'][0]['text'] = 'Hearthstone'
    elif defect == 'low_confidence': data['text'][3]['confidence'] = .3
    elif defect == 'body_title': data['text'][0]['box']['y'] = .75
    elif defect == 'duplicate_start': data['text'].append(data['text'][3].copy())
    elif defect == 'thumbnail': data['width'] = 258
    elif defect == 'ocr_failed': data['ocr_status'] = 'failed'
    result = probe.interpret_frame(data)
    assert result['phase'] == 'unknown'
    assert all(value is None for value in result['game_fields'].values())
