"""Mac-local, read-only Hearthstone capture and OCR explorer (no model actions)."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import plistlib
import signal
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP_INFO = Path('/Applications/Hearthstone/Hearthstone.app/Contents/Info.plist')
MODEL_MANIFEST = ROOT / 'runs/candidates/omnibob-v0.4-20261102/policy/manifest.json'
SOURCE = HERE / 'ScreenProbe.swift'
STOPPED = False


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def build():
    if platform.system() != 'Darwin':
        raise RuntimeError('This screen probe requires macOS14 or newer')
    version = subprocess.run(['xcrun', 'swiftc', '--version'], capture_output=True, text=True, check=True, timeout=15).stdout
    key = hashlib.sha256((sha(SOURCE) + version + platform.machine()).encode()).hexdigest()[:20]
    out = ROOT / 'runs/real-client-tools' / key
    out.mkdir(parents=True, exist_ok=True)
    binary = out / 'screen-probe'
    receipt = out / 'BUILD.json'
    if binary.exists() and receipt.exists():
        if json.loads(receipt.read_text())['binary_sha256'] == sha(binary):
            return binary
        raise RuntimeError('Cached helper changed; preserve it and inspect BUILD.json')
    command = ['xcrun', 'swiftc', '-parse-as-library', '-O', str(SOURCE), '-o', str(binary),
               '-module-cache-path', str(ROOT / 'runs/real-client-tools/module-cache')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (out / 'build.stdout').write_text(result.stdout)
    (out / 'build.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Native build failed; see {out / "build.stderr"}')
    write_new(receipt, {'source_sha256': sha(SOURCE), 'binary_sha256': sha(binary), 'compiler': version,
                        'utc': now(), 'command': command})
    return binary


def native(binary, *args, timeout=20):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f'Native helper returned invalid JSON: {result.stderr[:500]}') from error
    if result.returncode:
        raise RuntimeError(data.get('error', f'native exit{result.returncode}'))
    return data


def compatibility(client_build, model_environment):
    expected = model_environment.get('game_version')
    if not client_build or not expected:
        state = 'version_unknown'
    elif client_build != expected:
        state = 'version_mismatch'
    else:
        state = 'build_matches_semantics_unverified'
    return {'client_build': client_build, 'model_game_version': expected, 'status': state,
            'model_input_shape': {'features': 4017, 'cards': 447, 'action_mask': 245},
            'inference_ready': False,
            'missing': ['visible-state feature encoder', 'card identity and effect mapping',
                        'real-client legal-action reconstruction', 'current rules semantic validation'],
            'note': 'OCR words are evidence, not an ObservationV2. No missing feature is filled with zero.'}


def metadata(binary):
    app = plistlib.loads(APP_INFO.read_bytes()) if APP_INFO.exists() else {}
    model = json.loads(MODEL_MANIFEST.read_text()) if MODEL_MANIFEST.exists() else {}
    return {'utc': now(), 'system': platform.platform(), 'machine': platform.machine(),
            'app_info_path': str(APP_INFO), 'installed': bool(app),
            'client_bundle_id': app.get('CFBundleIdentifier'),
            'model_manifest': str(MODEL_MANIFEST),
            'compatibility': compatibility(app.get('CFBundleVersion'), model.get('environment', {})),
            'helper_binary': str(binary), 'helper_sha256': sha(binary),
            'python_source_sha256': sha(Path(__file__)), 'swift_source_sha256': sha(SOURCE),
            'readonly': True, 'actions_sent': 0}


def interpret_frame(data):
    """Small, evidence-backed lobby recognizer; never constructs a game observation.

    Regions only disambiguate recognized labels. They are not click targets or
    fallback guesses when OCR fails. Other phases intentionally stay unknown.
    """
    result = {'phase': 'unknown', 'evidence': [], 'ui_labels': {},
              'game_fields': {k: None for k in ('gold', 'tavern_tier', 'hand_count', 'board_count')},
              'game_fields_reason': 'not_observed', 'inference_ready': False,
              'recognizer_version': 'lobby-1'}
    if data.get('width', 0) < 640 or data.get('height', 0) < 400:
        result['reason'] = 'low_resolution'
        return result
    if data.get('ocr_status') != 'completed':
        result['reason'] = 'ocr_not_completed'
        return result
    labels = {
        'title': ({'酒馆战棋', '酒館戰棋', 'battlegrounds'}, .5, (0, 0, 1, .2)),
        'shop': ({'商店', 'shop'}, .8, (0, .4, .65, .85)),
        'collection': ({'收藏', 'collection'}, .8, (.2, .4, .7, .85)),
        'start': ({'开始', '開始', 'play'}, .8, (.65, .5, 1, .95)),
        'season_rewards': ({'赛季奖励', '賽季獎勵', 'seasonrewards'}, .5, (.65, .25, 1, .75)),
    }
    for index, item in enumerate(data.get('text', [])):
        text = ''.join(unicodedata.normalize('NFKC', item['text']).casefold().split())
        box = item['box']; x = box['x'] + box['width']/2; y = box['y'] + box['height']/2
        for name, (aliases, confidence, region) in labels.items():
            if (text in aliases and item['confidence'] >= confidence and
                    region[0] <= x <= region[2] and region[1] <= y <= region[3]):
                result['ui_labels'].setdefault(name, []).append({
                    'ocr_index': index, 'text': item['text'], 'box': box, 'confidence': item['confidence']})
    found = result['ui_labels']
    # All key labels must be unambiguous. A "start" label alone is insufficient.
    required = ('title', 'shop', 'collection', 'start')
    if all(len(found.get(name, [])) == 1 for name in required):
        result.update(phase='battlegrounds_lobby', reason='title_and_three_menu_labels',
                      evidence=[found[name][0]['ocr_index'] for name in required],
                      game_fields_reason='not_applicable_in_lobby')
    else:
        result['reason'] = 'missing_or_ambiguous_lobby_labels'
    return result


def render_report(directory, rows, result):
    cards = []
    for row in rows:
        boxes = []
        texts = []
        for item in row['text']:
            b = item['box']
            boxes.append(f'<rect x="{b["x"]*1000:.2f}" y="{b["y"]*1000:.2f}" '
                         f'width="{b["width"]*1000:.2f}" height="{b["height"]*1000:.2f}"/>')
            texts.append(f'<li>{html.escape(item["text"])} <small>{item["confidence"]:.2f}</small></li>')
        image = html.escape(row['image_file'], quote=True)
        interpreted = row.get('interpretation', {})
        semantic = html.escape(json.dumps(interpreted, ensure_ascii=False, indent=2))
        cards.append(f'<section><h2>Frame {row["frame"]}: {html.escape(interpreted.get("phase", "unknown"))}</h2><div class="frame"><img src="{image}">'
                     f'<svg viewBox="0 0 1000 1000" preserveAspectRatio="none">{"".join(boxes)}</svg>'
                     f'</div><ul>{"".join(texts)}</ul><details><summary>阶段判断依据与文字位置</summary><pre>{semantic}</pre></details></section>')
    content = ('<!doctype html><meta charset="utf-8"><title>OmniBob screen evidence</title>'
               '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#181b22;color:#eee}'
               '.frame{position:relative}.frame img{width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%}'
               'rect{fill:none;stroke:#70ffc1;stroke-width:1.5}small{color:#aaa}pre{white-space:pre-wrap}</style>'
               '<h1>OmniBob 只读画面记录</h1><p>绿色框是OCR文字位置。阶段判断仅支持大厅；文字框不是已验证的可点击区域。没有生成模型输入或发送游戏操作。</p>'
               f'<pre>{html.escape(json.dumps(result, ensure_ascii=False, indent=2))}</pre>' + ''.join(cards))
    (directory / 'index.html').write_text(content)


def archive_frame(directory, data, frame, started, input_image=None):
    image = directory / f'frame-{frame:04d}.png'
    if input_image is not None:
        with image.open('xb') as out:
            out.write(input_image.read_bytes())
    row = {**data, 'frame': frame, 'utc': now(), 'elapsed_seconds': time.monotonic()-started,
           'image_file': image.name, 'image_sha256': sha(image)}
    # A window thumbnail is useful capture evidence but unsuitable for parsing
    # card text. Passing this size check alone does not establish readability.
    row['observation_quality'] = (
        'low_resolution' if data.get('width', 0) < 640 or data.get('height', 0) < 400
        else 'requires_semantic_validation')
    row['interpretation'] = interpret_frame(row)
    # Use only relative archive names; metadata keeps capture geometry if available.
    row.pop('image', None)
    write_new(directory / f'frame-{frame:04d}.json', row)
    return row


def main():
    global STOPPED
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='inspect installed client, permissions and model compatibility')
    doctor.add_argument('--output', type=Path)
    capture = sub.add_parser('capture', help='capture only a Hearthstone window without changing focus; no input events')
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--window-id', type=int)
    capture.add_argument('--frames', type=int, default=1)
    capture.add_argument('--seconds', type=float, default=30)
    capture.add_argument('--interval', type=float, default=1)
    capture.add_argument('--delay', type=float, default=0, help='optional initial delay; does not activate or move any window')
    inspect = sub.add_parser('inspect', help='OCR an existing PNG locally without screen permission')
    inspect.add_argument('image', type=Path)
    inspect.add_argument('--output', type=Path, required=True)
    inspect.add_argument('--timeout', type=float, default=60, help='bounded OCR call; first model initialization may take tens of seconds')
    args = parser.parse_args()
    if args.command == 'capture' and not (1 <= args.frames <= 120 and 1 <= args.seconds <= 60 and .1 <= args.interval <= 30):
        parser.error('capture bounds: frames1..120, seconds1..60, interval0.1..30')
    if args.command == 'capture' and not (0 <= args.delay <= 15 and args.delay < args.seconds):
        parser.error('delay must be0..15 and smaller than seconds budget')
    if args.command == 'inspect' and (not args.image.is_file() or args.image.suffix.lower() != '.png'):
        parser.error('inspect requires an existing PNG')
    if args.command == 'inspect' and not 1 <= args.timeout <= 60:
        parser.error('OCR timeout must be1..60 seconds')
    if args.output is not None and args.output.exists():
        parser.error('output already exists; choose a new directory (archives are never overwritten)')
    try:
        binary = build()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f'{error}\n')
    if args.command == 'doctor':
        report = metadata(binary)
        try: report['screen'] = native(binary, 'doctor')
        except (OSError, RuntimeError, subprocess.SubprocessError) as error: report['screen_error'] = str(error)
        if args.output:
            args.output.mkdir(parents=True); write_new(args.output/'doctor.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2)); return
    out = args.output; out.mkdir(parents=True)
    write_new(out/'SESSION.json', {**metadata(binary), 'command': args.command,
                                  'requested_frames': getattr(args, 'frames', 1),
                                  'seconds_budget': getattr(args, 'seconds', getattr(args, 'timeout', 20))})
    def stop(*_):
        global STOPPED
        STOPPED = True
    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    started = time.monotonic(); rows = []; error = None; reason = 'completed'
    try:
        if args.command == 'inspect':
            tick = time.monotonic()
            data = native(binary, 'ocr', args.image.resolve(), timeout=args.timeout)
            data['ocr_status'] = 'completed'
            rows.append(archive_frame(out, data, 0, tick, args.image))
        else:
            deadline = started+args.seconds
            until = started+args.delay
            if args.delay:
                print(f'{args.delay:g}秒后采集炉石窗口，无需切换前台；Ctrl-C或输出目录STOP文件可停止。', flush=True)
            while time.monotonic() < until and not STOPPED and not (out/'STOP').exists():
                time.sleep(min(.05, max(0, until-time.monotonic())))
            if STOPPED or (out/'STOP').exists():
                reason = 'stopped'
                raise InterruptedError('capture stopped before screen preflight')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = 'deadline'
                raise InterruptedError('budget exhausted before screen preflight')
            doctor = native(binary, 'doctor', timeout=min(20, remaining))
            write_new(out/'SCREEN_PREFLIGHT.json', doctor)
            if not doctor.get('screen_recording_permission'):
                raise RuntimeError('screen_recording_permission_required: enable the launching app in macOS Privacy & Security > Screen & System Audio Recording, then retry in a new output directory')
            windows = doctor.get('windows', [])
            if args.window_id is None:
                if len(windows) != 1:
                    raise RuntimeError(f'expected one visible game window, found{len(windows)}; use doctor and --window-id')
                window_id = windows[0]['window_id']
            else:
                if args.window_id not in [w['window_id'] for w in windows]:
                    raise RuntimeError('selected window is not a visible Hearthstone window')
                window_id = args.window_id
            for i in range(args.frames):
                if STOPPED or (out/'STOP').exists(): reason = 'stopped'; break
                remaining = deadline-time.monotonic()
                if remaining <= 0: reason = 'deadline'; break
                tick = time.monotonic()
                data = native(binary, 'capture', window_id, (out/f'frame-{i:04d}.png').resolve(), timeout=min(20, remaining))
                # Preserve capture independently if OCR fails or exhausts the budget.
                try:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError('capture budget exhausted before OCR')
                    ocr = native(binary, 'ocr', (out/f'frame-{i:04d}.png').resolve(), timeout=min(60, remaining))
                    data.update(text=ocr['text'], ocr_status='completed', ocr_compute_requested=ocr.get('ocr_compute_requested'))
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    data.update(ocr_status='failed', ocr_error=str(exc))
                    rows.append(archive_frame(out, data, i, tick))
                    raise
                rows.append(archive_frame(out, data, i, tick))
                if len(rows) < args.frames:
                    until = min(deadline, tick+args.interval)
                    while time.monotonic() < until and not STOPPED and not (out/'STOP').exists():
                        time.sleep(min(.05, max(0, until-time.monotonic())))
    except InterruptedError:
        if reason != 'deadline': reason = 'stopped'
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        error = str(exc); reason = 'failed'
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    result = {'status': reason, 'frames': len(rows), 'seconds': time.monotonic()-started,
              'error': error, 'actions_sent': 0, 'inference_ready': False,
              'low_resolution_frames': sum(r['observation_quality'] == 'low_resolution' for r in rows),
              'phases': [r['interpretation']['phase'] for r in rows],
              'scope': 'Window pixels, OCR and lobby-only phase interpretation. No model observation or policy execution.'}
    write_new(out/'RESULT.json', result); render_report(out, rows, result)
    print(json.dumps({**result, 'report': str((out/'index.html').resolve())}, ensure_ascii=False, indent=2))
    if error: raise SystemExit(2)


if __name__ == '__main__':
    main()
