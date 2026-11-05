"""Check a deployment against recorded own observations, without server access.

These are observations from the recorded policy's trajectory. Replaying them
checks adapter equivalence and legality, never counterfactual placement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .learned import load_policy
from .observation_v2 import ObservationV2
from .repeat_guard import clone_runtime
from .runtime_state import save_runtime_state
from .train import load_checkpoint


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(checkpoint, recording, output):
    checkpoint, recording, output = map(Path, (checkpoint, recording, output))
    original_sha, policy_sha = sha256(recording), sha256(checkpoint)
    rows = [json.loads(line) for line in recording.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('recording contains no own observations')
    output.mkdir(parents=True, exist_ok=False)
    result = {'offline_only': True, 'queued': False, 'status': 'checking',
              'recording_sha256': original_sha, 'policy_sha256': policy_sha,
              'recorded_own_frames': len(rows), 'checked_frames': 0, 'action_mismatches': [],
              'illegal_actions': 0, 'independent_context_state_verified': False,
              'limitation': 'Recorded trajectory: adapter equivalence and legality only; '
                            'not candidate placement, counterfactual gameplay or server validation.'}
    # Do not inherit journaling destinations, and restore caller-local adapter state.
    old_environment = {key: os.environ.get(key) for key in ('OMNIBOB_CHECKPOINT', 'OMNIBOB_JOURNAL_DIR')}
    arena, old_definition = None, None
    try:
        os.environ['OMNIBOB_CHECKPOINT'] = str(checkpoint.resolve())
        os.environ.pop('OMNIBOB_JOURNAL_DIR', None)
        from . import arena
        old_definition = arena._learned
        state = load_checkpoint(checkpoint)
        forbidden = {'optimizer', 'environments', 'sampling_rng', 'torch_rng', 'reference_model'} & state.keys()
        if state.get('format') != 'omnibob-policy-v1' or forbidden:
            raise ValueError('expected inference-only export without training state')
        direct_definition = load_policy(checkpoint)
        arena.prepare(checkpoint)  # Checks the saved local protocol, no network.
        isolation = {'state': {}, 'match_id': 'offline-isolation-probe', 'seat': rows[0]['seat'], 'turn': rows[0]['turn']}
        arena.act(rows[0]['observation'], isolation)
        isolated_policy = isolation['state']['learned_policy']
        isolated_before = save_runtime_state(isolated_policy)
        contexts, direct = {}, {}
        with (output/'frames.jsonl').open('x') as frames:
            for i, row in enumerate(rows):
                key = (row['match_id'], row['seat'])
                if key not in contexts:
                    contexts[key] = {'state': {}, 'match_id': row['match_id'], 'seat': row['seat']}
                    direct[key] = clone_runtime(direct_definition)
                context = contexts[key]
                context['turn'] = row['turn']
                payload = json.loads(json.dumps(row['observation']))
                obs = ObservationV2(**payload)
                expected = int(direct[key].act(obs).action)
                actual = arena.act(payload, context)
                legal = 0 <= actual < len(obs.action_mask) and bool(obs.action_mask[actual])
                frames.write(json.dumps({'frame': i, 'match_id': row['match_id'], 'seat': row['seat'],
                                         'turn': row['turn'], 'direct_action': expected, 'adapter_action': actual,
                                         'legal': legal})+'\n')
                result['checked_frames'] += 1
                if actual != expected:
                    result['action_mismatches'].append(i)
                if not legal:
                    result['illegal_actions'] += 1
        result['independent_context_state_verified'] = (
            all(context['state']['learned_policy'] is not isolated_policy for context in contexts.values())
            and save_runtime_state(isolated_policy) == isolated_before)
        result['match_seat_contexts'] = len(contexts)
        result['runtime_config'] = state['runtime_config']
        result['environment'] = state['environment']
        result['deployment_fields_excluded'] = ['optimizer', 'environments', 'sampling_rng', 'torch_rng', 'reference_model']
        if sha256(recording) != original_sha or sha256(checkpoint) != policy_sha:
            raise ValueError('input changed during offline check')
        if result['action_mismatches'] or result['illegal_actions'] or not result['independent_context_state_verified']:
            raise ValueError('adapter equivalence/legality/context isolation check failed')
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if arena is not None:
            arena._learned = old_definition
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.checkpoint, args.recording, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'environment'}, indent=2))


if __name__ == '__main__':
    main()
