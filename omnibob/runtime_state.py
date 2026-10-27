"""Serializable observation-derived controller histories for frozen training opponents."""
from collections import Counter
from .move_guard import MoveGuardPolicy
from .repeat_guard import RepeatGuardPolicy


def _pack(value):
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, tuple):
        return {'tuple': [_pack(x) for x in value]}
    return value


def _unpack(value):
    if isinstance(value, dict):
        if set(value) == {'bytes_hex'}:
            return bytes.fromhex(value['bytes_hex'])
        if set(value) == {'tuple'}:
            return tuple(_unpack(x) for x in value['tuple'])
        raise ValueError('unsupported controller state')
    return value


def save_runtime_state(policy):
    if isinstance(policy, RepeatGuardPolicy):
        return {'kind':'repeat', 'runtime':policy.runtime_config, 'turn':policy.turn,
                'counts': [[state.hex(), action, count] for (state,action),count in sorted(policy.counts.items())],
                'base':save_runtime_state(policy.base)}
    if isinstance(policy, MoveGuardPolicy):
        return {'kind':'move', 'runtime':policy.runtime_config, 'turn':policy.turn,
                'moves':policy.moves, 'context':_pack(policy.context),
                'seen':[_pack(units) for units in sorted(policy.seen)], 'base':save_runtime_state(policy.base)}
    return {'kind':'stateless'}


def restore_runtime_state(policy, saved):
    expected = 'repeat' if isinstance(policy, RepeatGuardPolicy) else 'move' if isinstance(policy, MoveGuardPolicy) else 'stateless'
    if saved['kind'] != expected or expected != 'stateless' and saved['runtime'] != policy.runtime_config:
        raise ValueError('frozen opponent controller state does not match its deployment runtime')
    if expected == 'repeat':
        policy.turn = saved['turn']
        policy.counts = Counter({(bytes.fromhex(state), action): count for state,action,count in saved['counts']})
        restore_runtime_state(policy.base, saved['base'])
    elif expected == 'move':
        policy.turn, policy.moves = saved['turn'], saved['moves']
        policy.context = _unpack(saved['context'])
        policy.seen = {_unpack(units) for units in saved['seen']}
        restore_runtime_state(policy.base, saved['base'])
