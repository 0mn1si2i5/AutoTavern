"""Read-only player-visible Power.log view. Never exports arbitrary hidden entities.

Options/choices are exposed by the client to the local player. Other entity
records are ignored unless they describe that player's own visible entities.
This is a diagnostic adapter, not a network game-action API.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

if __package__:
    from .catalog import Catalog
else:
    from catalog import Catalog

ENTITY = re.compile(r'entityName=.*? id=(\d+) zone=(\w+) zonePos=(\d+) cardId=([^\s\]]*) player=(\d+)')
OPTION = re.compile(r'option (\d+) type=(\w+) mainEntity=(.*?) error=(\w+)')
TARGET = re.compile(r'target (\d+) entity=(.*?) error=(\w+)')
TAG = re.compile(r'TAG_CHANGE Entity=(.*?) tag=(\w+) value=(\S+)')
VISIBLE_TAGS = {'ATK', 'HEALTH', 'DAMAGE', 'ARMOR', 'COST', 'EXHAUSTED', 'ZONE_POSITION',
                'PLAYER_TECH_LEVEL', 'PLAYER_LEADERBOARD_PLACE', 'NUM_HERO_POWER_USES_THIS_TURN',
                'TAG_SCRIPT_DATA_NUM_1', 'TAG_SCRIPT_DATA_NUM_2', 'TAG_SCRIPT_DATA_NUM_3'}


def entity(text):
    match = ENTITY.search(text)
    if not match: return None
    return {'entity_id': int(match[1]), 'zone': match[2], 'position': int(match[3]),
            'card_id': match[4], 'controller': int(match[5])}


def snapshot(text, catalog):
    marker = 'GameState.DebugPrintPower() - CREATE_GAME'
    start = text.rfind(marker)
    if start < 0: raise ValueError('no complete CREATE_GAME boundary in input')
    lines = text[start:].splitlines()
    own_candidates = set(); shop_candidates = set()
    # Determine controller identity from the local player's own action sources.
    # Opponent hand/deck and raw FULL_ENTITY records do not participate.
    for line in lines:
        if 'GameState.DebugPrintOptions()' not in line or 'mainEntity=' not in line: continue
        item = entity(line)
        if not item or item['zone'] != 'PLAY': continue
        card = catalog.get(item['card_id']) or {}
        if item['card_id'].startswith('TB_BaconShopBob'):
            shop_candidates.add(item['controller'])
        elif card.get('battlegroundsHero') or card.get('battlegroundsSkinParentId'):
            own_candidates.add(item['controller'])
    if not own_candidates:
        for line in lines:
            if 'GameState.DebugPrintEntityChoices()' not in line: continue
            item = entity(line)
            card=(catalog.get(item['card_id']) or {}) if item else {}
            if item and (card.get('battlegroundsHero') or card.get('battlegroundsSkinParentId')):
                own_candidates.add(item['controller'])
    if len(own_candidates) != 1:
        raise ValueError(f'local controller not uniquely identified: {sorted(own_candidates)}')
    own = next(iter(own_candidates))
    shops = shop_candidates - {own}
    if len(shops) > 1: raise ValueError('ambiguous tavern controller')

    def public(item, choice=False):
        if not item or not item['card_id']: return None
        if choice:
            allowed = item['controller'] == own and item['zone'] in ('HAND', 'SETASIDE', 'PLAY')
        else:
            allowed = ((item['controller'] == own and item['zone'] in ('HAND', 'PLAY')) or
                       (item['controller'] in shops and item['zone'] == 'PLAY'))
        if not allowed: return None
        card = catalog.get(item['card_id'])
        return {**item, 'name': card.get('name') if card else None,
                'card_type': card.get('type') if card else None, 'metadata_known': bool(card)}

    own_declared = {}
    for line in lines:
        if 'GameState.DebugPrintOptions()' in line or 'GameState.DebugPrintEntityChoices()' in line:
            item = entity(line)
            if item and item['controller'] == own and item['zone'] in ('HAND', 'PLAY'):
                own_declared[item['entity_id']] = public(item)
    full_entity_id = None
    options = None; building = None; current_option = None; choice = None
    entities = {}; game = {}; option_time = None; last_line_time = None
    for line in lines:
        stamp = re.match(r'\w (\d\d:\d\d:\d\d\.\d+)', line)
        if stamp: last_line_time = stamp[1]
        if 'GameState.DebugPrintOptions() - ' in line:
            message = line.split('() - ', 1)[1].strip()
            if message.startswith('id='):
                if building: options = building
                building = {'id': int(message[3:]), 'time': last_line_time, 'options': [], 'complete': False}
                current_option = None
            elif (match := OPTION.match(message)) and building:
                item = public(entity(match[3]))
                current_option = {'index': int(match[1]), 'type': match[2], 'entity': item,
                                  'error': match[4], 'targets': []}
                if item or match[2] == 'END_TURN':
                    building['options'].append(current_option)
                    if item: entities[item['entity_id']] = {**item, 'tags': entities.get(item['entity_id'], {}).get('tags', {})}
            elif (match := TARGET.match(message)) and current_option:
                item = public(entity(match[2]))
                if item:
                    current_option['targets'].append({'index': int(match[1]), 'entity': item, 'error': match[3]})
                    entities[item['entity_id']] = {**item, 'tags': entities.get(item['entity_id'], {}).get('tags', {})}
            continue
        if building:
            building['complete'] = True; options = building; building = None; current_option = None
        if 'GameState.DebugPrintEntityChoices() - ' in line:
            message = line.split('() - ', 1)[1].strip()
            if (match := re.match(r'id=(\d+).*ChoiceType=(\w+)', message)):
                choice = {'id': int(match[1]), 'type': match[2], 'time': last_line_time, 'source': None, 'entities': []}
            elif choice and message.startswith('Source='):
                choice['source'] = public(entity(message), choice=True)
            elif choice and (match := re.match(r'Entities\[(\d+)\]=', message)):
                item = public(entity(message), choice=True)
                if item: choice['entities'].append({'index': int(match[1]), **item})
        if ('GameState.SendChoices() - id=' in line or 'GameState.DebugPrintEntitiesChosen() - id=' in line) and choice:
            match = re.search(r'id=(\d+)', line)
            if match and int(match[1]) == choice['id']: choice = None
        if 'GameState.DebugPrintPower()' not in line: continue
        created = re.search(r'FULL_ENTITY - Creating ID=(\d+) CardID=', line)
        if created:
            full_entity_id = int(created[1]) if int(created[1]) in own_declared else None
            continue
        initial = re.search(r'\(\) -\s+tag=(\w+) value=(\S+)$', line)
        if initial and full_entity_id is not None:
            if initial[1] in VISIBLE_TAGS:
                previous = entities.get(full_entity_id, {})
                entities[full_entity_id] = {**own_declared[full_entity_id], 'tags': {**previous.get('tags', {}), initial[1]: initial[2]}}
            continue
        full_entity_id = None
        match = TAG.search(line)
        if not match: continue
        raw, tag, value = match.groups()
        if raw in ('GameEntity', '1') and tag in ('STATE', 'STEP', 'TURN'):
            game[tag] = value
        item = entity(raw)
        # Own HAND/PLAY fields only; never import other players' entity tags.
        if item and item['controller'] == own and item['zone'] in ('HAND', 'PLAY') and tag in VISIBLE_TAGS:
            safe = public(item)
            if safe:
                previous = entities.get(item['entity_id'], {})
                entities[item['entity_id']] = {**safe, 'tags': {**previous.get('tags', {}), tag: value}}
    if building: options = building  # explicitly incomplete; cannot dispatch from this snapshot
    complete = game.get('STATE') == 'COMPLETE' or game.get('STEP') == 'FINAL_GAMEOVER'
    if complete: choice = None
    return {'local_controller': own, 'shop_controllers': sorted(shops), 'game': game, 'game_complete': complete,
            'latest_log_time': last_line_time, 'last_options': options, 'active_choice': choice,
            'entities': list(entities.values()), 'catalog_build': catalog.meta['build'],
            'visibility': 'local action/choice sources; own HAND/PLAY tags; tavern PLAY; no hidden opponent entities',
            'unknown_fields': ['gold resource reconciliation', 'initial tags not emitted as TAG_CHANGE',
                               'current screen geometry and animation completion'],
            'ready_for_automatic_dispatch': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    began = time.monotonic()
    before = args.log.stat()
    catalog = Catalog(args.db)
    with args.log.open(errors='replace') as stream:
        text = stream.read(before.st_size)
    data = snapshot(text, catalog); catalog.close()
    data.update(captured_at=datetime.now(timezone.utc).isoformat(), source_log=str(args.log.resolve()),
                source_bytes=before.st_size, parse_seconds=time.monotonic()-began)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(data, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: data[k] for k in ['local_controller','game_complete','catalog_build','parse_seconds','ready_for_automatic_dispatch']}))


if __name__ == '__main__': main()
