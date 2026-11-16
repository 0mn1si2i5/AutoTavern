"""Player-visible log projection, including current options and dynamic values.

Only entities actually disclosed in local options/choices are materialized.
Raw game seeds, other players' hands/decks and account identifiers are not exported.
"""
from __future__ import annotations
import hashlib
import re
from . import log_view

TAGS = log_view.VISIBLE_TAGS | {
    'CONTROLLER','ZONE','ZONE_POSITION','CARDTYPE','TECH_LEVEL','CARDRACE',
    'TAUNT','DIVINE_SHIELD','WINDFURY','REBORN','POISONOUS','VENOMOUS','STEALTH',
    'MAGNETIC','IMMUNE','PREMIUM','FROZEN','ATTACHED','CARD_TARGET',
    'MOVE_MINION_HOVER_TARGET_SLOT','HAS_ACTIVATE_POWER','NUM_TURNS_IN_PLAY',
    'BACON_TRIPLE_UPGRADE_MINION_ID','BACON_TRIPLED_BASE_MINION_ID',
    'BATTLEGROUNDS_PREMIUM','BATTLECRY','DEATHRATTLE','TAG_ONE_TURN_EFFECT',
    'BACON_LOCKED_MULLIGAN_HERO',
    '2442',  # visible DragBuy button -> shop entity, verified with SendOption target
}
PLAYER_TAGS = {'RESOURCES','RESOURCES_USED','TEMP_RESOURCES','PLAYER_TECH_LEVEL',
    'HERO_ENTITY','NUM_CARDS_PLAYED_THIS_TURN','NUM_MINIONS_PLAYED_THIS_TURN',
    'NUM_RESOURCES_SPENT_THIS_GAME','NUM_FRIENDLY_MINIONS_THAT_DIED_THIS_TURN',
    'NUM_HERO_POWER_USES_THIS_TURN','CURRENT_PLAYER','PLAYSTATE','NEXT_OPPONENT_PLAYER_ID'}
GAME_TAGS = {'TURN','STATE','STEP','BACON_TRINKETS_ACTIVE','BACON_DARK_GIFTS_ACTIVE'}


def snapshot(text, catalog):
    marker='GameState.DebugPrintPower() - CREATE_GAME'
    begin=text.rfind(marker)
    if begin<0: raise ValueError('no CREATE_GAME')
    text=text[begin:]; lines=text.splitlines()
    base=log_view.snapshot(text,catalog); own=base['local_controller']
    declared={}
    for line in lines:
        if 'GameState.DebugPrintOptions()' in line or 'GameState.DebugPrintEntityChoices()' in line:
            e=log_view.entity(line)
            if e and e['card_id'] and (e['controller']==own and e['zone'] in ('PLAY','HAND','SETASIDE') or
                    e['controller'] in base['shop_controllers'] and e['zone']=='PLAY'):
                declared[e['entity_id']]=e
    tags={k:{} for k in declared}; player={}; game={}; players={}; player_names={}
    game_id=None; initial=None
    for line in lines:
        if 'GameState.DebugPrintGame() - PlayerID=' in line:
            m=re.search(r'PlayerID=(\d+), PlayerName=(.*)',line)
            if m and int(m[1])==own: player_names[m[2].strip()]=own
        if 'GameState.DebugPrintPower() - ' not in line: continue
        msg=line.split('() - ',1)[1].strip()
        if m:=re.match(r'GameEntity EntityID=(\d+)',msg):
            game_id=int(m[1]); initial=('game',game_id); continue
        if m:=re.match(r'Player EntityID=(\d+) PlayerID=(\d+)',msg):
            players[int(m[1])]=int(m[2]);initial=('player',int(m[1]));continue
        if m:=re.search(r'(?:FULL_ENTITY - Creating ID=|SHOW_ENTITY - Updating Entity=)(\d+).*CardID=',msg):
            initial=('entity',int(m[1]));continue
        if msg.startswith('SHOW_ENTITY'):
            e=log_view.entity(msg);initial=('entity',e['entity_id']) if e else None;continue
        if m:=re.fullmatch(r'tag=(\w+) value=(\S+)',msg):
            if initial:
                kind,eid=initial;tag,val=m.groups()
                if kind=='entity' and eid in tags and tag in TAGS:tags[eid][tag]=val
                elif kind=='player' and players.get(eid)==own and tag in PLAYER_TAGS:player[tag]=val
                elif kind=='game' and tag in GAME_TAGS:game[tag]=val
            continue
        initial=None
        m=log_view.TAG.search(msg)
        if not m: continue
        raw,tag,val=m.groups();e=log_view.entity(raw)
        eid=e['entity_id'] if e else int(raw) if raw.isdigit() else None
        if eid in tags and tag in TAGS:tags[eid][tag]=val
        if (raw in player_names or players.get(eid)==own) and tag in PLAYER_TAGS:player[tag]=val
        if (raw=='GameEntity' or eid==game_id) and tag in GAME_TAGS:game[tag]=val
    def enrich(e):
        if not e:return None
        card=catalog.get(e['card_id']) or {}
        return {**e,'tags':tags.get(e['entity_id'],{}),'card_type':card.get('type'),
                'name':card.get('name'),'metadata_known':bool(card)}
    options=base['last_options']
    if options:
        for opt in options['options']:
            opt['entity']=enrich(opt['entity'])
            for target in opt['targets']:target['entity']=enrich(target['entity'])
    choice=base['active_choice']
    if choice:
        choice['source']=enrich(choice['source'])
        choice['entities']=[enrich(e) for e in choice['entities']]
    # Current options are authoritative for current zones; old observed entities
    # remain only in the identity/tag history and never fill current slots.
    current={}
    if options:
        for opt in options['options']:
            for e in [opt['entity']]+[t['entity'] for t in opt['targets']]:
                if e:current[e['entity_id']]=e
    def zone(controller,where,types):
        return sorted([e for e in current.values() if e['controller']==controller and e['zone']==where
                       and e['card_type'] in types],key=lambda e:e['position'])
    item_types=('MINION','SPELL','BATTLEGROUND_SPELL','BATTLEGROUND_TRINKET')
    board=zone(own,'PLAY',('MINION',));hand=zone(own,'HAND',item_types)
    shop=[]
    for controller in base['shop_controllers']:shop+=zone(controller,'PLAY',item_types)
    hero=next(iter(zone(own,'PLAY',('HERO',))),None)
    powers=zone(own,'PLAY',('HERO_POWER',))
    buttons=[e for e in current.values() if e['controller']==own and
             ('Button' in e['card_id'] or 'Drag' in e['card_id'])]
    turn=(int(game.get('TURN',0))+1)//2
    resources=int(player['RESOURCES']) if 'RESOURCES' in player else None
    gold=resources-int(player.get('RESOURCES_USED',0))+int(player.get('TEMP_RESOURCES',0)) if resources is not None else None
    base.update(game=game,player=player,round=turn,gold=gold,resources=resources,
                board=board,hand=hand,shop=shop,hero=hero,powers=powers,buttons=buttons,
                game_key=hashlib.sha256('\n'.join(lines[:3]).encode()).hexdigest()[:16])
    base['entities']=[enrich(e) for e in current.values()]
    base['game_complete']=game.get('STATE')=='COMPLETE' or game.get('STEP')=='FINAL_GAMEOVER'
    if base['game_complete']:base['active_choice']=None
    base['unknown_fields']=['unmapped simulator effect/counter semantics','opponent public summaries and combat memory']
    return base
