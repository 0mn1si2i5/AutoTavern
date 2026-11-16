"""Hero-specific decisions over versioned metadata and visible log fields."""
CARIEL = 'BG21_HERO_000p'
EFFECTS = {'BG21_HERO_000p_t1': 'targets', 'BG21_HERO_000_Buddyt': 'attack',
           'BG21_HERO_000_Buddyt2': 'health'}


def cariel_power(view, catalog):
    if view.get('game_complete'): return None
    options = view.get('last_options') or {}
    if not options.get('complete'): return None
    local = view['local_controller']
    source = next((e for e in view['entities'] if e['card_id']==CARIEL and e['controller']==local), None)
    if source is None: return None
    card = catalog.get(CARIEL)
    if not card or card.get('cost') != 0: return None  # verified free, untargeted profile only
    choice = view.get('active_choice')
    available = not choice and any(o['entity'] and o['entity']['entity_id']==source['entity_id'] and o['error']=='NONE'
                                  for o in options['options'])
    friendly = {o['entity']['entity_id'] for o in options['options'] if o['entity'] and
                o['entity']['controller']==local and o['entity']['zone']=='PLAY' and o['entity']['card_type']=='MINION'}
    tags = source.get('tags', {})
    return {'card_id':CARIEL,'cost':0,'available':available,'board_count':len(friendly),
            'attack':int(tags['TAG_SCRIPT_DATA_NUM_1']) if 'TAG_SCRIPT_DATA_NUM_1' in tags else None,
            'health':int(tags['TAG_SCRIPT_DATA_NUM_2']) if 'TAG_SCRIPT_DATA_NUM_2' in tags else None,
            'targets':int(tags['TAG_SCRIPT_DATA_NUM_3']) if 'TAG_SCRIPT_DATA_NUM_3' in tags else None}


def cariel_choice(view, catalog):
    choice = view.get('active_choice')
    if not choice or not choice.get('source') or choice['source']['card_id'] != CARIEL: return None
    power = cariel_power(view, catalog)
    if power is None or any(power[k] is None for k in ('attack','health','targets')): return None
    attack, health, targets = (power[k] for k in ('attack','health','targets'))
    # Explicit three-activation heuristic, assuming one extra board unit per turn.
    # This is not a global optimality claim or a learned win-probability model.
    counts = [min(7,power['board_count']+i) for i in range(3)]
    options = []
    for offered in choice['entities']:
        effect=EFFECTS.get(offered['card_id'])
        if effect is None or not catalog.get(offered['card_id']): continue
        delta=[]
        for count in counts:
            before=min(targets,count)*(attack+health)
            after=(min(targets+1,count)*(attack+health) if effect=='targets'
                   else min(targets,count)*(attack+health+1))
            delta.append(after-before)
        options.append({'index':offered['index'],'card_id':offered['card_id'],'effect':effect,
                        'attribute_gain_by_activation':delta,'score':sum(delta)})
    if not options: return None
    best=max(options,key=lambda o:(o['score'],o['effect']=='health',-o['index']))
    return {'selected':best,'options':options,'power':power,'assumed_board_counts':counts,
            'objective':'three-activation total attributes, equal attack/health weight',
            'global_optimality_established':False}
