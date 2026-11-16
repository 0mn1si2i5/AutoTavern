import json
from pathlib import Path
import pytest
from tools.real_client.catalog import Catalog, build
from tools.real_client import data_sync
from tools.real_client.log_view import snapshot


def cards(attack=2):
    return [{'id':'MINION','dbfId':1,'name':'测试','type':'MINION','attack':attack,'health':3}]


def test_atomic_updates_pin_open_readers_and_retain_previous_on_failure(tmp_path):
    version={'attack':2,'broken':False}
    def get(url,cached=None,**kwargs):
        if 'rizling' in url:
            return json.dumps({'environment':{'game_version':'36.6.3.253932'}}).encode(), {'url':url,'status':200}
        value=cards(version['attack'])
        if '/enUS/' in url and version['broken']: value[0]['attack']=999
        return json.dumps(value).encode(), {'url':url,'status':200}
    check=lambda: data_sync.check(tmp_path,getter=get,installed=('255790','37.0.255790'),latest_resolver=lambda:'255791')
    first=check();assert first['status']=='updated' and first['newer_public_build_available']
    old_pointer=json.loads((tmp_path/'CURRENT.json').read_text()); reader=Catalog(old_pointer['catalog'])
    version['attack']=4
    second=check();assert second['status']=='updated'
    new_pointer=json.loads((tmp_path/'CURRENT.json').read_text())
    assert new_pointer['revision']!=old_pointer['revision']
    assert reader.get('MINION')['attack']==2
    other=Catalog(new_pointer['catalog']);assert other.get('MINION')['attack']==4
    assert check()['status']=='unchanged'
    before=(tmp_path/'CURRENT.json').read_bytes()
    version.update(attack=6,broken=True)
    failed=check();assert failed['status']=='failed'
    assert (tmp_path/'CURRENT.json').read_bytes()==before
    assert Path(old_pointer['catalog']).exists()
    reader.close();other.close()


def test_catalog_preserves_ambiguous_names_and_refuses_overwrite(tmp_path):
    zh=tmp_path/'zh.json';en=tmp_path/'en.json';out=tmp_path/'catalog.sqlite'
    rows=cards()+[{'id':'MINION_G','dbfId':2,'name':'测试','type':'MINION','attack':4,'health':6}]
    zh.write_text(json.dumps(rows));en.write_text(json.dumps(rows))
    build(zh,en,out,'255790');c=Catalog(out)
    assert len(c.search('测试'))==2 and c.get(1)['id']=='MINION'
    c.close()
    with pytest.raises(FileExistsError):build(zh,en,out,'255790')


class FakeCatalog:
    meta={'build':'255790'}
    def get(self,card_id):
        return {'id':card_id,'name':card_id,'type':'HERO' if card_id=='HERO' else 'MINION',
                'battlegroundsHero':card_id=='HERO'}


def test_log_view_excludes_opponent_hand_and_deck_and_preserves_visible_tags():
    log='''D 01:00:00.000 GameState.DebugPrintPower() - CREATE_GAME
D 01:00:00.001 GameState.DebugPrintOptions() - id=1
D 01:00:00.001 GameState.DebugPrintOptions() - option 1 type=POWER mainEntity=[entityName=hero id=10 zone=PLAY zonePos=0 cardId=HERO player=2] error=NONE
D 01:00:00.001 GameState.DebugPrintOptions() - option 2 type=POWER mainEntity=[entityName=bob id=20 zone=PLAY zonePos=0 cardId=TB_BaconShopBob player=10] error=NONE
D 01:00:00.001 GameState.DebugPrintOptions() - option 3 type=POWER mainEntity=[entityName=own id=11 zone=HAND zonePos=1 cardId=OWN player=2] error=NONE
D 01:00:00.001 GameState.DebugPrintOptions() - option 4 type=POWER mainEntity=[entityName=secret id=90 zone=HAND zonePos=1 cardId=HIDDEN player=9] error=NONE
D 01:00:00.001 GameState.DebugPrintOptions() - target 1 entity=[entityName=secret id=91 zone=DECK zonePos=1 cardId=HIDDEN2 player=9] error=NONE
D 01:00:00.002 GameState.DebugPrintPower() - TAG_CHANGE Entity=[entityName=own id=11 zone=HAND zonePos=1 cardId=OWN player=2] tag=ATK value=7
D 01:00:00.003 GameState.DebugPrintOptions() - id=2
D 01:00:00.003 GameState.DebugPrintOptions() - option 1 type=POWER mainEntity=[entityName=own id=11 zone=HAND zonePos=1 cardId=OWN player=2] error=NONE
D 01:00:00.004 GameState.DebugPrintPower() - TAG_CHANGE Entity=GameEntity tag=STATE value=COMPLETE
'''
    result=snapshot(log,FakeCatalog())
    encoded=json.dumps(result)
    assert 'HIDDEN' not in encoded and 'secret' not in encoded
    assert result['game_complete'] and not result['ready_for_automatic_dispatch']
    assert next(e for e in result['entities'] if e['entity_id']==11)['tags']['ATK']=='7'
    assert result['last_options']['complete']


def test_cariel_prefers_more_targets_when_that_dominates_and_caps_at_board_size():
    from tools.real_client.hero_rules import cariel_choice
    class C:
        def get(self,key):return {'cost':0,'id':key}
    power={'entity_id':136,'card_id':'BG21_HERO_000p','controller':2,
           'tags':{'TAG_SCRIPT_DATA_NUM_1':'5','TAG_SCRIPT_DATA_NUM_2':'2','TAG_SCRIPT_DATA_NUM_3':'1'}}
    entities=[{'index':i,'card_id':card} for i,card in enumerate(
        ['BG21_HERO_000p_t1','BG21_HERO_000_Buddyt','BG21_HERO_000_Buddyt2'])]
    board=[{'entity':{'entity_id':i,'controller':2,'zone':'PLAY','card_type':'MINION'},'error':'NONE'} for i in range(2)]
    view={'local_controller':2,'entities':[power],'last_options':{'complete':True,'options':board},
          'active_choice':{'source':power,'entities':entities},'game_complete':False}
    result=cariel_choice(view,C())
    assert result['selected']['index']==0
    assert result['selected']['attribute_gain_by_activation'][0]==7
    assert not result['global_optimality_established']
    power['tags']['TAG_SCRIPT_DATA_NUM_3']='7'
    assert cariel_choice(view,C())['selected']['index']!=0
    view['game_complete']=True
    assert cariel_choice(view,C()) is None


