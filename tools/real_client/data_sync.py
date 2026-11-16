"""Refresh public card metadata into immutable, build-pinned local revisions.

No game input, model training, credentials, or runtime-source modifications.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import urllib.error
from urllib.request import Request, urlopen

if __package__:
    from . import cli
    from .catalog import build as build_catalog
else:
    import cli
    from catalog import build as build_catalog

ROOT = cli.ROOT / 'runs/real-client-data'
LABEL = 'local.omnibob.card-data-sync'
FIELDS = ('type','cost','attack','health','armor','techLevel','text','text_en','mechanics','races',
          'playRequirements','heroPowerDbfId','isBattlegroundsPoolMinion','isBattlegroundsPoolSpell',
          'battlegroundsHero','battlegroundsNormalDbfId','battlegroundsPremiumDbfId')


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); tmp.replace(path)


def client_build():
    info = plistlib.loads(cli.APP_INFO.read_bytes())
    version = str(info['CFBundleVersion'])
    if not re.fullmatch(r'[\d.]+', version): raise ValueError('unknown client build format')
    return version.split('.')[-1], version


def fetch(url, cached=None, limit=32*1024*1024):
    headers = {'User-Agent':'OmniBob-DataSync/0.1', 'Accept':'application/json'}
    if cached and cached.get('etag'): headers['If-None-Match'] = cached['etag']
    if cached and cached.get('last_modified'): headers['If-Modified-Since'] = cached['last_modified']
    try:
        with urlopen(Request(url, headers=headers), timeout=20) as response:
            data = response.read(limit+1)
            if len(data)>limit: raise ValueError('public response exceeded size budget')
            return data, {'url':url,'final_url':response.url,'status':response.status,
                          'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}
    except urllib.error.HTTPError as error:
        if error.code == 304 and cached and Path(cached['path']).is_file():
            data = Path(cached['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != cached['sha256']: raise ValueError('cached data changed')
            return data, {**cached,'status':304}
        raise


def latest_build():
    request = Request('https://api.hearthstonejson.com/v1/latest/',
                      headers={'User-Agent':'OmniBob-DataSync/0.1'})
    with urlopen(request, timeout=10) as response:
        html = response.read(65537).decode('utf8')
        if len(html)>65536: raise ValueError('latest index exceeded size budget')
        builds = set(re.findall(r'href=[\"\']/v1/(\d+)(?:/|[\"\'])', html))
        if len(builds)!=1: raise ValueError('latest index did not identify one numbered build')
        return next(iter(builds))


def card_diff(old_cards, new_cards):
    old={c['id']:c for c in old_cards}; new={c['id']:c for c in new_cards}
    changed=[]
    for card_id in sorted(old.keys() & new.keys()):
        fields={k:{'before':old[card_id].get(k),'after':new[card_id].get(k)} for k in FIELDS
                if old[card_id].get(k)!=new[card_id].get(k)}
        if fields: changed.append({'card_id':card_id,'fields':fields})
    return {'added':sorted(new.keys()-old.keys()), 'removed':sorted(old.keys()-new.keys()), 'changed':changed,
            'numeric_or_text_update_does_not_establish_model_compatibility':True}


def check(root=ROOT, getter=fetch, installed=None, latest_resolver=latest_build):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'sync.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'already_running'}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        report_dir=root/'checks'/stamp;report_dir.mkdir(parents=True)
        result={'utc':cli.now(),'status':'checking','events':[],'credentials_used':False,'game_actions':0}
        old_current=json.loads((root/'CURRENT.json').read_text()) if (root/'CURRENT.json').exists() else None
        cache=json.loads((root/'HTTP_CACHE.json').read_text()) if (root/'HTTP_CACHE.json').exists() else {}
        try:
            build_id,version=installed or client_build();result.update(client_build=build_id,client_version=version)
            if not re.fullmatch(r'\d+',build_id):raise ValueError('invalid build')
            try:
                result['latest_public_build']=latest_resolver()
                result['newer_public_build_available']=int(result['latest_public_build'])>int(build_id)
            except Exception as error:
                result['events'].append({'kind':'latest_build_check_failed','error':repr(error)})
            # The arena is a separate ruleset; a failure here does not invalidate
            # independently pinned live-client card data.
            arena_url='https://rizling.top/api/v1/manifest'
            try:
                data,receipt=getter(arena_url,cache.get(arena_url),limit=1024*1024)
                arena=json.loads(data);path=report_dir/'rizling-manifest.json';path.write_bytes(data)
                cache[arena_url]={**receipt,'path':str(path),'sha256':cli.sha(path)}
                result['rizling_game_version']=arena.get('environment',{}).get('game_version')
                result['rizling_catalog_sha256']=arena.get('catalog_sha256')
                result['same_client_version']=result['rizling_game_version']==version
            except Exception as error:
                result['events'].append({'kind':'arena_check_failed','error':repr(error)})
            sources={}; hashes={}; parsed={}
            for locale in ('zhCN','enUS'):
                url=f'https://api.hearthstonejson.com/v1/{build_id}/{locale}/cards.json'
                data,receipt=getter(url,cache.get(url))
                cards=json.loads(data)
                if not isinstance(cards,list) or not cards:raise ValueError('invalid card payload')
                path=report_dir/(locale+'.json');path.write_bytes(data)
                sources[locale]=path;hashes[locale]=cli.sha(path);parsed[locale]=cards
                cache[url]={**receipt,'path':str(path),'sha256':hashes[locale]}
            revision=build_id+'-'+hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
            dest=root/'revisions'/revision
            if not dest.exists():
                final_dest=dest
                dest=report_dir/'staged-revision'
                dest.mkdir(parents=True)
                for locale,path in sources.items():shutil.copyfile(path,dest/(locale+'.json'))
                build_catalog(dest/'zhCN.json',dest/'enUS.json',dest/'catalog.sqlite',build_id)
                old_cards=json.loads((Path(old_current['path'])/'zhCN.json').read_text()) if old_current else []
                diff=card_diff(old_cards,parsed['zhCN']);atomic(dest/'DIFF.json',diff)
                atomic(dest/'MANIFEST.json',{'build':build_id,'client_version':version,'revision':revision,
                       'source_hashes':hashes,'catalog_sha256':cli.sha(dest/'catalog.sqlite'),
                       'fetched_at':cli.now(),'source_urls':{k:f'https://api.hearthstonejson.com/v1/{build_id}/{k}/cards.json' for k in sources},
                       'model_compatibility':'not_established','policy_code_changed':False})
                final_dest.parent.mkdir(parents=True,exist_ok=True)
                dest.rename(final_dest);dest=final_dest
            manifest=json.loads((dest/'MANIFEST.json').read_text())
            if cli.sha(dest/'catalog.sqlite')!=manifest['catalog_sha256']:raise ValueError('revision database hash mismatch')
            for locale in ('zhCN','enUS'):
                if cli.sha(dest/(locale+'.json'))!=hashes[locale]:raise ValueError('revision raw hash mismatch')
                url=f'https://api.hearthstonejson.com/v1/{build_id}/{locale}/cards.json'
                cache[url]['path']=str(dest/(locale+'.json'))
            # Activate metadata only, after both locales and the index validate.
            # Existing open Catalog objects continue to use their immutable file.
            pointer={'revision':revision,'build':build_id,'path':str(dest),'catalog':str(dest/'catalog.sqlite'),
                     'manifest_sha256':cli.sha(dest/'MANIFEST.json'),'activated_at':cli.now()}
            changed=not old_current or old_current['revision']!=revision
            if changed:atomic(root/'CURRENT.json',pointer)
            atomic(root/'HTTP_CACHE.json',cache)
            atomic(report_dir/'SOURCES.json', {'hashes':hashes,'revision_path':str(dest)})
            for path in sources.values():
                path.unlink()  # duplicates created by this check; immutable originals retained
            result.update(status='updated' if changed else 'unchanged',revision=revision,
                          added=len(json.loads((dest/'DIFF.json').read_text())['added']) if changed else 0,
                          runtime_policy_or_model_modified=False)
        except Exception as error:
            result.update(status='failed',error=repr(error),previous_revision_retained=True)
        atomic(report_dir/'RESULT.json',result);atomic(root/'LAST_CHECK.json',result)
        return result


def install_hourly():
    # Project-local Python and data; only this named per-user launch job is installed.
    folder=Path.home()/'Library/LaunchAgents';folder.mkdir(exist_ok=True)
    target=folder/(LABEL+'.plist');ROOT.mkdir(parents=True,exist_ok=True)
    payload={'Label':LABEL,'ProgramArguments':[str(cli.ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'check'],
             'WorkingDirectory':str(cli.ROOT),'StartInterval':3600,
             'StandardOutPath':str(ROOT/'updater.stdout.log'),'StandardErrorPath':str(ROOT/'updater.stderr.log')}
    if target.exists() and plistlib.loads(target.read_bytes())!=payload:
        raise ValueError(f'existing different launch job preserved: {target}')
    if not target.exists():target.write_bytes(plistlib.dumps(payload))
    domain=f'gui/{os.getuid()}'
    exists=subprocess.run(['launchctl','print',domain+'/'+LABEL],capture_output=True).returncode==0
    if not exists:subprocess.run(['launchctl','bootstrap',domain,str(target)],check=True,capture_output=True)
    return {'label':LABEL,'interval_seconds':3600,'plist':str(target),'no_chat_or_llm_required':True,
            'uninstall':['launchctl','bootout',domain+'/'+LABEL]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','status','install-hourly','uninstall-hourly'])
    args=parser.parse_args()
    if args.command=='check':result=check()
    elif args.command=='install-hourly':result=install_hourly()
    elif args.command=='uninstall-hourly':
        target=Path.home()/'Library/LaunchAgents'/(LABEL+'.plist')
        if target.exists():
            job=plistlib.loads(target.read_bytes())
            if job.get('Label')!=LABEL or str(Path(__file__).resolve()) not in job.get('ProgramArguments',[]):
                raise ValueError('different launch job preserved')
            subprocess.run(['launchctl','bootout',f'gui/{os.getuid()}/'+LABEL],capture_output=True)
            target.unlink()
        result={'status':'uninstalled','data_retained':True}
    else:result=json.loads((ROOT/'LAST_CHECK.json').read_text())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if result.get('status')=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
