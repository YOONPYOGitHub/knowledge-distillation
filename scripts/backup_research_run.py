#!/usr/bin/env python3
"""Backup completed stage generations plus frozen corpus/config metadata.

The caller must use one writer per run/prefix. No VM allocation or credentials.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import checkpoint_blob as cb


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def backup_run(run,store_factory,status_path):
    run=Path(run).resolve();status_path=Path(status_path)
    if (run/'plan.json').is_file():
        return backup_sweep(run,store_factory,status_path)
    required=['config.json','prepared.json','raw/manifest.json','raw/train.jsonl','raw/validation.jsonl','raw/test.jsonl']
    if any(not (run/name).is_file() for name in required):
        return {'status':'waiting_for_frozen_data','run':str(run)}
    previous=json.loads(status_path.read_text()) if status_path.exists() else {}
    status={'run':str(run),'stages':{},'metadata_hash':previous.get('metadata_hash')}
    all_complete=True
    for stage in ['teacher','ce','kd']:
        source=run/stage;pointer_path=source/'latest.json'
        if not pointer_path.exists():all_complete=False;continue
        pointer=json.loads(pointer_path.read_text())
        marker=hashlib.sha256(pointer_path.read_bytes()).hexdigest()
        cached=previous.get('stages',{}).get(stage,{})
        if cached.get('local_pointer_sha256')==marker:
            status['stages'][stage]=cached
        else:
            remote=cb.publish_stage(source,store_factory(stage))
            status['stages'][stage]={'local_pointer_sha256':marker,'remote':remote}
        record_path=run/(stage+'.json')
        if not record_path.exists():all_complete=False;continue
        record=json.loads(record_path.read_text())['result']
        committed=status['stages'][stage]['remote']
        if record['status']!='completed' or record['final_step']!=committed['step'] or Path(record['best_checkpoint']).name!=committed['best_checkpoint']:
            all_complete=False
    names=list(required)
    if all_complete:
        names += [p for p in ['teacher.json','ce.json','kd.json','selection.json','test-results.json'] if (run/p).is_file()]
    snapshot=commit_metadata(run,names,store_factory,previous.get('metadata_hash'))
    status.update(status='verified',metadata_hash=snapshot,all_stages_completed=all_complete,
                  verified_at=datetime.now(timezone.utc).isoformat())
    atomic_json(status_path,status)
    return status


def commit_metadata(run,names,store_factory,previous_hash):
    files={name:{'size':(run/name).stat().st_size,'sha256':cb.digest(run/name)} for name in names}
    snapshot=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if snapshot==previous_hash:return snapshot
    store=store_factory('metadata')
    with tempfile.TemporaryDirectory(prefix='metadata-verify-') as temp:
        temp=Path(temp)
        for name in names:
            target=temp/name;target.parent.mkdir(parents=True,exist_ok=True)
            store.put(run/name,'snapshots/'+snapshot+'/'+name)
            store.get('snapshots/'+snapshot+'/'+name,target)
            if cb.digest(target)!=files[name]['sha256']:raise ValueError('Metadata readback mismatch: '+name)
        manifest=temp/'metadata-manifest.json';manifest.write_text(json.dumps({'files':files,'run':str(run)},sort_keys=True))
        store.put(manifest,'snapshots/'+snapshot+'/manifest.json')
        store.get('snapshots/'+snapshot+'/manifest.json',temp/'verified-manifest.json')
        if cb.digest(manifest)!=cb.digest(temp/'verified-manifest.json'):raise ValueError('Metadata manifest mismatch')
        pointer=temp/'metadata-latest.json';pointer.write_text(json.dumps({'snapshot':snapshot,'manifest_sha256':cb.digest(manifest)},sort_keys=True))
        store.put(pointer,'latest.json',overwrite=True)
        store.get('latest.json',temp/'verified-latest.json')
        if pointer.read_bytes()!=(temp/'verified-latest.json').read_bytes():raise ValueError('Metadata pointer mismatch')
    return snapshot


def backup_sweep(run,store_factory,status_path):
    plan=json.loads((run/'plan.json').read_text())
    previous=json.loads(status_path.read_text()) if status_path.exists() else {}
    status={'run':str(run),'trials':{},'base_provenance':plan.get('provenance',{})}
    all_complete=True
    for trial in plan['trials']:
        name=cb.safe_name(trial['id']);source=run/'trials'/name
        pointer_path=source/'latest.json'
        if not pointer_path.exists():all_complete=False;continue
        marker=hashlib.sha256(pointer_path.read_bytes()).hexdigest()
        cached=previous.get('trials',{}).get(name,{})
        if cached.get('local_pointer_sha256')==marker:status['trials'][name]=cached
        else:status['trials'][name]={'local_pointer_sha256':marker,'remote':cb.publish_stage(source,store_factory(name))}
        result_path=source/'result.json'
        if not result_path.exists():all_complete=False;continue
        record=json.loads(result_path.read_text())['result'];committed=status['trials'][name]['remote']
        if record['status']!='completed' or record['final_step']!=committed['step'] or Path(record['best_checkpoint']).name!=committed['best_checkpoint']:
            all_complete=False
    names=['plan.json']
    if all_complete:
        names += [name for name in ['selection.json','report.json','trials.csv','summary.md'] if (run/name).is_file()]
        names += ['trials/'+cb.safe_name(t['id'])+'/result.json' for t in plan['trials']]
    snapshot=commit_metadata(run,names,store_factory,previous.get('metadata_hash'))
    status.update(status='verified',metadata_hash=snapshot,all_trials_completed=all_complete,
                  verified_at=datetime.now(timezone.utc).isoformat())
    atomic_json(status_path,status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--account',required=True)
    p.add_argument('--container',default='kd-backups')
    p.add_argument('--prefix',required=True)
    p.add_argument('--status-path',required=True,type=Path)
    args=p.parse_args()
    if not os.path.ismount('/srv/kd'):raise RuntimeError('Persistent disk is not mounted')
    os.environ['TMPDIR']='/srv/kd';tempfile.tempdir='/srv/kd'
    key=hashlib.sha256(args.prefix.encode()).hexdigest()[:20]
    factory=lambda stage:cb.AzCopyStore(args.account,args.container,args.prefix+'/'+stage)
    with cb.prefix_lock('/srv/kd/.research-backup-'+key+'.lock'):
        result=backup_run(args.run_dir,factory,args.status_path)
    print('VERIFIED_RESEARCH_BACKUP '+json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
