#!/usr/bin/env python3
"""Commit immutable checkpoint generations only after remote readback verification.

No account keys/SAS. Azure CLI uses azcopy managed identity from inside the VM.
Single-writer prefix required: the training service must hold its run lock.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    if not isinstance(name,str) or name in {'.','..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+',name):
        raise ValueError('Unsafe checkpoint path component')
    return name


def valid_files(manifest):
    files=manifest.get('files')
    if not isinstance(files,dict) or not files or 'state.pt' not in files:
        raise ValueError('Checkpoint requires a files manifest including state.pt')
    for name, info in files.items():
        safe_name(name)
        if name in {'manifest.json','COMPLETE','latest.json'} or not isinstance(info,dict):
            raise ValueError('Reserved or malformed manifest entry')
        if not isinstance(info.get('size'),int) or info['size']<0 or not re.fullmatch(r'[a-f0-9]{64}',info.get('sha256','')):
            raise ValueError('Malformed checksum or size')
    return files


def validate(directory):
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Checkpoint must be a real directory')
    for name in ['manifest.json','COMPLETE']:
        if (directory/name).is_symlink() or not (directory/name).is_file():
            raise ValueError('Checkpoint incomplete or symlinked')
    marker=json.loads((directory/'COMPLETE').read_text())
    if not isinstance(marker,dict) or marker.get('manifest_sha256')!=digest(directory/'manifest.json'):
        raise ValueError('COMPLETE manifest hash mismatch')
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,info in valid_files(manifest).items():
        p=directory/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=info['size'] or digest(p)!=info['sha256']:
            raise ValueError('Checkpoint checksum/size mismatch: '+name)
    return manifest


class LocalStore:
    """Filesystem-backed transport for real offline protocol tests."""
    def __init__(self,root):
        self.root=Path(root)
    def _path(self,key):
        parts=key.split('/')
        for part in parts:safe_name(part)
        p=self.root.joinpath(*parts)
        # Never follow an attacker-controlled destination symlink.
        if any(q.is_symlink() for q in [p,*p.parents] if q != self.root.parent):
            raise ValueError('Symlinked store path')
        return p
    def put(self,source,key,overwrite=False):
        p=self._path(key);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and not overwrite:
            if digest(p)!=digest(source):raise FileExistsError('Immutable object already differs')
            return
        tmp=p.with_name(p.name+'.tmp')
        shutil.copyfile(source,tmp);os.replace(tmp,p)
    def current(self):
        p=self._path('latest.json')
        return json.loads(p.read_text()) if p.exists() else None
    def get(self,key,destination):
        source=self._path(key)
        shutil.copyfile(source,destination)


class AzCopyStore:
    def __init__(self,account,container,prefix):
        if not re.fullmatch(r'[a-z0-9]{3,24}',account) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]',container):
            raise ValueError('Invalid Azure account/container')
        for part in prefix.split('/'):safe_name(part)
        self.base=f'https://{account}.blob.core.windows.net/{container}/{prefix}'
    def _url(self,key):
        for part in key.split('/'):safe_name(part)
        return self.base+'/'+key
    def _copy(self,source,target,overwrite):
        env={**os.environ,'AZCOPY_AUTO_LOGIN_TYPE':'MSI',
             'AZCOPY_LOG_LOCATION':'/srv/kd/backup-logs','AZCOPY_JOB_PLAN_LOCATION':'/srv/kd/backup-plans'}
        subprocess.run(['azcopy','copy',str(source),str(target),'--overwrite='+str(overwrite).lower(),
                        '--put-md5','--output-level=essential'],env=env,check=True,timeout=1800)
    def _get_request(self,key):
        import urllib.request
        import urllib.parse
        query=urllib.parse.urlencode({'api-version':'2018-02-01','resource':'https://storage.azure.com/'})
        request=urllib.request.Request('http://169.254.169.254/metadata/identity/oauth2/token?'+query,headers={'Metadata':'true'})
        with urllib.request.urlopen(request,timeout=30) as response:
            token=json.loads(response.read())['access_token']
        return urllib.request.Request(self._url(key),headers={'Authorization':'Bearer '+token,'x-ms-version':'2023-11-03'})
    def current(self):
        # Only a confirmed authenticated 404 means absent. Other errors stop.
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(self._get_request('latest.json'),timeout=30) as response:data=response.read(65537)
        except urllib.error.HTTPError as exc:
            if exc.code==404:return None
            raise
        if len(data)>65536:raise ValueError('Oversized latest pointer')
        return json.loads(data)
    def put(self,source,key,overwrite=False):self._copy(source,self._url(key),overwrite)
    def get(self,key,destination):
        # AzCopy's Linux MD5 xattr storage is unsupported on this mounted disk.
        # Stream authenticated bytes directly, then callers verify SHA-256.
        import urllib.request
        with urllib.request.urlopen(self._get_request(key),timeout=60) as response, Path(destination).open('wb') as out:
            shutil.copyfileobj(response,out,length=8*1024*1024)
            out.flush();os.fsync(out.fileno())


def publish(generation,store):
    generation=Path(generation)
    safe_name(generation.name)
    manifest=validate(generation)
    step=manifest.get('step')
    if not isinstance(step,int) or step<0:raise ValueError('Invalid checkpoint step')
    previous=store.current()
    if previous is not None:
        safe_name(previous.get('checkpoint'))
        if not isinstance(previous.get('step'),int):raise ValueError('Malformed remote latest pointer')
        if previous['step']>step:return previous
        if previous['step']==step and previous['checkpoint']!=generation.name:
            raise ValueError('Divergent generation at same optimizer step')
    base='generations/'+generation.name+'/'
    # Upload mutable pointer only after all payloads have been verified remotely.
    with tempfile.TemporaryDirectory(prefix='checkpoint-verify-') as temp:
        temp=Path(temp)
        for name in [*valid_files(manifest),'manifest.json']:
            store.put(generation/name,base+name)
            received=temp/name
            store.get(base+name,received)
            if digest(received)!=digest(generation/name):raise ValueError('Remote readback mismatch: '+name)
        store.put(generation/'COMPLETE',base+'COMPLETE')
        store.get(base+'COMPLETE',temp/'COMPLETE')
        if digest(temp/'COMPLETE')!=digest(generation/'COMPLETE'):raise ValueError('Completion marker readback mismatch')
        pointer={'schema':1,'checkpoint':generation.name,'step':manifest.get('step'),
                 'manifest_sha256':digest(generation/'manifest.json')}
        p=temp/'latest.json';p.write_text(json.dumps(pointer,sort_keys=True))
        store.put(p,'latest.json',overwrite=True)
        store.get('latest.json',temp/'latest-verified.json')
        if (temp/'latest-verified.json').read_bytes()!=p.read_bytes():raise ValueError('Latest pointer readback mismatch')
    return pointer


def restore(store,destination):
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError('Restore destination must not exist')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-',dir=destination.parent) as temp:
        temp=Path(temp)
        store.get('latest.json',temp/'latest.json')
        pointer=json.loads((temp/'latest.json').read_text())
        name=safe_name(pointer['checkpoint']);base='generations/'+name+'/'
        received=temp/'generation';received.mkdir()
        store.get(base+'manifest.json',received/'manifest.json')
        if digest(received/'manifest.json')!=pointer.get('manifest_sha256'):raise ValueError('Remote manifest mismatch')
        manifest=json.loads((received/'manifest.json').read_text())
        for file in valid_files(manifest):store.get(base+file,received/file)
        store.get(base+'COMPLETE',received/'COMPLETE')
        validate(received)
        os.replace(received,destination)
    return destination


def _upload_generation(generation,store):
    generation=Path(generation);manifest=validate(generation)
    base='generations/'+safe_name(generation.name)+'/'
    with tempfile.TemporaryDirectory(prefix='stage-verify-') as temp:
        temp=Path(temp)
        for name in [*valid_files(manifest),'manifest.json','COMPLETE']:
            store.put(generation/name,base+name)
            store.get(base+name,temp/name)
            if digest(temp/name)!=digest(generation/name):raise ValueError('Remote generation readback mismatch')
    return digest(generation/'manifest.json')


def publish_stage(stage,store):
    """Commit latest AND its selected best dependency for a resumable stage."""
    from src.resume_checkpoint import load_checkpoint
    stage=Path(stage).resolve()
    local=json.loads((stage/'latest.json').read_text())
    latest=(stage/local['checkpoint']).resolve()
    if latest.parent!=(stage/'checkpoints').resolve():raise ValueError('Unsafe local stage pointer')
    state=load_checkpoint(latest)
    best=(stage/state['best_checkpoint']).resolve()
    if best.parent!=(stage/'checkpoints').resolve():raise ValueError('Unsafe best dependency')
    manifest=json.loads((latest/'manifest.json').read_text())
    load_checkpoint(best,expected_contract=manifest['contract'])
    contract_hash=hashlib.sha256(json.dumps(manifest['contract'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    previous=store.current()
    if previous is not None:
        if previous.get('schema')!=2 or previous.get('contract_sha256')!=contract_hash:
            raise ValueError('Remote stage contract differs')
        if previous['step']>state['step']:return previous
        if previous['step']==state['step'] and previous['checkpoint']!=latest.name:
            raise ValueError('Divergent remote stage generation')
    best_hash=_upload_generation(best,store)
    latest_hash=best_hash if latest==best else _upload_generation(latest,store)
    pointer={'schema':2,'checkpoint':latest.name,'manifest_sha256':latest_hash,
             'best_checkpoint':best.name,'best_manifest_sha256':best_hash,
             'contract_sha256':contract_hash,'step':state['step']}
    with tempfile.TemporaryDirectory(prefix='stage-pointer-') as temp:
        p=Path(temp)/'latest.json';p.write_text(json.dumps(pointer,sort_keys=True))
        store.put(p,'latest.json',overwrite=True)
        store.get('latest.json',Path(temp)/'verified.json')
        if p.read_bytes()!=(Path(temp)/'verified.json').read_bytes():raise ValueError('Stage pointer readback mismatch')
    return pointer


def restore_stage(store,destination):
    """Restore a verified stage into a NEW directory without a live-path overwrite."""
    from src.resume_checkpoint import load_checkpoint
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError('Restore destination must not exist')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage-restore-',dir=destination.parent) as temp:
        temp=Path(temp);store.get('latest.json',temp/'remote-latest.json')
        pointer=json.loads((temp/'remote-latest.json').read_text())
        if pointer.get('schema')!=2:raise ValueError('Not a stage commit')
        name=safe_name(pointer['checkpoint']);best=safe_name(pointer['best_checkpoint'])
        recovered=temp/'stage';(recovered/'checkpoints').mkdir(parents=True)
        for generation_name,expected_hash in {(name,pointer['manifest_sha256']),(best,pointer['best_manifest_sha256'])}:
            target=recovered/'checkpoints'/generation_name;target.mkdir()
            base='generations/'+generation_name+'/'
            store.get(base+'manifest.json',target/'manifest.json')
            if digest(target/'manifest.json')!=expected_hash:raise ValueError('Stage manifest mismatch')
            manifest=json.loads((target/'manifest.json').read_text())
            for file in valid_files(manifest):store.get(base+file,target/file)
            store.get(base+'COMPLETE',target/'COMPLETE');validate(target);load_checkpoint(target)
        (recovered/'latest.json').write_text(json.dumps({'checkpoint':'checkpoints/'+name,'manifest_sha256':pointer['manifest_sha256']}))
        state=load_checkpoint(recovered)
        if state['step']!=pointer['step'] or Path(state['best_checkpoint']).name!=best:raise ValueError('Stage dependency mismatch')
        latest_manifest=json.loads((recovered/'checkpoints'/name/'manifest.json').read_text())
        contract_hash=hashlib.sha256(json.dumps(latest_manifest['contract'],sort_keys=True,allow_nan=False).encode()).hexdigest()
        if contract_hash!=pointer['contract_sha256']:raise ValueError('Stage contract hash mismatch')
        load_checkpoint(recovered/state['best_checkpoint'],expected_contract=latest_manifest['contract'])
        os.replace(recovered,destination)
    return destination


@contextmanager
def prefix_lock(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['publish','restore','publish-stage','restore-stage'])
    parser.add_argument('path',type=Path)
    parser.add_argument('--account',required=True)
    parser.add_argument('--container',default='kd-backups')
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    if not os.path.ismount('/srv/kd'):raise RuntimeError('Persistent disk /srv/kd is not mounted')
    os.environ['TMPDIR']='/srv/kd'
    tempfile.tempdir='/srv/kd'
    store=AzCopyStore(args.account,args.container,args.prefix)
    lock=hashlib.sha256(args.prefix.encode()).hexdigest()[:20]
    with prefix_lock('/srv/kd/.blob-publish-'+lock+'.lock'):
        function={'publish':publish,'restore':restore,'publish-stage':publish_stage,'restore-stage':restore_stage}[args.action]
        result=function(args.path,store) if args.action.startswith('publish') else str(function(store,args.path))
    print(json.dumps({'action':args.action,'result':result},ensure_ascii=False))

if __name__=='__main__':main()
