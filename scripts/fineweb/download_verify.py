#!/usr/bin/env python3
"""Download pinned model snapshots with retries and verify weight files by SHA-256.

Exits non-zero unless every requested safetensors file matches the Hub's recorded hash.
usage: download_verify.py <repo>@<40-char revision> ...
"""
import hashlib
import sys
import time

from huggingface_hub import HfApi, snapshot_download

ATTEMPTS = 8


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo, revision):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return snapshot_download(repo, revision=revision)
        except Exception as exc:
            print(f'{repo}: attempt {attempt} failed: {type(exc).__name__}: {str(exc)[:160]}', flush=True)
            time.sleep(min(30 * attempt, 180))
    raise SystemExit(f'{repo}: download failed after {ATTEMPTS} attempts')


def main():
    api = HfApi()
    for item in sys.argv[1:]:
        repo, revision = item.split('@')
        folder = fetch(repo, revision)
        files = [s for s in api.model_info(repo, revision=revision, files_metadata=True).siblings
                 if s.rfilename.endswith('.safetensors')]
        if not files:
            raise SystemExit(f'{repo}: no safetensors files listed')
        for sibling in files:
            actual = sha256(f'{folder}/{sibling.rfilename}')
            if actual != sibling.lfs.sha256:
                raise SystemExit(f'{repo}/{sibling.rfilename}: hash mismatch')
        print(f'{repo}@{revision[:8]}: {len(files)} weight files verified', flush=True)
    print('ALL VERIFIED', flush=True)


main()
