#!/usr/bin/env python3
"""Download the colleague's generic-kd-v1 checkpoint ZIPs and verify each SHA-256.

Reads the private link list from a file and never prints a URL. usage: fetch_colleague.py <links.md> <out dir>
"""
import hashlib
import re
import sys
import time
import urllib.request
from pathlib import Path

doc, out = Path(sys.argv[1]).read_text(encoding='utf-8'), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
expected = dict(re.findall(r'- 파일: `([^`]+\.zip)`(?:.|\n)*?- ZIP SHA256: `([0-9a-f]{64})`', doc))
links = {Path(u.split('?')[0]).name: u for u in re.findall(r'\]\((https://[^)]+)\)', doc)}
print(f'{len(expected)} checkpoint ZIPs with SHA-256, {len(links)} links', flush=True)
missing = [n for n in expected if n not in links]
if missing:
    raise SystemExit(f'no link for {missing}')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


failed = []
for name, url in links.items():
    target = out / name
    want = expected.get(name)
    if target.exists() and (want is None or sha256(target) == want):
        print(f'ok (cached) {name}', flush=True)
        continue
    for attempt in range(1, 6):
        try:
            partial = target.with_suffix(target.suffix + '.part')
            with urllib.request.urlopen(url, timeout=120) as response, open(partial, 'wb') as stream:
                while chunk := response.read(1 << 22):
                    stream.write(chunk)
            if want and sha256(partial) != want:
                raise ValueError('SHA-256 mismatch')
            partial.rename(target)
            print(f'ok {name} {target.stat().st_size} bytes' + (' sha256 verified' if want else ' (no published hash)'), flush=True)
            break
        except Exception as exc:  # the message may not contain the URL; print only the type and status
            print(f'retry {attempt} {name}: {type(exc).__name__} {getattr(exc, "code", "")}', flush=True)
            time.sleep(20 * attempt)
    else:
        failed.append(name)
print('ALL VERIFIED' if not failed else f'FAILED {failed}', flush=True)
sys.exit(1 if failed else 0)
