"""Freeze existing native VAL/FINAL capture coordinates for physical evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from distill.common import HOLDOUT_SEEDS, REVISION, atomic_json, source_digest


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def capture_site(path):
    path=Path(path).expanduser().resolve()
    with np.load(path,allow_pickle=False) as sample:
        metadata=json.loads(str(sample['metadata']))
    split,seed=metadata.get('split'),metadata.get('seed')
    if (split not in ('val','final') or metadata.get('teacher_revision')!=REVISION
            or metadata.get('native_size')!=128):
        raise ValueError('Pinned native128 VAL/FINAL captures required; TRAIN is excluded.')
    if (split=='final')!=(seed in HOLDOUT_SEEDS):
        raise ValueError('Reserved-world / split mismatch.')
    x,y=metadata['x'],metadata['y']
    if not np.isfinite([x,y]).all() or x%32 or y%32:
        raise ValueError('Finite coordinates on the native32 grid required.')
    bank_path=path.parent/'manifest.json'
    bank=json.loads(bank_path.read_text())
    records=[r for r in bank['scenes'] if r['file']==path.name]
    if (bank['revision']!=REVISION or len(records)!=1 or
            any(metadata.get(k)!=v for k,v in records[0]['metadata'].items())):
        raise ValueError('Capture must match its pinned teacher bank.')
    record=records[0]
    return dict(name='native-'+split+'-'+path.stem,seed=seed,profile=metadata['profile'],
        x=(x+64)*240,y=(y+64)*240,kind=record['case_kind'],split=split,
        source_scene=str(path)),dict(path=str(path),sha256=digest(path),
        bank_manifest=str(bank_path),bank_sha256=digest(bank_path))


def freeze(paths,output):
    sites=[];receipts=[]
    for path in paths:
        site,receipt=capture_site(path)
        if any(s['name']==site['name'] for s in sites):raise ValueError('Duplicate native capture.')
        sites.append(site);receipts.append(receipt)
    if not sites:raise ValueError('Native capture bank is empty.')
    if Path(output).exists():raise ValueError('Preserve the frozen site manifest.')
    atomic_json(output,dict(schema='native-captures-v1',frozen=True,revision=REVISION,
        teacher_sources=source_digest(),native_captures=receipts,sites=sites))


def load_sites(path):
    manifest=json.loads(Path(path).read_text())
    if (manifest.get('schema')!='native-captures-v1' or not manifest.get('frozen')
            or manifest.get('revision')!=REVISION or manifest.get('teacher_sources')!=source_digest()):
        raise ValueError('Frozen native capture provenance differs.')
    sites=[]
    for receipt in manifest['native_captures']:
        site,actual=capture_site(receipt['path'])
        if actual!=receipt:raise ValueError('Native capture or teacher bank changed.')
        if any(s['name']==site['name'] for s in sites):raise ValueError('Duplicate native capture.')
        sites.append(site)
    if not sites or sites!=manifest['sites']:raise ValueError('Native site coordinates changed.')
    return sites
