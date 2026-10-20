"""Install the three immutable original Terrain Diffusion models outside Git.

The five selected distilled models are already supplied in distill/weights.
Use --check for an entirely offline installation/provenance check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.terrain_paths import HF_HUB_CACHE,model_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Check installed files without accessing the network.')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'distill/weights/manifest.json').read_text())
    for model in manifest['exports'].values():
        path=ROOT/'distill/weights'/model['file']
        if not path.is_file() or path.stat().st_size!=model['bytes']:
            raise SystemExit('Missing bundled inference weights: '+str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=model['sha256']:
            raise SystemExit('Bundled model changed: '+str(path))
    revision=manifest['teacher_revision']
    if not args.check:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=manifest['teacher_model'],revision=revision,cache_dir=str(HF_HUB_CACHE),
            allow_patterns=['config.json','coarse_model/*','base_model/*','decoder_model/*','LICENSE*','README.md'])
    snapshot=model_snapshot(revision)
    required=['config.json']+[f'{stage}/{name}' for stage in manifest['originals']['stages']
        for name in ['config.json','diffusion_pytorch_model.safetensors']]
    missing=[name for name in required if not (snapshot/name).is_file()]
    if missing:
        raise SystemExit('Original models not installed: '+', '.join(missing)+'. Run python tools/download_models.py')
    print('Five selected inference models verified; three original models installed at '+str(snapshot))
    print('Pinned upstream revision: '+revision)


if __name__=='__main__':
    main()
