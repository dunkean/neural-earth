"""Original full-pipeline VAL with a self-contained low-rank compressed base."""
import argparse
from pathlib import Path
import sys
import json

from distill.common import atomic_json
from distill.native_weight_quant import file_sha
from distill.tile40_native3_lora import load_residual
from distill import tile40_native3_evaluate as evaluator


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--packed',type=Path,required=True)
    args,_=parser.parse_known_args()
    evaluator.load_packed=load_residual
    evaluator.main()
    path=args.output/'report.json'
    report=json.loads(path.read_text())
    report.update(packed_bytes=sum(p.stat().st_size for p in args.packed.rglob('*') if p.is_file()),
        loader_producer_sha256=file_sha(Path(__file__)),
        residual_model_producer_sha256=file_sha(Path(__file__).with_name('tile40_native3_lora.py')),
        all_base_weights_included=True)
    if report['packed_bytes']>=100000000:raise ValueError('Complete residual base exceeds100MB.')
    atomic_json(path,report)


if __name__=='__main__':main()
