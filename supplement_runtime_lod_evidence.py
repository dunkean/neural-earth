"""Add current CPU provenance checks without changing historical GPU receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supplement(directory):
    root=Path(__file__).resolve().parent
    labels=('baseline','batched-inputs','final','paired-before-1','paired-final-1',
            'paired-final-2','paired-before-2','progression-baseline','progression-final')
    rows=[]
    for label in labels:
        receipt_path=directory/(label+'.json')
        array_path=directory/(label+'.npz')
        receipt=json.loads(receipt_path.read_bytes())
        reference_label='progression-baseline' if label.startswith('progression-') else 'baseline'
        reference_path=directory/(reference_label+'.npz')
        reference_receipt=directory/(reference_label+'.json')
        self_comparison=array_path.resolve()==reference_path.resolve()
        with np.load(array_path) as arrays,np.load(reference_path) as reference:
            comparisons={name:dict(byte_exact=arrays[name].shape==reference[name].shape and
                                  arrays[name].dtype==reference[name].dtype and
                                  arrays[name].tobytes(order='C')==reference[name].tobytes(order='C'))
                         for name in arrays.files}
        samples=receipt['samples']
        unique={(sample['seed'],sample['lod'],sample['tx'],sample['ty']) for sample in samples}
        source_matches={}
        for name,expected in receipt.get('source_sha256',{}).items():
            filename=name if name.endswith('.py') else name+'.py'
            candidates=[directory/'before'/filename,root/filename]
            source_matches[name]=[str(path.resolve()) for path in candidates if path.exists() and sha(path)==expected]
        row=dict(label=label,historical_receipt_path=str(receipt_path.resolve()),
                 historical_receipt_sha256=sha(receipt_path),array_path=str(array_path.resolve()),
                 array_sha256=sha(array_path),cpu_recomparison_reference=dict(path=str(reference_path.resolve()),
                 sha256=sha(reference_path),label=reference_label,receipt_path=str(reference_receipt.resolve()),
                 receipt_sha256=sha(reference_receipt)),self_comparison=self_comparison,
                 independent_reference_comparison=not self_comparison,comparisons=comparisons,
                 all_arrays_byte_exact=all(value['byte_exact'] for value in comparisons.values()),
                 coverage=dict(total_footprints=len(samples),unique_site_lod_results=len(unique),
                               repeated_site_lod_results=len(samples)-len(unique)),
                 historical_source_sha256=receipt.get('source_sha256',{}),matching_available_source_snapshots=source_matches,
                 method_scope_correction='Terrain server stopped and no concurrent Terrain NN worker; desktop graphics load uncontrolled. Historical wording is retained in the unmodified receipt.')
        summaries={}
        for lod in {sample['lod'] for sample in samples}:
            values=[sample['seconds'] for sample in samples if sample['lod']==lod and sample.get('phase')=='runtime-warm-new-NN-footprint']
            if values:
                summaries[str(lod)]=dict(sample_count=len(values),median_seconds=float(np.median(values)),
                                         max_seconds=max(values),min_seconds=min(values))
        row['short_sample_statistics']=summaries
        if label.startswith('paired-'):
            row['paired_timing_harness_archive_available']=bool(source_matches.get('benchmark_runtime_lod.py'))
            row['paired_timing_protocol_reconstructable_from_archives']=False
        rows.append(row)
    all_array_recomparisons_byte_exact=all(row['all_arrays_byte_exact'] for row in rows)
    result=dict(method='Current CPU byte recomparison and file hashing only; no GPU execution and no historical receipt rewrite.',
                limitations=[
                    'These reference identities describe this CPU recomparison; older GPU receipts did not record reference provenance.',
                    'Matching source snapshots are content matches, not retroactive proof of actual imported module paths.',
                    'Baseline and progression-baseline compare to themselves, verify artifact identity only, and are excluded from independent output-equality counts.',
                    'The exact intermediate harness source for the four paired runs was not archived; its execution and instrumentation protocol cannot be reconstructed reliably from available source snapshots.',
                    'Historical p95_seconds fields are interpolated quantiles of very small samples; this supplement reports median/min/max.',
                    'The hardened benchmark require-exact flag has not been exercised by a new GPU run.'
                ],supplement_source_sha256=sha(Path(__file__)),
                test_source_sha256={name:sha(root/name) for name in ('test_runtime_latent_batch.py','test_runtime_benchmark.py')},
                test_execution='Not performed by this supplement; source hashes alone are not test results.',
                independent_array_comparison_count=sum(len(row['comparisons']) for row in rows if not row['self_comparison']),
                self_array_comparison_count=sum(len(row['comparisons']) for row in rows if row['self_comparison']),
                all_array_recomparisons_byte_exact=all_array_recomparisons_byte_exact,receipts=rows)
    destination=directory/'cpu-evidence-supplement.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(destination),receipts=len(rows),
                         all_array_recomparisons_byte_exact=all_array_recomparisons_byte_exact)))
    if not all_array_recomparisons_byte_exact:
        raise ValueError('One or more saved-array CPU recomparisons differ; see written supplement')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',default='E:/TerrainDiffusionRuntime/runtime-lod-optimization')
    supplement(Path(parser.parse_args().directory).resolve())
