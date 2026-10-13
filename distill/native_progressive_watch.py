"""Sparse durable4090 progress, compaction/benchmark/seams, then Opus5.5 review."""
import argparse
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path
from distill.jobs import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-job', required=True)
    args = parser.parse_args()
    root = external_path(args.root)
    child, stopped = None, False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def execute(command, task):
        nonlocal child
        atomic_json(root/'watch-status.json', dict(status='running', task=task))
        child = subprocess.Popen([sys.executable, '-u', *command], cwd=REPO)
        code = child.wait();child = None
        if code or stopped:
            raise RuntimeError(f'{task} exited{code}; inspect retained outputs before explicit retry.')
    def plate(checkpoint, step, coarse, label='progressive160'):
        stem = DATA/'_base'/f'base-{label}-{step:06d}-{coarse}'
        if not (stem.with_suffix('.json').exists() and stem.with_suffix('.png').exists()):
            execute(['-m', 'distill.native_progressive_evaluate', '--checkpoint', str(checkpoint),
                     '--coarse', coarse, '--label', label], f'plate:{step}:{coarse}:{label}')
    try:
        for step in range(0, 20001, 2000):
            checkpoint = root/'anneal'/f'step{step:06d}.pt'
            while not checkpoint.exists() and not stopped:
                job = state(DATA/'jobs'/args.train_job)
                if not job['live'] and job['status'] in ('complete', 'failed'):
                    raise RuntimeError(f'Trainer ended before checkpoint{step}: {job["status"]}')
                atomic_json(root/'watch-status.json', dict(status='waiting', checkpoint_step=step))
                time.sleep(60)
            if stopped:
                return
            for coarse in ('coarse4', 'coarse2'):
                plate(checkpoint, step, coarse)
        # Do not benchmark until the trainer has finished saving/releasing CUDA.
        while state(DATA/'jobs'/args.train_job)['live'] and not stopped:
            time.sleep(60)
        job = state(DATA/'jobs'/args.train_job)
        if job['status'] != 'complete' or job['returncode'] != 0:
            raise RuntimeError('A completed first stage is required before the review.')
        compact = root/'compact'
        execute(['-m', 'distill.native_progressive_compact', '--checkpoint',
                 str(root/'anneal/step020000.pt'), '--output', str(compact)], 'compact160')
        for coarse in ('coarse4', 'coarse2'):
            plate(compact/'step020000.pt', 20000, coarse, 'progressive160compact')
        bench = DATA/'bench/native-progressive160-stage1-4090.json'
        execute(['-m', 'distill.benchmark_native_runtime', '--checkpoint', str(compact/'step020000.pt'),
                 '--output', str(bench), '--batches', '1,16', '--iterations', '20',
                 '--note', 'Stage1 trainer finished; 4090 Windows display shared, live app on5090. BF16 native teacher/compact160, same embeddings.'], 'benchmark160')
        seams = DATA/'eval/native-progressive160-stage1-cold-seams-4090'
        captures = [DATA/'crops/native-mixed-scenes-v2'/f'val-{index:04d}-coarse4.npz' for index in (685, 689, 696, 697)]
        captures += [DATA/'crops/native-wet-mountains-v1'/f'val-{index:04d}-coarse4.npz' for index in (6, 8, 10)]
        execute(['-m', 'distill.check_physical_seams', '--native-base', str(compact/'step020000.pt'),
                 '--conditioning-coarse', str(REPO/'distill/weights/coarse4.pt'),
                 '--conditioning-decoder', str(REPO/'distill/weights/decoder.pt'),
                 '--validation-scenes', *[str(path) for path in captures], '--lods', '3', '0',
                 '--no-preview', '--output', str(seams)], 'cold-seams160')
        lines = ['# Première étape de réduction progressive192 vers160', '',
                 '20 000 mises à jour terminées, trois blocs par niveau et deux passes conservés.',
                 'Teacher original comme cible ; B75 publié reste la référence de produit.',
                 'Ce palier160 est intermédiaire : cible finale128 et gain2× mesuré, pas une promesse.', '',
                 '## Preuves et code', '',
                 f'- Contrat : {root}/anneal/contract.json',
                 f'- Logs : {root}/anneal/train.jsonl',
                 f'- Préparation : {root}/preparation-v2/proof.json',
                 f'- Compactage réel : {compact}/audit.json',
                 f'- Benchmark : {bench}', f'- Raccords : {seams}/report.json',
                 '- Code : distill/native_progressive_width.py, native_progressive_train.py, native_progressive_prepare.py, native_progressive_compact.py, native_progressive_runtime.py.',
                 '- Évaluations : distill/native_progressive_evaluate.py ; teacher BF16 recalculé sur le même GPU/bruit, decoder original.', '',
                 '## Progression EMA', '',
                 '| Étape | Coarse | MAE LOD3 (m) | MAE LOD0 (m) | biais absolu LOD0 (m) |',
                 '|---|---|---:|---:|---:|']
        for step in range(0, 20001, 2000):
            for coarse in ('coarse4', 'coarse2'):
                path = DATA/'_base'/f'base-progressive160-{step:06d}-{coarse}.json'
                report = json.loads(path.read_text())
                mean = lambda lod, key, absolute=False:statistics.mean(abs(row[key]) if absolute else row[key] for row in report['rows'] if row['lod'] == lod)
                lines.append(f'| {step} | {coarse} | {mean(3, "mae_m"):.3f} | {mean(0, "mae_m"):.3f} | {mean(0, "height_bias_m", True):.3f} |')
        lines += ['', 'Les planches et CSV/JSON/TXT se trouvent en vrac dans '+str(DATA/'_base')+'.',
                  'Noms : base-progressive160-<step>-<coarse>, puis base-progressive160compact-020000-<coarse>.',
                  'Ce sont des champs natifs32/256px agrandis ; les planches finales512px de la version retenue viendront ensuite.', '',
                  '## Précautions pour interpréter', '',
                  'Les logs tous les100pas coïncident avec la strate auxiliaire montagne humide ; le sampler réel tourne sur cinq catégories et trois coarses tous les quatre pas.',
                  'Les kernels MP compactés sont contrôlés exactement ; les différences BF16 dues aux dimensions des opérations sont mesurées séparément.',
                  'Le benchmark indique des forwards isolés, pas le temps navigateur ou une vue complète.',
                  'La réduction n’est pas confondue avec un QAT NF4 : ce premier palier calcule en BF16 et garde les embeddings768/bruit192.',
                  'Programme restant : récupération en160, revue, puis réduction vers128 et nouvelle récupération. Examiner la prochaine recette avant de lancer son palier.']
        report_path = root/'STAGE1_REPORT.md'
        report_path.write_text('\n'.join(lines)+'\n')
        execute(['-m', 'distill.opus_stage_review', '--report', str(report_path),
                 '--output', str(DATA/'review/opus-progressive160-stage1')], 'opus5.5-stage1-review')
        atomic_json(root/'watch-status.json', dict(status='complete', review=str(DATA/'review/opus-progressive160-stage1/OPUS_REVIEW.md'),
                                                  next='Analyze full Opus review before recovery stage.'))
    except Exception as exc:
        atomic_json(root/'watch-status.json', dict(status='interrupted' if stopped else 'failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
