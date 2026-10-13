"""Persist a full read-only Opus5.5 review of a completed training stage."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import uuid

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_weight_quant import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the existing review; no automatic duplicate.')
    output.mkdir(parents=True)
    session = str(uuid.uuid4())
    companion = args.report.parent/'ADDITIONAL_EVIDENCE.md'
    additional = [dict(path=str(companion), sha256=file_sha(companion))] if companion.exists() else []
    contract = dict(report=str(args.report), report_sha256=file_sha(args.report),
                    requested_model='claude-opus-5-5', effort='xhigh', session_id=session,
                    additional_evidence=additional,
                    scope='Full read-only stage review; no training or model changes')
    atomic_json(output/'contract.json', contract)
    prompt = f'''Lis le rapport de fin d'étape {args.report}, puis le code et les preuves auxquels il renvoie.
Lis également les preuves complémentaires si présentes : {json.dumps(additional, ensure_ascii=False)}.
L'utilisateur demande ton avis après chaque grande étape pour accélérer le base tout en conservant un look correct.
Sa priorité est une forte accélération de toute la chaîne ; il accepte un peu de bruit si le relief reste correct.
N'attends pas la perfection numérique pour recommander un candidat utile. Distingue le bruit tolérable d'une perte
des vallées, de l'organisation du relief ou des côtes, et propose des compromis qualité/vitesse concrets.
Fais une revue complète, pas une version condensée. Distingue les faits des hypothèses.
Évalue la récupération du relief et des vallées humides, le grain des plaines, les côtes, le biais d'altitude,
la continuité prévue entre tuiles, et le gain de vitesse effectivement mesuré.
Audite la réduction progressive des canaux, les normalisations MP, skips, QKV et la supervision des deux passes.
Explique les défauts observés, les risques ou erreurs de protocole, et recommande concrètement la prochaine étape :
récupération prolongée en160, modification ciblée, ou passage ultérieur vers128. Ne conclue pas à une limite de
capacité à partir d'une courte récupération. Le base B75 publié est la référence de produit, le teacher original la cible.
Pour les paliers suivants, considère aussi moins de blocs et une passe réellement distillée vers la sortie du teacher
à deux passes ; le gain2× du réseau plus étroit est un premier objectif intermédiaire, pas la limite du programme.
Lis les résultats utiles et quelques planches. Ne lance aucun calcul, n'écris aucun fichier et ne change rien.
Donne l'avis final complet dans ta réponse, avec les chemins des preuves utilisées.'''
    (output/'prompt.txt').write_text(prompt)
    command = ['claude', '-p', '--model', contract['requested_model'], '--effort', contract['effort'],
               '--max-turns', '60', '--output-format', 'stream-json', '--verbose', '--session-id', session,
               '--allowedTools', 'Read', 'Glob', 'Grep', '--tools', 'Read', 'Glob', 'Grep',
               '--permission-mode', 'dontAsk', '--strict-mcp-config', '--add-dir', str(DATA)]
    atomic_json(output/'status.json', dict(status='running', started_at=time.time(), **contract))
    child, result, actual = None, None, None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    with (output/'stderr.log').open('w') as stderr, (output/'cli.jsonl').open('w') as transcript:
        child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=stderr, text=True, bufsize=1)
        child.stdin.write(prompt);child.stdin.close()
        for line in child.stdout:
            transcript.write(line);transcript.flush()
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'system' and event.get('subtype') == 'init':
                actual = event.get('model')
                print(json.dumps(dict(actual_model=actual, session_id=session)), flush=True)
            if event.get('type') == 'result':
                result = event
        code = child.wait()
    success = code == 0 and result and not result.get('is_error') and actual and contract['requested_model'] in actual
    if success:
        (output/'OPUS_REVIEW.md').write_text(result.get('result', ''))
    atomic_json(output/'status.json', dict(status='complete' if success else 'failed', actual_model=actual,
        actual_model_matches=bool(actual and contract['requested_model'] in actual), returncode=code,
        finished_at=time.time(), session_id=session))
    if not success:
        raise RuntimeError('Opus stage review failed; inspect retained transcript before explicit retry.')


if __name__ == '__main__':
    main()
