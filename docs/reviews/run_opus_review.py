"""Run an explicit Opus 5.5 read-only review and preserve CLI provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('block')
    parser.add_argument('focus')
    parser.add_argument('--round',type=int,default=1)
    args=parser.parse_args()
    prompt=f'''Review the completed terrain audit implementation block {args.block}.
Read docs/ASTRA_TERRAIN_AUDIT.md and docs/AUDIT_IMPLEMENTATION.md, then inspect these files and their tests: {args.focus}.
The product priority is credible terrain and fast NN navigation down to 30m/pixel; procedural engine integration is a later phase.
Read-only review. Do not edit files or run commands. Do not spawn subagents.
Find concrete correctness, determinism, dependency/fusion, physical-unit, cache identity, bounds, memory, numerical fidelity or misleading validation problems.
Report severity P0/P1/P2, file and line, reproduction and specific proposed fix. Distinguish proven defects from missing evidence; do not certify visual quality or performance without measurements.
Evaluate fit to this block; do not demand unrelated research/other phases. Finish with ACCEPT / ACCEPT WITH LIMITATIONS / CHANGES REQUIRED and outstanding gates.
'''
    destination=ROOT/'docs'/'reviews'
    stem=f'{args.block}-opus-r{args.round}'
    (destination/f'{stem}-prompt.txt').write_text(prompt,encoding='utf-8')
    started=time.time()
    result=subprocess.run(['claude','-p','--model','claude-opus-5-5','--effort','high',
        '--tools','Read,Glob,Grep','--allowedTools','Read,Glob,Grep','--permission-mode','dontAsk',
        '--no-session-persistence','--output-format','json'],input=prompt,cwd=ROOT,
        encoding='utf-8',capture_output=True,timeout=1200)
    try:
        report=json.loads(result.stdout)
    except json.JSONDecodeError:
        report={'raw_output':result.stdout,'is_error':True}
    report.update(cli_returncode=result.returncode,elapsed_seconds=time.time()-started,
                  requested_model='claude-opus-5-5',requested_effort='high',stderr=result.stderr)
    (destination/f'{stem}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    review=report.get('result',report.get('raw_output',''))
    (destination/f'{stem}.md').write_text(review,encoding='utf-8')
    print(json.dumps({'review':str(destination/f'{stem}.md'),'returncode':result.returncode,
                      'is_error':report.get('is_error',False)}))
    if result.returncode or report.get('is_error'):
        raise SystemExit(1)

if __name__=='__main__':
    main()
