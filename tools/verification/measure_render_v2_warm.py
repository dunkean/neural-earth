"""Incremental, CPU-only audit of native DEMs produced by the warm viewer.

Uses select_render_v2_reference.measure_native, the same implementation as
--measure --world-info. Never requests a tile, starts inference or edits a
selection. Working source labels do not imply visual acceptance.
"""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--world-info', required=True)
parser.add_argument('--warm-dir', required=True)
parser.add_argument('--watch', action='store_true')
parser.add_argument('--interval', type=float, default=15)
args = parser.parse_args()

# The canonical helper consumes --world-info from the same argv.
import select_render_v2_reference as ref

ROOT = ref.qa.ROOT
BASE = Path(args.warm_dir).resolve()
MANIFEST = json.loads((ROOT/'output/render-v2-reference/manifest.json').read_text(encoding='utf-8'))
assert MANIFEST['worldIdentity'] == ref.runtime_world['world_identity']
FAMILIES = dict(AF=['AF', 'AM'], CFB=['CFB', 'CFC'], BSK=['BSK', 'BSH'],
                ARID=['BWH', 'BWK'], DFC=['DFC', 'DSC'], POLAR=['ET', 'EF'])
RELIEFS = ['plain', 'rolling', 'plateau', 'mountains']
SITES = {s['id']: s for s in MANIFEST['sites']}
STATE_FILE = BASE/'native-incremental-metrics.json'
state = json.loads(STATE_FILE.read_text(encoding='utf-8')) if STATE_FILE.exists() else dict(
    label='CURRENT NATIVE CACHE AUDIT ONLY; no final render or visual acceptance',
    worldIdentity=MANIFEST['worldIdentity'], measurements=[], errors={})
assert state['worldIdentity'] == MANIFEST['worldIdentity']


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def audit(completed):
    measurements = {r['id']: r for r in state['measurements']}
    cells = []
    for family, codes in FAMILIES.items():
        for relief in RELIEFS:
            attempts = []
            for site in SITES.values():
                if site['biomeFamily'] != family or site.get('requestedRelief') != relief:
                    continue
                result = measurements.get(site['id'])
                fractions = result['stats']['biome_fractions_land'] if result else None
                share = sum(fractions.get(c, 0) for c in codes) if fractions is not None else None
                attempts.append(dict(id=site['id'], candidate=any(a['origin']=='candidate' for a in site['aliases']),
                    center_m=site['center_m'], nativeMeasured=result is not None,
                    nativeRelief=result['relief'] if result else None, biomeFractions=fractions,
                    familyDominance=share, strictBiomeMatched=share is not None and share>=.8,
                    representativeBiomeMatched=share is not None and share>=.7,
                    nativeReliefMatched=result is not None and result['relief']==relief,
                    visualAcceptance='not-assessed-working-cache-only'))
            strict = [r['id'] for r in attempts if r['nativeReliefMatched'] and r['strictBiomeMatched']]
            mixed = [r['id'] for r in attempts if r['nativeReliefMatched'] and r['representativeBiomeMatched']]
            candidates = [r for r in attempts if r['candidate']]
            cells.append(dict(biomeFamily=family, requestedRelief=relief, attempts=attempts,
                strictNativeMatched=strict, representativeNativeMatched=mixed,
                candidateCount=len(candidates), candidateMeasured=sum(r['nativeMeasured'] for r in candidates),
                status='native-match-visual-pending' if strict else 'mixed-only-strict-gap' if mixed else
                    'native-mismatch-gap' if any(r['nativeMeasured'] for r in attempts) else 'pending-native'))
    actual_cells = []
    for family, codes in FAMILIES.items():
        for relief in RELIEFS:
            actual = []
            for native in measurements.values():
                if native['relief'] != relief:
                    continue
                fractions = native['stats']['biome_fractions_land']
                share = sum(fractions.get(c, 0) for c in codes)
                if share<=0:
                    continue
                site = SITES[native['id']]
                source_share = sum(fractions.get(c, 0) for c in FAMILIES.get(site['biomeFamily'], []))
                actual.append(dict(id=site['id'], center_m=site['center_m'], nativeRelief=relief,
                    actualFamily=family, familyDominance=share, biomeFractions=fractions,
                    strictBiomeMatched=share>=.8, representativeBiomeMatched=share>=.7,
                    sourceFamily=site['biomeFamily'], sourceReliefTarget=site.get('requestedRelief'),
                    failedSourceTarget=site.get('requestedRelief') is not None and (
                        native['relief']!=site['requestedRelief'] or source_share<.8),
                    sourceProvenance=site['aliases'], visualAcceptance='pending-human-review'))
            strict = [r['id'] for r in actual if r['strictBiomeMatched']]
            mixed = [r['id'] for r in actual if r['representativeBiomeMatched']]
            actual_cells.append(dict(biomeFamily=family, actualRelief=relief, attempts=actual,
                strictNativeMatched=strict, representativeNativeMatched=mixed,
                visuallyAccepted=[], status='actual-native-strict-visual-pending' if strict else
                'actual-native-mixed-visual-pending' if mixed else 'no-qualifying-native-sample-yet'))
    result = dict(label=state['label'], worldIdentity=state['worldIdentity'], updatedAt=time.time(),
        completedWarmSites=completed, nativeMeasuredSites=len(measurements), requiredCells=24,
        strictNativeMatchedCells=sum(bool(c['strictNativeMatched']) for c in cells),
        representativeNativeMatchedCells=sum(bool(c['representativeNativeMatched']) for c in cells),
        visuallyAcceptedCells=0, cells=cells,
        requestedTargetMatches=dict(strictNativeMatchedCells=sum(bool(c['strictNativeMatched']) for c in cells),
            representativeNativeMatchedCells=sum(bool(c['representativeNativeMatched']) for c in cells),
            definition='Source requested family AND relief matched; failures keep their original labels.'),
        actualNativeCoverage=dict(strictNativeMatchedCells=sum(bool(c['strictNativeMatched']) for c in actual_cells),
            representativeNativeMatchedCells=sum(bool(c['representativeNativeMatched']) for c in actual_cells),
            visuallyAcceptedCells=0, cells=actual_cells,
            definition='Measured morphology and native biome fractions, independently of source target. Visual review still required.'),
        candidatePlan=[dict(biomeFamily=c['biomeFamily'], relief=c['requestedRelief'],
            selected=c['candidateCount'], measured=c['candidateMeasured'],
            strictNativeMatches=[r['id'] for r in c['attempts'] if r['candidate'] and r['nativeReliefMatched'] and r['strictBiomeMatched']])
            for c in cells if c['candidateCount']],
        limitations=['No request, inference, capture, manifest change or relabelling.',
            'Native morphology and modeled biome classification are measurements, not visual acceptance.',
            'Unclassified morphology remains null; strict80% and representative70% stay separate.'])
    write(BASE/'native-coverage.json', result)
    rows = '\n'.join('|'+ '|'.join([c['biomeFamily'], c['requestedRelief'],
        ', '.join(c['strictNativeMatched']) or '—', ', '.join(c['representativeNativeMatched']) or '—',
        str(c['candidateMeasured'])+'/'+str(c['candidateCount']), c['status']])+'|' for c in cells)
    actual_rows = '\n'.join('|'+ '|'.join([c['biomeFamily'], c['actualRelief'],
        ', '.join(c['strictNativeMatched']) or '—', ', '.join(c['representativeNativeMatched']) or '—',
        ', '.join(r['id']+' (cible '+str(r['sourceReliefTarget'])+')' for r in c['attempts']
            if r['representativeBiomeMatched'] and r['failedSourceTarget']) or '—'])+'|' for c in actual_cells)
    (BASE/'native-coverage.md').write_text(
        '# Audit natif incrémental du préchauffage\n\n'
        f"{len(measurements)} centres mesurés ; {result['strictNativeMatchedCells']}/24 cases strictes, "
        f"{result['representativeNativeMatchedCells']}/24 avec transitions ≥70 %. Aucune acceptation visuelle.\n\n"
        '|Famille|Relief demandé|Natif strict ≥80 %|Natif représentatif ≥70 %|Candidats mesurés/prévus|État|\n'
        '|---|---|---|---|---|---|\n'+rows+'\n\n'
        '## Couverture des reliefs réellement mesurés\n\n'
        f"{result['actualNativeCoverage']['strictNativeMatchedCells']}/24 cases strictes ; "
        f"{result['actualNativeCoverage']['representativeNativeMatchedCells']}/24 avec transitions ≥70 %. "
        'Cette seconde lecture ne dépend pas du relief source demandé. Les échecs de cible restent visibles ; '
        'une réutilisation pour valider le rendu exige encore un examen visuel. Aucun centre ne change de nom.\n\n'
        '|Famille native|Relief mesuré|Natif strict ≥80 %|Natif représentatif ≥70 %|Cible source échouée ou différente|\n'
        '|---|---|---|---|---|\n'+actual_rows+'\n', encoding='utf-8')
    for milestone in [20, 40, 60]:
        target = BASE/f'native-coverage-after-{milestone}.json'
        if completed>=milestone and not target.exists():
            write(target, result)
    return result


while True:
    try:
        report = json.loads((BASE/'report.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        if not args.watch:
            raise
        time.sleep(args.interval)
        continue
    assert report['selection']['world_identity'] == state['worldIdentity']
    ready = [r for r in report['results'] if r.get('ok') and len(r['frames'])==1]
    known = {r['id'] for r in state['measurements']}
    for entry in ready:
        if entry['id'] in known:
            continue
        site = SITES[entry['id']]
        assert entry['selection']['center_m'] == site['center_m']
        try:
            measured = ref.measure_native(dict(entry, site=site))
            state['measurements'].append(measured)
            state['errors'].pop(entry['id'], None)
            write(STATE_FILE, state)
            print(json.dumps(dict(id=entry['id'], nativeRelief=measured['relief'],
                measuredSites=len(state['measurements'])), ensure_ascii=False), flush=True)
        except Exception as error:
            state['errors'][entry['id']] = str(error)
            print(json.dumps(dict(id=entry['id'], error=str(error))), flush=True)
    write(STATE_FILE, state)
    coverage = audit(len(ready))
    print(json.dumps(dict(completed=len(ready), nativeMeasured=len(state['measurements']),
        strictCells=coverage['strictNativeMatchedCells'], representativeCells=coverage['representativeNativeMatchedCells'],
        actualStrictCells=coverage['actualNativeCoverage']['strictNativeMatchedCells'],
        actualRepresentativeCells=coverage['actualNativeCoverage']['representativeNativeMatchedCells'],
        errors=state['errors'])), flush=True)
    if not args.watch or len(report['results'])>=len(SITES):
        break
    time.sleep(args.interval)
