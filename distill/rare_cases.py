"""Teacher-only selection of held-out plains and coast stress cases.

Survey conditioning first, measure the proposed sites with the unchanged teacher,
then freeze a bank using teacher relief alone. Never read a student to choose sites.
The original seven-site acceptance bank remains mandatory and unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter, binary_erosion, distance_transform_edt, label

from distill.common import HOLDOUT_SEEDS, REPO, atomic_json, external_path, source_digest

KINDS = ('desert-plain', 'wet-complex-coast', 'low-plain-sea', 'temperate-plain')
SPAN = 256 * 30 * 8
WORLDS = ((42, 'orogen'), (101, 'natural'), (202, 'terrestrial-earthlike'),
          (303, 'natural'), (404, 'terrestrial-archipelago'))


def load_sites(path):
    manifest = json.loads(Path(path).read_text())
    sites = manifest['sites']
    names = set()
    if not sites:
        raise ValueError('The held-out site bank is empty.')
    for site in sites:
        if site['seed'] not in HOLDOUT_SEEDS or site['kind'] not in KINDS:
            raise ValueError('Rare cases must use reserved worlds and known case kinds.')
        if site['name'] in names or not np.isfinite([site['x'], site['y']]).all():
            raise ValueError('Site names must be unique and coordinates finite.')
        names.add(site['name'])
    if manifest['teacher_sources'] != source_digest():
        raise ValueError('Teacher sources differ from the site-selection manifest.')
    return sites


def field_stats(fields):
    e, t, _, p, _ = np.asarray(fields, np.float64)
    land = e > 0
    if not land.any():
        return None
    # Coast perimeter in sample edges, normalized by sqrt(area), excluding the
    # boundary of the image. A straight coast across a square has score near 1.
    edge = np.count_nonzero(land[1:] != land[:-1]) + np.count_nonzero(land[:, 1:] != land[:, :-1])
    return dict(land=float(land.mean()), land_height_mean=float(e[land].mean()),
                land_height_p90=float(np.quantile(e[land], .9)),
                height_std=float(e.std()), rain=float(p[land].mean()), temp=float(t[land].mean()),
                coast_complexity=float(edge / np.sqrt(e.size)))


def score(kind, stats):
    """Conditioning-only ranking; physical teacher qualification happens later."""
    if stats is None:
        return None
    s = stats
    if kind == 'desert-plain' and s['land'] > .95 and s['rain'] < 350:
        return s['height_std'] + max(0, s['land_height_mean'] - 1500) * .1
    if kind == 'temperate-plain' and s['land'] > .95 and 0 < s['temp'] < 22 and 400 < s['rain'] < 1800:
        return s['height_std'] + max(0, s['land_height_mean'] - 600) * .2
    if .15 < s['land'] < .85:
        if kind == 'wet-complex-coast' and s['rain'] > 1200:
            return -s['coast_complexity'] + s['height_std'] * .00001
        if kind == 'low-plain-sea':
            return s['land_height_p90'] + s['height_std'] * .05
    return None


def survey(output, count=8, warm_plains=False):
    from terrain_conditioning import make_conditioning_factory, WORLD_BOUNDS
    x0, y0, x1, y1 = WORLD_BOUNDS
    # One global sample per world; at 30.72 km spacing the arrays stay small.
    xs = (np.arange(np.ceil(x0 / (SPAN / 2)), np.floor(x1 / (SPAN / 2))) + .5) * (SPAN / 2)
    ys = (np.arange(np.ceil(y0 / (SPAN / 2)), np.floor(y1 / (SPAN / 2))) + .5) * (SPAN / 2)
    pools = {kind: [] for kind in KINDS}
    for seed, profile in WORLDS:
        factory = make_conditioning_factory(seed, profile)
        f = factory.sample(xs, ys).astype(np.float64)
        land = f[0] > 0
        lm = uniform_filter(land.astype(float), size=3)
        hm = uniform_filter(f[0], size=3)
        hs = np.sqrt(np.maximum(0, uniform_filter(f[0] ** 2, size=3) - hm ** 2))
        rain = uniform_filter(f[3] * land, size=3) / np.maximum(lm, 1e-6)
        temp = uniform_filter(f[1] * land, size=3) / np.maximum(lm, 1e-6)
        masks = {
            'desert-plain': (lm > .95) & (rain < 350),
            'temperate-plain': (lm > .95) & (rain > 400) & (rain < 1800) & (temp > 0) & (temp < 22),
            'wet-complex-coast': (lm > .1) & (lm < .9) & (rain > 1200),
            'low-plain-sea': (lm > .1) & (lm < .9),
        }
        if warm_plains:
            masks['desert-plain'] &= (temp >= 15) & (rain < 250)
            masks['temperate-plain'] &= temp >= 10
        for kind, mask in masks.items():
            mask[:2] = mask[-2:] = False
            mask[:, :2] = mask[:, -2:] = False
            rough_score = hs if 'plain' in kind and kind != 'low-plain-sea' else np.abs(hm) + hs
            indices = np.flatnonzero(mask)
            indices = indices[np.argsort(rough_score.ravel()[indices], kind='stable')[:600]]
            seen = set()
            for index in indices:
                iy, ix = np.unravel_index(index, mask.shape)
                tx, ty = int(np.floor(xs[ix] / SPAN)), int(np.floor(ys[iy] / SPAN))
                if (tx, ty) in seen:
                    continue
                seen.add((tx, ty))
                cx, cy = (tx + .5) * SPAN, (ty + .5) * SPAN
                offsets = np.linspace(-SPAN / 2, SPAN / 2, 17)
                stats = field_stats(factory.sample(cx + offsets, cy + offsets))
                value = score(kind, stats)
                archetype = ('warm-arid' if kind == 'desert-plain' else 'mild-temperate'
                             if kind == 'temperate-plain' else None) if warm_plains else None
                if archetype == 'warm-arid' and stats is not None and not (stats['temp'] >= 15 and stats['rain'] < 250):
                    value = None
                if archetype == 'mild-temperate' and stats is not None and stats['temp'] < 10:
                    value = None
                if value is not None:
                    prefix = kind + ('-' + archetype if archetype else '')
                    pools[kind].append(dict(name=f'{prefix}-{seed}-{tx}-{ty}', kind=kind, seed=seed,
                                           climate_archetype=archetype,
                                           profile=profile, x=cx, y=cy, conditioning=stats, selection_score=value))
        print(json.dumps(dict(seed=seed, profile=profile, proposed={k: len(v) for k, v in pools.items()})), flush=True)
    selected = []
    for kind, pool in pools.items():
        chosen = []
        for site in sorted(pool, key=lambda s: s['selection_score']):
            # Avoid eight neighbouring patches of the same large feature.
            if any(site['seed'] == p['seed'] and np.hypot(site['x'] - p['x'], site['y'] - p['y']) < 3 * SPAN for p in chosen):
                continue
            chosen.append(site)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise ValueError(f'Only {len(chosen)} candidates for {kind}; expand teacher-only survey.')
        selected.extend(chosen)
    atomic_json(external_path(output), dict(schema=1, frozen=False, teacher_sources=source_digest(),
                selection='conditioning-only proposals; teacher physical qualification pending', sites=selected))


def relief_stats(e, lod):
    e = np.asarray(e, np.float64)
    land = e > 0
    gy, gx = np.gradient(e, 30 * 2 ** lod)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    edge = np.count_nonzero(land[1:] != land[:-1]) + np.count_nonzero(land[:, 1:] != land[:, :-1])
    def largest_fraction(mask):
        components, count = label(mask)
        return float(np.bincount(components.ravel())[1:].max() / mask.sum()) if count else 0.
    return dict(land=float(land.mean()), land_height_mean=float(e[land].mean()) if land.any() else 0.,
                land_height_p90=float(np.quantile(e[land], .9)) if land.any() else 0.,
                land_height_std=float(e[land].std()) if land.any() else 0.,
                slope_land_p90=float(np.quantile(slope[land], .9)) if land.any() else 0.,
                coast_complexity=float(edge / np.sqrt(e.size)),
                land_largest_component_fraction=largest_fraction(land),
                sea_largest_component_fraction=largest_fraction(~land))


def qualifies(kind, stats, conditioning, archetype=None):
    s, c = stats, conditioning
    if archetype == 'warm-arid' and not (c['temp'] >= 15 and c['rain'] < 250):
        return False
    if archetype == 'mild-temperate' and not (10 <= c['temp'] < 22):
        return False
    if kind in ('desert-plain', 'temperate-plain'):
        climate = c['rain'] < 350 if kind == 'desert-plain' else 400 < c['rain'] < 1800 and 0 < c['temp'] < 22
        return climate and s['land'] > .95 and s['land_height_std'] < 100 and s['slope_land_p90'] < 5
    if kind == 'wet-complex-coast':
        return c['rain'] > 1200 and .1 < s['land'] < .9 and s['coast_complexity'] > 2
    return (.1 < s['land'] < .9 and s['land_height_p90'] < 100 and s['slope_land_p90'] < 5 and
            s.get('land_largest_component_fraction', 0) >= .75 and
            s.get('sea_largest_component_fraction', 0) >= .75)


def freeze(proposals, evaluation, output, per_kind=1):
    sites = load_sites(proposals)
    report_path, array_path = Path(evaluation) / 'report.json', Path(evaluation) / 'arrays.npz'
    report = json.loads(report_path.read_text())
    if report['sites'] != sites or report.get('site_manifest_digest') != hashlib.sha256(Path(proposals).read_bytes()).hexdigest():
        raise ValueError('Teacher survey does not correspond to these proposals.')
    if any(name != 'reference' for name in report['variants']):
        raise ValueError('Selection requires a teacher-only report without student measurements.')
    chosen = []
    with np.load(array_path) as arrays:
        for kind in KINDS:
            candidates = []
            for site in sites:
                if site['kind'] != kind:
                    continue
                stats = {str(lod): relief_stats(arrays[f"reference|{site['name']}|{lod}"], lod) for lod in (3, 0)}
                # Plains must be flat at both scales; both views of a coast must
                # contain sea and land. An offshore LOD 0 cannot validate a coast.
                if all(qualifies(kind, stats[str(lod)], site['conditioning'], site.get('climate_archetype')) for lod in (3, 0)):
                    value = (-sum(s['coast_complexity'] for s in stats.values()) if kind == 'wet-complex-coast'
                             else sum(s['land_height_std'] for s in stats.values()))
                    candidates.append((value, site | dict(teacher_relief=stats)))
            candidates.sort(key=lambda pair: pair[0])
            if len(candidates) < per_kind:
                raise ValueError(f'{kind}: {len(candidates)} teacher-qualified cases; more proposals needed, no bank frozen.')
            chosen.extend(site for _, site in candidates[:per_kind])
    payload = dict(schema=1, frozen=True, teacher_sources=source_digest(),
                selection='fixed qualification at both LODs, ranked on teacher relief only',
                teacher_report_digest=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                teacher_arrays_digest=hashlib.sha256(array_path.read_bytes()).hexdigest(), sites=chosen)
    output = external_path(output)
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError('A frozen bank is immutable; write a separately named bank for a changed definition.')
    atomic_json(output, payload)


def augment(original, addition, output):
    base = json.loads(Path(original).read_text())
    extra = json.loads(Path(addition).read_text())
    if not base['frozen'] or not extra['frozen']:
        raise ValueError('Augment only frozen teacher-qualified banks.')
    sites = load_sites(original)
    warm = [s for s in load_sites(addition) if s.get('climate_archetype') in ('warm-arid', 'mild-temperate')]
    if {s.get('climate_archetype') for s in warm} != {'warm-arid', 'mild-temperate'}:
        raise ValueError('Both warm arid and mild temperate plains are required.')
    if {s['name'] for s in sites} & {s['name'] for s in warm}:
        raise ValueError('An augmentation must add distinct cases without replacing old sites.')
    payload = dict(schema=1, frozen=True, teacher_sources=source_digest(),
                   selection='Immutable original rare bank plus teacher-qualified warmer plains; no student-driven selection.',
                   parent_banks={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in (original, addition)}, sites=sites+warm)
    output = external_path(output)
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError('A frozen augmented bank is immutable.')
    atomic_json(output, payload)


def local_errors(candidate, reference, lod):
    """Absolute, local diagnostics: avoid losing small low coasts in global MAE."""
    r, c = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    land = r > 0
    er = np.abs(c - r)
    boundary = land != binary_erosion(land, border_value=1)
    boundary |= ~land != binary_erosion(~land, border_value=1)
    coast_band = (distance_transform_edt(~boundary) * (30 * 2 ** lod) <= 300
                  if boundary.any() else np.zeros_like(land))
    masks = {'land': land, 'low_land_0_20m': (r > 0) & (r <= 20),
             'low_land_0_100m': (r > 0) & (r <= 100), 'coast_300m': coast_band}
    result = {}
    for name, mask in masks.items():
        result[name] = dict(pixels=int(mask.sum()), mae_m=float(er[mask].mean()) if mask.any() else None,
                            p99_m=float(np.quantile(er[mask], .99)) if mask.any() else None,
                            sign_flip=float(((c > 0) != land)[mask].mean()) if mask.any() else None)
    # Coherent row/column errors reveal aligned stripes even when global PSD is
    # plausible; compare these residuals with the numerical floor on the same site.
    d = c - r
    result['axis_error'] = dict(row_mean_std_m=float(d.mean(axis=1).std()),
                               column_mean_std_m=float(d.mean(axis=0).std()))
    return result


def audit(manifest_path, evaluation, variant, output):
    from distill.evaluate import audit_cases
    manifest = json.loads(Path(manifest_path).read_text())
    sites = load_sites(manifest_path)
    if not manifest['frozen'] or {s['kind'] for s in sites} != set(KINDS):
        raise ValueError('All four teacher-qualified rare cases must be frozen before student evaluation.')
    report = json.loads((Path(evaluation) / 'report.json').read_text())
    digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if report['sites'] != sites or report.get('site_manifest_digest') != digest:
        raise ValueError('Evaluation and frozen rare bank do not match.')
    result = audit_cases(report, variant)
    with np.load(Path(evaluation) / 'arrays.npz') as arrays:
        for site in sites:
            for lod in (3, 0):
                stats = relief_stats(arrays[f"reference|{site['name']}|{lod}"], lod)
                if not qualifies(site['kind'], stats, site['conditioning'], site.get('climate_archetype')):
                    raise ValueError(f"{site['name']} LOD {lod}: teacher does not qualify for this rare category.")
        for row in result['rows']:
            name, lod = row['site'], row['lod']
            ref = arrays[f'reference|{name}|{lod}']
            candidate = local_errors(arrays[f'{variant}|{name}|{lod}'], ref, lod)
            floor = local_errors(arrays[f'fp32base|{name}|{lod}'], ref, lod)
            row['local_errors'] = candidate
            row['local_floor'] = floor
            local_pass = all(candidate[key]['mae_m'] <= floor[key]['mae_m'] and
                             candidate[key]['sign_flip'] <= floor[key]['sign_flip']
                             for key in ('land', 'low_land_0_20m', 'low_land_0_100m', 'coast_300m')
                             if candidate[key]['pixels'])
            row['checks']['local_coast_and_height'] = local_pass
            row['passed'] = row['passed'] and local_pass
    result['physical_passed'] = all(row['passed'] for row in result['rows'])
    result.update(site_manifest_digest=digest, checkpoint_digests=report.get('checkpoint_digests', {}).get(variant),
                  gpu=report.get('gpu'), teacher_sources=report.get('source_digests'),
                  accepted=False, note='Additional stress-case evidence; the original 14 cases, speed and seam gates remain required.')
    atomic_json(external_path(output), result)
    return result


def additional_evidence(report, variant, manifest_path=None, evaluation=None):
    if manifest_path is None or evaluation is None:
        return dict(passed=False, reason='Frozen rare-case evaluation not supplied.')
    sites = load_sites(manifest_path)
    if not {'warm-arid', 'mild-temperate'}.issubset({s.get('climate_archetype') for s in sites}):
        return dict(passed=False, reason='Warm arid and mild temperate plains must supplement the original rare cases.')
    # Check the saved parents too: adding warm cases must not silently discard
    # an original difficult coast or plain, even if all four kind labels remain.
    parents = json.loads(Path(manifest_path).read_text()).get('parent_banks', {})
    original_preserved = False
    for parent, digest in parents.items():
        path = Path(parent)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return dict(passed=False, reason='A frozen parent bank is missing or changed.')
        parent_sites = load_sites(path)
        if all(s.get('climate_archetype') is None for s in parent_sites):
            original_preserved = ({s['kind'] for s in parent_sites} == set(KINDS) and
                                  all(s in sites for s in parent_sites))
    if not original_preserved:
        return dict(passed=False, reason='The original rare cases must remain intact in the augmented bank.')
    from tempfile import TemporaryDirectory
    # Compute from raw arrays/report instead of trusting a separately edited
    # acceptance JSON. The temporary audit stays outside the repository.
    with TemporaryDirectory() as directory:
        result = audit(manifest_path, evaluation, variant, Path(directory) / 'audit.json')
    checks = dict(physical=result['physical_passed'], gpu=result['gpu'] == report.get('gpu'),
                  weights=result['checkpoint_digests'] == report.get('checkpoint_digests', {}).get(variant),
                  teacher_sources=result['teacher_sources'] == report.get('source_digests'))
    return dict(checks=checks, passed=all(checks.values()), evidence=result)


def coverage(dataset, output):
    """Count rare morphology in stored targets, without touching held-out worlds.

    This is a low-frequency base-target proxy, not full decoded terrain and not
    a replacement for the physical case bank. No sampling weights are changed.
    """
    from collections import Counter
    root = Path(dataset)
    counts = Counter()
    rows = []
    paths = sorted((root / 'base').glob('*.npz'))
    if not paths:
        raise ValueError('No base targets to audit.')
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as sample:
            meta = json.loads(str(sample['metadata']))
            if meta['seed'] in HOLDOUT_SEEDS:
                raise ValueError('Reserved evaluation world found in training/validation data.')
            q = sample['target'][4].astype(np.float64) * 38.6 - 31.4
            height = np.sign(q) * q ** 2
            cy, cx = meta['y'] // 32 - meta['coarse_y'], meta['x'] // 32 - meta['coarse_x']
            n = meta['crop'] // 32
            climate = sample['coarse'][2:, cy:cy+n, cx:cx+n].astype(np.float64)
            climate = climate.repeat(32, axis=1).repeat(32, axis=2)
            fields = np.concatenate((height[None], climate), axis=0)
            conditioning = field_stats(fields)
            stats = relief_stats(height, 3)
            kinds = [kind for kind in KINDS if conditioning is not None and qualifies(kind, stats, conditioning)]
            split, profile = meta['split'], meta['profile']
            counts[(split, profile, 'all')] += 1
            for kind in kinds:
                counts[(split, profile, kind)] += 1
            rows.append(dict(file=path.name, seed=meta['seed'], profile=profile, split=split,
                             kinds=kinds, teacher_lowfreq=stats, climate=conditioning))
        if (index + 1) % 2000 == 0:
            print(json.dumps(dict(audited=index+1, total=len(paths))), flush=True)
    atomic_json(external_path(output), dict(schema=1, dataset_manifest_digest=hashlib.sha256(
        (root / 'manifest.json').read_bytes()).hexdigest(),
        note='Low-frequency base-target proxy at 240 m/pixel; decoder detail and physical coast validation are separate.',
        counts=[dict(split=s, profile=p, kind=k, count=c) for (s, p, k), c in sorted(counts.items())], rows=rows))


def sampling_policy(coverage_path, output, rare_fraction=.25):
    if not 0 <= rare_fraction <= .5:
        raise ValueError('Keep at least half of the sampler uniform.')
    payload = Path(coverage_path).read_bytes()
    coverage_report = json.loads(payload)
    rows = sorted((r for r in coverage_report['rows'] if r['split'] == 'train'), key=lambda r: r['file'])
    if not rows or any(r['seed'] in HOLDOUT_SEEDS for r in rows):
        raise ValueError('The sampler must contain training data only, without reserved seeds.')
    p = np.full(len(rows), (1 - rare_fraction) / len(rows), dtype=np.float64)
    counts = {}
    for kind in KINDS:
        mask = np.array([kind in row['kinds'] for row in rows])
        counts[kind] = int(mask.sum())
        if not mask.any():
            raise ValueError(f'No training examples for {kind}; do not substitute evaluation worlds.')
        p[mask] += rare_fraction / len(KINDS) / mask.sum()
    validation_files = []
    for kind in KINDS:
        candidates = sorted(r['file'] for r in coverage_report['rows'] if r['split'] == 'val' and kind in r['kinds'])
        if candidates:
            validation_files.extend(candidates[i] for i in np.linspace(0, len(candidates)-1, min(4, len(candidates))).round().astype(int))
    atomic_json(external_path(output), dict(schema=1, stage='base',
        dataset_manifest_digest=coverage_report['dataset_manifest_digest'],
        coverage_digest=hashlib.sha256(payload).hexdigest(), rare_fraction=rare_fraction,
        counts=counts, train_files=[r['file'] for r in rows], probabilities=p.tolist(),
        validation_files=sorted(set(validation_files))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('survey')
    scan.add_argument('output', type=Path)
    scan.add_argument('--count', type=int, default=8)
    scan.add_argument('--warm-plains', action='store_true')
    select = sub.add_parser('freeze')
    select.add_argument('proposals', type=Path)
    select.add_argument('evaluation', type=Path)
    select.add_argument('output', type=Path)
    select.add_argument('--per-kind', type=int, default=1)
    check = sub.add_parser('audit')
    check.add_argument('manifest', type=Path)
    check.add_argument('evaluation', type=Path)
    check.add_argument('variant')
    check.add_argument('output', type=Path)
    cover = sub.add_parser('coverage')
    cover.add_argument('dataset', type=Path)
    cover.add_argument('output', type=Path)
    balance = sub.add_parser('sampling-policy')
    balance.add_argument('coverage', type=Path)
    balance.add_argument('output', type=Path)
    balance.add_argument('--rare-fraction', type=float, default=.25)
    extend = sub.add_parser('augment')
    extend.add_argument('original', type=Path)
    extend.add_argument('addition', type=Path)
    extend.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'survey':
        survey(args.output, args.count, args.warm_plains)
    elif args.command == 'freeze':
        freeze(args.proposals, args.evaluation, args.output, args.per_kind)
    elif args.command == 'audit':
        result = audit(args.manifest, args.evaluation, args.variant, args.output)
        print(json.dumps(dict(physical_passed=result['physical_passed'], accepted=False)))
    elif args.command == 'coverage':
        coverage(args.dataset, args.output)
    elif args.command == 'sampling-policy':
        sampling_policy(args.coverage, args.output, args.rare_fraction)
    else:
        augment(args.original, args.addition, args.output)


if __name__ == '__main__':
    main()
