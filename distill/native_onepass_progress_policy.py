"""Paired terrain evidence for continuing NF4 training beyond the20k milestone."""
import math
from collections import defaultdict


KINDS = frozenset(('desert-plain', 'temperate-plain', 'wet-complex-coast',
                   'low-plain-sea', 'wet-high-mountain'))
COASTS = frozenset(('wet-complex-coast', 'low-plain-sea'))
POLICY = dict(extension_steps=5000, plateau_patience=2, minimum_improvement=.005,
    maximum_kind_regression=.10, maximum_coast_regression=.02,
    weights=dict(height=.45, morphology=.35, grain=.05, coast=.15),
    floors=dict(height=1., morphology=1., grain=.05, coast=.01),
    note='Deterministic paired validation proxies;not a declaration of visual acceptance.')


def finite(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid terrain evidence: {name}')
    return float(value)


def evidence(reports):
    rows, gpus, selections, steps = {}, set(), set(), set()
    for report in reports:
        if (report.get('status') != 'complete' or report.get('representation') != 'packed-nf4' or
                not 0 < report['weight_bytes'] < 100000000):
            raise ValueError('Complete real packed NF4 reports under100MB required.')
        gpus.add(report['gpu']); selections.add(report['selected_weights']); steps.add(report['step'])
        for row in report['rows']:
            key = (row['kind'], row['seed'], row['coarse'], row['lod'])
            if key in rows:
                raise ValueError('Duplicate validation scene.')
            errors = dict(height=finite(row['mae_m'], 'height'))
            tpi = [finite(value['tpi_mae_m'], 'morphology') for value in
                   row['shape']['topographic_position'] if value.get('available')]
            if tpi:
                errors['morphology'] = sum(tpi)/len(tpi)
            ratio = row.get('high_frequency_rms_ratio')
            if ratio is not None:
                ratio = finite(ratio, 'grain')
                if ratio <= 0:
                    raise ValueError('Positive high-frequency ratio required.')
                errors['grain'] = abs(math.log(ratio))
            if row['kind'] in COASTS:
                errors['coast'] = finite(row['coast_disagreement'], 'coast')
                if errors['coast'] > 1:
                    raise ValueError('Coast disagreement exceeds1.')
            rows[key] = errors
    if (len(gpus) != 1 or len(selections) != 1 or len(steps) != 1 or
            {key[0] for key in rows} != KINDS or {key[2] for key in rows} != {'coarse2', 'coarse4'} or
            {key[3] for key in rows} != {0, 3}):
        raise ValueError('Matched GPU/selection/step and all climates, coarses and LODs required.')
    scenes = {(kind, seed) for kind, seed, _, _ in rows}
    expected = {(kind, seed, coarse, lod) for kind, seed in scenes
                for coarse in ('coarse2', 'coarse4') for lod in (0, 3)}
    if set(rows) != expected:
        raise ValueError('Every scene needs both coarse variants and both LODs.')
    return dict(rows=rows, gpu=next(iter(gpus)), selection=next(iter(selections)), step=next(iter(steps)))


def compare(candidate, reference, policy=POLICY):
    if candidate['gpu'] != reference['gpu'] or set(candidate['rows']) != set(reference['rows']):
        raise ValueError('Progress must compare the same scenes on the same GPU.')
    kinds, components, coast_regressions = defaultdict(list), defaultdict(list), []
    for key, actual in candidate['rows'].items():
        previous = reference['rows'][key]
        if set(actual) != set(previous):
            raise ValueError('Progress metrics changed availability.')
        ratios = {name:max(actual[name], policy['floors'][name])/max(previous[name], policy['floors'][name])
                  for name in actual}
        weight = sum(policy['weights'][name] for name in ratios)
        kinds[key[0]].append(sum(ratio*policy['weights'][name] for name, ratio in ratios.items())/weight)
        for name, ratio in ratios.items():
            components[(key[0], name)].append(ratio)
        if 'coast' in actual and actual['coast']-previous['coast'] > policy['maximum_coast_regression']:
            coast_regressions.append(dict(scene=list(key), increase=actual['coast']-previous['coast']))
    kind_scores = {name:sum(values)/len(values) for name, values in kinds.items()}
    score = sum(kind_scores.values())/len(kind_scores)
    regressions = [dict(kind=kind, metric=name, ratio=sum(values)/len(values))
                   for (kind, name), values in components.items() if name in ('height', 'morphology') and
                   sum(values)/len(values) > 1+policy['maximum_kind_regression']]
    eligible = not regressions and not coast_regressions
    return dict(relative_score=score, kind_scores=kind_scores, eligible=eligible,
                improved=eligible and score <= 1-policy['minimum_improvement'],
                regressions=regressions, coast_regressions=coast_regressions)


def select(candidates, reference, policy=POLICY):
    scored = [(candidate, compare(candidate, reference, policy)) for candidate in candidates]
    eligible = [pair for pair in scored if pair[1]['eligible']]
    return min(eligible or scored, key=lambda pair:pair[1]['relative_score'])
