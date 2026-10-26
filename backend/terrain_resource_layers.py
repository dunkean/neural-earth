"""Informative resource layers for the viewer, served like Orogen diagnostics.

Modes are named ``orogen-resources-<group>-<name>`` so that the existing
source-diagnostic path (PNG tiles, map and globe, menu enabling) serves them
unchanged. The world authority (terrain_resources.compile_world at LOD 10) is
computed once per Orogen atlas on first request, kept in memory and on disk.
Displayed values are bilinear between LOD-10 cell centres; the canonical value
is the cell's. Legends name the unit, the status (derived / synthetic) and
"proxy" when a value comes from a Render v2 appearance field.
"""
import hashlib
import json
import math
import os
import threading
from collections import OrderedDict
import time
from pathlib import Path

import numpy as np

PREFIX = 'orogen-resources-'
VERSION = 'resource-layers-v1'
SEASONS = (('djf', 'déc.–fév.', (11, 0, 1)), ('mam', 'mars–mai', (2, 3, 4)),
           ('jja', 'juin–août', (5, 6, 7)), ('son', 'sept.–nov.', (8, 9, 10)))
SEQ = ['10141f', '2c3e6b', '2f7f86', '6fbf73', 'e8e36a']          # dark -> light, sequential
WARM = ['1a1410', '5b2c1d', 'a5532a', 'e0a050', 'f6e7a8']
GREEN = ['171a12', '2e4a22', '4f7f2f', '93b552', 'e2ec9c']
BLUE = ['0c1420', '163a5f', '2a6fa0', '5fb2d6', 'c9ecf5']
ICE = ['1b1f24', '4b5a66', '8aa1b0', 'c6d7df', 'ffffff']
CONTEXT_COLORS = ['e0533d', '9b59b6', 'f0a330', 'ff2d55', '3fa7a0', '2e8b57', '8c6d46', '6fa8dc', 'c9b27c', '1d3557']
ROCK_COLORS = ['d48099', '8ccbd3', 'e5b65d', '99a669', '6e65ad', 'b07aa1', '3c7a46', 'f2f2f2']
FAMILY_COLORS = {
    'porphyry_cu': 'e07b39', 'epithermal_au_ag': 'ffd84d', 'sn_w_granite': 'c0c0d8', 'polymetallic_vein': '9aa5b1',
    'vms': 'd9534f', 'orogenic_au': 'f4c430', 'mvt_pb_zn': '7f8c8d', 'sediment_cu': 'c46a2b', 'skarn_fe_cu': 'a0522d',
    'bif_iron': '8b3a3a', 'ironstone': 'b5651d', 'laterite_fe': 'cd5c5c', 'evaporite_salt': 'ffffff', 'coal': '303030',
    'placer_au': 'ffe680', 'placer_sn': 'dfe4f0', 'fantasy_star_metal': '9b5de5'}
FAMILY_LABELS = {
    'porphyry_cu': 'Porphyre Cu-Au (Cox & Singer 17/20c)', 'epithermal_au_ag': 'Épithermal Au-Ag (25a–c)',
    'sn_w_granite': 'Sn-W de granite, filons et greisens (15a–c)', 'polymetallic_vein': 'Filons polymétalliques Pb-Zn-Ag (22c)',
    'vms': 'Sulfures massifs volcanogènes (24a, 28a)', 'orogenic_au': 'Or orogénique en quartz (36a)',
    'mvt_pb_zn': 'Pb-Zn type Mississippi (32a–b)', 'sediment_cu': 'Cu sédimentaire (30b)',
    'skarn_fe_cu': 'Skarn Fe-Cu (18b)', 'bif_iron': 'Fer rubané Superior/Algoma (34a, 28b)',
    'ironstone': 'Fer oolithique (34f)', 'laterite_fe': 'Fer latéritique', 'evaporite_salt': 'Sel et gypse évaporitiques (35a)',
    'coal': 'Charbon', 'placer_au': 'Placers aurifères (39a)', 'placer_sn': 'Placers stannifères (39e)',
    'fantasy_star_metal': 'Fantasy · métal stellaire'}
METAL_LABELS = dict(iron='fer', copper='cuivre', tin='étain', lead='plomb', zinc='zinc', silver='argent', gold='or',
                    salt='sel', gypsum='gypse', coal='charbon', tungsten='tungstène', star_metal='métal stellaire')
_LOCK = threading.RLock()


def fantasy_enabled():
    return os.environ.get('TERRAIN_RESOURCES_FANTASY', '') == '1'


def _specs():
    """(mode, spec) in menu order. spec: label, unit, range, colors, status, proxy, value(f, grid)."""
    import terrain_resources as R
    out = []

    def add(group, name, label, unit, lo, hi, colors, value, status='derived', proxy=False, log=False, kind='ramp', classes=None):
        out.append((PREFIX + group + '-' + name, dict(group=group, label=label, unit=unit, range=[lo, hi], colors=colors,
                   status=status, proxy=proxy, log=log, kind=kind, classes=classes, value=value)))

    per_km2 = lambda key, scale=1.: (lambda f, g: np.asarray(f[key], np.float64) / np.maximum(g.area_m2, 1) * 1e6 * scale)
    field = lambda key: (lambda f, g: np.asarray(f[key], np.float64))
    # Geology
    add('geo', 'context', 'Contexte tectonique · composition', 'fraction', 0, 1, CONTEXT_COLORS,
        lambda f, g: np.stack([f['context_' + c] for c in R.CONTEXTS]), kind='mix',
        classes=list(zip(R.CONTEXT_LABELS, CONTEXT_COLORS)))
    add('geo', 'rock', 'Roche mère · composition (8 classes)', 'fraction', 0, 1, ROCK_COLORS,
        lambda f, g: np.stack([f['rock_' + r] for r in R.ROCKS]), kind='mix', classes=list(zip(R.ROCK_LABELS, ROCK_COLORS)),
        status='derived+synthetic')
    for c, label in zip(R.CONTEXTS, R.CONTEXT_LABELS):
        add('geo', 'context-' + c, 'Contexte · ' + label, 'fraction', 0, 1, WARM, field('context_' + c))
    for r, label in zip(R.ROCKS, R.ROCK_LABELS):
        add('geo', 'rock-' + r, 'Roche · ' + label, 'fraction', 0, 1, WARM, field('rock_' + r), status='derived+synthetic')
    # Deposits
    add('dep', 'points', 'Gisements discrets · famille, tonnage, accessibilité', 'symbole', 0, 1, SEQ, None, kind='points',
        classes=[(FAMILY_LABELS[k], FAMILY_COLORS[k]) for k in FAMILY_LABELS if fantasy_enabled() or not k.startswith('fantasy')],
        status='derived (tirages adressés)')
    families = list(R.FAMILIES) + list(R.PLACERS) + (list(R.FANTASY) if fantasy_enabled() else [])
    for fam in families:
        add('dep', 'prospect-' + fam.replace('_', '-'), 'Prospectivité · ' + FAMILY_LABELS[fam], 'indice 0–1', 0, 1, WARM,
            field('prospect_' + fam), status='synthetic' if fam in R.FANTASY else 'derived')
    metals = ['iron', 'copper', 'tin', 'lead', 'zinc', 'silver', 'gold', 'salt', 'gypsum', 'coal', 'tungsten'] + (['star_metal'] if fantasy_enabled() else [])
    for m in metals:
        unit = 'kg/km²' if m in R.GRAM_METALS else 't/km²'
        add('dep', 'stock-' + m.replace('_', '-'), f'Stock contenu · {METAL_LABELS[m]}', unit, 1e-3, 1e5, WARM,
            (lambda key: (lambda f, g: np.asarray(f.get(key, np.zeros(g.shape)), np.float64) / np.maximum(g.area_m2, 1) * 1e6))('stock_' + m), log=True)
        add('dep', 'recoverable-' + m.replace('_', '-'), f'Récupérable médiéval · {METAL_LABELS[m]}', unit, 1e-3, 1e5, WARM,
            (lambda key: (lambda f, g: np.asarray(f.get(key, np.zeros(g.shape)), np.float64) / np.maximum(g.area_m2, 1) * 1e6))('recoverable_' + m), log=True)
    # Agriculture and herding
    add('agri', 'cereal-suitability', 'Céréales · aptitude (Ecocrop × sols × relief)', 'indice 0–1', 0, 1, GREEN, field('cereal_suitability'))
    add('agri', 'cereal-class', 'Céréales · classes GAEZ (NS, vmS, mS, MS, S, VS)', 'classe', 0, 5,
        ['202020', '6b4f2a', 'a37a3c', 'c9b458', '8fbf4d', '3f8f3a'], field('cereal_class'), kind='classes',
        classes=list(zip(['NS', 'vmS <20 %', 'mS 20–40 %', 'MS 40–60 %', 'S 60–80 %', 'VS ≥80 %'], ['202020', '6b4f2a', 'a37a3c', 'c9b458', '8fbf4d', '3f8f3a'])))
    add('agri', 'cereal-yield', 'Céréales · rendement brut médiéval', 't/ha', 0, .8, GREEN, field('cereal_yield_t_ha'))
    add('agri', 'cereal-ratio', 'Céréales · rapport récolte/semence', 'grain/grain', 1, 5, GREEN, field('cereal_yield_ratio'))
    add('agri', 'arable-share', 'Terres arables potentielles · part de la cellule', 'fraction', 0, 1, GREEN,
        lambda f, g: np.asarray(f['arable_area_m2']) / np.maximum(g.area_m2, 1), proxy=True)
    add('agri', 'cereal-net', 'Céréales · production nette potentielle (assolement)', 't/km²/an', 0, 60, GREEN, per_km2('cereal_net_t_year'), proxy=True)
    add('agri', 'growing-months', 'Saison de croissance (T ≥ 5 °C, ETR ≥ ½ ETP)', 'mois', 0, 12, GREEN, field('growing_months'))
    add('agri', 'harvest-month', 'Mois de récolte (fin de la plus longue saison)', 'mois', 0, 11, ['3f7fbf', '4fa3a5', '6fbf73', 'b5c95a', 'e8c547', 'e89a3c', 'd9603b', 'b53f6a', '8a3f9a', '5f4fb5', '3f5fbf', '3f6fbf'],
        lambda f, g: np.where(np.asarray(f['harvest_month']) == 255, np.nan, f['harvest_month']).astype(np.float64), kind='cyclic')
    add('agri', 'lsu-year', 'Pâture · capacité annuelle avec foin', 'UGB/km²', 0, 120, GREEN, field('lsu_year_km2'), proxy=True)
    add('agri', 'lsu-lean', 'Pâture · capacité du mois le plus maigre', 'UGB/km²', 0, 60, GREEN, field('lsu_lean_km2'), proxy=True)
    add('agri', 'transhumance', 'Pâture · potentiel de transhumance', 'indice 0–1', 0, 1, WARM, field('transhumance'), proxy=True)
    # Forest, wildlife, fisheries, water
    add('bio', 'npp', 'Production primaire nette (modèle Miami de Lieth)', 'g MS/m²/an', 0, 2000, GREEN, field('npp_g_m2'))
    add('bio', 'tree-cover', 'Couvert ligneux potentiel (Render v2)', 'fraction', 0, 1, GREEN, field('tree_cover'), proxy=True)
    add('bio', 'canopy', 'Hauteur de canopée (Render v2)', 'm', 0, 35, GREEN, lambda f, g: np.asarray(f['v2_canopy_m']) * (np.asarray(f['tree_cover']) > 0), proxy=True)
    add('bio', 'conifer', 'Part résineuse (Render v2)', 'fraction', 0, 1, BLUE, field('conifer_share'), proxy=True)
    add('bio', 'timber-share', "Part bois d'œuvre du stock", 'fraction', 0, 1, WARM, field('timber_share'), proxy=True)
    add('bio', 'wood-increment', 'Bois · accroissement annuel', 'm³/ha/an', 0, 8, GREEN, field('wood_increment_m3_ha'), proxy=True)
    add('bio', 'wood-stock', 'Bois · stock sur pied initial', 'm³/ha', 0, 400, GREEN, per_km2('wood_stock_m3', 1e-2), proxy=True)
    add('bio', 'game-density', 'Gibier · densité (Damuth, 80 kg)', 'ind./km²', 0, 8, WARM, field('game_per_km2'))
    add('bio', 'game-harvest', 'Gibier · prélèvement durable', 'kg/km²/an', 0, 60, WARM, per_km2('game_harvest_kg_year'))
    add('bio', 'fur', 'Fourrures · potentiel', 'indice 0–1', 0, 1, ICE, field('fur_potential'), proxy=True)
    add('bio', 'fish-inland', 'Pêche continentale (plaines inondables)', 'kg/km²/an', 0, 300, BLUE,
        lambda f, g: np.asarray(f['fish_inland_month_t'], np.float64).sum(axis=0) / np.maximum(g.area_m2, 1) * 1e9)
    add('bio', 'fish-marine', 'Pêche côtière (plateau < 200 m)', 't/km²/an', 0, 3, BLUE, per_km2('fish_marine_t_year'))
    add('bio', 'water-yield', 'Eau · écoulement annuel', 'mm/an', 0, 1500, BLUE, lambda f, g: np.asarray(f['runoff_mm'], np.float64).sum(axis=0))
    add('bio', 'perennial-snow', 'Neige pérenne (Render v2)', 'fraction', 0, 1, ICE, field('v2_perennial_snow'), proxy=True)
    # Materials, salt, peat, bog iron
    add('mat', 'stone', 'Pierre de taille · potentiel', 'indice 0–1', 0, 1, WARM, field('stone_ashlar'))
    add('mat', 'lime', 'Chaux (calcaire) · potentiel', 'indice 0–1', 0, 1, WARM, field('lime'))
    add('mat', 'slate', 'Ardoise et schiste · potentiel', 'indice 0–1', 0, 1, WARM, field('slate'))
    add('mat', 'clay', 'Argile · potentiel', 'indice 0–1', 0, 1, WARM, field('clay'))
    add('mat', 'sand', 'Sables et graviers · potentiel', 'indice 0–1', 0, 1, WARM, field('sand_gravel'))
    add('mat', 'sea-salt', 'Sel marin solaire', 't/km²/an', 0, 50, ICE,
        lambda f, g: np.asarray(f['sea_salt_month_t'], np.float64).sum(axis=0) / np.maximum(g.area_m2, 1) * 1e6)
    add('mat', 'peat-share', 'Tourbe · part de la cellule', 'fraction', 0, .3, WARM, lambda f, g: np.asarray(f['peat_area_m2']) / np.maximum(g.area_m2, 1))
    add('mat', 'peat-stock', 'Tourbe · volume', 'm³/km²', 0, 4e5, WARM, per_km2('peat_stock_m3'))
    add('mat', 'bog-iron', 'Fer des marais · renouvellement', 't Fe/km²/an', 0, .4, WARM, per_km2('bog_iron_t_year'))
    # Seasons (Orogen's northern calendar months; southern seasons are inverted)
    for key, label, months in SEASONS:
        m = list(months)
        add('season', 'npp-' + key, f'Production primaire · {label}', 'g MS/m²/mois', 0, 250, GREEN,
            (lambda m: lambda f, g: np.asarray(f['npp_month_g_m2'])[m].mean(axis=0))(m))
        add('season', 'forage-' + key, f'Pâture · capacité {label}', 'UGB/km²', 0, 150, GREEN,
            (lambda m: lambda f, g: np.asarray(f['lsu_month_km2'])[m].mean(axis=0))(m), proxy=True)
        add('season', 'snow-' + key, f'Neige au sol · {label} (Render v2)', 'fraction', 0, 1, ICE,
            (lambda m: lambda f, g: np.asarray(f['v2_snow_month'])[m].mean(axis=0) * f['land'])(m), proxy=True)
        add('season', 'fish-' + key, f'Pêche · {label}', 'kg/km²/mois', 0, 100, BLUE,
            (lambda m: lambda f, g: (np.asarray(f['fish_inland_month_t'])[m].mean(axis=0)
                                     + np.asarray(f['fish_marine_month_t'])[m].mean(axis=0)) / np.maximum(g.area_m2, 1) * 1e9)(m))
        add('season', 'salt-' + key, f'Sel marin · {label}', 't/km²/mois', 0, 10, ICE,
            (lambda m: lambda f, g: np.asarray(f['sea_salt_month_t'])[m].mean(axis=0) / np.maximum(g.area_m2, 1) * 1e6)(m))
        add('season', 'felling-' + key, f'Abattage et débardage · {label}', 'part des mois', 0, 1, WARM,
            (lambda m: lambda f, g: (np.asarray(f['felling_months'])[m].mean(axis=0) + np.asarray(f['sledge_months'])[m].mean(axis=0)) / 2)(m), proxy=True)
        add('season', 'runoff-' + key, f'Écoulement · {label}', 'mm/mois', 0, 150, BLUE,
            (lambda m: lambda f, g: np.asarray(f['runoff_mm'])[m].mean(axis=0))(m))
    return out


_SPECS = None


def specs():
    global _SPECS
    if _SPECS is None:
        _SPECS = dict(_specs())
    return _SPECS


def legend_text(spec):
    proxy = ' · proxy de rendu' if spec['proxy'] else ''
    lo, hi = spec['range']
    scale = ' (log)' if spec['log'] else ''
    return f"Ressources · {spec['label']} · {lo:g}–{hi:g} {spec['unit']}{scale} · {spec['status']}{proxy}"


def legends():
    return {mode: legend_text(spec) for mode, spec in specs().items()}


def legend_specs():
    """Structured legends for the viewer panel (JSON)."""
    return {mode: {k: v for k, v in spec.items() if k != 'value'} for mode, spec in specs().items()}


# ---------------------------------------------------------------------------
# World cache: memory, then disk, then compile

def _cache_root():
    from terrain_paths import RUNTIME_ROOT, configured_path
    return configured_path('TERRAIN_RESOURCES_CACHE', RUNTIME_ROOT / 'resources') / VERSION


def _identity(atlas):
    import terrain_resources as R
    source = getattr(atlas, 'source', atlas)
    meta = getattr(source, 'metadata', {}) or {}
    # Only the compiler defines the world authority; display code may change freely.
    code = hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest()
    world = meta.get('cache_key') or hashlib.sha256(np.ascontiguousarray(source.height_m).tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(dict(world=world, code=code, fantasy=fantasy_enabled(),
                                          seed=str(getattr(source, 'seed', 0))), sort_keys=True).encode()).hexdigest()


# Intermediate monthly arrays not served by any view nor export; dropped after
# compilation to keep a world near 0.3 GB in memory and on disk.
_TRANSIENT = ('pet_mm', 'aet_mm', 'precipitation_mm', 'temperature_c', 'snow_mm', 'v2_moisture_month',
              'v2_leaf_month', 'v2_grass_month', 'v2_ice_month', 'forage_month_kg_ha')


def _slim(fields):
    import terrain_resources as R
    out = {}
    for k, v in fields.items():
        if k in _TRANSIENT:
            continue
        v = np.asarray(v)
        if v.dtype == np.float64 and k not in R.EXTENSIVE and k != 'area_m2':
            v = v.astype(np.float32)   # committed extensive totals stay float64 (exact rounding)
        out[k] = v
    return out


def world_resources(atlas):
    """(grid, fields, deposits, seconds, origin) for an Orogen atlas, cached."""
    import terrain_resources as R
    source = getattr(atlas, 'source', atlas)
    bounds = tuple(float(b) for b in getattr(atlas, 'bounds', R.BOUNDS))
    if any(abs(a - b) > 1 for a, b in zip(bounds, R.BOUNDS)):
        raise ValueError('Resource layers currently require the default planet bounds')
    key = _identity(atlas)
    with _LOCK:
        cached = getattr(source, '_terrain_resources', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        started = time.perf_counter()
        path = _cache_root() / (key + '.npz')
        grid = R.Lattice(R.WORLD_LOD)
        result = None
        if path.is_file() and path.with_suffix('.json').is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    fields = _slim({k: data[k] for k in data.files if k not in _TRANSIENT})
                deposits = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                result = (grid, fields, deposits, time.perf_counter() - started, 'disk')
            except (OSError, ValueError, KeyError):
                result = None
        if result is None:
            grid, fields, deposits = R.compile_world(source.layers, source.height_m, int(getattr(source, 'seed', 0)),
                                                     fantasy=fantasy_enabled())
            arrays = _slim({k: np.asarray(v) for k, v in fields.items() if isinstance(v, np.ndarray)})
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.stem + '.tmp.npz')
                np.savez(temp, **arrays)
                temp.replace(path)
                path.with_suffix('.json').write_text(json.dumps(deposits), encoding='utf-8')
            except OSError:
                pass
            fields = arrays
            result = (grid, fields, deposits, time.perf_counter() - started, 'compiled')
        source._terrain_resources = (key, result)
        return result


# ---------------------------------------------------------------------------
# Rendering

def _hex(colors):
    return np.array([tuple(bytes.fromhex(c)) for c in colors], np.float64) / 255


def _sample(grid, value, xs, ys):
    """Bilinear between LOD-10 cell centres; longitude wraps, latitude clamps."""
    s = grid.size
    period = (grid.i[-1] - grid.i[0] + 1)
    x0, y0, x1, y1 = (-2e7, -1e7, 2e7, 1e7)
    xs = (np.asarray(xs, np.float64) - x0) % (x1 - x0) + x0
    gx = xs / s - .5 - grid.i[0]
    gy = np.clip(np.asarray(ys, np.float64) / s - .5 - grid.j[0], 0, len(grid.j) - 1)
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    tx, ty = (gx - ix)[None, :], (gy - iy)[:, None]
    i0, i1 = np.clip(ix, 0, period - 1), np.clip(ix + 1, 0, period - 1)
    j0, j1 = iy, np.minimum(iy + 1, len(grid.j) - 1)
    stacked = value.ndim == 3
    v = value if stacked else value[None]
    a = v[:, j0][:, :, i0] * (1 - tx) + v[:, j0][:, :, i1] * tx
    b = v[:, j1][:, :, i0] * (1 - tx) + v[:, j1][:, :, i1] * tx
    out = a * (1 - ty) + b * ty
    return out if stacked else out[0]


def _ramp(t, colors):
    c = _hex(colors)
    t = np.clip(np.nan_to_num(t, nan=0), 0, 1) * (len(c) - 1)
    k = np.minimum(t.astype(int), len(c) - 2)
    w = (t - k)[..., None]
    return c[k] * (1 - w) + c[k + 1] * w


def _base(grid, fields, xs, ys):
    land = _sample(grid, np.asarray(fields['land'], np.float64), xs, ys)
    return land, (land[..., None] * np.array([.16, .16, .15]) + (1 - land[..., None]) * np.array([.04, .07, .12]))


def render(atlas, mode, xs, ys):
    spec = specs().get(mode)
    if spec is None:
        raise ValueError(f'Unknown resource layer {mode!r}')
    grid, fields, deposits, _, _ = world_resources(atlas)
    land, base = _base(grid, fields, xs, ys)
    if spec['kind'] == 'points':
        return draw_deposits(grid, deposits, xs, ys, base)
    value = np.asarray(spec['value'](fields, grid), np.float64)
    if spec['kind'] == 'mix':
        mix = _sample(grid, value, xs, ys)
        rgb = np.einsum('kyx,kc->yxc', mix, _hex(spec['colors']))
        return np.clip(rgb * np.maximum(land, .35)[..., None], 0, 1)
    if spec['kind'] in ('classes', 'cyclic'):
        # Categorical: nearest cell, never interpolated.
        s = grid.size
        xw = (np.asarray(xs, np.float64) + 2e7) % 4e7 - 2e7
        ix = np.clip(np.floor(xw / s).astype(int) - grid.i[0], 0, len(grid.i) - 1)
        iy = np.clip(np.floor(np.asarray(ys) / s).astype(int) - grid.j[0], 0, len(grid.j) - 1)
        cat = value[iy[:, None], ix[None, :]]
        if spec['kind'] == 'cyclic':
            hue = np.nan_to_num(cat / 12, nan=0)
            rgb = .5 + .5 * np.cos(2 * np.pi * (hue[..., None] + np.array([0, 1 / 3, 2 / 3])))
            rgb = np.where(np.isnan(cat)[..., None], base, rgb * .9)
        else:
            rgb = _hex(spec['colors'])[np.clip(np.nan_to_num(cat).astype(int), 0, len(spec['colors']) - 1)]
        return np.where((land > .5)[..., None], rgb, base)
    v = _sample(grid, np.nan_to_num(value), xs, ys)
    lo, hi = spec['range']
    t = (np.log10(np.maximum(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) if spec['log'] else (v - lo) / (hi - lo)
    rgb = _ramp(t, spec['colors'])
    shown = v > (lo if spec['log'] else lo + 1e-9 * (hi - lo))
    marine = 'fish-marine' in mode or mode.endswith(tuple('fish-' + k for k, _, _ in SEASONS))
    mask = shown if marine else shown & (land > .5)
    return np.where(mask[..., None], rgb, base)


def draw_deposits(grid, deposits, xs, ys, base):
    """Map symbols: hue = family, radius = log ore tonnage, brightness = medieval access."""
    import terrain_resources as R
    rgb = base.copy()
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    dx = abs(xs[1] - xs[0]) if len(xs) > 1 else 1.
    if not deposits:
        return rgb
    pos = np.array([[(d['position_i'] + .5) * R.cell_size(d['position_lod']), (d['position_j'] + .5) * R.cell_size(d['position_lod'])]
                    for d in deposits])
    span = 4e7
    xc = (xs[0] + xs[-1]) / 2
    px = pos[:, 0] + np.round((xc - pos[:, 0]) / span) * span
    # Symbols shrink in overviews (about 1-2 px at 40 km/px) so provinces stay readable.
    zoom = min(1., max(.4, 4000. / dx))
    radius_px = np.array([(1.2 + .8 * max(0, math.log10(max(d['ore_t'], 1)) - 4)) * zoom + .3 for d in deposits])
    margin = (radius_px.max() + 1) * dx
    keep = (px > xs[0] - margin) & (px < xs[-1] + margin) & (pos[:, 1] > ys[0] - margin) & (pos[:, 1] < ys[-1] + margin)
    order = np.argsort([-d['ore_t'] for d in deposits])   # large symbols first, small on top
    for k in order:
        if not keep[k]:
            continue
        d = deposits[k]
        cx, cy, r = (px[k] - xs[0]) / dx, (pos[k, 1] - ys[0]) / dx, radius_px[k]
        x0, x1 = max(0, int(cx - r - 1)), min(len(xs), int(cx + r + 2))
        y0, y1 = max(0, int(cy - r - 1)), min(len(ys), int(cy + r + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)
        color = _hex([FAMILY_COLORS.get(d['family'], 'ffffff')])[0] * (.45 + .55 * d['medieval_accessible_share'])
        disc = np.clip(r + .5 - dist, 0, 1)[..., None]
        ring = (np.clip(r + 1.5 - dist, 0, 1) - np.clip(r + .5 - dist, 0, 1))[..., None]
        patch = rgb[y0:y1, x0:x1]
        patch[:] = patch * (1 - disc) + color * disc
        patch[:] = patch * (1 - ring) + np.array([0., 0., 0.]) * ring
    return rgb


def inspect(atlas, x, y, radius_m=60000.):
    """Values under the pointer and nearby deposits, for the viewer panel."""
    import terrain_resources as R
    grid, fields, deposits, seconds, origin = world_resources(atlas)
    s = grid.size
    xw = (float(x) + 2e7) % 4e7 - 2e7
    i = int(np.clip(math.floor(xw / s) - grid.i[0], 0, len(grid.i) - 1))
    j = int(np.clip(math.floor(float(y) / s) - grid.j[0], 0, len(grid.j) - 1))
    values = {}
    for mode, spec in specs().items():
        if spec['kind'] in ('points', 'mix'):
            continue
        v = float(np.asarray(spec['value'](fields, grid), np.float64)[j, i])
        values[mode] = dict(label=spec['label'], value=None if not math.isfinite(v) else round(v, 4), unit=spec['unit'],
                            status=spec['status'], proxy=spec['proxy'])
    dominant = dict(context=R.CONTEXT_LABELS[int(np.argmax([fields['context_' + c][j, i] for c in R.CONTEXTS]))],
                    rock=R.ROCK_LABELS[int(np.argmax([fields['rock_' + r][j, i] for r in R.ROCKS]))])
    near = []
    for d in deposits:
        px = (d['position_i'] + .5) * R.cell_size(d['position_lod'])
        py = (d['position_j'] + .5) * R.cell_size(d['position_lod'])
        dx = (px - xw + 2e7) % 4e7 - 2e7
        dist = math.hypot(dx, py - float(y))
        if dist <= radius_m:
            near.append(dict(d, distance_m=round(dist), label=FAMILY_LABELS.get(d['family'], d['family'])))
    near.sort(key=lambda d: d['distance_m'])
    return dict(cell=dict(lod=grid.lod, i=int(grid.i[i]), j=int(grid.j[j]), id=str(int(grid.ids[j, i])),
                          area_km2=round(float(grid.area_m2[j, i]) / 1e6, 1)),
                dominant=dominant, values=values, deposits=near[:12], cache=origin, seconds=round(seconds, 2))


# ---------------------------------------------------------------------------
# Multi-scale tiles: descent with noise and conservation (terrain_resource_detail)

VEIN = ('polymetallic_vein', 'epithermal_au_ag', 'sn_w_granite', 'orogenic_au')
STRATA = dict(bif_iron=50., ironstone=8., mvt_pb_zn=15., sediment_cu=4., evaporite_salt=30., coal=2.)
AMAS = dict(porphyry_cu=300., vms=40., skarn_fe_cu=60., laterite_fe=8., fantasy_star_metal=50.)
PLACER = ('placer_au', 'placer_sn')
PROSPECT_ALL = PREFIX + 'dep-prospect-all'
R_WORLD = 4e7
_AUTH_LOCK = threading.RLock()


class _World:
    """Per-world state shared by tiles: authority arrays, deposits as arrays, descent engine."""

    def __init__(self, grid, fields, deposits, seed):
        import terrain_resources as R
        from terrain_resource_detail import Detail
        from scipy.ndimage import distance_transform_edt
        self.grid, self.fields, self.deposits, self.seed = grid, fields, deposits, seed
        self.cache = OrderedDict()
        land = np.asarray(fields['land'], bool)
        # Coastal extrapolation: nearest land (or sea) cell value, so a fine coast
        # inside an authority sea cell still has a parent value (doc §16).
        self.nearest_land = distance_transform_edt(~land, return_distances=False, return_indices=True) if (~land).any() else None
        self.nearest_sea = distance_transform_edt(land, return_distances=False, return_indices=True) if land.any() else None
        self.detail = Detail(seed, grid, fields, deposits, self.authority)
        self.png = OrderedDict()
        n = len(deposits)
        self.dep_family = np.array([d['family'] for d in deposits]) if n else np.array([], str)
        offs = [R.key(seed, ['resources', 'deposit-lod0', d['id']]) for d in deposits]
        span = np.array([float(1 << d['position_lod']) for d in deposits]) if n else np.zeros(0)
        pi = np.array([d['position_i'] for d in deposits], np.float64) * span if n else np.zeros(0)
        pj = np.array([d['position_j'] for d in deposits], np.float64) * span if n else np.zeros(0)
        oi = np.array([(k % 997) / 997 for k in offs]) if n else np.zeros(0)
        oj = np.array([((k >> 20) % 991) / 991 for k in offs]) if n else np.zeros(0)
        # LOD-0 cell centre inside the deposit's LOD-3 cell (keyed, never re-drawn).
        self.dep_x = (pi + np.floor(oi * span) + .5) * 30.
        self.dep_y = (pj + np.floor(oj * span) + .5) * 30.
        self.dep_ore = np.array([d['ore_t'] for d in deposits], np.float64) if n else np.zeros(0)
        self.dep_access = np.array([d['medieval_accessible_share'] for d in deposits]) if n else np.zeros(0)
        self.dep_gossan = np.array([d['gossan'] for d in deposits], bool) if n else np.zeros(0, bool)
        jit = np.array([((k >> 40) % 1000) / 1000 - .5 for k in offs]) if n else np.zeros(0)
        strike = self.detail.strike()
        if n:
            ci = (np.floor(self.dep_x / grid.size).astype(int) - grid.i[0]) % len(grid.i)
            cj = np.clip(np.floor(self.dep_y / grid.size).astype(int) - grid.j[0], 0, len(grid.j) - 1)
            self.dep_theta = strike[cj, ci] + .5 * jit
        else:
            self.dep_theta = np.zeros(0)
        self.dep_mass = {}
        for kind in ('contained', 'medieval_recoverable'):
            for d_idx, d in enumerate(deposits):
                for m, v in d[kind].items():
                    self.dep_mass.setdefault((kind, m), np.zeros(n))[d_idx] = v

    def authority(self, mode):
        with _AUTH_LOCK:
            if mode in self.cache:
                self.cache.move_to_end(mode)
                return self.cache[mode]
        if mode == PROSPECT_ALL:
            v = np.clip(sum(np.asarray(self.fields[k], np.float64) for k in self.fields if k.startswith('prospect_')), 0, 1)
        else:
            v = np.asarray(specs()[mode]['value'](self.fields, self.grid), np.float64)
        v = np.nan_to_num(v)
        idx = self.nearest_sea if 'fish-marine' in mode else self.nearest_land
        if idx is not None and not mode.startswith(PREFIX + 'geo-'):
            v = v[..., idx[0], idx[1]]
        with _AUTH_LOCK:
            self.cache[mode] = v
            while len(self.cache) > 24:
                self.cache.popitem(last=False)
        return v


def world_state(atlas):
    source = getattr(atlas, 'source', atlas)
    grid, fields, deposits, _, _ = world_resources(atlas)
    with _LOCK:
        state = getattr(source, '_terrain_resources_state', None)
        if state is None or state.fields is not fields:
            state = _World(grid, fields, deposits, int(getattr(source, 'seed', 0)))
            source._terrain_resources_state = state
        return state


def _bounds_of(spec):
    if spec['unit'] in ('fraction', 'indice 0–1') or spec['label'].startswith('Prospectivité'):
        return (0., 1.)
    if spec['unit'] == 'mois':
        return (0., 12.)
    return None


def _shade(feat, lod):
    if feat is None:
        return None
    e = np.maximum(feat['elevation'], 0)
    gy, gx = np.gradient(e, 30. * 2 ** lod)
    light = (-gx * .6 - gy * .8 + 1) / np.sqrt(1 + gx * gx + gy * gy)
    return np.clip(.78 + 1.2 * (light - 1), .55, 1.1)


def render_tile(atlas, mode, lod, tx, ty, inputs):
    """RGB (256, 256) of a resource layer at any LOD; cached per (mode, lod, tile)."""
    from terrain_resource_detail import HALO, TILE, tile_axes, multi_noise
    state = world_state(atlas)
    key = (mode, lod, tx, ty)
    with _LOCK:
        if key in state.png:
            state.png.move_to_end(key)
            return state.png[key]
    spec = specs()[mode]
    if lod < -6:
        raise ValueError('Resource layers stop at LOD -6')
    det = state.detail
    xs, ys, _, _ = tile_axes(lod, tx, ty)
    inner = (slice(HALO, HALO + TILE), slice(HALO, HALO + TILE))
    fine = lod < det.lod
    feat = det.features(lod, tx, ty, inputs) if fine else None
    if feat is not None:
        land = feat['land'][inner].astype(np.float64)
    else:
        land = np.asarray(_sample(state.grid, np.asarray(state.fields['land'], np.float64), xs, ys))[inner]
    base = land[..., None] * np.array([.16, .16, .15]) + (1 - land[..., None]) * np.array([.04, .07, .12])
    values = lambda m, b: (det.values(m, lod, tx, ty, inputs, 'ramp', b) if fine else det.authority_tile(m, lod, tx, ty))[inner]
    kind = spec['kind']
    if kind == 'points':
        # Background only: smooth bilinear prospectivity (the conserved fields are
        # the per-family layers); exact zeros of 30 km cells would draw rectangles.
        prospect = np.clip(_sample(state.grid, state.authority(PROSPECT_ALL), xs, ys)[inner], 0, 1)
        rgb = base * (1 - .55 * prospect[..., None]) + _hex(['5a3b1c'])[0] * .55 * (prospect * land)[..., None]
        rgb = draw_shapes(state, xs[HALO:HALO + TILE], ys[HALO:HALO + TILE], rgb, lod, feat)
    elif mode.startswith(PREFIX + 'dep-stock-') or mode.startswith(PREFIX + 'dep-recoverable-'):
        rgb = _colorize(spec, mode, stock_density(state, mode, lod, xs, ys)[inner], base, land)
    elif kind == 'mix':
        frac = (det.values(mode, lod, tx, ty, inputs, 'mix', None) if fine else det.authority_tile(mode, lod, tx, ty))[(slice(None),) + inner]
        q = np.maximum(frac, 0) ** 2.5 if fine else frac
        q = q / np.maximum(q.sum(axis=0), 1e-12)
        rgb = np.einsum('kyx,kc->yxc', q, _hex(spec['colors'])) * np.maximum(land, .35)[..., None]
    elif kind == 'classes':
        suit = values(PREFIX + 'agri-cereal-suitability', (0., 1.))
        cls = np.select([suit >= .8, suit >= .6, suit >= .4, suit >= .2, suit > 0], [5, 4, 3, 2, 1], 0)
        rgb = np.where((land > .5)[..., None], _hex(spec['colors'])[cls], base)
    elif kind == 'cyclic':
        value = det.authority_value(mode)
        g = state.grid
        warp = g.size * .45 * np.stack([multi_noise(state.seed, 'harvest-warp-' + a, max(lod, -6), min(g.lod, max(lod, -6) + 3), xs, ys) for a in 'xy'])
        X, Y = np.meshgrid(xs, ys)
        ii = (np.floor((X + warp[0]) / g.size).astype(int) - g.i[0]) % len(g.i)
        jj = np.clip(np.floor((Y + warp[1]) / g.size).astype(int) - g.j[0], 0, len(g.j) - 1)
        cat = value[jj, ii][inner]
        rgb = .5 + .5 * np.cos(2 * np.pi * (np.nan_to_num(cat / 12)[..., None] + np.array([0, 1 / 3, 2 / 3])))
        rgb = np.where((np.isfinite(cat) & (land > .5))[..., None], rgb * .9, base)
    else:
        rgb = _colorize(spec, mode, values(mode, _bounds_of(spec)), base, land)
    shade = _shade(feat, lod)
    if shade is not None:
        rgb = rgb * shade[inner][..., None]
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    with _LOCK:
        state.png[key] = rgb
        while len(state.png) > 256:
            state.png.popitem(last=False)
    return rgb


def _colorize(spec, mode, v, base, land):
    lo, hi = spec['range']
    if spec['log']:
        t = (np.log10(np.maximum(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        shown = v > lo
    else:
        t = (v - lo) / (hi - lo)
        shown = v > lo + 1e-9 * (hi - lo)
    rgb = _ramp(t, spec['colors'])
    marine = 'fish-marine' in mode or mode.endswith(tuple('fish-' + k for k, _, _ in SEASONS))
    mask = shown if marine else shown & (land > .5)
    return np.where(mask[..., None], rgb, base)


def stock_density(state, mode, lod, xs, ys):
    """Contained (or recoverable) mass per km2 as a sum of deposit kernels.

    Each deposit spreads its exact mass with a normalised Gaussian of width
    max(physical footprint, 0.75 sample): integrated over space it equals the
    deposit, so coarser aggregates equal the stock layers (doc §16).
    """
    metal = (mode.split('-stock-')[-1] if '-stock-' in mode else mode.split('-recoverable-')[-1]).replace('-', '_')
    kind = 'contained' if '-stock-' in mode else 'medieval_recoverable'
    mass = state.dep_mass.get((kind, metal))
    out = np.zeros((len(ys), len(xs)))
    if mass is None or not len(mass):
        return out
    r = 30. * 2 ** lod
    sigma = np.maximum(np.sqrt(state.dep_ore / 2.7 / 30.) / 2, .75 * r)
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    px = state.dep_x + np.round(((x0 + x1) / 2 - state.dep_x) / R_WORLD) * R_WORLD
    keep = (mass > 0) & (px > x0 - 4 * sigma) & (px < x1 + 4 * sigma) & \
           (state.dep_y > y0 - 4 * sigma) & (state.dep_y < y1 + 4 * sigma)
    for k in np.flatnonzero(keep):
        s = sigma[k]
        cx = slice(max(0, int(np.searchsorted(xs, px[k] - 4 * s))), int(np.searchsorted(xs, px[k] + 4 * s)) + 1)
        cy = slice(max(0, int(np.searchsorted(ys, state.dep_y[k] - 4 * s))), int(np.searchsorted(ys, state.dep_y[k] + 4 * s)) + 1)
        gx = np.exp(-((xs[cx] - px[k]) ** 2) / (2 * s * s))
        gy = np.exp(-((ys[cy] - state.dep_y[k]) ** 2) / (2 * s * s))
        out[cy, cx] += mass[k] * 1e6 / (2 * np.pi * s * s) * gy[:, None] * gx[None, :]
    return out


def smooth01(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def draw_shapes(state, xs, ys, rgb, lod, feat):
    """Deposit footprints: oriented veins, ellipsoidal bodies, strata lenses,
    placers on valley floors and canonical rivers, prospectivity halo and
    gossan rim. Below 1.5 px a footprint becomes a map symbol."""
    from terrain_resource_detail import HALO, TILE
    if not len(state.dep_x):
        return rgb
    r = 30. * 2 ** lod
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    px = state.dep_x + np.round(((x0 + x1) / 2 - state.dep_x) / R_WORLD) * R_WORLD
    ore, fam = state.dep_ore, state.dep_family
    vein, placer, strata = np.isin(fam, VEIN), np.isin(fam, PLACER), np.isin(fam, list(STRATA))
    thick = np.array([AMAS.get(f, STRATA.get(f, 30.)) for f in fam])
    aspect = np.where(strata, 5., 1.6)
    a = np.sqrt(ore / 2.7 / thick * aspect / np.pi)
    b = a / aspect
    a = np.where(vein, np.clip(120 * (ore / 1e4) ** .45, 60, 5000) / 2, a)
    b = np.where(vein, np.clip(1.5 * (ore / 1e4) ** .25, 1, 15) / 2, b)
    a = np.where(placer, np.clip(400 * (ore / 1e5) ** .5, 200, 4000) / 2, a)
    b = np.where(placer, 60., b)
    halo_r = np.minimum(np.maximum(a, 2 * r), 4000.)
    reach = np.maximum(halo_r * 2.5, a + 2 * r)
    keep = (px > x0 - reach) & (px < x1 + reach) & (state.dep_y > y0 - reach) & (state.dep_y < y1 + reach)
    # Placer trap: canonical rivers, else flat valley floors (slope < 4 degrees).
    valley = (np.maximum(feat['river'] > .3, smooth01(feat['valley'], .7, .95) * (feat['slope'] < 4))
              [HALO:HALO + TILE, HALO:HALO + TILE]) if feat is not None else None
    for k in np.flatnonzero(keep)[np.argsort(-ore[keep])]:
        color = _hex([FAMILY_COLORS.get(fam[k], 'ffffff')])[0] * (.45 + .55 * state.dep_access[k])
        ia = slice(max(0, int(np.searchsorted(xs, px[k] - reach[k]))), int(np.searchsorted(xs, px[k] + reach[k])) + 1)
        ja = slice(max(0, int(np.searchsorted(ys, state.dep_y[k] - reach[k]))), int(np.searchsorted(ys, state.dep_y[k] + reach[k])) + 1)
        if ia.start >= ia.stop or ja.start >= ja.stop:
            continue
        dx = xs[ia][None, :] - px[k]
        dy = ys[ja][:, None] - state.dep_y[k]
        patch = rgb[ja, ia]
        if 2 * b[k] / r < 1.5 and not placer[k] or 2 * a[k] / r < 1.5:
            # Map symbol of constant screen size (veins thinner than a sample, far views).
            rpx = (1.2 + .8 * max(0, math.log10(max(ore[k], 1)) - 4)) * min(1., max(.4, 4000. / r)) + .3
            if vein[k] and 2 * a[k] / r >= 3:
                c, s = math.cos(state.dep_theta[k]), math.sin(state.dep_theta[k])
                u, v = dx * c + dy * s, -dx * s + dy * c
                body = np.clip(1.2 - np.abs(v) / r, 0, 1) * np.clip((a[k] - np.abs(u)) / r, 0, 1)
                patch[:] = patch * (1 - body[..., None]) + color * body[..., None]
                continue
            dist = np.hypot(dx, dy) / r
            disc = np.clip(rpx + .5 - dist, 0, 1)[..., None]
            ring = (np.clip(rpx + 1.5 - dist, 0, 1) - np.clip(rpx + .5 - dist, 0, 1))[..., None]
            patch[:] = patch * (1 - disc) + color * disc
            patch[:] = patch * (1 - ring)
            continue
        c, s = math.cos(state.dep_theta[k]), math.sin(state.dep_theta[k])
        u, v = dx * c + dy * s, -dx * s + dy * c
        q = np.sqrt((u / a[k]) ** 2 + (v / b[k]) ** 2)
        if vein[k]:
            body = np.clip((b[k] + .5 * r - np.abs(v)) / r, 0, 1) * np.clip((a[k] - np.abs(u)) / (r + .05 * a[k]), 0, 1)
        elif placer[k] and valley is not None:
            body = .8 * np.clip(1.2 - np.hypot(dx, dy) / a[k], 0, 1) * valley[ja, ia]
        else:
            body = np.clip((1 - q) * b[k] / r, 0, 1)
        halo = .2 * np.exp(-(np.hypot(dx, dy) / halo_r[k]) ** 2)
        patch[:] = patch * (1 - halo[..., None]) + _hex(['c9a64a'])[0] * halo[..., None]
        if state.dep_gossan[k]:
            rim = np.clip((1.6 - q) / .6, 0, 1) * (1 - body)
            patch[:] = patch * (1 - .7 * rim[..., None]) + _hex(['b5532a'])[0] * .7 * rim[..., None]
        patch[:] = patch * (1 - body[..., None]) + color * body[..., None]
    return rgb
