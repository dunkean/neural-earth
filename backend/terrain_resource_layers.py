"""Informative resource layers for the viewer, served like Orogen diagnostics.

Modes are named ``resources-<group>-<name>``. The browser evaluates them in
WGSL inside the renderer's shade pass (web/terrain_resources_gpu.js) from the
physical tiles already on the GPU plus a per-mode package served once per
world (LOD-10 fields as half floats, deposits, buckets). The NumPy mirror
(terrain_resource_gpu.py) is the reference. The world authority
(terrain_resources.compile_world at LOD 10) is computed once per Orogen
atlas in a serialized background thread, published stage by
stage, kept in memory and on disk; requests never wait for it (202 with
progress). Legends name the unit, the status (derived / synthetic) and
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

PREFIX = 'resources-'
VERSION = 'resource-layers-v3'
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
    'small_coal': '555166',
    'bog_iron': 'cf7042', 'local_ironstone': 'dc8545', 'small_pb_ag': 'becbdf',
    'salt_spring': '67e4e9', 'gypsum_quarry': 'fff2c5', 'small_sn': 'b9c9ff',
    'porphyry_cu': 'e07b39', 'epithermal_au_ag': 'ffd84d', 'sn_w_granite': 'c0c0d8', 'polymetallic_vein': '9aa5b1',
    'vms': 'd9534f', 'orogenic_au': 'f4c430', 'mvt_pb_zn': '7f8c8d', 'sediment_cu': 'c46a2b', 'skarn_fe_cu': 'a0522d',
    'bif_iron': '8b3a3a', 'ironstone': 'b5651d', 'laterite_fe': 'cd5c5c', 'evaporite_salt': 'ffffff', 'coal': '303030',
    'placer_au': 'ffe680', 'placer_sn': 'dfe4f0', 'fantasy_star_metal': '9b5de5'}
FAMILY_LABELS = {
    'small_coal': 'Petit affleurement de charbon',
    'bog_iron': 'Fer des marais', 'local_ironstone': 'Petit fer sédimentaire', 'small_pb_ag': 'Petits filons Pb–Ag',
    'salt_spring': 'Source salée / saumure', 'gypsum_quarry': 'Carrière de gypse', 'small_sn': 'Petits filons Sn–W',
    'porphyry_cu': 'Porphyre Cu-Au (Cox & Singer 17/20c)', 'epithermal_au_ag': 'Épithermal Au-Ag (25a–c)',
    'sn_w_granite': 'Sn-W de granite, filons et greisens (15a–c)', 'polymetallic_vein': 'Filons polymétalliques Pb-Zn-Ag (22c)',
    'vms': 'Sulfures massifs volcanogènes (24a, 28a)', 'orogenic_au': 'Or orogénique en quartz (36a)',
    'mvt_pb_zn': 'Pb-Zn type Mississippi (32a–b)', 'sediment_cu': 'Cu sédimentaire (30b)',
    'skarn_fe_cu': 'Skarn Fe-Cu (18b)', 'bif_iron': 'Fer rubané Superior/Algoma (34a, 28b)',
    'ironstone': 'Fer oolithique (34f)', 'laterite_fe': 'Fer latéritique', 'evaporite_salt': 'Sel et gypse évaporitiques (35a)',
    'coal': 'Charbon', 'placer_au': 'Placers aurifères (39a)', 'placer_sn': 'Placers stannifères (39e)',
    'fantasy_star_metal': 'Fantasy · métal stellaire'}
# Provisional display ranges of per-metal potentials (t or kg per km2).
# These colour ranges are not physical bounds or measured quantiles.
POTENTIAL_RANGE = dict(iron=(.05, 500.), copper=(5e-4, 5.), tin=(1e-5, .1), lead=(1e-4, 1.), zinc=(2e-4, 2.),
                       silver=(5e-4, 5.), gold=(3e-5, .3), salt=(.05, 500.), gypsum=(3e-3, 30.), coal=(.02, 200.),
                       tungsten=(3e-6, .03), star_metal=(1e-4, 1.))
METAL_LABELS = dict(iron='fer', copper='cuivre', tin='étain', lead='plomb', zinc='zinc', silver='argent', gold='or',
                    salt='sel', gypsum='gypse', coal='charbon', tungsten='tungstène', star_metal='métal stellaire')
_LOCK = threading.RLock()
try:                                   # segfaults leave a Python traceback in the server log
    import faulthandler
    faulthandler.enable()
except (ImportError, RuntimeError, ValueError):
    pass


def fantasy_enabled():
    return os.environ.get('TERRAIN_RESOURCES_FANTASY', '') == '1'


def _specs():
    """(mode, spec) in menu order. spec: label, unit, range, colors, status, proxy, value(f, grid)."""
    import terrain_resources as R
    out = []

    def add(group, name, label, unit, lo, hi, colors, value, status='derived', proxy=False, log=False, kind='ramp', classes=None,
            seasonal=False, physical_bounds=None):
        out.append((PREFIX + group + '-' + name, dict(group=group, label=label, unit=unit, range=[lo, hi], colors=colors,
                   status=status, proxy=proxy, log=log, kind=kind, classes=classes, value=value, seasonal=seasonal,
                   physical_bounds=physical_bounds)))

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
    add('dep', 'points', 'Tous les gisements', 'symbole', 0, 1, SEQ, None, kind='points',
        classes=[(FAMILY_LABELS[k], FAMILY_COLORS[k]) for k in FAMILY_LABELS if fantasy_enabled() or not k.startswith('fantasy')],
        status='derived (tirages adressés)')
    families = list(R.FAMILIES) + list(R.PLACERS) + (list(R.FANTASY) if fantasy_enabled() else [])
    for fam in families:
        add('dep', 'prospect-' + fam.replace('_', '-'), 'Prospectivité · ' + FAMILY_LABELS[fam] + ' (zones)', 'indice 0–1', 0, 1, WARM,
            field('prospect_' + fam), status='synthetic' if fam in R.FANTASY else 'derived')
    metals = list(R.METALS) + (['star_metal'] if fantasy_enabled() else [])

    def per_deposit(key):
        def value(f, g):
            if not any(k.startswith('prospect_') for k in f):
                raise KeyError(key)            # deposits stage not compiled yet
            return np.asarray(f.get(key, np.zeros(g.shape)), np.float64) / np.maximum(g.area_m2, 1) * 1e6
        return value

    for m in metals:
        unit = 'kg/km²' if m in R.GRAM_METALS else 't/km²'
        # Linear metal-specific scale: weak permissive background should not
        # look as important as a fertile belt after four decades of log gain.
        lo, hi = 0., .12 * sum(R.metal_potential_weights(m, fantasy_enabled()).values())
        add('dep', 'potential-' + m.replace('_', '-'), f'Potentiel · {METAL_LABELS[m]} (zones)', unit, lo, hi, WARM,
            (lambda metal: (lambda f, g: R.metal_potential(f, metal, fantasy_enabled())))(m),
            physical_bounds=(0., sum(R.metal_potential_weights(m, fantasy_enabled()).values())),
            status='synthetic' if m == 'star_metal' else 'derived')
        add('dep', m.replace('_', '-'), METAL_LABELS[m].capitalize(), 'tonnes récupérables', 0, 1, SEQ,
            None, kind='points', classes=[(FAMILY_LABELS[k], FAMILY_COLORS[k]) for k in families
                if m in {**R.FAMILIES, **R.PLACERS, **R.FANTASY}[k]['metals']])
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
    # Seasonal layers: 12 monthly fields, continuous season slider (Orogen's
    # calendar months; southern seasons follow their own climate).
    month = lambda key, scale=1., area=False: (lambda f, g: np.asarray(f[key], np.float64) * scale /
                                                (np.maximum(g.area_m2, 1)[None] if area else 1))
    add('season', 'npp', 'Production primaire mensuelle', 'g MS/m²/mois', 0, 250, GREEN, month('npp_month_g_m2'), seasonal=True)
    add('season', 'forage', 'Pâture · capacité du mois', 'UGB/km²', 0, 150, GREEN, month('lsu_month_km2'), proxy=True, seasonal=True)
    add('season', 'snow', 'Neige au sol (Render v2)', 'fraction', 0, 1, ICE,
        lambda f, g: np.asarray(f['v2_snow_month'], np.float64) * f['land'][None], proxy=True, seasonal=True)
    add('season', 'fish', 'Pêche du mois', 'kg/km²/mois', 0, 100, BLUE,
        lambda f, g: (np.asarray(f['fish_inland_month_t'], np.float64) + np.asarray(f['fish_marine_month_t'], np.float64))
        / np.maximum(g.area_m2, 1)[None] * 1e9, seasonal=True)
    add('season', 'salt', 'Sel marin du mois', 't/km²/mois', 0, 10, ICE, month('sea_salt_month_t', 1e6, True), seasonal=True)
    add('season', 'felling', 'Abattage et débardage possibles', 'indice 0–1', 0, 1, WARM,
        lambda f, g: (np.asarray(f['felling_months'], np.float64) + np.asarray(f['sledge_months'], np.float64)) / 2,
        proxy=True, seasonal=True)
    add('season', 'runoff', 'Écoulement du mois', 'mm/mois', 0, 150, BLUE, month('runoff_mm'), seasonal=True)
    # Diagnostic maps are placement inputs, not inventories. Keep stable mode IDs.
    diagnostic = [item for item in out if '-potential-' in item[0] or '-prospect-' in item[0]]
    for _, spec in diagnostic:
        spec['group'] = 'diagnostic'
    return [item for item in out if item not in diagnostic] + diagnostic


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
    if spec['kind'] == 'points':
        return f"Ressources · {spec['label']} · aire logarithmique du tonnage récupérable · diamètre 6–14 px + contour"
    diagnostic = ' · intensité de placement, pas une carte des ressources' if spec['group'] == 'diagnostic' else ''
    return f"Ressources · {spec['label']} · {lo:g}–{hi:g} {spec['unit']}{scale} · {spec['status']}{proxy}{diagnostic}"


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


def _render_v2_code():
    """The Render v2 evaluator is an external input of the authority: its source names the cache."""
    import terrain_render_v2
    return hashlib.sha256(Path(terrain_render_v2.__file__).read_bytes()).hexdigest()[:16]


def _world_key(atlas):
    source = getattr(atlas, 'source', atlas)
    meta = getattr(source, 'metadata', {}) or {}
    return meta.get('cache_key') or hashlib.sha256(np.ascontiguousarray(source.height_m).tobytes()).hexdigest()


def _identity(atlas):
    """Cache key of the compiled authority: world inputs, seed, options and
    COMPILER_VERSION (bumped by hand when canonical outputs change; display
    edits never invalidate it), plus the Render v2 evaluator it consumes."""
    import terrain_resources as R
    source = getattr(atlas, 'source', atlas)
    return hashlib.sha256(json.dumps(dict(world=_world_key(atlas), compiler=R.COMPILER_VERSION, v2=_render_v2_code(),
                                          bounds=_atlas_bounds(atlas), fantasy=fantasy_enabled(), seed=str(getattr(source, 'seed', 0))),
                                     sort_keys=True).encode()).hexdigest()


def _v2_identity(atlas):
    import terrain_resources as R
    source = getattr(atlas, 'source', atlas)
    return hashlib.sha256(json.dumps(dict(world=_world_key(atlas), inputs=R.V2_INPUTS_VERSION, v2=_render_v2_code(),
                                          bounds=_atlas_bounds(atlas), seed=str(getattr(source, 'seed', 0))), sort_keys=True).encode()).hexdigest()


# Intermediate monthly arrays not served by any view nor export; dropped after
# compilation. Stocks are recomputed from the deposits on load (exact sums).
_TRANSIENT = ('pet_mm', 'aet_mm', 'precipitation_mm', 'temperature_c', 'snow_mm', 'v2_moisture_month',
              'v2_leaf_month', 'v2_grass_month', 'v2_ice_month', 'forage_month_kg_ha')
_DERIVED = ('stock_', 'recoverable_')


def _slim(fields):
    import terrain_resources as R
    out = {}
    for k, v in fields.items():
        if k in _TRANSIENT or not isinstance(v, np.ndarray):
            continue
        v = np.asarray(v)
        if v.dtype == np.float64 and k not in R.EXTENSIVE and k != 'area_m2':
            v = v.astype(np.float32)   # committed extensive totals stay float64 (exact rounding)
        out[k] = v
    return out


class ResourcesPending(Exception):
    """The world authority (or the stage a layer needs) is still being compiled."""

    def __init__(self, status):
        super().__init__(status.get('message', 'compiling'))
        self.status = status


# Stage durations measured in the compile process while the viewer opens the
# world (RESOURCE_LAYERS.md §15), for the ETA shown in the legend.
STAGES = (('climate', 'productivité climatique', .15),
          ('tectonics', 'tectonique, roches attendues, climat, eau', 11.),
          ('environment', 'environnement Render v2 et biologie', 11.),
          ('geology', 'géologie réalisée et matériaux', 8.),
          ('deposits', 'gisements et stocks', 10.))
_COMPILES = {}          # world key -> _Compile (one background job per world)


ATLAS_LAYERS = ('crust', 'boundaries', 'convergence', 'hotspot', 'backArc', 'uplift', 'temperature_summer',
                'temperature_winter', 'precip_summer', 'precip_winter', 'soil_sand')


class _Compile:
    """One serialized background compiler; immutable stages stay in memory.

    NumPy/SciPy release the GIL for bulk work. HTTP never waits for compilation,
    and neither atlas arrays nor growing stage snapshots cross a process boundary.
    """
    _compute_lock = threading.Lock()

    def __init__(self, key, atlas):
        self.key, self.root = key, _cache_root()
        self.bounds = _atlas_bounds(atlas)
        self.started = time.perf_counter()
        self._fields, self._result, self._error = None, None, None
        self._stages, self._lock = [], threading.Lock()
        self._v2_cached = False
        self._thread = threading.Thread(target=self._run, args=(atlas,), name='resources-compile', daemon=True)
        self._thread.start()

    def _run(self, atlas):
        import terrain_resources as R
        from terrain_resource_cache import save_world
        source = getattr(atlas, 'source', atlas)
        try:
            with self._compute_lock:
                v2 = _load_npz(self.root / f'{_v2_identity(atlas)}-v2.npz')
                self._v2_cached = v2 is not None
                def publish(stage, fields):
                    snapshot = _slim(fields)
                    with self._lock:
                        self._fields = snapshot
                        self._stages.append([stage, time.perf_counter() - self.started])
                grid, fields, deposits = R.compile_world(source.layers, source.height_m, int(source.seed),
                    fantasy=fantasy_enabled(), bounds=self.bounds, v2_fields=v2, progress=publish)
                seconds = time.perf_counter() - self.started
                with self._lock:
                    self._result = grid, _slim(fields), deposits, seconds, 'compiled'
            # Atomic durable cache publication is outside the compute lock and
            # outside the first-layer critical path. Errors cannot discard a
            # successfully compiled world already available to HTTP clients.
            try:
                save_world(self.root, self.key, self._result[1], deposits)
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Cannot save resource cache %s', self.key)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Cannot compile resources %s', self.key)
            self._error = f'{type(exc).__name__}: {exc}'

    @property
    def error(self):
        return self._error

    @property
    def result(self):
        with self._lock:
            return self._result

    @property
    def fields(self):
        with self._lock:
            return self._fields

    def status(self):
        with self._lock:
            stages = list(self._stages)
            ready = self._result is not None
        done = [n for n, _ in stages]
        current = next((n for n, _, _ in STAGES if n not in done), None)
        label = dict((n, l) for n, l, _ in STAGES).get(current, '')
        elapsed = time.perf_counter() - self.started
        eta = sum(sec for n, _, sec in STAGES if n not in done and not (n == 'environment' and self._v2_cached))
        state = 'error' if self.error else 'ready' if ready else 'compiling'
        return dict(state=state, stage=current if state == 'compiling' else state, stages_done=done, stage_label=label,
                    elapsed_s=round(elapsed, 1), eta_s=round(eta, 1), progress=round(len(done)/len(STAGES), 2), error=self.error,
                    message=self.error or (f'Calcul des ressources du monde : {label} (étape {len(done)+1}/{len(STAGES)}, '
                                           f'~{eta:.0f} s)' if state == 'compiling' else 'prêt'))


def _load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) > 20:
                # ZipFile serializes short reads, while zlib releases the GIL
                # during inflation. Arrays remain lossless and read-only tasks
                # share one archive; no extra file handles or pickling.
                from terrain_resource_workers import thread_map
                return dict(zip(data.files, thread_map(data.__getitem__, data.files)))
            return {k: data[k] for k in data.files}
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _from_disk(key, bounds, stage=False):
    """The compiled world from its cache, or (stage=True) from the final stage
    snapshot while the compressed cache is still being written."""
    import terrain_resources as R
    root = _cache_root()
    started = time.perf_counter()
    candidates = [(root / f'{key}.npz', root / f'{key}.json')]
    if stage:
        candidates.insert(0, (root / f'{key}-stage.npz', root / f'{key}-deposits.json'))
    for npz, deps in candidates:
        if not npz.is_file() or not deps.is_file():
            continue
        fields = _load_npz(npz)
        try:
            deposits = json.loads(deps.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if fields is None:
            continue
        grid = R.Lattice(R.WORLD_LOD, bounds=bounds)
        R.stock_layers(grid, fields, deposits)
        return grid, _slim(fields), deposits, time.perf_counter() - started, 'disk'
    return None


def _atlas_bounds(atlas):
    import terrain_resources as R
    return tuple(float(b) for b in getattr(atlas, 'bounds', R.BOUNDS))


def compile_status(atlas):
    """Start (or report) the world's resources: never waits for the compile."""
    source = getattr(atlas, 'source', atlas)
    key = _identity(atlas)
    with _LOCK:
        cached = getattr(source, '_terrain_resources', None)
        if cached is not None and cached[0] == key:
            return dict(state='ready', stage='ready', progress=1., eta_s=0., message='prêt', origin=cached[1][4])
        job = _COMPILES.get(key)
        if job is not None and job.result is not None:
            source._terrain_resources = (key, job.result)
            del _COMPILES[key]
            return dict(state='ready', stage='ready', progress=1., eta_s=0., message='prêt', origin='compiled')
        if job is None or job.error:
            result = _from_disk(key, _atlas_bounds(atlas))
            if result is not None:
                source._terrain_resources = (key, result)
                return dict(state='ready', stage='ready', progress=1., eta_s=0., message='prêt', origin='disk')
            if job is None or job.error and time.perf_counter() - job.started > 30:
                job = _COMPILES[key] = _Compile(key, atlas)
        return job.status()


def world_resources(atlas, partial=False):
    """(grid, fields, deposits, seconds, origin) for an Orogen atlas, cached.

    Never compiles in the caller's thread: while the background compile runs,
    raises ResourcesPending, or with ``partial`` returns the fields of the
    stages done so far (origin 'partial', deposits [] until the last stage)."""
    import terrain_resources as R
    status = compile_status(atlas)
    source = getattr(atlas, 'source', atlas)
    if status['state'] == 'ready':
        return source._terrain_resources[1]
    job = _COMPILES.get(_identity(atlas))
    if partial and job is not None and job.fields is not None:
        return R.Lattice(R.WORLD_LOD, bounds=job.bounds), job.fields, [], time.perf_counter() - job.started, 'partial'
    raise ResourcesPending(status)


# ---------------------------------------------------------------------------
# Rendering

def _hex(colors):
    return np.array([tuple(bytes.fromhex(c)) for c in colors], np.float64) / 255


def _sample(grid, value, xs, ys):
    """Bilinear between LOD-10 cell centres; longitude wraps, latitude clamps."""
    import terrain_resources as R
    return R.sample_lattice(grid, value, xs, ys)



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
    try:
        grid, fields, deposits, _, _ = world_resources(atlas, partial=True)
    except ResourcesPending:              # nothing compiled yet: neutral grey
        return np.full((len(ys), len(xs), 3), .16)
    land, base = _base(grid, fields, xs, ys)
    try:
        return _render(spec, mode, grid, fields, deposits, land, base, xs, ys)
    except KeyError:                      # stage not compiled yet: plain land and sea
        return base


def _render(spec, mode, grid, fields, deposits, land, base, xs, ys):
    if spec['kind'] == 'points':
        return draw_deposits(grid, deposits, xs, ys, base, mode.removeprefix(PREFIX+'dep-').replace('-', '_'))
    value = np.asarray(spec['value'](fields, grid), np.float64)
    if spec.get('seasonal'):
        value = value.mean(axis=0)          # overview: annual mean of the monthly fields
    if spec['kind'] == 'mix':
        mix = _sample(grid, value, xs, ys)
        rgb = np.einsum('kyx,kc->yxc', mix, _hex(spec['colors']))
        return np.clip(rgb * np.maximum(land, .35)[..., None], 0, 1)
    if spec['kind'] in ('classes', 'cyclic'):
        # Categorical: nearest cell, never interpolated.
        s = grid.size
        xw = (np.asarray(xs, np.float64) - grid.bounds[0]) % (grid.bounds[2] - grid.bounds[0]) + grid.bounds[0]
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


def draw_deposits(grid, deposits, xs, ys, base, metal='points'):
    """CPU overview fallback: recoverable-mass symbols, filtered by resource."""
    import terrain_resources as R
    rgb = base.copy()
    mass_of = lambda d: (sum(v/(1000 if m in R.GRAM_METALS else 1) for m,v in d['medieval_recoverable'].items())
                         if metal == 'points' else d['medieval_recoverable'].get(metal,0)/(1000 if metal in R.GRAM_METALS else 1))
    deposits = [d for d in deposits if mass_of(d)>0]
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    dx = abs(xs[1] - xs[0]) if len(xs) > 1 else 1.
    if not deposits:
        return rgb
    pos = np.array([[(d['position_i'] + .5) * R.cell_size(d['position_lod']), (d['position_j'] + .5) * R.cell_size(d['position_lod'])]
                    for d in deposits])
    span = grid.bounds[2] - grid.bounds[0]
    xc = (xs[0] + xs[-1]) / 2
    px = pos[:, 0] + np.round((xc - pos[:, 0]) / span) * span
    radius_px = np.array([math.sqrt(9+5*min(8,max(0,math.log10(max(mass_of(d),1))))) for d in deposits])
    margin = (radius_px.max() + 1) * dx
    keep = (px > xs[0] - margin) & (px < xs[-1] + margin) & (pos[:, 1] > ys[0] - margin) & (pos[:, 1] < ys[-1] + margin)
    order = np.argsort([mass_of(d) for d in deposits], kind='stable')
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
        color = _hex([FAMILY_COLORS.get(d['family'], 'ffffff')])[0]
        disc = np.clip(r + .5 - dist, 0, 1)[..., None]
        ring = (np.clip(r + 1.5 - dist, 0, 1) - np.clip(r + .5 - dist, 0, 1))[..., None]
        patch = rgb[y0:y1, x0:x1]
        patch[:] = patch * (1 - disc) + color * disc
        patch[:] = patch * (1 - ring) + np.array([0., 0., 0.]) * ring
    return rgb


def inspect(atlas, x, y, radius_m=60000., mode=None):
    """Values under the pointer and nearby deposits, for the viewer panel."""
    import terrain_resources as R
    grid, fields, deposits, seconds, origin = world_resources(atlas)
    s = grid.size
    xw = (float(x) - grid.bounds[0]) % (grid.bounds[2] - grid.bounds[0]) + grid.bounds[0]
    i = int(np.clip(math.floor(xw / s) - grid.i[0], 0, len(grid.i) - 1))
    j = int(np.clip(math.floor(float(y) / s) - grid.j[0], 0, len(grid.j) - 1))
    values = {}
    for value_mode, spec in specs().items():
        if spec['kind'] in ('points', 'mix'):
            continue
        arr = np.asarray(spec['value'](fields, grid), np.float64)
        v = float(arr[:, j, i].mean() if arr.ndim == 3 else arr[j, i])     # seasonal: annual mean
        values[value_mode] = dict(label=spec['label'], value=None if not math.isfinite(v) else round(v, 4), unit=spec['unit'],
                            status=spec['status'], proxy=spec['proxy'])
    dominant = dict(context=R.CONTEXT_LABELS[int(np.argmax([fields['context_' + c][j, i] for c in R.CONTEXTS]))],
                    rock=R.ROCK_LABELS[int(np.argmax([fields['rock_' + r][j, i] for r in R.ROCKS]))])
    near = []
    state = world_state(atlas)
    for n, d in enumerate(deposits):
        px = (d['position_i'] + .5) * R.cell_size(d['position_lod'])
        py = (d['position_j'] + .5) * R.cell_size(d['position_lod'])
        px, py = (state.dep_di[n] + .5) * 30, (state.dep_dj[n] + .5) * 30
        span = grid.bounds[2] - grid.bounds[0]
        dx = (px - xw + span / 2) % span - span / 2
        dist = math.hypot(dx, py - float(y))
        if dist <= radius_m + state.dep_a[n]:
            near.append(dict(d, distance_m=round(dist), label=FAMILY_LABELS.get(d['family'], d['family']),
                             footprint_m=[round(2*state.dep_a[n], 1), round(2*state.dep_b[n], 1)]))
    if mode and mode.startswith(PREFIX + 'dep-') and mode != PREFIX + 'dep-points':
        metal = mode.removeprefix(PREFIX + 'dep-').replace('-', '_')
        near = [d for d in near if d['medieval_recoverable'].get(metal, 0) > 0]
    near.sort(key=lambda d: d['distance_m'])
    return dict(cell=dict(lod=grid.lod, i=int(grid.i[i]), j=int(grid.j[j]), id=str(int(grid.ids[j, i])),
                          area_km2=round(float(grid.area_m2[j, i]) / 1e6, 1)),
                dominant=dominant, values=values, deposits=near[:12], cache=origin, seconds=round(seconds, 2))


# ---------------------------------------------------------------------------
# GPU packages (served once per world and mode) and the CPU fallback tile

VEIN = ('small_pb_ag', 'small_sn', 'polymetallic_vein', 'epithermal_au_ag', 'sn_w_granite', 'orogenic_au')
STRATA = dict(small_coal=2., local_ironstone=3., gypsum_quarry=3., salt_spring=10., bif_iron=50., ironstone=8., mvt_pb_zn=15., sediment_cu=4., evaporite_salt=30., coal=2.)
AMAS = dict(bog_iron=1., porphyry_cu=300., vms=40., skarn_fe_cu=60., laterite_fe=8., fantasy_star_metal=50.)
PLACER = ('placer_au', 'placer_sn')
PROSPECT_ALL = PREFIX + 'dep-prospect-all'
CRYSTALLINE = ('ultramafic', 'basalt', 'metamorphic', 'granite')
SEDIMENTARY = ('evaporite', 'marl', 'limestone', 'sandstone')
CONTEXT_ORDER = ('oceanic', 'passive', 'platform', 'craton', 'backarc', 'rift', 'hotspot', 'arc', 'ophiolite', 'collision')
_AUTH_LOCK = threading.RLock()


def profile_of(mode):
    m = mode.split(PREFIX, 1)[-1]
    rules = [(('potential', 'prospect', 'dep-points'), 'mineral'), (('cereal', 'arable'), 'arable'), (('lsu', 'forage', 'transhumance'), 'pasture'),
             (('tree', 'canopy', 'conifer', 'timber', 'wood', 'felling'), 'forest'),
             (('peat', 'bog-iron', 'fish-inland'), 'wet'), (('fish-marine',), 'marine'), (('season-fish',), 'wet'),
             (('stone', 'slate', 'lime'), 'rock'), (('clay', 'sand', 'salt'), 'lowflat'), (('snow', 'fur'), 'cold'),
             (('growing',), 'season'), (('water-yield', 'runoff'), 'runoff'), (('prospect', 'dep-points'), 'mineral')]
    for keys, name in rules:
        if any(k in m for k in keys):
            return name
    return 'neutral'


class _World:
    """Per-world state: authority arrays (LRU), structure, deposits as arrays, packages."""

    def __init__(self, grid, fields, deposits, seed, identity):
        import terrain_resources as R
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        self.grid, self.fields, self.deposits, self.seed, self.identity = grid, fields, deposits, seed, identity
        self.cache, self.packages = OrderedDict(), OrderedDict()
        land = np.asarray(fields['land'], bool)
        # Coastal extrapolation: nearest land (or sea) cell value, so a fine coast
        # inside an authority sea cell still has a parent value.
        self.nearest_land = distance_transform_edt(~land, return_distances=False, return_indices=True) if (~land).any() else None
        self.nearest_sea = distance_transform_edt(land, return_distances=False, return_indices=True) if land.any() else None
        # Parallel to orogenic belts (stored by the compiler; recomputed for older caches).
        self.strike = (np.asarray(fields['structural_strike'], np.float64) if 'structural_strike' in fields
                       else R.structural_strike(fields) if 'context_arc' in fields else None)
        self.seed32 = R.key(seed, ['resources', 'gpu']) & 0xffffffff
        n = len(deposits)
        self.dep_family = np.array([d['family'] for d in deposits]) if n else np.array([], str)
        offs = [R.key(seed, ['resources', 'deposit-lod0', d['id']]) for d in deposits]
        span = np.array([1 << d['position_lod'] for d in deposits], np.int64) if n else np.zeros(0, np.int64)
        pi = np.array([d['position_i'] for d in deposits], np.int64) * span if n else np.zeros(0, np.int64)
        pj = np.array([d['position_j'] for d in deposits], np.int64) * span if n else np.zeros(0, np.int64)
        # LOD-0 cell inside the deposit's LOD-3 cell (keyed, never re-drawn).
        self.dep_di = pi + np.array([k % 997 for k in offs], np.int64) * span // 997 if n else pi
        self.dep_dj = pj + np.array([(k >> 20) % 991 for k in offs], np.int64) * span // 991 if n else pj
        self.dep_ore = np.array([d['ore_t'] for d in deposits], np.float64) if n else np.zeros(0)
        self.dep_access = np.array([d['medieval_accessible_share'] for d in deposits]) if n else np.zeros(0)
        self.dep_gossan = np.array([d['gossan'] for d in deposits], bool) if n else np.zeros(0, bool)
        jit = np.array([((k >> 40) % 1000) / 1000 - .5 for k in offs]) if n else np.zeros(0)
        ci = (self.dep_di >> 10) - grid.i[0]
        cj = np.clip((self.dep_dj >> 10) - grid.j[0], 0, len(grid.j) - 1)
        self.dep_theta = self.strike[cj, ci % len(grid.i)] + .5 * jit if n else np.zeros(0)
        # Canonical regional relief bends the lobe chain. Placer axes follow
        # descent; primary veins retain the structural strike. Fine relief is
        # read by the shader for appearance, without multiplying stock density.
        gy, gx = np.gradient(np.asarray(fields['height_m'], np.float64))
        self.dep_bend = np.tanh((gx[cj, ci % len(grid.i)] * np.cos(self.dep_theta)
                                + gy[cj, ci % len(grid.i)] * np.sin(self.dep_theta)) / 300.)
        fam = self.dep_family
        thick = np.array([AMAS.get(f, STRATA.get(f, 30.)) for f in fam]) if n else np.zeros(0)
        strata = np.isin(fam, list(STRATA))
        aspect = np.where(strata, 5., 1.6)
        a = np.sqrt(self.dep_ore / 2.7 / np.maximum(thick, 1) * aspect / np.pi)
        b = a / aspect
        vein, placer = np.isin(fam, VEIN), np.isin(fam, PLACER)
        self.dep_theta = np.where(placer, np.arctan2(gy[cj, ci % len(grid.i)], gx[cj, ci % len(grid.i)]), self.dep_theta)
        self.dep_a = np.where(placer, np.clip(400 * (self.dep_ore / 1e5) ** .5, 200, 4000) / 2,
                              np.where(vein, np.clip(120 * (self.dep_ore / 1e4) ** .45, 60, 5000) / 2, a))
        self.dep_b = np.where(placer, 60., np.where(vein, np.clip(1.5 * (self.dep_ore / 1e4) ** .25, 1, 15) / 2, b))
        self.dep_shape = np.where(vein, 0, np.where(placer, 3, np.where(strata, 2, 1)))
        names = list(FAMILY_LABELS)
        self.dep_color = np.array([names.index(f) if f in names else 0 for f in fam], np.int32)
        self.dep_mass = {}
        for kind in ('contained', 'medieval_recoverable'):
            for d_idx, d in enumerate(deposits):
                for m, v in d[kind].items():
                    if (kind, m) not in self.dep_mass:
                        self.dep_mass[kind, m] = np.zeros(n)
                    self.dep_mass[kind, m][d_idx] = v

    def authority(self, mode):
        with _AUTH_LOCK:
            if mode in self.cache:
                self.cache.move_to_end(mode)
                return self.cache[mode]
        if mode == PROSPECT_ALL:
            keys = [k for k in self.fields if k.startswith('prospect_')]
            if not keys:
                raise KeyError('prospect_')
            v = np.clip(sum(np.asarray(self.fields[k], np.float64) for k in keys), 0, 1)
        else:
            v = np.asarray(specs()[mode]['value'](self.fields, self.grid), np.float64)
        v = np.nan_to_num(v)
        idx = self.nearest_sea if 'fish-marine' in mode else self.nearest_land
        if idx is not None and not mode.startswith(PREFIX + 'geo-') and 'harvest' not in mode:
            v = v[..., idx[0], idx[1]]
        with _AUTH_LOCK:
            self.cache[mode] = v
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
        return v


def world_state(atlas):
    source = getattr(atlas, 'source', atlas)
    grid, fields, deposits, _, _ = world_resources(atlas, partial=True)
    with _LOCK:
        state = getattr(source, '_terrain_resources_state', None)
        if state is None or state.fields is not fields:
            state = _World(grid, fields, deposits, int(getattr(source, 'seed', 0)), _identity(atlas))
            source._terrain_resources_state = state
        return state


def _bounds_of(spec):
    if spec.get('physical_bounds') is not None:
        return tuple(spec['physical_bounds'])
    if spec['unit'] in ('fraction', 'indice 0–1') or spec['label'].startswith('Prospectivité'):
        return (0., 1.)
    if spec['unit'] == 'mois':
        return (0., 12.)
    return None


def _mean_preserving(v, bounded=None):
    """One pycnophylactic iteration: bilinear cell means of the result ~ v (wrap in longitude)."""
    q = np.concatenate((v[..., -1:], v, v[..., :1]), axis=-1)
    q = np.concatenate((q[..., :1, :], q, q[..., -1:, :]), axis=-2)
    rows = (q[..., :-2, :] + 6 * q[..., 1:-1, :] + q[..., 2:, :]) / 8
    s = (rows[..., :, :-2] + 6 * rows[..., :, 1:-1] + rows[..., :, 2:]) / 8
    out = np.maximum(2 * v - s, 0)
    if bounded is not None:
        out = np.clip(out, bounded[0], bounded[1])
    return out


def _kind_of(mode, spec):
    if '-stock-' in mode or '-recoverable-' in mode:
        return 'stock'
    if spec['kind'] in ('mix', 'classes', 'cyclic', 'points'):
        return spec['kind']
    return 'log' if spec['log'] else 'ramp'


def gpu_package(atlas, mode):
    """(header dict, binary payload) for the browser evaluator; cached per world and mode.
    Raises ResourcesPending while the stage this layer needs is still compiling."""
    state = world_state(atlas)
    if specs()[mode]['kind'] == 'points' and not state.deposits:
        # The deposits stage publishes prospectivities just before its site
        # JSON is committed. Never cache that brief partial state as an empty
        # inventory in the browser for the rest of the session.
        status = compile_status(atlas)
        if status['state'] != 'ready':
            raise ResourcesPending(status)
    try:
        return _gpu_package(state, mode)
    except KeyError as exc:
        status = compile_status(atlas)
        if status['state'] == 'ready':
            raise
        raise ResourcesPending(dict(status, waiting_for=str(exc))) from None


def _gpu_package(state, mode):
    import terrain_resources as R
    import terrain_resource_gpu as G
    with _LOCK:
        if mode in state.packages:
            state.packages.move_to_end(mode)
            return state.packages[mode]
    spec = specs()[mode]
    kind = _kind_of(mode, spec)
    grid = state.grid
    land = np.asarray(state.fields['land'], bool)
    bounded = _bounds_of(spec)
    groups, classes = [], 0
    if kind == 'mix' and mode.endswith('geo-rock'):
        # Realized geology: the canonical shares come from the same function (§5.1, §16.3).
        names = R.ROCKS
        expected = np.stack([np.asarray(state.fields['rock_expected_' + n], np.float64) for n in names])
        field = R.realized_rock_field(expected, state.strike, state.fields['rock_bias'])
        classes = len(names)
    elif kind == 'mix':
        names = R.CONTEXTS
        frac = np.asarray(state.authority(mode), np.float64)
        # Display only: the oceanic context keeps a 2 % floor on land in the
        # authority (soft membership); land cells show the other contexts.
        ocean = names.index('oceanic')
        rest = frac.sum(axis=0) - frac[ocean]
        frac[ocean] = np.where(land & (rest > 0), 0., frac[ocean])
        frac = frac / np.maximum(frac.sum(axis=0), 1e-12)
        field = np.concatenate((frac, np.cos(2 * state.strike)[None], np.sin(2 * state.strike)[None])).astype(np.float32)
        # Context fractions are soft memberships (distance kernels to plate
        # boundaries), not area shares: shown as the dominant context.
        classes = len(names)
    elif kind == 'stock':
        field = np.zeros((1,) + grid.shape, np.float32)
    elif kind == 'points':
        if not state.deposits and not any(k.startswith('prospect_') for k in state.fields):
            raise KeyError('prospect_')
        field = np.zeros((1,) + grid.shape, np.float32)
        bounded = (0., 1.)
    elif kind == 'classes':
        field = _mean_preserving(state.authority(PREFIX + 'agri-cereal-suitability')[None], (0., 1.)).astype(np.float32)
        bounded = (0., 1.)
    elif kind == 'cyclic':
        v = np.asarray(state.authority(mode), np.float64)
        field = np.where(land, np.where(v >= 255, -1, v), -1)[None].astype(np.float32)
    else:
        v = np.asarray(state.authority(mode), np.float64)
        v = v if v.ndim == 3 else v[None]
        field = _mean_preserving(v, bounded).astype(np.float32)
    names_all = list(specs())
    realized = kind == 'mix' and mode.endswith('geo-rock')
    layer = R.ROCK_LAYER if realized else names_all.index(mode) + 1
    coef, sigma = G.PROFILES[profile_of(mode)]
    pkg = dict(seed32=state.seed32, layer=layer, field=G.Field(field, grid.i[0], grid.j[0], grid.bounds), classes=classes,
               groups=groups, seasonal=bool(spec.get('seasonal')), season=.5, profile=profile_of(mode),
               range=tuple(spec['range']), bounded=bounded is not None, bounds=tuple(bounded or spec['range']))
    if realized:
        pkg.update(aligned=R.ROCK_ALIGNED, groups=R.ROCK_GROUPS, group_aligned=R.ROCK_GROUP_ALIGNED, beta=G.REALIZED_BETA)
    deposits = buckets = None
    if kind in ('points', 'stock'):
        deposits, buckets = _deposit_buffers(state, mode, kind)
    palette = [tuple(bytes.fromhex(c)) for c in spec['colors']]
    if kind == 'points':
        palette = [tuple(bytes.fromhex(FAMILY_COLORS[k])) for k in FAMILY_LABELS]
    # Realized geology, in `order`: per class (belt flag << 4 | group), then per group belt flag.
    order = ([int(a) << 4 | g for a, g in zip(R.ROCK_ALIGNED, R.ROCK_GROUPS)] + [int(a) for a in R.ROCK_GROUP_ALIGNED]
             if realized else [])
    header = dict(mode=mode, kind=G.KINDS[kind], lattice=[int(grid.i[0]), int(grid.j[0]), int(grid.shape[1]), int(grid.shape[0])],
                  world_bounds=list(grid.bounds), layers=int(field.shape[0]), classes=classes, seasonal=pkg['seasonal'], range=list(spec['range']),
                  bounded=pkg['bounded'], bounds=list(pkg['bounds']), log=bool(spec['log']), profile=G.PROFILE_NAMES.index(profile_of(mode)),
                  coef=list(coef), sigma=sigma, seed32=int(state.seed32), layer=layer,
                  groups=[], realized=realized, beta=list(G.REALIZED_BETA) if realized else [0, 0], order=order, thresholds=0, scale=0., max_footprint=float(np.max(deposits[:, 4:6], initial=0) * 2.4) if deposits is not None else 0.,
                  palette=[[c / 255 for c in rgb] for rgb in palette], deposits=0 if deposits is None else int(len(deposits)),
                  field_shape=list(field.shape))
    parts = [field.astype(np.float16).tobytes()]
    if deposits is not None:
        size = sum(len(x) for x in parts)
        parts.append(bytes(-size % 4))            # 32-bit alignment of the deposit words
        parts.extend([deposits.tobytes(), buckets.tobytes()])
    result = (header, pkg, b''.join(parts))
    with _LOCK:
        state.packages[mode] = result
        while len(state.packages) > 6:
            state.packages.popitem(last=False)
    return result


def _deposit_buffers(state, mode, kind):
    """Deposits sorted by LOD-10 bucket: (N, 12) 32-bit words and (H, W, 2) start/count."""
    grid = state.grid
    if kind == 'stock':
        metal = (mode.split('-stock-')[-1] if '-stock-' in mode else mode.split('-recoverable-')[-1]).replace('-', '_')
        mass = state.dep_mass.get(('contained' if '-stock-' in mode else 'medieval_recoverable', metal), np.zeros(len(state.dep_ore)))
        keep = mass > 0
    else:
        import terrain_resources as R
        metal = mode.removeprefix(PREFIX + 'dep-').replace('-', '_')
        if metal == 'points':
            mass = sum((v / (1000 if m in R.GRAM_METALS else 1) for (k, m), v in state.dep_mass.items()
                        if k == 'medieval_recoverable'), np.zeros(len(state.dep_ore)))
        else:
            mass = state.dep_mass.get(('medieval_recoverable', metal), np.zeros(len(state.dep_ore)))
            if metal in R.GRAM_METALS:
                mass = mass / 1000  # common physical tonnes for symbol scale
        keep = mass > 0
    idx = np.flatnonzero(keep)
    ci = ((state.dep_di[idx] >> 10) - grid.i[0]) % len(grid.i)
    cj = np.clip((state.dep_dj[idx] >> 10) - grid.j[0], 0, len(grid.j) - 1)
    cell = cj * len(grid.i) + ci
    order = np.argsort(cell, kind='stable')
    idx, cell = idx[order], cell[order]
    words = np.zeros((len(idx), 12), np.float32)
    ints = words.view(np.int32)
    ints[:, 0], ints[:, 1] = state.dep_di[idx], state.dep_dj[idx]
    ints[:, 2], ints[:, 3] = state.dep_shape[idx], state.dep_color[idx]
    words[:, 4], words[:, 5], words[:, 6] = state.dep_a[idx], state.dep_b[idx], state.dep_theta[idx]
    words[:, 7] = state.dep_bend[idx] if kind == 'stock' else state.dep_access[idx]
    words[:, 8] = state.dep_gossan[idx]
    words[:, 9] = np.sqrt(9. + 5. * np.clip(np.log10(np.maximum(mass[idx], 1.)), 0., 8.))
    words[:, 10] = np.sqrt(state.dep_ore[idx] / 2.7 / 30.) / 2
    words[:, 11] = mass[idx]
    buckets = np.zeros((len(grid.j), len(grid.i), 2), np.uint32)
    if len(idx):
        uniq, start, count = np.unique(cell, return_index=True, return_counts=True)
        buckets.reshape(-1, 2)[uniq, 0] = start
        buckets.reshape(-1, 2)[uniq, 1] = count
    return words, buckets


def package_deposits(pkg_bytes_header, words):
    """Deposit records as dicts for the NumPy mirror."""
    ints = words.view(np.int32)
    return [dict(di=int(ints[k, 0]), dj=int(ints[k, 1]), shape=int(ints[k, 2]), color=int(ints[k, 3]), a=float(words[k, 4]),
                 b=float(words[k, 5]), theta=float(words[k, 6]), access=float(words[k, 7]), gossan=float(words[k, 8]),
                 symbol=float(words[k, 9]), sigma=float(words[k, 10]), mass=float(words[k, 11])) for k in range(len(words))]
