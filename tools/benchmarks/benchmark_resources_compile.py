"""Resource authority timings, excluding Orogen generation; no shared cache deletion.

First run --prepare to retain the exact viewer atlas inputs, then --label before
or after. Reports include cold subprocess, warm process, disk load, and cProfile.
"""
from pathlib import Path
import argparse
import cProfile
import hashlib
import json
import os
import pstats
import subprocess
import sys
import time

PROCESS_STARTED = time.perf_counter()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import activate
activate()
import numpy as np


def fingerprint(fields, deposits):
    arrays = {}
    for name, value in sorted(fields.items()):
        if isinstance(value, np.ndarray):
            arrays[name] = dict(dtype=value.dtype.str, shape=list(value.shape),
                                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    return dict(arrays=arrays, deposits=len(deposits),
                metadata={k:v for k,v in fields.items() if not isinstance(v,np.ndarray)},
                deposits_sha256=deposits.fingerprint() if hasattr(deposits,'fingerprint') else hashlib.sha256(json.dumps(deposits, sort_keys=True).encode()).hexdigest())


def load_inputs(path):
    from terrain_resource_layers import ATLAS_LAYERS
    with np.load(path, allow_pickle=False) as data:
        return ({k[6:]: data[k] for k in data.files if k.startswith('layer_') and
                 (k[6:] in ATLAS_LAYERS or k[6:].startswith('pedology_'))},
                data['height_m'], int(data['seed']), tuple(data['bounds']))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', default='E:/TerrainDiffusionRuntime/resources/perf-inputs')
    parser.add_argument('--output', default='output/resources-perf')
    parser.add_argument('--seeds', default='42,20261027')
    parser.add_argument('--label', default='before')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--profile',default='orogen',help='Orogen profile whose retained atlas supplies the benchmark')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--serial', action='store_true', help='Disable worker processes (recorded in report)')
    parser.add_argument('--reference', action='store_true', help='Load the pinned compiler via git show, without editing the worktree')
    args = parser.parse_args()
    directory, output = Path(args.directory), Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.prepare:
        from terrain_orogen import get_heightmap
        from terrain_generation import resolve_generation
        from terrain_geometry import world_bounds
        descriptor = resolve_generation(args.profile)
        for seed in seeds:
            from terrain_resource_layers import ATLAS_LAYERS
            atlas=get_heightmap(seed,descriptor.bootstrap_style,options=descriptor.bootstrap_options)
            height,layers=atlas.height_m,atlas.layers
            path = directory / f'{seed}.npz'
            np.savez(path, height_m=height, seed=np.int64(seed), bounds=world_bounds(descriptor.settings),
                     **{'layer_' + k: v for k, v in layers.items() if k in ATLAS_LAYERS or k.startswith('pedology_')})
            print('Prepared', path, flush=True)
        return
    if not args.child:
        results = []
        for seed in seeds:
            started = time.perf_counter()
            child=subprocess.Popen([sys.executable, __file__, '--child', '--seeds', str(seed), '--label', args.label,
                            '--directory', str(directory), '--output', str(output)] +
                           (['--serial'] if args.serial else []) + (['--reference'] if args.reference else []),
                           stdout=subprocess.PIPE,text=True)
            cold_executable=None
            for line in child.stdout:
                message=json.loads(line)
                if message.get('event') == 'cold_compile_ready':cold_executable=time.perf_counter()-started
                else:print(line.strip(),flush=True)
            if child.wait():raise subprocess.CalledProcessError(child.returncode,child.args)
            result = json.loads((output / f'{args.label}-{seed}.json').read_text())
            result['cold_executable_wall_s']=cold_executable
            result['benchmark_batch_wall_s'] = time.perf_counter() - started
            (output / f'{args.label}-{seed}.json').write_text(json.dumps(result,indent=2))
            results.append(result)
        (output / f'{args.label}.json').write_text(json.dumps(results, indent=2))
        print(json.dumps([{k:v for k,v in r.items() if k not in ('fingerprint','statistics','statistical_deltas')} for r in results], indent=2))
        return
    import terrain_resources as R
    import terrain_resource_workers as W
    import terrain_resource_layers as L
    if args.reference:
        # All three canonical modules are pinned, including imports used inside
        # reference functions. Run serially so no child can import edited code.
        import types
        for name in ('terrain_resource_gpu', 'terrain_resources'):
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / 'backend' / (name+'.py'))
            source = subprocess.check_output(['git','show',f'e2046f9:backend/{name}.py'], cwd=ROOT)
            sys.modules[name] = module
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
        R = sys.modules['terrain_resources']
    from types import SimpleNamespace
    if not args.reference:
        from terrain_resource_acceleration import prepare
        prepare()
    seed = seeds[0]
    # Production runs in its background thread without subdividing geology:
    # these small arrays are slower when NumPy calls compete across threads.
    parallel = None
    layers, height, _, bounds = load_inputs(directory / f'{seed}.npz')
    stages, last = {}, time.perf_counter()
    def progress(name, fields):
        nonlocal last
        now = time.perf_counter()
        stages[name] = now - last
        last = now
    started = last
    grid, fields, deposits = R.compile_world(layers, height, seed, bounds=bounds, parallel=parallel, progress=progress)
    cold = time.perf_counter() - started
    cold_wall = time.perf_counter() - PROCESS_STARTED
    print(json.dumps(dict(event='cold_compile_ready',seed=seed)),flush=True)
    from tools.verification.resource_statistics import world_statistics,compare_statistics
    statistics=world_statistics(fields,deposits)
    input_identity=fingerprint(dict(layers,height_m=height),[])
    references=json.loads((ROOT/'tests/fixtures/resources-statistics-round1.json').read_text())['worlds']
    expected=next((entry for entry in references.values() if entry['inputs']==input_identity),None)
    deltas=compare_statistics(expected['statistics'],statistics) if expected is not None else None
    fp = fingerprint(fields, deposits)
    started = time.perf_counter()
    _, warm_fields, warm_deposits = R.compile_world(layers, height, seed, bounds=bounds, parallel=parallel)
    warm = time.perf_counter() - started
    assert fingerprint(warm_fields, warm_deposits) == fp, 'non-deterministic compile'
    profiler = cProfile.Profile()
    profiler.runcall(R.compile_world, layers, height, seed, bounds=bounds)
    profiler.dump_stats(str(output / f'{args.label}-{seed}.prof'))
    with (output / f'{args.label}-{seed}-profile.txt').open('w') as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumtime').print_stats(45)
    # Exercise the production loader with an isolated cache, including stock reconstruction.
    os.environ['TERRAIN_RESOURCES_CACHE'] = str(output.resolve() / 'cache' / args.label / str(seed))
    root = L._cache_root()
    root.mkdir(parents=True, exist_ok=True)
    arrays = {k:v for k,v in L._slim(fields).items() if isinstance(v, np.ndarray) and not k.startswith(L._DERIVED)}
    started = time.perf_counter()
    if args.reference:
        np.savez_compressed(root / 'benchmark.npz', **arrays)
        (root / 'benchmark.json').write_text(json.dumps(deposits))
    else:
        from terrain_resource_cache import save_world
        save_world(root, 'benchmark', arrays, deposits)
    save = time.perf_counter() - started
    started = time.perf_counter()
    disk = L._from_disk('benchmark', bounds)
    disk_s = time.perf_counter() - started
    assert disk is not None
    result = dict(seed=seed, shape=grid.shape, serial=args.serial or args.reference, reference=args.reference,
                  cold_process_wall_s=cold_wall, compile_cold_s=cold, compile_warm_s=warm,
                  setup_and_input_s=cold_wall-cold,stages_s=stages, disk_load_s=disk_s, disk_save_s=save, fingerprint=fp,
                  statistics=statistics,statistical_deltas=deltas)
    (output / f'{args.label}-{seed}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('fingerprint','statistics','statistical_deltas')}), flush=True)


if __name__ == '__main__':
    main()

