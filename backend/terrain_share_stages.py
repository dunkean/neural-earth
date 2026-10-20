"""Portable recipes for retained Orogen stages, including their older parents."""
import hashlib
import json


def export_stages(settings):
    keys = [settings.get('orogen_' + stage + '_stage') for stage in ('relief', 'erosion', 'climate')]
    if not any(keys):
        return {}
    import numpy as np
    from terrain_orogen_stages import _path, _recall, _json
    recipes = {}

    def visit(key):
        if not key or key in recipes:
            return
        path = _path(key)
        artifact = _recall(path)
        if artifact:
            namespace = artifact['metadata']['namespace']
        else:
            with np.load(path, allow_pickle=False) as saved:
                namespace = json.loads(str(saved['metadata']))['namespace']
        if hashlib.sha256(_json(namespace)).hexdigest() != key:
            raise ValueError('Stage recipe identity mismatch')
        recipes[key] = {name: namespace[name] for name in
                        ('stage', 'seed', 'style', 'width', 'height', 'parent', 'settings')}
        visit(namespace['parent'])

    for key in keys:
        visit(key)
    return recipes


def restore_stages(generation):
    recipes = generation.get('stages', {})
    if not recipes:
        if any(generation['settings'].get('orogen_' + s + '_stage') for s in ('relief', 'erosion', 'climate')):
            raise ValueError('Portable recipe is missing retained generation stages')
        return generation
    if not isinstance(recipes, dict) or len(recipes) > 64:
        raise ValueError('Invalid portable stage graph')
    from terrain_generation import _descriptor, _normalize
    import terrain_orogen as core
    from terrain_orogen_stages import load_stage, build_stage, STAGE_KEYS
    seed = int(generation['seed'])
    restored, visiting, configurations = {}, set(), {}

    def visit(key):
        if key in restored:
            return restored[key]
        if key in visiting or key not in recipes:
            raise ValueError('Invalid portable stage dependency')
        visiting.add(key)
        recipe = recipes[key]
        stage = recipe['stage']
        if (stage not in STAGE_KEYS or recipe['seed'] != seed
                or recipe['width'] != core.WIDTH or recipe['height'] != core.HEIGHT
                or set(recipe['settings']) != STAGE_KEYS[stage]):
            raise ValueError('Invalid portable stage settings')
        parent_key = recipe['parent']
        parent = visit(parent_key) if parent_key else None
        if (stage == 'relief') != (parent is None):
            raise ValueError('Invalid portable stage parent')
        settings = dict(configurations[parent_key] if parent_key else generation['settings'])
        settings.update(recipe['settings'])
        for name in STAGE_KEYS:
            settings['orogen_' + name + '_stage'] = ''
        settings = _normalize(generation['profile'], settings)
        configurations[key] = settings
        try:
            artifact = load_stage(key, stage, seed=seed, width=core.WIDTH, height=core.HEIGHT)
        except ValueError:
            descriptor = _descriptor('portable-stage', generation['profile'], settings)
            config = core.style_config(descriptor.bootstrap_style, descriptor.bootstrap_options)
            ancestor = parent_key
            while ancestor and recipes[ancestor]['parent']:
                ancestor = recipes[ancestor]['parent']
            relief = restored.get(ancestor)
            artifact, _ = build_stage(stage, seed, descriptor.bootstrap_style, config,
                                      core.WIDTH, core.HEIGHT, parent, relief)
        restored[key] = artifact
        visiting.remove(key)
        return artifact

    settings = dict(generation['settings'])
    with core._LOCK:
        for stage in STAGE_KEYS:
            key = settings.get('orogen_' + stage + '_stage')
            if key:
                settings['orogen_' + stage + '_stage'] = visit(key)['id']
    return dict(generation, settings=settings)
