"""Curated application model choices and immutable neural-cache identities."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import threading

from distill.common import DATA, REPO, REVISION, atomic_json

STAGES = ('coarse', 'base', 'decoder')
DEFAULT_SELECTION = dict(coarse='solver8', base='teacher', decoder='distilled200k')
RETIRED_CHOICES = {
    'coarse': {'solver4': 'solver8', 'solver1': 'solver2'},
    'base': {key: 'teacher' for key in ('initial128','coupled128','refined128','refined192','nf4_70k')},
    'decoder': {},
}


def catalog(root=DATA, bundled_root=REPO/'distill/weights'):
    """An allowlist: research files can never add application model choices."""
    bundled_root = Path(bundled_root)
    specifications = {
        'coarse': [('solver8','Coarse8 · 25k · qualité','coarse8.pt'),
                   ('solver2','Coarse2 · 200k · rapide','coarse2.pt')],
        'base': [],  # Residual8k belongs to the separate finite40km workflow.
        'decoder': [('distilled200k','Décodeur64 · 200k · qualité','decoder.pt'),
                    ('compact10k','Décodeur16 · 10k · compact','decoder16-10k.pt')],
    }
    manifest_path = bundled_root/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    by_file = {value['file']:value for value in manifest['exports'].values()} if manifest else {}
    result = {}
    for stage, entries in specifications.items():
        result[stage] = {'teacher': dict(label='Original · référence',path=None)}
        for key,label,name in entries:
            path = bundled_root/name
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = by_file.get(name)
            if manifest and (expected is None or expected['sha256']!=digest or expected['stage']!=stage):
                raise ValueError('Bundled model differs from its manifest: '+name)
            result[stage][key] = dict(label=label,path=str(path),sha256=digest,bytes=path.stat().st_size)
    return result


def validate_selection(value, options):
    if not isinstance(value, dict) or set(value) != set(STAGES):
        raise ValueError('Select exactly coarse, base and decoder.')
    if any(not isinstance(value[stage], str) or value[stage] not in options[stage] for stage in STAGES):
        raise ValueError('Unknown model choice.')
    return dict(value)


def prepare_config(config_path, options):
    """Start with the approved models and retire previous base experiments."""
    import uuid
    path = Path(config_path)
    if not path.exists():
        missing = [f'{stage}/{choice}' for stage, choice in DEFAULT_SELECTION.items()
                   if choice not in options[stage]]
        if missing:
            raise ValueError('Missing default model exports: ' + ', '.join(missing) +
                             '. Install the weights described in distill/LIVE.md.')
        config = dict(selection=dict(DEFAULT_SELECTION), revision=uuid.uuid4().hex, reset='initial')
        atomic_json(path, config)
    else:
        config = json.loads(path.read_text())
        selection = config.get('selection')
        if isinstance(selection, dict):
            migrated = {stage:RETIRED_CHOICES[stage].get(selection.get(stage),selection.get(stage)) for stage in STAGES}
            if migrated != selection:
                validate_selection(migrated, options)
                config = dict(config,selection=migrated,revision=uuid.uuid4().hex)
                atomic_json(path,config)
    validate_selection(config['selection'], options)
    return config


class LiveModels:
    def __init__(self, config_path, options=None):
        self.path = Path(config_path)
        self.options = catalog() if options is None else options
        self.config = prepare_config(self.path, self.options)
        self.selection = validate_selection(self.config['selection'], self.options)
        self.models = {}
        self.lock = threading.Lock()

    def identity(self):
        # A reset changes only neural namespaces. Orogen's request/identity is
        # unchanged and its persisted bootstrap remains reusable.
        files = ('student.py','features.py','inference.py','coarse_solver.py','live_runtime.py')
        return dict(teacher_revision=REVISION, reset=self.config['reset'],
            selection=self.selection,
            checkpoints={stage: self.options[stage][key].get('sha256')
                         for stage, key in self.selection.items()},
            code={name: hashlib.sha256((REPO/'distill'/name).read_bytes()).hexdigest() for name in files})

    def install(self, world):
        import torch
        from distill.inference import install
        from distill.student import load_student
        models = {}
        with self.lock:
            for stage, choice in self.selection.items():
                if choice == 'teacher':
                    continue
                option = self.options[stage][choice]
                key = (str(world.device), stage, choice)
                if key not in self.models:
                    payload = Path(option['path']).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != option['sha256']:
                        raise ValueError('Model export changed since startup;restart the server.')
                    with torch.inference_mode(False):
                        model, _ = load_student(io.BytesIO(payload), world.device)
                    if model.config.stage != stage:
                        raise ValueError('Model export stage differs from selected stage.')
                    self.models[key] = model.requires_grad_(False)
                models[stage] = self.models[key]
        if models:
            install(world, models, tile_size=512)

    def public(self):
        return dict(selection=self.selection, revision=self.config['revision'],
            pid=os.getpid(), reset=self.config['reset'],
            tile40_url='/tile40/', options={stage: [dict(id=key, label=value['label']) for key, value in choices.items()]
                     for stage, choices in self.options.items()})

    def save_request(self, data):
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object.')
        selection = validate_selection(data.get('selection'), self.options)
        if not isinstance(data.get('reset', False), bool):
            raise ValueError('Reset must be boolean.')
        import uuid
        config = dict(selection=selection, revision=uuid.uuid4().hex,
                      reset=uuid.uuid4().hex if data.get('reset') else self.config['reset'])
        atomic_json(self.path, config)
        return config


_current = None


def current():
    global _current
    if _current is None:
        from backend.terrain_paths import OUTPUT_ROOT
        path = os.environ.get('DISTILL_LIVE_CONFIG') or os.environ.get('TERRAIN_MODELS_CONFIG') or OUTPUT_ROOT/'models.json'
        _current = LiveModels(path)
    return _current
