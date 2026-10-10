"""Explicit model choices and immutable identities for the local review server."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import threading

from distill.common import DATA, REPO, REVISION, atomic_json

STAGES = ('coarse', 'base', 'decoder')


def catalog(root=DATA):
    weights = Path(root)/'final/weights'
    specifications = {
        'coarse': [('solver8', 'Distillé · 8 étapes', weights/'candidate-1/coarse.pt')],
        'base': [
            ('initial128', '128 initial · 53 Mo · plus lisse', weights/'candidate-1/base.pt'),
            ('coupled128', '128 couplé · 53 Mo', weights/'candidate-2/base.pt'),
            ('refined128', '128 affiné · 53 Mo', weights/'candidate-3/base.pt'),
            ('refined192', '192 affiné · 120 Mo', weights/'candidate-4/base.pt'),
        ],
        'decoder': [('distilled200k', 'Distillé · 13 Mo', weights/'candidate-1/decoder.pt')],
    }
    result = {}
    for stage, entries in specifications.items():
        result[stage] = {'teacher': dict(label='Original · référence', path=None)}
        for key, label, path in entries:
            if path.is_file():
                result[stage][key] = dict(label=label, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def validate_selection(value, options):
    if not isinstance(value, dict) or set(value) != set(STAGES):
        raise ValueError('Select exactly coarse, base and decoder.')
    if any(not isinstance(value[stage], str) or value[stage] not in options[stage] for stage in STAGES):
        raise ValueError('Unknown model choice.')
    return dict(value)


class LiveModels:
    def __init__(self, config_path, options=None):
        self.path = Path(config_path)
        self.options = catalog() if options is None else options
        self.config = json.loads(self.path.read_text())
        self.selection = validate_selection(self.config['selection'], self.options)
        self.models = {}
        self.lock = threading.Lock()

    def identity(self):
        # A reset changes only neural namespaces. Orogen's request/identity is
        # unchanged and its persisted bootstrap remains reusable.
        files = ('student.py', 'features.py', 'inference.py', 'coarse_solver.py', 'live_runtime.py')
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
                        raise ValueError('Model export changed since server startup; restart the review server.')
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
            options={stage: [dict(id=key, label=value['label']) for key, value in choices.items()]
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
        _current = LiveModels(os.environ['DISTILL_LIVE_CONFIG'])
    return _current
