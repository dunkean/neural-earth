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
DEFAULT_SELECTION = dict(coarse='solver4', base='teacher', decoder='distilled200k')
RETIRED_BASE_CHOICES = {'initial128', 'coupled128', 'refined128', 'refined192'}


def catalog(root=DATA, bundled_root=REPO/'distill/weights'):
    weights = Path(root)/'final/weights'
    def weight_path(name, fallback):
        bundled = Path(bundled_root)/name
        return bundled if bundled.is_file() else fallback

    def short_coarse_path(evaluations):
        full = Path(root)/'ckpt/coarse-full'/f'coarse{evaluations}'/'export/coarse.pt'
        delivery = full.parent.parent/'delivery.json'
        # A half-finished export must not replace the reviewed pilot.
        if full.is_file() and delivery.is_file():
            proof = json.loads(delivery.read_text())
            if (proof.get('status') == 'complete' and proof.get('export_max_abs') == 0. and
                    proof.get('export', {}).get('export', {}).get('sha256') ==
                    hashlib.sha256(full.read_bytes()).hexdigest()):
                return full
        return Path(root)/f'live/weights/coarse{evaluations}/coarse.pt'

    specifications = {
        'coarse': [
            ('solver8', 'Distillé · 8 étapes', weight_path('coarse8.pt', weights/'candidate-1/coarse.pt')),
            ('solver4', 'Distillé · 4 étapes', weight_path('coarse4.pt', Path(root)/'live/weights/coarse4/coarse.pt')),
            ('solver2', 'Distillé · 2 étapes', weight_path('coarse2.pt', short_coarse_path(2))),
            ('solver1', 'Distillé · 1 étape', weight_path('coarse1.pt', short_coarse_path(1))),
        ],
        'base': [],
        'decoder': [('distilled200k', 'Distillé · 13 Mo', weight_path('decoder.pt', weights/'candidate-1/decoder.pt'))],
    }
    native=Path(root)/'ckpt/native-weight-nf4-100k/step070000.pt'
    proof=Path(root)/'eval/native-nf4-artifact70000-5090-audit.json'
    if native.is_file() and proof.is_file():
        audit=json.loads(proof.read_text())
        if (audit.get('status')=='complete' and audit.get('tensor_count')==174 and
                audit.get('checkpoint_sha256')==hashlib.sha256(native.read_bytes()).hexdigest()):
            specifications['base'].append(('nf4_70k','Essai NF4 · 70k · 150 Mo',native))
    result = {}
    for stage, entries in specifications.items():
        result[stage] = {'teacher': dict(label='Original · référence', path=None)}
        for key, label, path in entries:
            if path.is_file():
                result[stage][key] = dict(label=label, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if stage=='base':result[stage][key]['format']='native'
    from distill.compact_release import HASHES, RELEASE, verified
    compact = Path(root)/'releases'/RELEASE
    if verified(compact):
        result['base']['nf4_b75'] = dict(
            label='Distillé · NF4 B75 · 150 Mo', path=str(compact/'weights.bin'),
            directory=str(compact), sha256=HASHES['weights.bin'], format='scalar-packed')
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
        if isinstance(selection, dict) and selection.get('base') in RETIRED_BASE_CHOICES:
            selection = dict(selection, base='teacher')
            validate_selection(selection, options)
            config = dict(config, selection=selection, revision=uuid.uuid4().hex)
            atomic_json(path, config)
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
        files = ('student.py', 'features.py', 'inference.py', 'coarse_solver.py', 'live_runtime.py',
                 'native_live_runtime.py','native_inference.py','native_weight_runtime.py','native_weight_quant.py','native_base.py',
                 'native_scalar_pack.py', 'native_scalar_precision.py', 'compact_release.py')
        return dict(teacher_revision=REVISION, reset=self.config['reset'],
            selection=self.selection,
            checkpoints={stage: self.options[stage][key].get('sha256')
                         for stage, key in self.selection.items()},
            code={name: hashlib.sha256((REPO/'distill'/name).read_bytes()).hexdigest() for name in files})

    def students(self, device):
        """Selected students resident on ``device``, shared by all its worlds."""
        import torch
        from distill.student import load_student
        models = {}
        with self.lock:
            for stage, choice in self.selection.items():
                if choice == 'teacher':
                    continue
                option = self.options[stage][choice]
                key = (str(device), stage, choice)
                if key not in self.models:
                    payload = Path(option['path']).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != option['sha256']:
                        raise ValueError('Model export changed since server startup; restart the review server.')
                    with torch.inference_mode(False):
                        if option.get('format') == 'scalar-packed':
                            from distill.compact_release import verified
                            from distill.native_scalar_pack import runtime_scalar_packed
                            if stage != 'base' or not verified(option['directory']):
                                raise ValueError('Compact B75 export changed; reinstall its release.')
                            model = runtime_scalar_packed(option['directory'], device)
                        elif option.get('format') == 'native':
                            if stage != 'base':
                                raise ValueError('Native export must be a base.')
                            from distill.native_inference import load_native
                            model, _ = load_native(io.BytesIO(payload), device)
                        else:
                            model, _ = load_student(io.BytesIO(payload), device)
                    if option.get('format') not in ('native', 'scalar-packed') and model.config.stage != stage:
                        raise ValueError('Model export stage differs from selected stage.')
                    self.models[key] = model.requires_grad_(False)
                models[stage] = self.models[key]
        return models

    def install(self, world):
        from distill.inference import install
        models = dict(self.students(world.device))
        native_base = None
        if self.options['base'][self.selection['base']].get('format') in ('native', 'scalar-packed'):
            native_base = models.pop('base')

        if models:
            install(world, models, tile_size=512)
        if native_base is not None:
            from distill.native_live_runtime import install as install_native
            # configure_world prepared the original stack before this hook.
            # Never normalize the already baked native NF4 kernels again.
            install_native(world,native_base)

    def prewarm(self, world, stages, *, coarse_streams=1):
        """Capture the selected students' graphs; teacher stages they replace need none."""
        from distill.inference import prewarm
        models = {stage: model for stage, model in self.students(world.device).items() if stage in stages
                  and self.options[stage][self.selection[stage]].get('format') not in ('native', 'scalar-packed')}
        return prewarm(models, world.device, getattr(world, '_terrain_profile', None),
                       decoder_size=world.decoder_tile_size, coarse_streams=coarse_streams)

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
