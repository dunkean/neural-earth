"""Lazy, resident Python inference for the finite-tile quality viewer."""
import json
import threading
import time

from backend.terrain_paths import REPO_ROOT, model_snapshot
from distill.tile40_prepare import sha


class Tile40Python:
    def __init__(self):
        self.lock = threading.Lock()
        self.generator = None
        self.key = None

    def catalog(self, decoder='decoder16'):
        if decoder not in ('decoder16', 'decoder64'):
            raise ValueError('Décodeur inconnu.')
        distribution = json.loads((REPO_ROOT/'distill/weights/manifest.json').read_text())
        models = {stage: distribution['exports'][key] for stage, key in
                  [('coarse', 'coarse2'), ('encoder', 'base_residual8k'), ('decoder', decoder)]}
        return dict(models=models, total_model_bytes=sum(v['bytes'] for v in models.values()),
                    selected_models=dict(coarse='coarse2', base='base_residual8k', decoder=decoder))

    @staticmethod
    def validate(payload):
        import numpy as np
        from distill.tile40_conditioning import CLIMATES
        if not isinstance(payload, dict):
            raise ValueError('Paramètres de génération requis.')
        decoder, provider = payload.get('decoder'), payload.get('provider', 'auto')
        if decoder not in ('decoder16', 'decoder64') or provider not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Décodeur ou moteur Python inconnu.')
        inputs = payload.get('input')
        if not isinstance(inputs, dict):
            raise ValueError('Croquis, climat et graine requis.')
        seed, climate = inputs.get('seed'), inputs.get('climate')
        if type(seed) is not int or not 0 <= seed <= 4294950000:
            raise ValueError('Graine invalide.')
        if not isinstance(climate, str) or climate not in CLIMATES:
            raise ValueError('Climat inconnu.')
        try:
            sketch = np.asarray(inputs.get('sketch_m'), dtype=np.float32)
        except (ValueError, TypeError):
            raise ValueError('Croquis invalide.') from None
        if sketch.shape != (16, 16) or not np.isfinite(sketch).all() or (sketch < -1000).any() or (sketch > 8000).any():
            raise ValueError('Le croquis doit contenir 16 × 16 altitudes entre -1000 et 8000 m.')
        return decoder, provider, dict(sketch_m=sketch, climate=climate, seed=seed)

    def _load(self, decoder, device):
        from distill.student import load_student
        from distill.tile40_export_utils import ExportCoarse
        from distill.tile40_flow_runtime import FiniteFlowEncoder
        from distill.tile40_runtime import Tile40
        source = REPO_ROOT/'distill/weights'
        metadata = self.catalog(decoder)
        for item in metadata['models'].values():
            if sha(source/item['file']) != item['sha256']:
                raise RuntimeError('Les poids diffèrent du manifeste des modèles sélectionnés.')
        distribution = json.loads((source/'manifest.json').read_text())
        # CoarseSolver fetches the pinned original configuration when missing.
        coarse, _ = load_student(source/'coarse2.pt')
        parameters = json.loads((model_snapshot(distribution['teacher_revision'])/'config.json').read_text())
        parameters['histogram_raw'] = parameters['histogram_raw'] or [0.]*5
        encoder, saved = load_student(source/'base-residual8k.pt')
        if saved['contract']['coarse_sha256'] != metadata['models']['coarse']['sha256']:
            raise RuntimeError('Le résiduel 8k exige les poids Coarse2 sélectionnés.')
        encoder = FiniteFlowEncoder(encoder, saved['contract'], steps=4)
        final, _ = load_student(source/metadata['models']['decoder']['file'])
        return Tile40(ExportCoarse(coarse.solver), encoder, final, parameters, device,
                      whole_decoder=decoder == 'decoder16')

    def generate(self, payload):
        decoder, provider, inputs = self.validate(payload)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Une génération Python est déjà en cours. Réessaie après son calcul.')
        try:
            import torch
            started = time.perf_counter()
            if provider == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA indisponible sur le serveur ; choisis Python CPU.')
            device = 'cpu'
            if torch.get_num_threads() > 8:
                torch.set_num_threads(8)
            if provider != 'cpu' and torch.cuda.is_available():
                from backend.terrain_device import select_cuda_device, current_cuda_index
                select_cuda_device()
                device = f'cuda:{current_cuda_index()}'
            key = (decoder, device)
            if self.key != key:
                # Keep one resident configuration, including when comparing decoders.
                self.generator = None
                self.key = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.generator = self._load(decoder, device)
                self.key = key
            generated = self.generator.generate(**inputs)
            metadata = self.catalog(decoder)
            timing = generated['timing']
            header = dict(provider='python-cuda' if device.startswith('cuda') else 'python-cpu',
                          pixels=1334, sideM=generated['side_m'], spacingM=generated['spacing_m'],
                          timings=dict(totalMs=timing['native_total_seconds']*1000,
                                       **{k: v for k, v in timing.items() if k.endswith('_seconds')}),
                          requestMs=(time.perf_counter()-started)*1000, manifest=metadata)
            return generated['height_m'].astype('<f4').tobytes(), header
        finally:
            self.lock.release()
