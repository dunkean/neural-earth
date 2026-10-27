"""Optional CUDA graph / CPU kernel setup, before world compilation."""
import os
import threading

_evaluator=None
_prepared=False
_lock=threading.Lock()


def prepare():
    global _prepared,_evaluator
    with _lock:
        if _prepared: return
        from terrain_resource_distance import prepare as prepare_distance
        prepare_distance()
        from terrain_resource_water import prepare as prepare_water
        prepare_water()
        from terrain_resource_noise import prepare as prepare_noise
        prepare_noise()
        if os.environ.get('TERRAIN_RESOURCES_DEVICE','auto') != 'cpu':
            try:
                import torch
                if torch.cuda.is_available():
                    from terrain_resource_cuda import realized_class
                    # Create the context/allocator once; no world fields or JIT.
                    torch.zeros(1,device='cuda').cpu()
                    _evaluator=realized_class
                    import numpy as np
                    import terrain_resource_gpu as G
                    import terrain_resources as R
                    pkg=dict(seed32=42,layer=R.ROCK_LAYER,classes=len(R.ROCKS),aligned=R.ROCK_ALIGNED,
                        groups=R.ROCK_GROUPS,group_aligned=R.ROCK_GROUP_ALIGNED,
                        field=G.Field(np.zeros((len(R.ROCKS)+2,4,8),np.float32),-4,-2,
                                      (-122880.,-61440.,122880.,61440.)))
                    gi,gj=np.meshgrid(np.arange(-8,8),np.arange(-4,4))
                    realized_class(pkg,gi,gj,9)
                    from terrain_resource_cuda import prepare_graph
                    prepare_graph()
            except ImportError:
                pass
            except RuntimeError as exc:
                # CUDA is optional, including when another application has
                # already committed its memory. Keep server/library CPU usable.
                import logging
                logging.getLogger(__name__).warning('Resource CUDA setup failed; using CPU: %s',exc)
                _evaluator=None
        _prepared=True


def evaluator():
    if os.environ.get('TERRAIN_RESOURCES_DEVICE','auto') == 'cpu': return None
    return _evaluator
