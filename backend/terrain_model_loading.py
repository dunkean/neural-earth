"""Serialize constructors that use Accelerate's process-wide meta patches.

Loading weights on independent GPUs can otherwise overlap init_empty_weights
with another model constructor: ordinary Parameters then become meta tensors.
Inference and CUDA graph warmup remain independent after construction.
"""
from threading import RLock

MODEL_LOAD_LOCK = RLock()
