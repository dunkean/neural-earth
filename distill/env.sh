#!/usr/bin/env bash
# Source this file before running distillation tools. No jobs are started here.
_distill_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export DISTILL_REPO="$_distill_repo"
export DISTILL_ROOT="${DISTILL_ROOT:-$HOME/data/distill}"
export TERRAIN_RUNTIME_ROOT="${TERRAIN_RUNTIME_ROOT:-$HOME/data/runtime}"
export TERRAIN_RUNTIME="$TERRAIN_RUNTIME_ROOT"
export TERRAIN_OUTPUT_ROOT="${TERRAIN_OUTPUT_ROOT:-$DISTILL_ROOT/runtime-output}"
# Reuse the installed snapshot rather than downloading another copy.
export HF_HOME="${HF_HOME:-$HOME/.cache/neural-earth/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Ordinal zero inside each CUDA_VISIBLE_DEVICES assignment below.
export TERRAIN_CUDA_DEVICE=0
export TERRAIN_GPU_MODE=single
export TERRAIN_CUDA_DEVICES=0
export MPLBACKEND=Agg
_distill_node=$(printf '%s\n' "$HOME"/.nvm/versions/node/v22.* | sort -V | tail -n 1)
if [[ -z "$_distill_node" || ! -x "$_distill_node/bin/node" ]]; then
    echo 'Node 22 is required; install it with nvm install 22.' >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="$_distill_repo/.venv/bin:$_distill_node/bin:$PATH"
unset _distill_repo _distill_node
