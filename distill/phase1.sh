#!/usr/bin/env bash
# Defaults to printing the plan. --run starts only the reference measurements.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$DISTILL_REPO"

case "${1:---dry-run}" in
    --dry-run)
        cat <<'PLAN'
After the top, stop the Neural Earth server and free both GPUs, then run:
  bash distill/phase1.sh --run

This creates a tmux session named neural-earth-distill-reference:
  GPU 0: BF16/FP16 base-model benchmark (batch 16 and 32)
  GPU 1: BF16/FP16 base-model benchmark (batch 16 and 32)
  GPU 1: reference/fp32base physical comparison, all seven sites and both LODs
  CPU:   comparison sheets and per-site BF16/FP32 error floors

Outputs: ~/data/distill/{bench,eval,logs}
FP8 raw-kernel measurements remain a separate Phase 1 item: the existing
base-model benchmark does not implement FP8. No teacher crops or training
are started by this script. Inspect logs even if the tmux session exits.
PLAN
        exit 0
        ;;
    --run)
        command -v tmux >/dev/null
        command -v nvidia-smi >/dev/null
        if tmux has-session -t neural-earth-distill-reference 2>/dev/null; then
            echo 'The reference tmux session already exists.' >&2
            exit 1
        fi
        # WSL may report Windows PIDs as [Not Found]; these still occupy the GPU.
        _distill_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
        if [[ -n "$_distill_processes" ]]; then
            echo 'GPUs have active compute processes; stop them before measuring.' >&2
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
            exit 1
        fi
        mkdir -p "$DISTILL_ROOT"/{bench,eval,logs,ckpt,crops}
        tmux new-session -d -s neural-earth-distill-reference -c "$DISTILL_REPO" \
            -e "DISTILL_ROOT=$DISTILL_ROOT" \
            -e "TERRAIN_RUNTIME_ROOT=$TERRAIN_RUNTIME_ROOT" \
            -e "TERRAIN_OUTPUT_ROOT=$TERRAIN_OUTPUT_ROOT" \
            -e "HF_HOME=$HF_HOME" -e "HF_HUB_CACHE=$HF_HUB_CACHE" \
            "bash distill/phase1.sh --worker"
        echo 'Started: tmux attach -t neural-earth-distill-reference'
        echo "Log: $DISTILL_ROOT/logs/phase1.log"
        ;;
    --worker)
        # Used by tmux only. Preserve failures through tee and write a status file.
        mkdir -p "$DISTILL_ROOT"/{bench,eval,logs}
        exec > >(tee -a "$DISTILL_ROOT/logs/phase1.log") 2>&1
        trap 'printf "%s\n" "$?" > "$DISTILL_ROOT/logs/phase1.exit"' EXIT
        rm -f "$DISTILL_ROOT/logs/phase1.exit"
        date -Is
        CUDA_VISIBLE_DEVICES=0 python tools/benchmarks/benchmark_base_model.py \
            --output "$DISTILL_ROOT/bench/base-4090.json"
        CUDA_VISIBLE_DEVICES=1 python tools/benchmarks/benchmark_base_model.py \
            --output "$DISTILL_ROOT/bench/base-5090.json"
        CUDA_VISIBLE_DEVICES=1 python tools/verification/compare_base_variants.py \
            run "$DISTILL_ROOT/eval/fp32-5090" --variants reference fp32base
        CUDA_VISIBLE_DEVICES='' python tools/verification/compare_base_variants.py \
            sheet "$DISTILL_ROOT/eval/fp32-5090"
        CUDA_VISIBLE_DEVICES='' python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ['DISTILL_ROOT'])
report = json.loads((root / 'eval/fp32-5090/report.json').read_text())
expected = {(site['name'], lod) for site in report['sites'] for lod in (3, 0)}
if len(expected) != 14:
    raise SystemExit('Expected seven sites at two LODs; inspect the report.')
for variant in ('reference', 'fp32base'):
    entries = report['variants'][variant]
    if len(entries) != 14 or {(e['site'], e['lod']) for e in entries} != expected:
        raise SystemExit(f'{variant}: incomplete reference measurement.')
    if any('error' in entry for entry in entries):
        raise SystemExit(f'{variant}: failed measurements; inspect report.json.')
failed = []
for entry in report['variants']['fp32base']:
    mae = entry.get('vs_reference', {}).get('mae')
    print(f"{entry['site']} LOD {entry['lod']}: MAE={mae} m", flush=True)
    # 7–36 m was an observation on the 3090, not a hardware-independent cap.
    # The two Blackwell cases >36 m were rechecked and exactly reproduced.
    if mae is None or not math.isfinite(mae) or mae < 0:
        failed.append(entry)
if failed:
    raise SystemExit('STOP: invalid BF16/FP32 measurement; investigate before Phase 2.')
print('Complete finite reference measurements. Use each same-site MAE as the '
      'student error floor; student slopes and spectral bands still require ±5%.')
PY
        ;;
    *) echo 'Usage: bash distill/phase1.sh [--dry-run|--run]' >&2; exit 2 ;;
esac
