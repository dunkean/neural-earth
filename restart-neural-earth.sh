#!/usr/bin/env sh
# Kill every running Neural Earth server (and its child processes), then start a fresh one.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$root/.venv/bin/python" ]; then
    echo "Create .venv and install dependencies first; see docs/installation.md." >&2
    exit 1
fi

pattern='terrain_server\.py|run_terrain_benchmark_server\.py|launch_terrain\.py'

server_pids() {
    {
        pgrep -f "$pattern" || true
        if command -v lsof >/dev/null 2>&1; then
            lsof -t -iTCP:8765 -iTCP:8766 -sTCP:LISTEN 2>/dev/null || true
        fi
    } | grep -v "^$$\$" | sort -u
}

with_children() {
    for pid in "$@"; do
        echo "$pid"
        with_children $(pgrep -P "$pid" || true)
    done
}

echo "Stopping Neural Earth servers..."
pids=$(with_children $(server_pids) | sort -u)
if [ -z "$pids" ]; then
    echo "  none running"
else
    echo "  terminating PIDs:" $pids
    kill -TERM $pids 2>/dev/null || true
    for _ in $(seq 20); do
        alive=$(for pid in $pids; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done || true)
        [ -z "$alive" ] && break
        sleep 0.25
    done
    if [ -n "${alive:-}" ]; then
        echo "  force killing PIDs:" $alive
        kill -KILL $alive 2>/dev/null || true
    fi
fi

echo "Starting a new Neural Earth server..."
exec "$root/.venv/bin/python" "$root/launch_terrain.py" --gpu "$@"
