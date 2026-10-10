#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$root/.venv/bin/python" ]; then
    echo "Create .venv and install dependencies first; see docs/installation.md." >&2
    exit 1
fi
exec "$root/.venv/bin/python" "$root/launch_terrain.py" "$@"
