#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -f wasm/dist/index.html ]; then
  npm run build:wasm
fi
exec npm run serve:wasm
