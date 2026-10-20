"""Export canonical Python control metadata; no models/GPU/server are loaded."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'backend'))
from terrain_generation import generator_schema

destination = root / 'wasm' / 'generation-schema.json'
destination.write_text(json.dumps(generator_schema(), indent=2) + '\n', encoding='utf-8')
