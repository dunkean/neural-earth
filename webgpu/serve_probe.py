"""Standalone static server for the WebGPU feasibility page (no terrain worker)."""

from pathlib import Path

from flask import Flask, abort, redirect, send_from_directory

ROOT = Path(__file__).resolve().parent
MODELS = Path("E:/TerrainDiffusionRuntime/webgpu-models")
app = Flask(__name__)


@app.get("/")
def index():
    return redirect("/webgpu/probe.html")


@app.get("/webgpu/<path:name>")
def assets(name: str):
    return send_from_directory(ROOT, name)


@app.get("/webgpu-models/<path:name>")
def models(name: str):
    if Path(name).name != name or not name.endswith((".onnx", ".data", ".json", ".bin")):
        abort(404)
    return send_from_directory(MODELS, name, conditional=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8770, threaded=True)
