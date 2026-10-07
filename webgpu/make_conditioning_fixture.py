"""Extract a small real first-base-window conditioning fixture from the capture."""

from pathlib import Path
import json
import numpy as np

ARTIFACTS = Path("E:/TerrainDiffusionRuntime/webgpu-models")
MANIFEST = json.loads((ARTIFACTS / "crop64-coast-manifest.json").read_text())
OUT = Path(__file__).with_name("conditioning-fixture.json")


def read(spec):
    return np.fromfile(ARTIFACTS / spec["file"], dtype=np.float32).reshape(spec["shape"])


def main():
    entry = MANIFEST["windows"]["base_pass1"][0]
    _, ty, tx = entry["ctx"]
    y0, x0 = ty - 1, tx - 1
    sums = np.zeros((6, 4, 4), np.float32)
    weights = np.zeros((4, 4), np.float32)
    contributors = []
    for coarse in MANIFEST["windows"]["coarse"]:
        _, cy, cx = coarse["ctx"]
        cy, cx = cy * 48, cx * 48
        tile = read(coarse["tile"])
        hit = False
        for iy in range(4):
            for ix in range(4):
                sy, sx = y0 + iy - cy, x0 + ix - cx
                if 0 <= sy < 64 and 0 <= sx < 64:
                    sums[:, iy, ix] += tile[:6, sy, sx]
                    weights[iy, ix] += tile[6, sy, sx]
                    hit = True
        if hit:
            contributors.append(coarse["ctx"])
    if np.any(weights <= 0):
        raise AssertionError("Missing coarse tile coverage")
    coarse4 = sums / weights[None]
    expected = read(MANIFEST["first_real_forward"]["base_model"]["cond_0"]).reshape(-1)
    OUT.write_text(json.dumps({
        "crop_manifest": "crop64-coast-manifest.json",
        "ctx": entry["ctx"],
        "coarse_contributors": contributors,
        "coarse4": coarse4.reshape(-1).tolist(),
        "expected_cond_0": expected.tolist(),
    }, indent=2))
    print(OUT, "last component", expected[-1])


if __name__ == "__main__":
    main()
