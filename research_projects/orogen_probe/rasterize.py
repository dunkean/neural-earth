"""Research-only sphere nearest-neighbor previews, preserving float heights."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
w, h = 1024, 512
lat = np.pi / 2 - (np.arange(h) + .5) * np.pi / h
lon = -np.pi + (np.arange(w) + .5) * 2 * np.pi / w
yy, xx = np.meshgrid(lat, lon, indexing="ij")
sample_xyz = np.stack([np.cos(yy)*np.sin(xx), np.sin(yy), np.cos(yy)*np.cos(xx)], axis=-1)
levels = np.array([-9000, -5000, -1000, 0, 100, 1000, 3000, 6000])
colors = np.array([[5,22,48], [12,57,92], [48,109,137], [60,128,109], [102,146,84], [156,148,96], [131,105,94], [239,239,234]])
sheet = Image.new("RGB", (w, (h+48)*3), "#131c24")
draw = ImageDraw.Draw(sheet)
records=[]
for i, name in enumerate(["default", "continental", "oceanic"]):
    xyz = np.fromfile(root / f"{name}-xyz.f32", dtype=np.float32).reshape(-1, 3)
    height = np.fromfile(root / f"{name}-height-m.f32", dtype=np.float32)
    _, idx = cKDTree(xyz).query(sample_xyz.reshape(-1,3), workers=4)
    raster = height[idx].reshape(h,w)
    np.save(root / f"{name}-coarse.npy", raster)
    rgb = np.stack([np.interp(raster,levels,colors[:,ch]) for ch in range(3)],axis=-1).astype(np.uint8)
    preview = Image.fromarray(rgb)
    preview.save(root / f"{name}-coarse.png")
    sheet.paste(preview,(0,i*(h+48)+48))
    draw.text((12,i*(h+48)+10), f"Orogen {name} | 20,001 spherical nodes | nearest-neighbor preview | land {(height>0).mean():.1%}",fill="white")
    records.append({"name":name,"height_sha256":hashlib.sha256(height.tobytes()).hexdigest(),"raster_shape":[h,w],"height_min_m":float(height.min()),"height_max_m":float(height.max()),"finite":bool(np.isfinite(height).all()),"raster_land_fraction":float((raster>0).mean()),"rasterization":"nearest spherical neighbor; research preview, not production interpolation"})
sheet.save(root / "comparison.png")
(root / "raster-report.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
print(json.dumps(records))
