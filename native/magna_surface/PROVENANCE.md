# Magna Urbis Satellite and near relief

Imported from the user's sibling `../magna-urbis`, repository
<https://github.com/dunkean/magna-urbis>, under the retained
[GNU Affero GPL version 3](LICENSE). This is separate from the older neural
engine submodule and the City erosion adaptation.

The exact snapshot is recorded in [source.json](source.json), including its
base Git revision, SHA-256 of each input and modified source paths. The sibling
contained local changes and advanced during integration, so the snapshot hash,
rather than its base commit alone, identifies the imported implementation.
The recorded snapshot is `50bf0371f2f772a4a41b0c011b2a878510c2a7937556d6852a5e287e0fa1227d`.

`satellite.wgsl` retains the generated material dependency closure rooted at
`v2GroundEnv`, `v2ShadeGround` and `v2DisplayShoulder`: mineral/snow materials,
ground microdetail, BRDF and population-weighted canopy responses. Identifiers
are namespaced. Host boundaries use Neural Earth's retained spatial climate,
forest population, ground bands, seasonal calendar, hydrology and light.
Manually selected finite-map climate zones, occupation and city objects are
disabled. Neural Earth's existing precision-safe crowns are retained.

Finite-map microdetail is evaluated in overlapping, globally anchored 1024 m
domains, with quintic blending and variance compensation. Source microdetail
and ground-cover functions operate on small coordinates. Physical ground bands
and snow grain use split coordinates. The coarse path evaluates the same
Satellite material as unresolved population responses. These host adaptations
make this a planetary port, not pixel-identical output of the city viewer.

`terrain3dNoise.ts` and `terrainGround.ts` are unmodified CPU references.
`backend/terrain_magna.py` ports the default derivative-attenuated ridged fBm
and physical displacement to tensors. Its base is the immutable 30 m neural
DEM with Neural Earth's existing continuous coastal reconstruction. The
canonical geometry footprint is 30/64 m; finer ground bands remain shader
detail. No model weights or terrain generator are changed by this option.

Rebuild the browser asset from this retained snapshot and local adapters:

```powershell
.venv/Scripts/python.exe tools/assets/import_magna_satellite.py
```

Import a new sibling snapshot deliberately with `--source ../magna-urbis`.
Review the new source hashes and update the tensor port/reference tests when
upstream noise or ground algorithms change. Never edit the generated runtime
asset directly.
