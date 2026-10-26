# Amplification DEM fixtures

Unmodified copies of `data/heightfields/montserrat_256.png`, `mountains.png`,
and `new_zealand.png` from [H-Schott/MultiScaleErosion](https://github.com/H-Schott/MultiScaleErosion),
the reference code for Schott et al., *Terrain Amplification using Multi-Scale
Erosion*, TOG 2024. Copyright (c) 2023 Schott; the upstream MIT license is
retained in [LICENSE](LICENSE). No upstream revision was available in the
provided local source snapshot, so none is invented.

These are reference terrain heightfields, not georeferenced survey products.
The tests read 16-bit PNGs at their native precision and the RGB mountains
image as luminance. Each pixel represents a 30 m native DEM cell. Normalized
source ranges map to 30–1230 m (Montserrat), 100–2500 m (mountains), and
−40–2160 m (New Zealand). The finite DEM edges are extended by clamping;
benchmark centres stay inside the original image. Drainage touching an edge
is consequently truncated and must not be treated as a real catchment.

Unmodified source SHA-256 fingerprints:

- `montserrat_256.png`: `7650c25e6907c133cb03de530f1d575bc11de494a8c90a1048356b9f936c9551`
- `mountains.png`: `65dc378482cc3ef3832cd4fb5137efd5232aa6cd6712b1ee0a73b08f77635c0b`
- `new_zealand.png`: `c34200774f275aba8135d6507272e1aa60ba9b20adb22c60476bfd079f45a779`
