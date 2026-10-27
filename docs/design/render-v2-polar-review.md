# Real-world polar review before the Render v2 freeze

[French original](render-v2-polar-review_FR.md).

## Reproduction and evidence

World `bd36737badf0cedf99200ba94bf7dc761070aca686a98784373247c86efe87dd`, real ocean center `[14657326.440143395, 7263946.615556731]`, hydrology disabled. The user's original nearby center is on land; the 12 km shift allows the sea ice to be examined. Captures use the actual 30 m decoder caches, forced LOD0, and the same camera for V2 and Render3D.

- Before correction: `output/render-v2-polar-current/captures/0e195ba26a64-d11cd03a-a911f822/`, six views at 40 km and 10 km.
- After correction: `output/render-v2-polar-current/captures/db083bf3841a-6c77484e-a911f822/`, nine views at 40 km, 10 km and 5 m/pixel, for V2 in July, Render3D in July and V2 in January.
- The last footprint measures 5.12 km over 1024 pixels. It enlarges the 30 m DEM to examine the material; it is not new geometry at 5 m.
- Each pass report records the sources, resolutions, camera, month, cache keys and production hash. The nine final views satisfy the protocol; this does not establish photorealistic validation of every environment.

## Observed defect and bounded correction

The first pass showed very pale, smooth pack ice, winding dark filaments and a second gray network around the floes. The cellular basis used `F2-F1` as fracture thickness. This difference is not a uniform distance to the plane separating the two sites, producing wide openings and varying thickness in some configurations.

For compact marine pack ice only, the same traversal of 27 neighbors now retains the vectors for the two nearest sites. With distances `d1`, `d2` and positions `A`, `B`, the field becomes:

```text
bisector_distance = (d2² - d1²) / (2 × |B - A|)
```

This is the distance to the plane equidistant from those two sites, derived directly from Euclidean geometry. It preserves the `d1=d2` loci, sites, domain distortions, concentrations and floe selection. It does not claim to calculate distance to the entire Voronoi boundary after distortion. No second cellular traversal is added.

The nominal width of compact-pack leads is reduced to a few meters and integrated over the pixel footprint. Subpixel fractures contribute by their coverage fraction instead of becoming opaque lines. The gray of refrozen ice is toned down; its previous contrast drew an overly visible second network. Narrow compression shading and finer filtered grain replace some of this variation without adding large albedo marbling. Lakes and rivers retain their previous cellular field and appearance; marginal pack ice and icebergs also retain their existing path.

The cells are a procedural adaptation of the cellular texture functions in [Steven Worley, *A Cellular Texture Basis Function*, SIGGRAPH 1996](https://itn-web.it.liu.se/~stegu76/TNM084-2019/worley-originalpaper.pdf), not an ice dynamics simulation. The concepts of joined sea ice, fractures, leads and refrozen ice draw on [NSIDC descriptions](https://nsidc.org/learn/parts-cryosphere/sea-ice/science-sea-ice). The compression relief is a material cue; no ridge geometry or spectral photometry is claimed.

## Results and limitations

In the actual 40 km and 10 km views, the second network is subdued and fractures look less like wire. Large floes remain joined. The surface remains light and matte; some winding patterns in the cellular field are still visible in detail. Land relief is preserved, but patches of partial snow visible on land are separate from this marine correction.

The continental view `output/render-v2-floor/final-overview-lod6/` is approximately 3152 m/pixel with a coarse source at 7680 m: it does not resolve kilometer-scale floes. Its very broad blue–white front represents the average freezing field at that scale and does not, by itself, demonstrate a plastic-looking material. Render3D water and its uniform freezing are not a reference for physical floes either.

Checks performed: 37 Render v2 CPU tests pass, including eight polar tests. At 4096 positions, the four material outputs are strictly identical before and after for marginal pack ice, icebergs, lakes and rivers. The real comparison does not conceal the limitations of the procedural proxy. GPU parity gates, costs and the full campaign are handled separately by the global pass after the freeze.
