# Render v2: coastal glacier provenance

Render v2 uses a canonical source field to localize its procedural iceberg
appearance. This field supplies a plausible coastal origin. It does not model
glacier flow, calving, iceberg transport or iceberg bathymetry.

## Inputs and algorithm

The retained source atlas supplies physical elevation in metres, two normalized
seasonal temperatures (`T = 90 t - 45`, Celsius), and two half-year precipitation
fields (`P = 1000 p`, millimetres). Eligible cells are land above 20 m with both
seasonal temperatures below -1 Celsius and total precipitation above 80 mm.
An eligible cell must adjoin a source-ocean cell (eight-neighbour adjacency).
Longitude wraps on a sphere; latitude and planar edges do not wrap.

These thresholds are appearance heuristics. In particular, a cold wet coastal
cell is a potential glacial origin, not an observation of a tidewater glacier.
There is no source when required climate data are absent.

A cached `scipy.spatial.cKDTree` contains eligible source-cell centres. Planar
worlds use physical Cartesian metres. Spherical worlds use three-dimensional
directions multiplied by the world's radius; rotated polar requests first map
their directions back to the same canonical planet.

For nearest-source distance `d`, the influence is:

```text
t = clamp(d / 80,000 m, 0, 1)
glacier_proximity = 1 - t² (3 - 2t)
```

Spherical distances are chords. At the 80 km support on an Earth-sized world,
the chord/arc difference is below a metre; this approximation is larger on
small spherical worlds. The compact support is a visual neighborhood, not a
modeled iceberg lifetime or drift distance.

## Sampling and transport

The same source tree serves every tile and LOD, independent of local DEM
derivatives or the currently visible coast. Cropping a query cannot create or
remove a source. The field is evaluated at the appearance transport's sample
coordinates and appended as zero-based climate plane **50** (51 planes total).
Older 46/50-plane payloads remain readable and have no glacier provenance.
The source-only legacy Pedology response keeps its original 50-plane payload.

The CPU material calls `sample_glacier_proximity` directly. The GPU interpolates
the transported source field and evaluates the small iceberg appearance only
when resolved. Its fine shape remains procedural. Local terrain/climate detail
cannot create a new calving source absent from the source atlas.

## Validation and limits

`python -m unittest discover -s tests/python -p test_terrain_glacier.py` checks
that cold open ocean, inland land, warm coasts and missing climate do not supply
icebergs; influence has compact support and crop invariance; and antimeridian
and rotated-chart queries agree at the same geographic position.

Source resolution can omit narrow glaciated fjords or include a coarse coastal
glacier candidate where finer terrain has moved the coast. This is preferable
to deriving a different source from each rendered tile, but remains a proxy.
It changes no terrain, hydrology or climate state.

Reference: [NOAA, What is an iceberg?](https://oceanservice.noaa.gov/facts/iceberg.html)
describes icebergs as glacier/ice-shelf fragments; cold sea ice alone is not a
glacial source. Implementation: `backend/terrain_glacier.py`.
