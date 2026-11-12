# Optional Orogen v3 CUDA spatial projection

The active optional kernel is `nearest.cu`, used by
`backend/terrain_orogen_gpu.py` for exact Euclidean k=4 lookup on the spherical
graph. The index is built from CPU input arrays with NumPy, then the bounded
query runs on CUDA. This avoids requiring CuPy's optional CUB headers.
CPU cKDTree is the fallback when the optional runtime is unavailable.
The existing torch CUDA barycentric projection remains in
`backend/terrain_orogen_cuda.py`.

The V1 AST/source-rewriting adapter, its Acorn bundle and its experimental graph
propagation/erosion kernels were removed for v3. Generation, finish and climate
run the pinned upstream CPU algorithms. There are no advertised GPU toggles for
unported v3 stages. Source adaptation remains covered by the retained GPL-3.0
Orogen license. Delaunator retains its separate ISC license.
