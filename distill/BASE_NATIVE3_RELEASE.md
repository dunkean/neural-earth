# Climate-conditioned native base under 100 MB

The experimental **native3-base99m-v1** release contains the complete native
253,688,389-parameter base, selected at training step 5,000 (raw weights).
It retains the original 58 conditioning values and two solver passes.

- [Release and verification reports](https://github.com/dunkean/neural-earth/releases/tag/native3-base99m-v1)
- [Download the complete base package](https://github.com/dunkean/neural-earth/releases/download/native3-base99m-v1/native3-base99m-v1.tar)
- [Download comparisons, ten mountain images and evaluation evidence](https://github.com/dunkean/neural-earth/releases/download/native3-base99m-v1/native3-base99m-v1-evidence.zip)
- [SHA256 checksums](https://github.com/dunkean/neural-earth/releases/download/native3-base99m-v1/SHA256SUMS.txt)

The weights and manifest occupy **99,215,527 bytes**. The complete uncompressed
TAR, including the standalone loader, provenance, instructions and upstream
license, occupies **99,235,840 bytes**, below the strict 100,000,000-byte limit.
Its SHA256 is:

```text
247d2f8f8424b11de1af3dd4be954ebeec4ec7b43b4dd2b00ec433fb88227f5c
```

Convolutions use 3-bit storage, embedding projections use 4-bit storage,
sensitive tensors remain BF16, and block scales use FP16. The loader reconstructs
BF16 weights for inference. The size limit applies to stored base weights;
working RAM/VRAM, the coarse model and the original decoder are additional.
This package differs from the older 149.6 MB NF4 B75 release used by the browser
WASM edition. It does not replace that runtime or the application's saved models.

## Download and load

Install the project's pinned Python dependencies and initialize its submodules
as described in [installation.md](../docs/installation.md). From the repository
root, download and extract outside Git:

```bash
gh release download native3-base99m-v1 --repo dunkean/neural-earth \
  --pattern native3-base99m-v1.tar --pattern SHA256SUMS.txt \
  --dir "$HOME/data/distill/releases/native3-base99m-v1"
cd "$HOME/data/distill/releases/native3-base99m-v1"
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xf native3-base99m-v1.tar
```

From the repository root, load the extracted base with the virtual environment:

```python
from pathlib import Path
import sys

repository = Path.cwd()
package = Path.home() / "data/distill/releases/native3-base99m-v1/base"
sys.path.insert(0, str(repository / "terrain-diffusion"))
sys.path.insert(0, str(package))
from load_base import load_packed

base, manifest = load_packed(package, device="cuda")
```

The loader verifies the binary's size and SHA256, constructs the pinned
`terrain_diffusion` architecture and loads every tensor. It neither downloads
nor reads original base weights. Use the original pipeline's conditioning,
noise, blending and two solver passes when integrating it.

## Evidence and limitations

The candidate was selected using 30 validation cases, frozen with hashes, then
evaluated on 30 independent test cases. Test mountain morphology distance is
0.03287 (lower is better); detail correlations across climates are 0.9642–0.9762.
These diagnostics are not percentages of visual quality. Climate-response
relative RMSE remains 0.328–0.460, and the original-quality acceptance criterion
was not met. The ten mountain comparisons cover two geometries and five climates.

The evidence ZIP opens locally through `comparison.html` and includes its images,
CSV tables, held-out report, candidate freeze, visual review and source snapshot.
The release also provides the inventory and loader verification separately.
The original RTX 5090 verification found all reconstructed tensors and a 64×64
forward exactly equal to the evaluated model, with parent weight loading and
downloads forbidden. Publication rechecks the archive and all six payload hashes.

Teacher: `xandergos/terrain-diffusion-30m`, revision
`9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Its MIT license and provenance are
included in the package. Training checkpoints, optimizer state and generated
evaluation data remain outside Git; distribution artifacts are release assets.
