# Archived distilled models

Storage for distilled checkpoints worth keeping that are not shipped with the app. Model files
here (`*.pt`, `*.bin`, `*.safetensors`, `*.nf4`, `*.tar`, `*.zip`) go through Git LFS, and
`.lfsconfig` excludes this folder from LFS downloads: clones and pulls get small pointer files only.

- Fetch on demand: `git lfs pull --include="distill/archive/**"`
- Add a model: copy it here with a short note (source run, training step, sha256) in this README.

Not stored here:

- Runtime weights used by the app stay in `distill/weights/` as regular Git files.
- Releasable packages under the size target stay GitHub Release assets:
  [`native3-base99m-v1`](https://github.com/dunkean/neural-earth/releases/tag/native3-base99m-v1),
  [`base-nf4-b75-v1`](https://github.com/dunkean/neural-earth/releases/tag/base-nf4-b75-v1).
- The original `xandergos/terrain-diffusion-30m` checkpoints are downloaded from Hugging Face.
- Earlier versions of `coarse1.pt`, `coarse4.pt` and `coarse8.pt` (the `coarse8_legacy` export,
  revision `73d80e7`) remain in Git history.
