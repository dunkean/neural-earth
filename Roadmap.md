Now:
- map/view compressed UID
- Exports
- UI cleaning
- Repo cleaning

Short term:
- InifiniteGeneration Distillation (80% done)
- Negative LOD finetuning: either Decoder retraining - fake heightmap normalization - homemade/third-party noise + erosion (magna urbis rust)
- Orogen optimization pass: LOD ? tuning of best ratio cell / pixel / quality. Moar GPU porting
- Basins then multiscale rivers: try Magna Urbis gpu erosions rivers (fix FP16 straight lines / test on IG reliefs)
- Render layer optimization: either efficient gpu pipeline/shaders or surrogation in a UNET or UNET on GEarth (less probable)

Mid/Long term:
- Brush map layer
- Multiscale ressources / roads / settlements implantation
- Full WASM engine
- UI refactoring
- 3D view: fly, walk/street view
- View rendering with Gen AI + ControlNet
- Alien planets
- Magna Urbis generation integration
- + Modern/SF/futuristic cities: modern, cyberpunk, space opera, aliens
- + 3D cities