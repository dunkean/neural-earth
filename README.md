# Terrain Diffusion — monde terrestre

Cloner le projet avec ses dépendances et références épinglées :

```powershell
git clone --recurse-submodules https://github.com/dunkean/infinite_map.git
```

Les sous-modules conservent les dépôts upstream à leurs commits de référence.
Les poids, caches, données générées et environnements locaux sont exclus de Git.
Le dépôt voisin `world-builder-rs` reste requis au commit indiqué dans
[le contrat natif](docs/TERRAIN_BOOTSTRAP_NATIVE.md).

Ouvrir **http://127.0.0.1:8765** ou lancer **start-terrain.cmd**.

La page ouvre un monde terrestre avec une nouvelle seed aléatoire. Le bootstrap
est un heightmap signé de **1024 × 512**, sur une carte de **40 000 × 20 000 km**.
Il est généré sur CPU par le cœur natif de `world-builder-rs` : plaques,
continentalité, bruit fractal et relief tectonique. Les distributions ETOPO
calibrent séparément les altitudes et les profondeurs en mètres. Aucune carte
géographique réelle ni silhouette fabriquée à la main n'est copiée.

Le menu propose **Gondwana**, **grands continents séparés**, **monde terrestre**
et **archipel mondial**. **Naturel · référence NN** conserve le conditionnement
original du checkpoint pour comparaison. Les anciens Earth et Macro A1–A4
sont retirés des API et de l'interface ; leur code est archivé dans
`docs/archive/rejected_bootstrap_20261007/`. Le nouveau bootstrap n'utilise
aucun code ou procédé de city-generator.

La vue mondiale présente les **entrées du NN**, avec une résolution source
de **39,06 km** ; le réseau les échantillonne à **7,68 km**. Les régions visibles
passent ensuite par le coarse appris, les latents à 240 m et le DEM natif à
30 m. Le NN peut déplacer les côtes : aucun masque ne remet artificiellement
le relief appris dans la silhouette initiale.

- Molette / + / − : zoom ; glisser ou flèches : déplacement.
- **Vue mondiale** revient à l'ensemble du monde ; **30 m / pixel** montre le détail natif.
- Le dézoom s'arrête au **LOD 11**, soit 61,44 km/pixel. Un zoom fort présente **4 → 3 → 2 → 1 → 0**.
- **Nouveau monde** change la seed. Un lien avec `?seed=…&profile=terrestrial-earthlike` est reproductible, y compris pour une seed u64.
- **Carte** sélectionne relief, biomes, température ou précipitations.
- **Préparer le monde NN** lance la préparation coarse mondiale en fond ; **Pause préparation** suspend les prochains blocs. Les demandes visibles sont prioritaires.

Ouvrir **Paramètres de génération** pour choisir séparément l'altitude et le
climat, puis ajuster fréquence, octaves et bruit permis pour chacun des cinq
canaux. **Appliquer paramètres** conserve la position et le zoom ; les curseurs
seuls ne lancent aucune génération. **Mémoriser A**, puis appliquer une variante,
permet de basculer entre A et B sur la même région. Le lien de la page conserve
les paramètres, et **Rétablir le profil par défaut** retire les personnalisations.

Le préréglage **Naturel continental** conserve le signal régional Natural et
modifie sa composante à grande échelle, avec une force et un rayon de lissage
réglables. C'est un essai expérimental, sans garantie de géomorphologie réaliste.
Le préréglage **Naturel · référence** restaure le chemin original du checkpoint.
Chaque configuration possède une identité distincte pour ses caches et ses
fenêtres NN. [Utilisation et contrat des réglages](docs/GENERATION_CONTROLS.md).

L'exploration calcule les régions visitées à la demande. La préparation de
toute la planète est explicite, via le bouton ou `?prepare=1`. Le benchmark
antérieur occupait environ 707 Mo par monde complètement préparé ; le quota
coarse est par monde, sans éviction globale entre seeds. Le cache de heightmaps
est également persistant, sans quota global.

Les cinq canaux gardent leurs unités physiques. Le climat initial utilise
les distributions WorldClim par latitude et un gradient thermique en altitude ;
il ne simule pas encore la circulation atmosphérique. Le relief et le climat
appris passent par le chemin CUDA existant, avec les optimisations de fenêtres,
poids et batches conservées. WebGPU affiche les altitudes FP32 dans le navigateur.

Le heightmap est immutable, persisté et vérifié. Les identités comprennent la
seed demandée, le candidat déterministe retenu, les paramètres natifs, les sources,
le binaire Rust, les poids NN et les règles de calibration. Les anciennes identités
de cache restent séparées et ne sont pas réutilisées par le nouveau générateur.

Pré requis locaux : Python 3.12 / environnement `.venv`, PyTorch CUDA, checkpoint
`xandergos/terrain-diffusion-30m`, rasters ETOPO/WorldClim, Cargo et dépôt voisin
`../world-builder-rs` au commit documenté. Le bridge compile automatiquement en
release au premier démarrage si nécessaire. Les dépendances Rust sont verrouillées.
Les poids, l'environnement et les résultats résident dans **E:\TerrainDiffusionRuntime**.

```powershell
.\.venv\Scripts\python.exe launch_terrain.py --no-open
```

Le serveur écoute sur `127.0.0.1:8765`. Journaux : `server.log`, `server-error.log`.
Les tests GPU se lancent séparément, serveur arrêté.

[Nouveau bootstrap et validation](docs/TERRAIN_BOOTSTRAP_RESTART.md) ·
[Générateur natif et contrat](docs/TERRAIN_BOOTSTRAP_NATIVE.md) ·
[Conditionnement physique](docs/TERRAIN_BOOTSTRAP_CONDITIONING.md) ·
[Qualité CPU et NN](docs/TERRESTRIAL_BOOTSTRAP_QA.md) ·
[Étude world-builder-rs](docs/TERRAIN_BOOTSTRAP_WORLD_BUILDER_STUDY.md) ·
[Étude des autres projets](docs/TERRAIN_BOOTSTRAP_LOCAL_STUDY.md) ·
[Littérature](docs/TERRAIN_BOOTSTRAP_LITERATURE.md) ·
[Reviews](docs/reviews/).

Les anciens rapports de navigation et de bootstrap sont des preuves historiques.
Le nouveau bootstrap ne certifie ni une planète entièrement apprise ni une
génération NN froide instantanée. L'hydrologie, les glaciers et le raccord au
reste du moteur procédural sont les étapes suivantes.
