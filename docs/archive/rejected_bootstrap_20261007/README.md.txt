# Terrain Diffusion — monde terrestre

Ouvrir **http://127.0.0.1:8765** ou double-cliquer sur **start-terrain.cmd**.

La page démarre sur une carte plane de **40 000 × 20 000 km**, avec la seed 42 et **Continents A4**, un initialiseur expérimental qui organise de grandes masses terrestres, côtes et chaînes avant le NN. Le profil **Naturel** conserve la référence originale du checkpoint 30 m, avec ses cinq champs ETOPO/WorldClim et son bruit conditionnant 0,5. **Macro A1/A2/A3** et **Earth** restent accessibles pour comparaison. La carte globale est un aperçu des entrées du NN ; sa légende distingue cet aperçu du terrain appris. Les validations A4 portent sur des entrées mondiales et des régions NN ; elles ne certifient pas une planète apprise complète.

- Molette / + / − : zoom ; glisser ou flèches : déplacement.
- Les détails des régions visibles sont générés automatiquement sur GPU, puis mis en cache.
- **Vue mondiale** revient aux continents ; **30 m / pixel** affiche le détail natif au centre de la vue.
- Le dézoom s'arrête au **LOD 11**, soit 61,44 km/pixel. Un zoom fort sur une zone neuve présente **LOD 4 → 3 → 2 → 1 → 0** avant le détail natif.
- **Carte** choisit relief, biomes, température ou précipitations. Les biomes sont une classification visuelle, sans simulation glaciaire ou hydrologique.
- La préparation coarse NN démarre progressivement en fond à l'ouverture. **Préparer le monde NN** reprend ce travail ; **Pause préparation** arrête les prochains blocs. Les demandes visibles passent avant le prochain bloc de fond. Une seule préparation mondiale est active à la fois.
- **Région initiale** revient à l'export historique lorsqu'il existe.
- **Nouveau monde** change la seed ; **Tuiles** affiche la grille.
- Au-delà de 30 m/pixel, le zoom agrandit les pixels natifs.

Les fenêtres pondérées sont fusionnées avec tous leurs contributeurs. Le coarse à 7,68 km et les latents à 240 m sont des approximations apprises du DEM final. Lorsque tous les enfants natifs nécessaires sont disponibles, les LOD 1 et 2 utilisent une moyenne exacte des altitudes finales signées ; leur climat conserve sa grille compacte issue du coarse. Les poids GPU sont partagés entre les mondes. Au dézoom, les tuiles détaillées précédemment visitées restent en cache mais ne se superposent plus au LOD courant.

Le batch latent de référence est fixe à 16. L'amont BF16 peut encore présenter une faible dépendance à la partition des requêtes. `TERRAIN_CANONICAL_LATENTS=1` sélectionne un profil expérimental à batch un, avec une autre identité ; il ne constitue pas une optimisation exacte de la référence.

Fichiers actifs : `terrain_server.py`, `terrain_jobs.py`, `terrain_inference.py`, `terrain_world.py`, `terrain_climate.py`, `terrain_lod.js`, `terrain_renderer.js`, `index.html`, `launch_terrain.py`, `start-terrain.cmd`. `terrain_app.py` fournit le chargement CUDA et conserve l’export original de rectangle. Résultats : `generated/natural-v1/<profil numérique>/<earth ou natural>/`, avec cache physique des altitudes et du climat partagé par le PNG et le rendu WebGPU. Les profils séparent les conditionnements et les chemins numériques.

Installation : Python 3.12, PyTorch 2.11.0 + CUDA 12.8, RTX 3090, modèle `xandergos/terrain-diffusion-30m`, inférence bfloat16. Source : https://github.com/xandergos/terrain-diffusion ; article : https://arxiv.org/abs/2512.08309.

L’environnement, les poids et les résultats résident dans **E:\TerrainDiffusionRuntime** ; `.venv` et `generated` sont des jonctions vers E:. Versions installées : `requirements-lock.txt`.

Lancement manuel :

```powershell
.\.venv\Scripts\python.exe terrain_server.py
```

Vérification du monde, puis du profil original :

```powershell
node verify_earth_viewer.cjs
node verify_natural.cjs
```

Le serveur écoute sur `127.0.0.1:8765`. Journaux : `server.log`, `server-error.log`.

## Navigation et calcul GPU

[Implémentation, résultats et vérifications](docs/REALTIME_IMPLEMENTATION.md).

[Géographie mondiale, climat, limites du modèle et LOD](docs/WORLD_GENERATION.md).

[Optimisation runtime LOD 3/2 et navigation progressive](docs/RUNTIME_OPTIMIZATION.md) · [Initialisation A4, preuves NN et adaptation des formules Rust](docs/CONTINENTAL_BOOTSTRAP.md) · [Corrections après reviews Astra/Opus](docs/reviews/RUNTIME-corrections.md).

La caméra déplace immédiatement la carte ; elle publie ses besoins au serveur. Une file partagée donne la priorité à la couverture visible, puis aux détails et au préchargement. Les demandes abandonnées sont retirées avant calcul ; un travail GPU déjà lancé termine son bloc. Revenir dans une région réutilise ses altitudes et ses textures, sans relancer le NN.

Le rendu WebGPU calcule le relief et la palette dans le navigateur à partir des altitudes FP32, puis conserve les textures sur GPU pour le déplacement et le zoom. Le chemin PNG/Canvas reste disponible. Les NN tournent sur CUDA dans le serveur Python local. Le port des NN eux-mêmes vers WebGPU est une expérience isolée : les vrais forwards et le crop navigateur présentent actuellement des erreurs numériques trop grandes. Il n'est pas utilisé pour générer le terrain affiché.

Les CUDA Graphs accélèrent le coarse et les petits batches latents après validation numérique exacte à la première capture. La sélection du GPU CUDA est automatique par défaut ; `TERRAIN_CUDA_DEVICE=0` ou `1` force une carte au démarrage. Cette version emploie une carte CUDA à la fois. Les caches navigateur et renderer sont bornés à **192 Mio chacun** ; un très grand écran peut afficher un LOD plus large que 30 m pour respecter ce budget, ce que la légende indique. `TERRAIN_DISK_CACHE_GIB` règle le quota du profil physique actif (16 Gio par défaut). Chaque monde possède un quota coarse de 2 Gio ; les anciennes identités sont conservées séparément. Le stockage total de plusieurs mondes n'a donc pas de quota global automatique. `/api/status` expose les captures/replis, le GPU sélectionné et les files de calcul.

Pour vérifier la navigation et capturer les diagnostics :

```powershell
node verify_realtime.cjs
```

Pour mesurer les quatre cas HTTP (poids éventuellement déjà chargés), fermer les autres onglets terrain et choisir une seed encore absente du cache :

```powershell
.\.venv\Scripts\python.exe benchmark_navigation.py --seed 2026100702 --output E:\TerrainDiffusionRuntime\realtime-qa\http.json
```

Le benchmark NN avec comparaison numérique se lance séparément, **serveur arrêté**, pour éviter une concurrence GPU : `python benchmark_inference.py`. Ses rapports et tableaux restent sur E:. `launch_terrain.py --no-open` démarre le serveur sans ouvrir un onglet.

## Design du moteur temps réel

[Audit terrain](docs/ASTRA_TERRAIN_AUDIT.md) · [Suivi des cinq blocs et décisions](docs/AUDIT_IMPLEMENTATION.md) · [Validation finale et limites mesurées](docs/VALIDATION_FINAL.md) · [Macro et corpus](docs/MACRO_IMPLEMENTATION.md) · [Fenêtres et persistence](docs/WINDOW_IMPLEMENTATION.md) · [Expérience WebGPU](docs/WEBGPU_IMPLEMENTATION.md) · [Reviews Astra et Opus](docs/reviews/).

Sur la RTX 3090, une préparation coarse mondiale de la seed 42 a terminé 6 160 fenêtres en 7 min 20 s, pour environ 675 Mio de payload persistant. Ce cache ne contient pas le DEM 30 m de toute la planète. La première reconstruction mondiale mesurée prend encore 8,3 s sans NN ; la dernière revisite navigateur native prend 208 ms sur cache chaud. Deux pans continus à 6 km/s sur cache chaud ont un segment CPU de dessin p95 de 0,4–0,6 ms ; le temps GPU/présentation complet reste non mesuré. Les coûts froids et les portes expérimentales ouvertes sont détaillés dans la validation finale.

Les vérifications GPU doivent être exécutées séquentiellement, serveur arrêté : `verify_coarse_persistence.py`, `verify_terrain_windows.py`, `verify_canonical_latents.py`. Avec le serveur seul sur le GPU, `benchmark_navigation.py` mesure le transport HTTP et `node benchmark_browser.cjs` mesure une session navigateur à 30 m. Les rapports distinguent cache chaud, calcul froid et intervalle de présentation ; ils ne prouvent pas une planète NN froide instantanée. Le raccord au moteur procédural, l'hydrologie régionale, les glaciers et le détail sous 30 m restent la phase suivante.

[Document Markdown](docs/TERRAIN_REALTIME_DESIGN.md) · [Version HTML de lecture](docs/TERRAIN_REALTIME_DESIGN.html) · [Mesures et références de session](docs/audit-session.json).

L’analyse donne la priorité à l’agent WASM/WebGPU du navigateur, avec un agent natif C++/CUDA/TensorRT Linux/Windows optionnel et distribuable en open source si son gain est démontré. Elle couvre l’ordonnancement, les caches, l’annulation, l’adaptation aux GPU des visiteurs et aux 3090/4090/5090, le mode natif à deux GPU et les modes réseau facultatifs. Toutes les sources de relief et de climat sont conservées ; les objectifs de performance sont distingués des mesures actuelles.

L’HTML est autonome et se lit hors ligne. Pour le régénérer depuis le Markdown : `python docs/render_design.py` avec `Markdown==3.8.2` installé ; l’option `--markdown-lib` accepte un dossier de dépendances séparé. L’environnement de l’application n’a pas reçu cette dépendance.
