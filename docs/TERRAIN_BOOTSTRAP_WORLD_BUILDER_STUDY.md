# Étude de World Builder pour le nouveau bootstrap terrestre

Date : 2026-10-07. Dépôt étudié : `D:/Workspace/Self/RPG/world-builder-rs`, remote `https://github.com/dunkean/world-builder-rs.git`, commit **009efe18c457757da12ffc8a6215806efb87f012**, `GENERATOR_VERSION = prototype-0.11.0-hybrid`. Licence MIT. Checkout propre avant et après essais. Aucun `AGENTS.md` applicable trouvé dans les parents et les deux dépôts. Le pull SSH a échoué faute de clé; `git fetch https://github.com/dunkean/world-builder-rs.git main` puis `git merge --ff-only FETCH_HEAD` ont réussi et confirmé ce commit déjà à jour. Aucun changement dans les sources upstream.

## Recommandation

World Builder fournit une **vraie altitude globale signée en mètres**, déterministe, calculée sur CPU avant exploration. Son atlas natif constitue une source appropriée pour conditionner Terrain Diffusion. Les extrêmes testés donnent réellement un supercontinent ou des centaines d'îles. Un bridge minimal doit utiliser `World::generate`, exporter les cellules physiques, puis reconstruire un raster fini 1024×512 ou 2048×1024. Il ne doit pas utiliser le viewer, le sampler enrichi, le backend CUDA régional ou les anciennes recettes du bootstrap.

La validation de style doit mesurer les composantes terrestres: les valeurs des paramètres seules ne garantissent pas la topologie de toutes les graines. Il faut conserver une identité/version pour la source, le binaire, les paramètres, le raster, la convention de coordonnées et les éventuels essais déterministes de sélection. L'export de l'altitude doit rester séparé de l'inférence et du LOD déjà validés.

## Fonctions exactes et mécanisme initial

Les lignes ci-dessous correspondent au commit indiqué, chemins relatifs au dépôt upstream.

| Source | Rôle réel |
| --- | --- |
| `crates/world-core/src/lib.rs:25` | Version du générateur. |
| `crates/world-core/src/lib.rs:63` `WorldConfig` | Seed u64, rayon, résolution/face, fraction océan, relief, plaques, fréquence/fragmentation continentale et options physiques. |
| `crates/world-core/src/lib.rs:423` `hash64`, `random` | Mixage entier SplitMix64; conversion des 53 bits hauts en flottant [0,1). |
| `crates/world-core/src/lib.rs:434` `noise` | Value noise 3D de huit sommets, interpolation cubic smoothstep. Coordonnées sphériques globales, pas un bruit par tuile. |
| `crates/world-core/src/lib.rs:467` `fbm` | Fractal additif, amplitude ×0.5, fréquence ×2, seed +97/octave. |
| `crates/world-core/src/lib.rs:479` `atlas_fbm` | Filtre amplitudes par `max(0,1-(frequency/(0.35*resolution))^4)`. |
| `crates/world-core/src/lib.rs:494` `plates` | Sites Fibonacci à rotation azimutale seedée, vitesses angulaires 3D et continentalité seedée par plaque. |
| `crates/world-core/src/lib.rs:514` `raw_height` | Champ initial signé combinant continentalité, fBm, convergence/divergence et texture. |
| `crates/world-core/src/lib.rs:556` `tectonic_uplift` | Soulèvement ultérieur près des frontières convergentes. |
| `crates/world-core/src/lib.rs:581` `choose_sea_level` | Flood minimax depuis le minimum global; quantile des seuils de connexion pondéré par aire, ou volume marin explicite. |
| `crates/world-core/src/lib.rs:699` `ClimateField::compile` | Climat paramétrique latitude/obliquité/bruit spatial; pas GCM. |
| `crates/world-core/src/lib.rs:911` `World::generate` | Assemblage atlas, datum marin, climat, érosion, second datum, hydrologie, classification, validation. |
| `crates/world-core/src/geometry.rs:5` `FACE_BASES` | Orientation des six faces du cube. Axe Y vers le nord; x/z portent longitude. |
| `crates/world-core/src/geometry.rs:61` `direction`, `face_uv`, `cell_area` | Projection cube-sphère et aire exacte par triangles sphériques. |
| `crates/world-core/src/hydrology.rs:44` `neighbours8` | Adjacence huit voisins entre faces, y compris rotation aux coutures. |
| `crates/world-core/src/hydrology.rs:106` `mark_ocean` | Composante connexe sous zéro contenant le minimum global. |
| `crates/world-core/src/hydrology.rs:130` `route_potential` | Priority Flood, drainage avec descentes réelles préférées et départage des plats. |
| `crates/world-core/src/hydrology.rs:393` `physical_water` | Réservoirs aux niveaux résultant des budgets de pluie/évaporation/infiltration. |
| `crates/world-core/src/erosion.rs:158` `evolve` | Incision stream-power implicite n=1, soulèvement, transport/dépôt, diffusion versants; bilans de masse. |
| `crates/world-app/src/main.rs:294` `generate_cli`, `save_world` | CLI CPU et export complet `atlas.json`, checksum BLAKE3, rapport physique. |

Forme exacte de l'altitude avant datum:

```text
C = 1.4 * (continentalite_lissee - 0.5)
    + continent_fragmentation * fbm(seed^0xabc1,p,2.8*continent_scale,4)
boundary = exp(-((score_plaque1-score_plaque2)/0.065)^2)
ridge = (1-abs(noise(seed^0x3729,p,23)))^3
mountain = max(convergence,0)*boundary*(0.30+0.70*ridge)
trench = min(convergence,0)*boundary*0.25
height = relief_m*(C + 0.65*mountain + trench + 0.10*atlas_fbm(...))
```

La continentalité lissée utilise toutes les plaques avec poids `exp(12*(dot(p,site)-1))`. La convergence vient de la différence des vitesses `omega × p` des deux plaques les plus proches, projetée vers la différence de leurs sites. Il s'agit d'un modèle de tectonique heuristique, pas d'une simulation évolutive des plaques. Le champ initial n'a pas de domain warp. Le terme ridge module les chaînes mais ce n'est pas un ridged multifractal complet.

## Ce que les captures du viewer ne prouvent pas

Les captures `docs/screenshots/0_10/archipelago.png` et `docs/screenshots/coherent-0.7-early.png` ont été ouvertes visuellement. Elles montrent un rendu détaillé distinct de l'atlas.

`archipelago_density` ne change pas `raw_height`. `landforms.rs:350` `island_region`, `IslandField::new:358`, `compile_bucket:449`, puis `terrain:978` ajoutent les îles au sampler. Leurs sept rayons vont de 60 km à 16 m. Ce slider augmente des provinces visuelles d'îles; il ne crée pas seul une géographie globale d'archipel dans l'atlas physique.

`coast.rs` `CoastWarp` compose des rotations sphériques inversibles seedées pour le détail de côte. `sampler.rs` `World::sample` reconstruit l'atlas et ajoute détails de relief, terres locales, lacs et rivières. Ces chemins sont coûteux et intègrent des ajouts que l'hydrologie coarse n'a pas simulés. Pour le nouveau bootstrap, exporter `world.cells[*].height_m`, pas `World::sample`/`World::tile` ni l'API `/preview`.

## Essais CPU et artefacts réellement générés

Compilation release native Windows à partir du dépôt: `C:/Users/grego/.cargo/bin/cargo.exe`. Aucun GPU utilisé par les appels de génération. `world-app` compile une dépendance CUDA à chargement dynamique mais le chemin `generate` utilise le générateur CPU de `world-core`.

Commandes existantes exécutées:

```powershell
cargo run --release -p world-core --example generation_benchmark -- 64 42 0
target/release/examples/generation_benchmark.exe 128 42 8
cargo run --release -p world-app -- --data study-output generate --seed 42 --resolution 64
target/release/worldctl.exe --data artifacts/bootstrap-study/<case> generate --config artifacts/bootstrap-study/<case>.json
```

Benchmark atlas seul (16 threads Rayon): 64²×6, seed42, sans érosion **27.78 ms**; 128²×6 avec 8 étapes **2.003 s**. CLI 64²×6 avec érosion, validation et sauvegarde **218 ms**. Variantes 128²×6 sans érosion, rayon terrestre 6 371 km, CLI avec export **320–339 ms** dans le premier trio. Temps ponctuels de cette machine, sans garantie sur une autre configuration.

Les trois styles suivants ont ensuite été exécutés pour les graines 0, 42 et 1739571689, sans érosion. Les composantes sont calculées sur les huit voisins natifs du cube, avec aire sphérique, pour `~ocean` (lacs et bassins continentaux inclus).

| Style testé | Plaques / échelle / fragmentation / océan | Plus gros bloc (% des terres), seed0 /42 /1739571689 | Nombre de blocs, mêmes graines |
| --- | --- | --- | --- |
| Gondwana | 3 / 0.35 / 0.12 / 0.65 | **93.82 / 95.88 / 99.04** | 67 / 93 / 28 |
| Continents | 14 / 1.0 / 0.8 / 0.71 | **60.17 / 33.75 / 76.69** | 52 / 39 / 39 |
| Grand archipel | 28 / 3.0 / 1.6 / 0.82 | **9.57 / 5.80 / 9.40** | 244 / 283 / 273 |

Un premier essai « grand continent » avec 8 plaques/.35/.25/.65 ne donnait que 51.01 % dans le plus gros bloc seed42; il a été écarté pour Gondwana. Ces résultats étayent le choix de trois plaques pour le supercontinent. Le cas continents seed1739571689 est trop dominant pour garantir plusieurs grandes masses: mesurer et sélectionner un candidat déterministe si ce style doit être strict.

Artefacts ignorés par git upstream, conservés sous **`D:/Workspace/Self/RPG/world-builder-rs/artifacts/bootstrap-study/`**:

- `coarse-comparison.png`: trois cartes signées du premier essai, ouverte et inspectée.
- `coarse-comparison-multi-seed.png`: neuf cartes signées des styles et graines.
- `topology-summary.json`, `topology-summary-multi-seed.json`: métriques et IDs natifs.
- `<case>.json`, `<case>-report.json`, `<case>/worlds/<id>/atlas.json`: configurations et données natives intégrales.
- `<case>-coarse.npz`: altitude float32(6,128,128), océan bool, aire float64, configuration JSON; aucun réseau exécuté.
- `analyze_coarse.py`: export/projection equirectangulaire et mesure de composantes; Python de `infinite_map/.venv` utilisé pour NumPy/SciPy/Matplotlib.

Les images emploient l'échantillonnage cellule la plus proche pour diagnostic; leur pixelisation aux pôles et coutures n'est pas une discontinuité du champ analytique initial. La courbe noire marque `height=0`, ce qui diffère parfois du masque physique océan.

## Limites matérielles à prendre en compte

1. **Hypsométrie et bathymétrie.** Aucun modèle continental shelf/slope/abyss ni distribution bimodale explicite. Les variants fragmentés multiplient fortement l'amplitude: seed42 atteint −19 045/+11 668 m. Gondwana est plus doux (environ −1 977/+2 636 m). Pour conditionnement terrestre stable, utiliser une transformation monotone documentée avec échelle mer/terre distincte et plafonds doux, ou des paramètres relief adaptés à chaque style; ne jamais transformer le relief selon la caméra.
2. **Altitude négative et terre.** L'océan est connexe: un bassin continental isolé peut avoir `height<0` sans être marin. Exemples min hors océan: −415 m Gondwana42, −2 313 m continents42. Une interface qui ne garde que l'altitude interprétera ces bassins comme mer. Décider explicitement si on conserve un masque coarse auxiliaire ou si la convention finale est simplement mer sous zéro.
3. **Fréquences coarse.** Seule la texture est filtrée au Nyquist. Le fBm continental (octaves jusqu'à 22.4×scale) et le ridge fréquence23 ne sont pas filtrés. n=128 supporte mieux les styles fragmentés que n=32/64; préférer n≥128 avant export 1024×512. Une exportation plus dense ne recrée pas de structure coarse perdue.
4. **Résolution, datum et identité.** `resolution` modifie à la fois le filtre de texture, la topologie et le niveau marin, donc le monde; les résolutions de génération et d'export doivent appartenir à l'identité. Le hash natif tronqué à 20 caractères couvre config+version, pas les octets source ou binaire: le bridge doit ajouter SHA256 source et executable.
5. **Climat.** Température annuelle latitude/tilt avec gradient 6.5 °C/km; humidité par bandes latitudinales et fBm seedé. Pas vents, ombres pluviométriques ni couplage océan/atmosphère. Les quatre canaux climatiques Terrain Diffusion demandent une adaptation explicite distincte.
6. **Détail et LOD.** Le terrain appris doit utiliser le même raster coarse persisté à toutes les requêtes. L'upstream viewer/hybrid et ses îles fines ne doivent pas remplacer le chemin NN/LOD validé du projet.

## Contrat pratique du bridge proposé

Entrée: seed u64, style et contrôles numériques, résolution native/face, rayon, résolution raster. Source primaire `World::generate(WorldConfig)` du commit étudié. Export en raster float32 signé, projection equirectangulaire, grille aux centres des pixels, x périodique en longitude et y borné aux pôles. Reconstruire l'atlas par interpolation entre centres avec voisinage inter-face; ne pas interpoler séparément chaque face en clampant ses bords.

Adresser les échantillons par coordonnées globales en mètres, domaine existant ±20 000 km horizontal/±10 000 km vertical, nord pour y négatif. Les cellules NN de 7.68 km interrogent ce raster par bilinéaire sans réensemencer. Cache immutable vérifié par empreinte de config, upstream commit/version/SHA source, bridge source, binaire, grille et octets du raster. Fournir des mesures de topologie et la configuration physique effective pour chaque monde créé. Un monde régénéré avec même identité doit être identique, quel que soit l'ordre de lecture, la caméra et le nombre de threads.
