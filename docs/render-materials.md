# Render et Sols

`Biomes` conserve la palette et les transitions d’altitude informationnelles.
La pente ne remplace plus ses couleurs par de la roche. `Render` et `Sols`
suivent le même DEM et les mêmes LOD que le relief, sur carte et globe.

## Sol généré

`terrain_soil.py` compose un atlas à la résolution initiale : fractions de sable,
argile et humus normalisées, couleur du sol et couleur régionale de roche.
Le climat, l’altitude, le soulèvement et deux bruits sphériques déterministes
guident ce mélange. C’est une composition visuelle approximative.

Chaque composition de monde enregistre ces neuf rasters et leurs empreintes
dans l’atlas Orogen. Changer le relief ou le climat les actualise sans ajouter
d’étape neurale. Les anciens stages immuables restent utilisables ; un ancien
atlas dépourvu de sol peut dériver ce champ une fois en mémoire. Le layer Sols
montre sa couleur, avec un masque marin fourni par le DEM affiché.

## Matières

Le même `surfaceColor` WGSL est compilé dans le calcul des tuiles fines et dans
le fragment des blocs coarse. `terrain_render.py` fournit la référence CPU et
le fallback PNG.

- Huit familles de teintes végétales regroupent les trente classes Köppen.
  Leur climat régional de référence précède l’ajustement d’altitude ; les
  masques l’appliquent continûment, pour éviter des anneaux de neige au passage
  dans une classe alpine ou polaire.
- La végétation mêle massifs, clairières et herbes basses ; le sol reste visible.
  L’humidité, l’humus, l’altitude et la pente règlent leurs couvertures.
- La roche apparaît progressivement sur les pentes exposées et les sommets.
  Une courbure locale distingue les crêtes des creux.
- La neige dépend des saisons existantes, de l’altitude, de l’exposition et de
  la pente. Les montagnes froides la retiennent sur des pentes plus fortes,
  mais les falaises restent partiellement ou entièrement découvertes.
- Les entrées futures `water = (proximité de berge, humidité)` sont actuellement
  nulles. Elles peuvent alimenter le mélange sans inventer de réseau fluvial.

Les bruits de 12 km, 1,6 km, 180 m, 25 m et 12 m partagent le même seed et les
mêmes coordonnées géographiques 3D. Une déformation du bruit des massifs
utilise le bruit régional déjà calculé. Chaque octave disparaît progressivement
entre un quart et une moitié de sa longueur d’onde par pixel, puis son calcul
est supprimé. Le chart polaire est ramené dans le même repère géographique.

Les tuiles fines conservent leurs couleurs sur GPU. Le coarse adapte les
octaves à l’empreinte du pixel écran. Aucun calcul neural, téléchargement de
texture ou readback GPU supplémentaire n’est nécessaire pour changer de layer
ou de réglage de matière.

## Transport et réglages

Les cinq canaux physiques restent inchangés. Le transport d’affichage contient
46 plans : les 36 plans relief/climat/biomes existants, les neuf champs du sol
aux indices 36–44, puis les métadonnées exactes au plan 45. La ligne zéro de
ce dernier contient les bornes du monde, la topologie, le chart et le seed.

La saison, les variations, les massifs, la pente rocheuse, la neige et les trois
couleurs Render sont des réglages d’affichage, persistés dans `render_settings`
de l’URL. Le reset Render et le reset global restaurent aussi les RGB exacts,
indépendamment de l’aperçu hexadécimal des sélecteurs de couleur.

## Vérification

`test_terrain_render.py` vérifie la composition générée, les saisons opposées,
la neige sur les falaises, les clairières et les raccords géographiques.
`test_terrain_orogen_stages.py` vérifie la persistance et l’actualisation du sol
lors des générations indépendantes. `verify_terrain_render.cjs` compare dix
cas réels GPU/CPU, les deux chemins GPU et le reset exact des couleurs.
`verify_generator_sync.cjs` couvre aussi le reset global et le rechargement.
`verify_terrain_render_live.cjs` vérifie carte, globe et PNG sur le serveur réel.

Mesure indicative sur RTX 3090, Chrome headless, frame coarse de 960 × 540 :
environ 3–4 ms pour Biomes comme pour Render, aux empreintes de
1 000 m, 30 m et 3,75 m. Ce sont des médianes de soumission + complétion de
queue sur 30 frames, pas des durées isolées de shader ni une mesure de FPS.

## Sources des techniques

- [Terragen Distribution Shader](https://docs.planetside.co.uk/wiki/Distribution_Shader_v4) : masques altitude/pente, transitions et rupture fractale.
- [Gaea Colorizing and Textures](https://docs.gaea.app/using/using-gaea/colorizing-and-textures/index.html) : matières guidées par les propriétés du terrain.
- [Gaea Snow](https://docs.gaea.app/reference/nodes/simulate/snow) : altitude, adhérence, pente et exposition.
- [Unreal Landscape Materials](https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-materials-in-unreal-engine) : mélange des couches et détail selon la distance.
- [NVIDIA GPU Gems, Improved Perlin Noise](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-5-implementing-improved-perlin-noise) : coordonnées continues et filtrage fréquentiel des détails.
