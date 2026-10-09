# Biome layer / Layer Biomes

Implementation overview as inspected on 2026-10-09. Values below are the source defaults; saved worlds and custom climate settings may use different values.

- [English](#english)
- [Français](#français)

## English

### What the visible layer represents

The **Biomes** entry in the current layer menu selects `orogen-biomes`. It combines a Köppen climate class with an altitude-dependent color treatment. There are **30 land classes plus the ocean**, with additional rock and snow colors.

Orogen supplies the shared climate for the Noise, Tectonic and Custom relief generators. The layer represents climate-associated landscape colors. It does not independently simulate vegetation, forests, grasslands, wetlands, soils, glaciers or hydrology.

### Construction pipeline

1. **Generate relief and apply the selected erosion.** The resulting source terrain feeds the climate stage. Stages can also be generated independently; generating the complete chain updates relief, erosion and climate together.
2. **Calculate climate on a spherical mesh.** Orogen computes winds, ocean currents, precipitation and temperature for two seasons. Latitude, altitude, ocean influence and rain shadows contribute to these fields.
3. **Assign a Köppen class to each mesh region.** Temperature and precipitation determine the class. The two simulated seasons act as proxies for monthly climate; this is an approximation of Köppen classification, not a classification from a complete monthly time series. Local warm and cold seasons are identified from temperature so precipitation subtypes work in both hemispheres.
4. **Sample the class and appearance settings for each physical tile.** The class supplies a base color and alpine/snow thresholds. Newly exposed neural land inherits the nearest source land class; ocean pixels follow the current DEM sign. The nearest-land extension respects longitude wrapping on spherical worlds.
5. **Calculate the color from the current relief at every LOD.** Darkening, alpine rock and snow transitions are evaluated on the displayed physical DEM. At LOD 0 and finer, a slope override is applied after snow and vegetation. The class atlas remains **2,048 × 1,024 pixels**, in an equirectangular projection; compact sampled base colors and altitude lines are interpolated for rendering. The separate **Köppen** layer keeps categorical sampling and its diagnostic palette. The historical precomputed RGB atlas is retained, but the visible Biomes layer no longer renders it directly.

The main classification rules, with default thresholds, are:

| Group | Meaning | Main criterion |
|---|---|---|
| A | Tropical | Cold-season temperature ≥ 18 °C, unless classified as arid |
| B | Arid | Annual precipitation below a temperature- and seasonality-dependent threshold |
| C | Temperate | Cold season ≥ 0 °C, warm season ≥ 10 °C; outside tropical and arid classes |
| D | Continental | Cold season < 0 °C and warm season ≥ 10 °C; outside arid classes |
| ET | Tundra | Warm-season temperature ≥ 0 °C and < 10 °C |
| EF | Ice cap | Warm-season temperature < 0 °C |

Polar classes are assigned before the aridity test. For the other bands, the aridity threshold is `max(0, 20 × annual_mean_temperature + seasonal_addition)` in mm/year. The addition is 280 mm when at least 70% of rain falls in the local warm half-year, 0 mm when at most 30% does, and 140 mm otherwise. Below half this threshold, the class is desert; otherwise it is steppe. Arid climates are hot or cold according to whether annual mean temperature reaches 18 °C.

Within C and D, `f` means no dry season, `s` a dry summer and `w` a dry winter. Temperature letters distinguish hot summers (`a`), warm summers (`b`), short/cool summers (`c`) and extreme continental winters (`d`). These distinctions use seasonal proxies and an interpolated shoulder-month temperature.

### Default base colors

These are the **Biomes** palette colors before elevation adjustments and spatial interpolation. Hex values are rounded from the configured floating-point RGB values; they are not guaranteed to equal the final rendered pixel values.

| Code | Category | Base color |
|---|---|---|
| Af | Tropical rainforest | Deep green `#0D4D0D` |
| Am | Tropical monsoon | Dense green `#145412` |
| Aw | Tropical savanna | Yellow-green `#6B802E` |
| BWh | Hot desert | Sandy tan `#D1B880` |
| BWk | Cold desert | Gray-brown `#998C7A` |
| BSh | Hot steppe | Dry gold `#B89E4D` |
| BSk | Cold steppe | Olive-tan `#8C8552` |
| Cfa | Humid subtropical, hot summer | Medium green `#2E6B1F` |
| Cfb | Oceanic, warm summer | Rich green `#1F611A` |
| Cfc | Subpolar oceanic | Dark green `#1A471A` |
| Csa | Mediterranean, hot summer | Khaki `#737A38` |
| Csb | Mediterranean, warm summer | Olive green `#667333` |
| Csc | Mediterranean, cold summer | Dark olive `#596633` |
| Cwa | Subtropical, dry winter, hot summer | Medium green `#337024` |
| Cwb | Subtropical highland, dry winter | Green `#26661F` |
| Cwc | Cold subtropical highland, dry winter | Dark green `#1F521A` |
| Dfa | Humid continental, hot summer | Forest green `#1F5C14` |
| Dfb | Humid continental, warm summer | Dark forest green `#1A5214` |
| Dfc | Humid subarctic | Spruce green `#0F3814` |
| Dfd | Subarctic, extreme winter | Very dark green `#0D2E12` |
| Dsa | Continental, hot dry summer | Olive-brown `#61612E` |
| Dsb | Continental, warm dry summer | Dark olive-brown `#59592B` |
| Dsc | Subarctic, dry summer | Dark green `#143814` |
| Dsd | Extreme subarctic, dry summer | Very dark green `#0F2E12` |
| Dwa | Continental, dry winter, hot summer | Forest green `#245C1A` |
| Dwb | Continental, dry winter, warm summer | Dark forest green `#1F5217` |
| Dwc | Subarctic, dry winter | Spruce green `#123814` |
| Dwd | Extreme subarctic, dry winter | Very dark green `#0D2E12` |
| ET | Tundra | Earthy brown `#595238` |
| EF | Ice cap | Blue-gray white `#C7CCD6` |

Additional elevation colors:

- **Alpine rock:** brown-gray `#6B6152`.
- **Snow:** bluish white `#EBEDF5`.
- **Ocean:** a depth gradient, from approximately `#0A0F4D` in deep water to `#4D6B99` just below sea level. It is handled separately from the land palette.

### Elevation adjustments

| Climate group | Alpine rock transition starts | Snow transition starts |
|---|---:|---:|
| Tropical | 3,500 m | 5,500 m |
| Arid | 3,000 m | 5,000 m |
| Temperate | 2,000 m | 3,500 m |
| Continental with hot or warm summers | 1,500 m | 3,000 m |
| Subarctic | 800 m | 2,000 m |
| Tundra | 400 m | 1,500 m |
| Ice cap | Disabled | 500 m |

Below 200 m, the base color darkens by up to 7%, with maximum darkening at sea level. Between 200 m and the alpine line, it progressively darkens by up to 15%.

Above the alpine line, the color blends toward rock using a squared transition factor. The transition spans the interval between the alpine and snow lines, or a 2 km fallback interval when the snow line is not above the alpine line. Above the snow line, a separate squared blend adds snow. By default, the full snow color is reached **2,500 m above the snow threshold**. The threshold marks the start of the transition, not immediate complete snow cover.

### Slope rule and LOD behavior

The visible Biomes layer now follows the same physical terrain LODs as Relief, including neural coarse cells, latent previews, the 30 m DEM and experimental LODs −1 to −3. Its continuous refinement and forced LOD controls work like those of Relief. Other climate diagnostics retain their fixed LOD 9.

At **LOD 0 and finer** (sample spacing ≤ 30 m), terrain steeper than **40° by default** is rendered as rock, even in snowy or vegetated areas. Slope is measured with central differences of the current DEM in physical metres, using the tile halo. It does not use blurred hillshade or lighting exaggeration. The ocean mask takes precedence over the slope rule.

The threshold can be changed from **Rendering → Biomes → Pente rocheuse (°)**, between 1° and 89°. It is persisted as `biome_slope` in the URL. GPU tiles recolor without a neural inference or DEM upload; PNG rendering reuses the physical terrain cache. CPU and WebGPU use the same palette and altitude settings.

Biome colors, including snow, are multiplied directly by the terrain's hillshade intensity. This applies to physical tiles, native coarse rendering and the initial worldwide overview. Light is independent of the altitude palette: the former conversion from relief RGB luminance, multiplied by 1.8 and clipped, saturated on white summits and erased their shadows. The direct hillshade preserves the distinction between illuminated and shaded snow slopes while respecting the existing strength, ambient light, contrast and sunlight controls. The default snow base color remains `#EBEDF5`.

The transport retains the original five physical climate channels and appends sixteen display planes: base RGB, alpine/snow lines and appearance constants. The response advertises `X-Terrain-Climate-Layers: 21`. These display planes do not alter neural conditioning or the stored five-channel physical climate cache.

### Remaining limitations relevant to future improvements

- **Climate classes remain global.** Local terrain changes drive the surface appearance and coastline, but do not run a new climate simulation or assign a new Köppen class at every fine pixel.
- **Climate classes are used as landscape proxies.** Vegetation coverage, soil, wetlands and local ecological variation have no separate simulation in this layer.
- **Many categories use similar greens.** This suits the satellite-style palette, but makes climate classes difficult to distinguish visually.
- **Two biome implementations coexist.** The older `biomes` mode is hidden in the visible menu. It blends temperature, precipitation, seasonality, elevation, rock and snow continuously and follows the current neural terrain. It does not use the 30-class Orogen biome palette described above.

Colors, classification thresholds, altitude lines and transition settings are exposed through the climate controls. They take effect when the climate stage is regenerated; independently retained stages can still reflect previous inputs.

## Français

### Ce que représente le layer visible

L'entrée **Biomes** du menu actuel sélectionne `orogen-biomes`. Elle associe une classe climatique Köppen à un traitement de couleur dépendant de l'altitude. Elle contient **30 classes terrestres plus l'océan**, avec des couleurs supplémentaires pour la roche et la neige.

Orogen fournit le climat commun aux générateurs de relief Noise, Tectonique et Custom. Le layer représente des couleurs de paysage associées au climat. Il ne simule pas séparément la végétation, les forêts, les prairies, les marais, les sols, les glaciers ou l'hydrologie.

### Chaîne de construction

1. **Générer le relief et appliquer l'érosion sélectionnée.** Le terrain source obtenu alimente l'étape climat. Les étapes peuvent aussi être générées indépendamment ; générer toute la chaîne met à jour ensemble relief, érosion et climat.
2. **Calculer le climat sur un maillage sphérique.** Orogen calcule vents, courants océaniques, précipitations et températures pour deux saisons. Latitude, altitude, influence océanique et ombres de pluie contribuent à ces champs.
3. **Attribuer une classe Köppen à chaque région du maillage.** Température et précipitations déterminent la classe. Les deux saisons simulées servent d'approximation du climat mensuel ; il ne s'agit pas d'une classification à partir d'une série mensuelle complète. Les saisons chaude et froide locales sont identifiées à partir des températures pour traiter correctement les sous-types de précipitations dans les deux hémisphères.
4. **Échantillonner la classe et les réglages d'apparence pour chaque tuile physique.** La classe fournit une couleur de base et les seuils alpin/neigeux. Les nouvelles terres neuronales héritent de la classe terrestre source la plus proche ; les pixels océaniques suivent le signe du DEM courant. L'extension vers la classe terrestre la plus proche respecte le bouclage en longitude des mondes sphériques.
5. **Calculer la couleur avec le relief courant à chaque LOD.** Assombrissement, roche alpine et neige sont évalués sur le DEM physique affiché. Au LOD 0 et aux niveaux plus fins, une règle de pente s'applique après la neige et la végétation. L'atlas des classes reste à **2 048 × 1 024 pixels**, en projection équirectangulaire ; les couleurs de base et seuils d'altitude échantillonnés sur une grille compacte sont interpolés pour le rendu. Le layer **Köppen** séparé conserve son échantillonnage catégoriel et sa palette de diagnostic. L'atlas RGB historique précalculé est conservé, mais le layer Biomes visible ne l'affiche plus directement.

Les règles principales de classification, avec les seuils par défaut, sont :

| Groupe | Signification | Critère principal |
|---|---|---|
| A | Tropical | Température de la saison froide ≥ 18 °C, sauf classement aride |
| B | Aride | Précipitations annuelles sous un seuil dépendant de la température et de leur saisonnalité |
| C | Tempéré | Saison froide ≥ 0 °C, saison chaude ≥ 10 °C ; hors classes tropicales et arides |
| D | Continental | Saison froide < 0 °C et saison chaude ≥ 10 °C ; hors classes arides |
| ET | Toundra | Température de la saison chaude ≥ 0 °C et < 10 °C |
| EF | Calotte glaciaire | Température de la saison chaude < 0 °C |

Les classes polaires sont attribuées avant le test d'aridité. Pour les autres bandes, le seuil d'aridité vaut `max(0, 20 × température_moyenne_annuelle + ajout_saisonnier)` en mm/an. L'ajout est de 280 mm si au moins 70 % de la pluie tombe pendant le semestre chaud local, de 0 mm si au plus 30 % y tombe, et de 140 mm sinon. Sous la moitié de ce seuil, la classe est désertique ; sinon, elle est steppique. Les climats arides sont chauds ou froids selon que la température moyenne annuelle atteint ou non 18 °C.

Dans les groupes C et D, `f` signifie sans saison sèche, `s` un été sec et `w` un hiver sec. Les lettres de température distinguent les étés chauds (`a`), doux (`b`), courts/frais (`c`) et les hivers continentaux extrêmes (`d`). Ces distinctions utilisent les approximations saisonnières et une température interpolée de mois intermédiaire.

### Couleurs de base par défaut

Il s'agit de la palette **Biomes**, avant les effets d'altitude et l'interpolation spatiale. Les valeurs hexadécimales sont arrondies depuis les RGB flottants configurés ; elles ne correspondent pas nécessairement aux pixels finalement affichés.

| Code | Catégorie | Couleur de base |
|---|---|---|
| Af | Forêt tropicale humide | Vert profond `#0D4D0D` |
| Am | Tropical de mousson | Vert dense `#145412` |
| Aw | Savane tropicale | Vert jaune `#6B802E` |
| BWh | Désert chaud | Sable `#D1B880` |
| BWk | Désert froid | Gris brun `#998C7A` |
| BSh | Steppe chaude | Ocre doré `#B89E4D` |
| BSk | Steppe froide | Olive beige `#8C8552` |
| Cfa | Subtropical humide, été chaud | Vert moyen `#2E6B1F` |
| Cfb | Océanique, été doux | Vert soutenu `#1F611A` |
| Cfc | Océanique subpolaire | Vert sombre `#1A471A` |
| Csa | Méditerranéen, été chaud | Kaki `#737A38` |
| Csb | Méditerranéen, été doux | Vert olive `#667333` |
| Csc | Méditerranéen, été froid | Olive sombre `#596633` |
| Cwa | Subtropical, hiver sec, été chaud | Vert moyen `#337024` |
| Cwb | Subtropical d'altitude, hiver sec | Vert `#26661F` |
| Cwc | Subtropical d'altitude froid, hiver sec | Vert sombre `#1F521A` |
| Dfa | Continental humide, été chaud | Vert forestier `#1F5C14` |
| Dfb | Continental humide, été doux | Vert forestier sombre `#1A5214` |
| Dfc | Subarctique humide | Vert conifère `#0F3814` |
| Dfd | Subarctique, hiver extrême | Vert très sombre `#0D2E12` |
| Dsa | Continental, été chaud et sec | Olive brun `#61612E` |
| Dsb | Continental, été doux et sec | Olive brun sombre `#59592B` |
| Dsc | Subarctique, été sec | Vert sombre `#143814` |
| Dsd | Subarctique extrême, été sec | Vert très sombre `#0F2E12` |
| Dwa | Continental, hiver sec, été chaud | Vert forestier `#245C1A` |
| Dwb | Continental, hiver sec, été doux | Vert forestier sombre `#1F5217` |
| Dwc | Subarctique, hiver sec | Vert conifère `#123814` |
| Dwd | Subarctique extrême, hiver sec | Vert très sombre `#0D2E12` |
| ET | Toundra | Brun terreux `#595238` |
| EF | Calotte glaciaire | Blanc gris bleuté `#C7CCD6` |

Couleurs supplémentaires liées à l'altitude :

- **Roche alpine :** brun gris `#6B6152`.
- **Neige :** blanc bleuté `#EBEDF5`.
- **Océan :** dégradé selon la profondeur, d'environ `#0A0F4D` en eau profonde à `#4D6B99` juste sous le niveau marin. Il est traité séparément de la palette terrestre.

### Effets de l'altitude

| Groupe climatique | Début de transition rocheuse | Début de transition neigeuse |
|---|---:|---:|
| Tropical | 3 500 m | 5 500 m |
| Aride | 3 000 m | 5 000 m |
| Tempéré | 2 000 m | 3 500 m |
| Continental à été chaud ou doux | 1 500 m | 3 000 m |
| Subarctique | 800 m | 2 000 m |
| Toundra | 400 m | 1 500 m |
| Calotte glaciaire | Désactivée | 500 m |

Sous 200 m, la couleur de base s'assombrit de 7 % au maximum, avec l'assombrissement maximal au niveau marin. Entre 200 m et le seuil alpin, elle s'assombrit progressivement de 15 % au maximum.

Au-dessus du seuil alpin, la couleur se mélange à celle de la roche avec un facteur de transition au carré. La transition couvre l'intervalle entre les seuils alpin et neigeux, ou un intervalle de secours de 2 km lorsque le seuil neigeux n'est pas au-dessus du seuil alpin. Au-dessus du seuil neigeux, un autre mélange au carré ajoute la neige. Par défaut, sa couleur complète est atteinte **2 500 m au-dessus du seuil neigeux**. Ce seuil marque le début de la transition, pas une couverture neigeuse immédiatement complète.

### Règle de pente et comportement des LOD

Le layer Biomes visible suit maintenant les mêmes LOD physiques que Relief : cellules coarse neuronales, aperçus latents, DEM à 30 m et LOD expérimentaux −1 à −3. Le raffinement continu et le LOD forcé fonctionnent comme dans Relief. Les autres diagnostics climatiques restent au LOD 9 fixe.

Au **LOD 0 et aux niveaux plus fins** (espacement des échantillons ≤ 30 m), une pente supérieure à **40° par défaut** est affichée en roche, même sous un biome enneigé ou végétalisé. La pente est mesurée par différences centrales sur le DEM courant en mètres physiques, avec le halo de la tuile. Elle n'utilise ni le relief lissé de l'éclairage ni son exagération. Le masque océanique reste prioritaire sur cette règle.

Le seuil se règle dans **Rendu → Biomes → Pente rocheuse (°)**, entre 1° et 89°. Il est conservé dans l'URL sous `biome_slope`. Les tuiles GPU sont recolorées sans inférence neuronale ni renvoi du DEM ; le rendu PNG réutilise le cache du terrain physique. CPU et WebGPU utilisent la même palette et les mêmes réglages d'altitude.

Les couleurs des biomes, neige comprise, sont multipliées directement par l'intensité de l'éclairage du relief. Cela s'applique aux tuiles physiques, au rendu coarse natif et à la vue mondiale initiale. La lumière est indépendante de la palette d'altitude : l'ancienne conversion depuis la luminance RGB du relief, multipliée par 1,8 puis plafonnée, saturait sur les sommets blancs et effaçait leurs ombres. L'éclairage direct conserve la différence entre versants enneigés éclairés et ombragés, tout en respectant les réglages existants d'intensité, de lumière ambiante, de contraste et de soleil. La couleur de base de la neige reste `#EBEDF5`.

Le transport conserve les cinq canaux climatiques physiques et ajoute seize plans d'affichage : RGB de base, seuils alpin/neigeux et constantes d'apparence. La réponse indique `X-Terrain-Climate-Layers: 21`. Ces plans ne modifient ni le conditionnement neuronal ni le cache climatique physique à cinq canaux.

### Limites restantes pour les améliorations futures

- **Les classes climatiques restent globales.** Les changements du terrain local pilotent l'apparence et les côtes, mais ne relancent pas une simulation climatique ou une attribution de classe Köppen pour chaque pixel fin.
- **Les classes climatiques servent d'approximation du paysage.** Couverture végétale, sols, marais et variations écologiques locales n'ont pas de simulation séparée dans ce layer.
- **Beaucoup de catégories utilisent des verts proches.** Cela convient à la palette de type satellite, mais rend les classes climatiques difficiles à distinguer visuellement.
- **Deux implémentations de biome coexistent.** L'ancien mode `biomes` est masqué dans le menu visible. Il mélange continuellement température, précipitations, saisonnalité, altitude, roche et neige, et suit le terrain neuronal courant. Il n'utilise pas la palette Orogen à 30 classes décrite ici.

Les couleurs, seuils de classification, limites d'altitude et réglages de transition sont accessibles dans les contrôles du climat. Ils prennent effet lorsque l'étape climat est régénérée ; les étapes conservées indépendamment peuvent encore refléter des entrées précédentes.

## Source references / Références du code

- [Visible layer menu / Menu des layers](../terrain_toolbar.js): maps **Biomes** to `orogen-biomes` and hides the older `biomes` mode.
- [Climate classification / Classification climatique](../native/orogen/vendor/koppen.js): class definitions and two-season classification rules.
- [Biome color calculation / Calcul des couleurs](../native/orogen/vendor/color-map.js): base palette mapping, physical elevation conversion, darkening, rock, snow and ocean gradients.
- [Default climate parameters / Paramètres climatiques par défaut](../native/orogen/climate-parameters.json): configurable classification thresholds, palette and altitude settings.
- [Native pipeline / Pipeline natif](../native/orogen/pipeline.mjs): calculates and exports `biome_0`, `biome_1`, `biome_2` and separate Köppen diagnostic colors.
- [Atlas dimensions / Dimensions de l'atlas](../terrain_orogen.py): production dimensions and spherical atlas construction.
- [Atlas rasterization / Rasterisation de l'atlas](../terrain_orogen_cuda.py): interpolated biome colors and categorical Köppen fields.
- [Independent generation stages / Étapes de génération indépendantes](../terrain_orogen_stages.py): retained relief, erosion and climate fields.
- [Adaptive biome rendering / Rendu adaptatif des biomes](../terrain_biomes.py): source class sampling, nearest-land extension, current DEM altitude and slope coloring.
- [Source diagnostic rendering / Rendu des diagnostics source](../terrain_orogen_layers.py): historical RGB atlas and categorical Köppen sampling.
- [Browser LOD selection / Sélection du LOD dans le navigateur](../index.html): physical LOD selection for Biomes, fixed LOD 9 for climate diagnostics, and the slope control.
- [Older biome palette / Ancienne palette de biome](../terrain_climate.py) and [GPU renderer / Rendu GPU](../terrain_renderer.js): continuous climate/elevation coloring for the hidden `biomes` mode.

## Validation

- `python -m unittest test_terrain_biomes test_terrain_tile_rendering test_terrain_climate`: current coastlines and elevations, all 30 classes under the slope override, snow preservation below the threshold, LOD gating, custom settings and adjacent DEM halos.
- `node verify_terrain_biomes.cjs`: real WebGPU pixels compared with CPU outputs, including illuminated and shaded snowy slopes, threshold recoloring and native coarse rendering.
- `node test_terrain_coarse_gpu.cjs`: browser LOD routing, GPU layer switching, slope URL persistence and fine biome LODs.
