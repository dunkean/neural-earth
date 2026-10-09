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

**Rendu → Calcul neuronal → Streams coarse** choisit 1, 2, 4, 8 ou 16
fenêtres traitées en parallèle, avec 4 par défaut. Le réglage est partagé
entre les onglets du même moteur et conservé dans les liens. Le batch du
réseau reste à 1 et les fenêtres calculées restent réutilisables après un
changement. Plus de streams augmente la mémoire et la durée des groupes
avant de rendre la main au premier plan. Le serveur peut revenir au calcul
séquentiel si la capture ne peut pas être admise ; le panneau indique alors
le nombre effectif. `TERRAIN_COARSE_STREAMS` règle la valeur au démarrage.
Les [mesures sur les trois modèles](docs/performance_streams/README.txt)
montrent des gains modestes pour base/décodeur, dont les streams restent
expérimentaux.

La vue d'ensemble du layer est chargée et affichée avant les nouvelles tiles
et la préparation coarse. Les LOD parents déjà prêts restent visibles pendant
le zoom et sont remplacés progressivement par les blocs plus fins disponibles.
Les vues larges utilisent leurs tiles de LOD parent ; les blocs coarse natifs
prennent le relais aux LOD 7 à 4. Un changement d'éclairage ou de courbes garde
la vue d'ensemble du même layer jusqu'à l'arrivée de son remplacement.

Les boutons **Carte** et **Globe** sont directement dans la barre de menus.
Sur le globe, glisser fait orbiter la caméra ; la vitesse diminue avec le zoom.
La molette, +/− et « Vue du monde » règlent le zoom ; les flèches font tourner
le globe. Le globe affiche le layer sélectionné et demande le relief neuronal
progressif aux mêmes LOD et au même moteur NN que la carte, y compris le
raffinement et les deux côtés du méridien. Carte et globe partagent le cache
d'altitudes et les textures rendues (ou les PNG en secours) : changer de vue
réutilise les tiles disponibles à la même résolution sans les retélécharger.

Au-delà de 60° de latitude et en zoom local, le globe calcule les détails NN
dans une grille sphérique tournée de 90° : les pôles géographiques y sont à
l'équateur. Les latents et le DEM gardent ainsi leurs distances physiques,
sans être pincés en éventail. La caméra charge une zone locale de cette grille,
avec un cache distinct ; l'atlas des continents et la carte restent inchangés.
Les paramètres du monde et le SNR en latitude utilisent la latitude géographique.

**Settings → Relief → Monde · création** choisit le **diamètre en km** et la
**géométrie** avant génération. Le diamètre par défaut est 40 000 / π km,
soit environ 12 732,40 km ; une planète utilise une carte équirectangulaire
π × diamètre de large et π × diamètre / 2 de haut. Le mode **Carte plane ·
non projetée** utilise un carré de côté égal au diamètre, des distances
cartésiennes et des bords finis sans bouclage ; le globe y est désactivé.
Les générateurs sélectionnés restent les sources du relief et du climat,
y compris le générateur tectonique Orogen issu d’un maillage sphérique.
Changer la géométrie ou le diamètre puis générer crée une identité et des
caches distincts. Ces paramètres sont conservés dans les liens.

**Rendu → Rendu → Éclairage** règle immédiatement les ombres, la lumière
ambiante, leur contraste, le relief de l’éclairage et la direction/hauteur du
soleil. Choisir **Global** ou un **LOD** permet notamment de régler 3, 2, 1 et 0
séparément ; les LOD héritent du global tant que leur héritage reste coché.
**Sans ombres**, **Doux** et **Défaut** sont des presets. Les réglages sont
conservés dans les liens, sans régénérer les altitudes ni les caches NN ; ils
fonctionnent avec WebGPU et le rendu PNG de secours. Le globe applique le
global à son éclairage sphérique.

**Layer visible → Styles de carte** reprend les styles de `city_generator` :
Parchemin, Atlas, Classique (Watabou), Gravure, Cadastre, Blueprint, Enluminure,
Topographique et Nuit, ainsi que **MNE · Copernicus (Rust)** du prototype Rust.
Les palettes de terrain, d'eau et de courbes sont communes à WebGPU et au PNG
de secours ; les styles suivent le relief disponible à chaque LOD, sur carte
et globe. Leur plage de couleur est fixée à 0–4 500 m pour rester cohérente
entre tuiles et niveaux de zoom.
Le grain du papier et les hachures suivent la résolution du terrain, y compris
aux LOD négatifs, pour éviter les gros blocs de texture en zoom local.
Changer de style recolore les textures physiques en cache, sans recharger les
altitudes ni relancer le NN. Les vues d'ensemble déjà rendues sont réutilisées.
**Rendu → Rendu → Courbes de niveau** active les lignes et règle leur
équidistance, automatiquement selon le zoom ou en mètres en mode manuel.
Un slider logarithmique règle la densité automatique (3,125–200 %). L'ancien
minimum de 25 % est au milieu et devient la valeur par défaut ; la moitié gauche
permet d'espacer davantage les courbes. Un autre slider règle la largeur de
0,3 à 4 px (0,9 px par défaut).
Chaque cinquième ligne est une courbe principale. Les courbes sont des segments
interpolés sur les altitudes affichées, continus aux frontières des tuiles,
sans régénérer le relief. Les pentes fortes conservent leurs lignes. Les liens
conservent le style, l'activation, l'intervalle, le mode automatique, la densité et la largeur.
Les tokens visuels de `terrain_styles.json` sont adaptés de
`city_generator/web/src/render/styles.ts` et `rust/bridge/terrainRender.ts`
(GPL-3.0). Les couches urbaines et leurs symboles restent propres au générateur
de villes.

La page ouvre un monde terrestre avec une nouvelle seed aléatoire. Le bootstrap
est un heightmap signé de **1024 × 512**, sur une carte de **40 000 × 20 000 km** avec le diamètre par défaut.
Il est généré sur CPU par le cœur natif de `world-builder-rs` : plaques,
continentalité, bruit fractal et relief tectonique. Les distributions ETOPO
calibrent séparément les altitudes et les profondeurs en mètres. Aucune carte
géographique réelle ni silhouette fabriquée à la main n'est copiée.

La combo **Generator** choisit la source du monde. Les paramètres proposent
**Gondwana**, **grands continents séparés**, **monde terrestre** et
**archipel mondial**. **Naturel · référence NN** conserve le conditionnement
original du checkpoint pour comparaison. Les anciens Earth et Macro A1–A4
sont retirés des API et de l'interface ; leur code est archivé dans
`docs/archive/rejected_bootstrap_20261007/`. Le bootstrap continental conserve
le moteur world-builder-rs ; City intervient seulement comme érosion GPU optionnelle.

**Orogen · tectonic chains** ajoute la génération initiale Orogen d'origine,
avec superplaques et mouvements tectoniques. Le menu **Génération** propose **NN**, **Tectonique** et **Custom** pour le relief initial.
Le menu **Settings** propose les onglets **Relief**, **Érosion** et **Climat**.
L’érosion offre **Désactivée**, **Orogen · GPU**, **Orogen · CPU** et **City · GPU**.
Chaque onglet génère uniquement son étape ; les autres résultats sont conservés,
avec une indication si leurs entrées ont changé. **Générer toute la chaîne**, le
bouton général et **Random** exécutent relief → érosion → climat. Random choisit
une nouvelle seed et génère immédiatement. City réutilise le moteur GPU du prototype
`city_generator/rust` : incision hydraulique, relaxation thermique et diffusion,
avec dose, itérations, talus et échelle de drainage indépendants. Le climat est
recalculé après l'érosion. Pour installer son runtime optionnel :
`python -m pip install -r requirements-city-gpu.txt` (déjà installé dans la `.venv` locale).
Le pipeline atlas/NN conserve son prérequis CUDA ; cette option ne rend pas
l'application complète compatible avec les GPU AMD/Intel ou sans GPU.
**Orogen historique reste le défaut.** Le relief tectonique expose des options
GPU pour le relief, la propagation et la recherche spatiale ; l’érosion GPU se
choisit dans sa combo, avec un post-traitement GPU optionnel. Le climat a son
propre interrupteur GPU. Ils sont tous désactivés par défaut. Le relief et l'érosion parallèles
sont expérimentaux ; Save A / Show A permet de comparer les résultats. Installer
le runtime NVIDIA optionnel avec `python -m pip install -r requirements-orogen-gpu.txt`.
Sans ce runtime, ces interrupteurs se replient vers Orogen CPU ; le reçu de
génération indique la raison. City utilise son propre runtime WebGPU.
Le menu **Rendu** regroupe **Rendu** et **SNR**. Le SNR propose un choix
**Global / Par LOD** et un interrupteur **SNR adaptatif** ; désactiver ce dernier
conserve ses réglages mais ignore les règles altitude/climat/latitude.
**Appliquer SNR** ne régénère aucune étape physique. L’onglet **Climat** expose
le climat Orogen commun, ses saisons, vents, pluie et biomes. Les paramètres, vues de diagnostic et comparaison A/B
permettent de tester les combinaisons. La simulation globale est mise en cache ;
la projection de l'atlas et les détails neuronaux utilisent le GPU.
[Fonctionnement, diagnostics et benchmark Orogen](docs/OROGEN_GENERATION.md).

La vue mondiale présente les **entrées du NN**, avec une résolution source
de **39,06 km** ; le réseau les échantillonne à **7,68 km**. Les régions visibles
passent ensuite par le coarse appris, les latents à 240 m et le DEM natif à
30 m. Le NN peut déplacer les côtes : aucun masque ne remet artificiellement
le relief appris dans la silhouette initiale.

- Molette / + / − : zoom ; glisser ou flèches : déplacement.
- **Vue mondiale** revient à l'ensemble du monde ; **30 m / pixel** montre le détail natif.
- Le dézoom s'arrête au **LOD 11**, soit 61,44 km/pixel. Un zoom fort présente **4 → 3 → 2 → 1 → 0**.
- **Random**, à côté de la graine, choisit une nouvelle seed ; **Generate** ouvre
  le monde avec la graine et le générateur sélectionnés. Un lien avec `?seed=…&profile=terrestrial-earthlike` est reproductible, y compris pour une seed u64.
- **Carte** sélectionne relief, biomes, température ou précipitations.
- **Tiles visible**, dans la barre du haut, affiche les limites et les index des
  zones réellement rendues, y compris lorsque plusieurs LOD coexistent.
- **Raffinement · profondeur LOD**, dans **Rendering**, vaut **0** par défaut (désactivé).
  Chaque niveau est terminé sur toute la vue avant le suivant : LOD 5 et profondeur 2
  calcule LOD 4, puis LOD 3. Les tiles arrivent progressivement à l'écran.
- La fenêtre **Rendering** regroupe les options de rendu. **Render** lance une progression
  unique jusqu'au LOD choisi, sans modifier le zoom ; bouger la caméra annule ce choix.
  **Cache autorisé (Go)** règle le budget des tuiles du navigateur et du rendu GPU,
  à **2 Go par défaut**. Il est conservé dans les liens (`cache_gib`) et limite
  la profondeur effective du raffinement.
  **Cache · LOD gap** autorise par défaut 3 niveaux d'écart : un cache LOD 3 peut
  s'afficher depuis LOD 6. À 0, aucun historique plus fin n’est conservé ; les parents plus grossiers disponibles restent utilisables.
  Le raffinement continu et le rendu manuel font progresser ce LOD de rendu.
  Pendant un zoom, le rendu plus grossier déjà affiché et ses cadres restent présents
  jusqu'à leur remplacement effectif par les nouvelles tiles.
- **Coarse on GPU**, coché par défaut dans **Rendu**, charge des blocs fixes du
  coarse à 7,68 km par cellule. Interpolation, éclairage et filtrage des vues
  éloignées sont calculés sur le GPU du navigateur : les LOD 4 à 11 réutilisent
  les mêmes blocs pendant le zoom. Les régions nouvelles doivent être chargées ;
  l'aperçu procédural reste affiché avant la disponibilité du coarse appris.
  Le base model et le decoder prennent ensuite le relais aux LOD plus fins.
  Décocher cette option rétablit les tiles calculées par LOD sur le serveur ;
  sans WebGPU, ce chemin reste utilisé automatiquement.
- Une nouvelle génération annule la précédente, arrête le coarse et vide les
  calculs de l'ancienne carte. Relief, érosion et climat disposent du GPU sans
  concurrence des NN ; la préparation mondiale reprend après la dernière
  génération terminée uniquement si elle a été activée. L'annulation respecte le bloc GPU déjà soumis.
- Le coarse de toute la carte est **facultatif et désactivé par défaut**.
  Seule la zone visible est calculée ; **Prefetch** prépare les voisins lorsque
  les tiles visibles sont prêtes. Dans **Outils**, **Préparer toute la planète**
  ou **Prepare neural world** active explicitement la préparation mondiale en
  fond ; **Pause preparation** la désactive. Les demandes visibles restent
  prioritaires. `prepare_world=1` conserve cette activation dans un lien ; les
  anciens liens `coarse_prepare=1` n'activent plus automatiquement le calcul global.
- **Mer max LOD**, réglé à **9**, arrête les calculs des tiles sans relief au-dessus
  de −10 m dès ce LOD et aux niveaux plus fins. Les derniers rendus restent affichés.
  Avec un seuil 3, LOD 4 et au-dessus rendent la mer normalement ; LOD 3 et en dessous
  la gèlent. LOD 10 et 11 rendent toujours tout. **Rendre la mer**, décoché par défaut,
  désactive le filtre lorsqu'il est coché. L'altitude de l'intérieur rendu fait foi ;
  l'aperçu physique apporte une classification conservatrice aux zones inconnues.
- La file privilégie les côtes, puis les terres, puis la mer, du centre vers les bords
  dans chaque catégorie. **Prefetch** anticipe uniquement les déplacements.
- **Préparer le monde NN** lance la préparation coarse mondiale en fond ; **Pause préparation** suspend les prochains blocs. Les demandes visibles sont prioritaires.

Les menus à icônes **Génération**, **Climat**, **Bruit du relief · SNR** et
**Rendering** partagent un brouillon unique. **Apply** conserve position et zoom ;
modifier les réglages ne déclenche pas de génération. Changer le relief initial
conserve les paramètres communs. Les layouts continentaux sont des presets :
plaques pour Tectonique, paramètres de bruit pour NN, atlas pour Custom. Dans
Custom, le mode atlas ou continents procéduraux reste sélectionnable.

Le menu SNR ne règle que le relief. Le SNR global contraint les entrées du NN ;
les multiplicateurs adaptatifs utilisent altitude, température (ou autre champ
climatique) et latitude absolue. Un multiplicateur de 1 désactive une règle.
Les seuils et le nombre de niveaux des rampes sont éditables. Le réglage par LOD
(-3 à 11) agit expérimentalement sur le résidu local du relief reconstruit après
le NN ; 0 hérite du SNR global. Il ne crée pas un réseau indépendant pour chaque
LOD. Les réglages sont persistés dans les liens et identités de cache, avec
lecture compatible des anciennes configurations.

Aux LOD −1, −2 et −3, la bande côtière suit une interpolation bilinéaire du
DEM neuronal natif à 30 m. Les corrections de moyenne par blocs y sont
désactivées : elles créaient des pixels de terre ou d'eau détachés des côtes.
Le détail du décodeur reste limité pour conserver le signe de cette surface.
La conservation des moyennes reste exacte hors de cette bande ; les îlots
déjà présents dans le DEM natif peuvent rester visibles.

Le layer visible se choisit dans la barre de menus. **Biomes** utilise la palette
du climat Orogen commun et suit le relief à chaque LOD : côtes, altitude et neige.
Au LOD 0 et aux niveaux plus fins, les pentes supérieures à 40° deviennent
rocheuses, même sous la neige ; ce seuil est réglable dans **Rendu → Biomes**.
L'éclairage du relief s'applique directement aux couleurs, y compris à la neige,
pour conserver les ombres des versants dès la vue mondiale initiale.
[Documentation des biomes en anglais et en français](docs/BIOME_LAYER.md).
Préchargement des tiles voisines et
profondeur du raffinement sont réunis dans Rendering ; ils restent indépendants.
Les tiles intermédiaires disponibles remplacent progressivement leur parent,
même si la caméra vise déjà un LOD plus fin. Les cadres suivent les tiles rendues.

L'exploration calcule les régions visitées en priorité. Lorsqu'elle est activée,
la préparation de toute la planète avance sur le GPU disponible. Le benchmark
antérieur occupait environ 707 Mo par monde complètement préparé ; le quota
coarse est par monde, sans éviction globale entre seeds. Le cache de heightmaps
est également persistant, sans quota global.

Les cinq canaux gardent leurs unités physiques. Le climat commun utilise
le pipeline saisonnier Orogen, sa circulation atmosphérique et ses courants océaniques. Le relief et le climat
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
