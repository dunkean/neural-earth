# Audit Astra — Terrain Diffusion, moteur procédural et navigation mondiale

**Date : 7 octobre 2026. Statut : analyse et propositions, aucune modification du runtime.**

Périmètre : monde plat de **40 000 × 20 000 km**, génération neuronale native à **30 m**, navigateur **WebGPU prioritaire**, agent local open source Linux/Windows optionnel, raffinement procédural éventuel sous 30 m. Référence qualitative : le terrain neuronal original, notamment ses reliefs et ses côtes. Le profil expérimental `earth` n'est pas considéré comme une nouvelle référence validée.

Cet audit examine conjointement trois problèmes : fabriquer un meilleur **conditionnement spatial à cinq champs** avant le premier réseau ; rendre l'exploration réellement utilisable du monde entier au détail ; prolonger les hauteurs neuronales dans le moteur de `city_generator` sans effacer leurs qualités. Il distingue **constat du code**, **mesure déjà disponible**, **inférence de l'audit** et **proposition à expérimenter**. Aucun benchmark GPU, entraînement, changement de paramètres, arrêt ou redémarrage de serveur n'a été effectué pour cet audit.

## 1. Décision recommandée et problèmes déterminants

**La direction est viable, mais les deux moteurs ne se branchent pas directement l'un sur l'autre et le temps réel mondial n'est pas démontré.** Le moteur Rust/WebGPU apporte de vrais procédés de relief, de drainage et de reconstruction. Il faut en extraire des composants, pas étirer son banc régional aux dimensions de la Terre ni appliquer sa génération complète sur le DEM neuronal.

Les décisions recommandées sont les suivantes, dans cet ordre :

1. **Figer une référence `natural` mesurable et visuellement approuvée.** Séparer toute modification du monde de toute optimisation d'exécution. La vitesse d'un aperçu procédural ne valide ni sa géographie ni l'accélération des réseaux.
2. **Construire le monde macro comme une entrée structurée, puis conserver le coarse appris.** Réutiliser les idées de crêtes, bassins, formes finies et côtes de `city_generator`, mais à travers un nouveau compositeur géographique global. Les cinq champs doivent rester compatibles avec les unités, transformations et distributions du checkpoint.
3. **Prouver le port WebGPU avec les modèles actuels avant de choisir une compression.** Un export ONNX existe ; une inférence complète Terrain Diffusion dans le navigateur n'existe pas dans l'application auditée. Porter le scheduler, le bruit, les fenêtres, leurs fusions et le décodage est autant nécessaire que porter les U-Net.
4. **Ordonnancer les fenêtres neuronales et leurs dépendances, pas uniquement les tuiles HTTP.** C'est la prochaine amélioration structurelle. Préparer un coarse mondial appris en tâche amortie, servir ses niveaux de détail, puis raffiner les régions effectivement visitées.
5. **Ajouter une frontière DEM absolu au moteur Rust.** L'entrée actuelle `new_mixed_with_source` refuse les altitudes inférieures à 0,3 m ; elle ne peut pas recevoir telle quelle une côte neuronale signée. Le futur adaptateur doit conserver mètres, niveau marin, coordonnées globales, halo et provenance.
6. **Traiter séparément reconstruction littorale, détail sous 30 m et géomorphologie glaciaire.** L'antialiasing peut supprimer des marches visuelles. Il ne fabrique pas un fjord. Un fjord absent du relief macro nécessite une forme régionale contrôlée ou un modèle/processus glaciaire, pas quelques octaves supplémentaires.

La distillation du coarse mérite une expérience si les téléportations et la préparation mondiale restent dominées par ses 20 évaluations. Le base et le decoder sont déjà des modèles de consistance ; leur distillation supplémentaire vise surtout **une architecture plus petite**, pas la suppression miraculeuse de dizaines d'étapes inexistantes. **MoE n'est pas une optimisation prioritaire** de ces U-Net denses : il demande un nouvel entraînement, augmente potentiellement le téléchargement et complique les transitions spatiales.

## 2. État réellement audité et limites des preuves

### 2.1 Sources et versions

- `infinite_map` n'est pas un dépôt Git à sa racine dans cet environnement. Le sous-dépôt `terrain-diffusion` annonce `e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230` ; des fichiers locaux peuvent différer de ce commit.
- `city_generator` annonce `ba9f305e5e1c49311b29cee7778cf6a257ef77f5`, avec des modifications et fichiers hydrologiques non suivis déjà présents. L'audit porte sur **les fichiers de travail lus**, pas uniquement sur ce commit.
- Modèle chargé : `xandergos/terrain-diffusion-30m`, révision épinglée `9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Le chargement résout ce snapshot avant de charger les sous-modèles. [terrain_app.py](../terrain_app.py), lignes 32–49.
- Configuration installée : `native_resolution=30`, compression latente 8, `residual_std=0.7`, `coarse_pooling=1`, `cond_snr=[0.5]*5`, fréquences originales `[1]*5`. Les valeurs par défaut du constructeur upstream, notamment 90 m et un autre écart-type résiduel, **ne doivent pas remplacer celles du checkpoint**.
- Le papier demandé est maintenant **arXiv v4, 3 mai 2026**, intitulé *InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation*. Sa description de dataset utilise MERIT 90 m et exclut les tuiles au-delà de ±60°. La fiche du checkpoint installé annonce Copernicus GLO-30. Le manifeste exhaustif d'entraînement du 30 m n'est pas installé : l'exclusion polaire publiée ne prouve pas à elle seule chaque détail de son corpus. [Papier v4](https://arxiv.org/html/2512.08309v4), [fiche du modèle 30 m](https://huggingface.co/xandergos/terrain-diffusion-30m).

Les instructions applicables de `city_generator/AGENTS.md` et `rust/AGENTS.md`, `rust/README.md` et `rust/docs/INTEGRATION.md` ont été consultées. Le présent travail est un audit, sans intervention sur leur moteur.

Les empreintes SHA-256 des fichiers locaux cités et des configurations installées sont consignées dans [ASTRA_AUDIT_SOURCES.json](ASTRA_AUDIT_SOURCES.json). Les numéros de ligne désignent l'état lu le jour de l'audit. Les documentations web ont été consultées le 7 octobre 2026 ; celles sur `main` ou `latest` sont des références mobiles, et une implémentation devra épingler sa version réelle.

Les documents [TERRAIN_REALTIME_DESIGN.md](TERRAIN_REALTIME_DESIGN.md), [REALTIME_IMPLEMENTATION.md](REALTIME_IMPLEMENTATION.md) et [WORLD_GENERATION.md](WORLD_GENERATION.md) sont utiles comme historique. **Leurs recommandations ne sont pas des preuves d'implémentation.** Le grand design précède notamment certains rejets numériques du batching ; ses listes de batches candidats ne constituent plus des réglages acceptables par défaut.

### 2.2 Matrice « livré / partiel / proposé »

| Mécanisme | État observé | Conséquence |
|---|---|---|
| Inférence des trois réseaux sur CUDA | Livrée, Python/PyTorch, un worker de calcul | Le site dépend encore de l'agent local pour générer le NN |
| Poids MP normalisés en cache, embeddings constants, fenêtres GPU | Livrés | Gains déjà pris ; ne pas les recompter comme futurs gains |
| CUDA Graphs | Coarse et base de batch ≤4 ; pas decoder de production | Réduit les lancements, pas le nombre d'opérations du modèle |
| Déduplication, priorités, epochs, annulation en file | Livrés à l'échelle des requêtes | Pas encore un DAG partagé de fenêtres neuronales |
| WebGPU navigateur | Rendu, filtres, hillshade, textures résidentes | Ne prouve aucune inférence NN WebGPU |
| LOD coarse/latent/decoder | Livré, sources différentes selon le niveau | Ce n'est pas une pyramide exacte du DEM final partout |
| Aperçu global `earth` | Conditionnement procédural, sans NN à LOD ≥7 | Vitesse et qualité incomparables à un aperçu coarse appris |
| Export ONNX upstream | Export des forwards avec adaptations et comparaison simple | Point de départ ; pipeline, placement GPU et fidélité complète non validés |
| Agent natif hors Torch | Proposé | Ni ORT natif ni TensorRT complet mesurés ici |
| Plusieurs GPU simultanés | Non livré | `auto` choisit un seul GPU par heuristique matérielle |
| Relief/érosion WebGPU de `city_generator` | Livré sur des domaines régionaux finis | Une base de procédés, pas un générateur planétaire prêt |
| Hydrologie du banc | Drainage GPU FP32 + finalisation Rust, ou CPU | Réseau régional, sans contrat de bassin mondial entre tuiles |
| Import DEM neuronal signé | Absent comme contrat général | Adaptateur spécifique requis |
| Nouvelle géométrie physique au zoom sous la grille préparée | Absente dans le banc régional | Le zoom échantillonne une surface conservée |
| Glaciation, transport océanique, tectonique physique | Non observés | Exigent un nouveau procédé ou des données/modèles adaptés |

Preuves principales : [terrain_jobs.py](../terrain_jobs.py), lignes 1–6 et 43–188 ; [terrain_inference.py](../terrain_inference.py), lignes 30–69 et 358–398 ; [terrain_server.py](../terrain_server.py), lignes 188–235 ; [terrain_device.py](../terrain_device.py), lignes 19–51 ; [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 101–217 et 513–528.

## 3. Ce que le réseau reçoit réellement

### 3.1 Trois notions à ne plus confondre

**La seed scalaire** identifie les flux pseudo-aléatoires. Elle n'est pas une heightmap. **Le conditionnement spatial** est un raster à cinq champs ; il organise les attentes géographiques du premier réseau. **Le bruit de diffusion** est l'état aléatoire que le réseau débruite. Une meilleure « seed initiale » au sens du besoin produit signifie ici une meilleure ébauche spatiale, pas le remplacement de l'entier `seed` ni du bruit gaussien par un paysage Rust.

Le coarse reçoit **11 canaux d'image** : six canaux de l'état débruité et cinq canaux conditionnants. Ses cinq entrées conditionnelles scalaires décrivent les niveaux de bruit de ces cinq champs, pas cinq nouveaux rasters. Sa sortie à six champs comprend une statistique d'altitude, une statistique de bas relief/p5 et quatre variables climatiques. Le pipeline convertit la seconde sortie par différence avec la première avant de la donner au niveau latent. [world_pipeline.py](../terrain-diffusion/terrain_diffusion/inference/world_pipeline.py), lignes 1007–1088.

| Canal de conditionnement | Source obligatoire | Contrat physique avant encodage |
|---|---|---|
| 0 | ETOPO | Altitude et bathymétrie signées, en mètres ; puis `sign(h)·sqrt(abs(h))` |
| 1 | WorldClim BIO1 | Température moyenne annuelle, °C |
| 2 | WorldClim BIO4 | Saisonnalité thermique : écart-type mensuel ×100 |
| 3 | WorldClim BIO12 | Précipitations annuelles, mm/an |
| 4 | WorldClim BIO15 | Saisonnalité des précipitations : coefficient de variation, % |

BIO4 n'est ni une température brute ni une variance ; BIO15 n'est pas une quantité de pluie. Le suffixe `10m` des fichiers climat correspond à des **minutes d'arc**, pas à une grille de terrain de dix mètres. Les cinq sorties de climat finales ne représentent pas cinq BIO indépendants : on y trouve les quatre BIO et le gradient thermique calculé. [WorldClim, définitions officielles](https://www.worldclim.org/data/bioclim.html) ; [synthetic_map.py](../terrain-diffusion/terrain_diffusion/inference/synthetic_map.py), lignes 45–86 et 232–270 ; [world_pipeline.py](../terrain-diffusion/terrain_diffusion/inference/world_pipeline.py), lignes 1413–1463.

### 3.2 Le Perlin original est déjà calibré par des données réelles

Le chemin original ne donne pas cinq Perlin arbitraires au réseau. Il transforme leurs quantiles vers les distributions ETOPO/WorldClim ; il corrige la température par un gradient dépendant de la pluie, reconstruit BIO4 depuis un résidu de régression, ajuste BIO15 et applique la racine carrée signée à l'altitude. Le factory original utilise une fréquence de base `0.05`, plusieurs octaves et un réglage `drop_water_pct=0.5` qui change la population utilisée pour les quantiles d'altitude. [synthetic_map.py](../terrain-diffusion/terrain_diffusion/inference/synthetic_map.py), lignes 45–130 et 182–270.

**Inférence :** remplacer Perlin par Simplex, ou augmenter le nombre d'octaves, peut changer l'apparence mais ne résout pas à lui seul l'organisation des continents, chaînes et bassins. Le gain recherché doit venir de **relations spatiales et géographiques explicites**, tout en préservant les distributions et unités que le réseau sait interpréter.

Deux points d'intégration demandent une attention particulière :

- Le chemin `set_custom_conditioning_import` applique seulement la racine signée à l'altitude et **ne passe pas par `finalize`**. Lui donner les résidus BIO4 ou la température au niveau marin du factory original serait incorrect. Avec cinq imports complets, il peut éviter Perlin ; avec des imports partiels, le mélange requiert une normalisation sémantique explicite. [world_pipeline.py](../terrain-diffusion/terrain_diffusion/inference/world_pipeline.py), lignes 959–1003.
- Le cache JSON de statistiques synthétiques ne vérifie pas dans son chargement une identité complète des paramètres ayant servi à sa création. Un nouveau compositeur doit posséder son propre manifeste versionné de statistiques, fréquences et transformations, pas réutiliser implicitement un cache trouvé sur disque. [synthetic_map.py](../terrain-diffusion/terrain_diffusion/inference/synthetic_map.py), lignes 134–190.

### 3.3 Échelles réellement apprises et échelles seulement affichées

| Étape du checkpoint 30 m | Échantillonnage | Fenêtre/stride observés | Rôle |
|---|---:|---|---|
| Conditionnement | 7 680 m par cellule coarse | Fournit les cinq champs sur les fenêtres demandées | Ébauche, non terrain final |
| Coarse appris | 7 680 m | 64², stride 48 ; 20 pas du solveur | Géographie/climat appris à basse résolution |
| Base latent | 240 m | 64², stride 32 ; deux passes avec fusion | Quatre canaux latents + une basse fréquence, pas une heightmap détaillée à cinq canaux |
| Decoder | 30 m | 512², stride 384 ; un forward de consistance | Résidu haute résolution conditionné par les quatre latents |
| Reconstruction | 30 m | Halo et débruitage Laplacien | Combine résidu et basse fréquence, puis inverse la racine signée |

Ces échelles viennent de `30×256` et `30×8`, pas du 90 m cité dans la publication. Le base parcourt une fenêtre de 15,36 km et avance de 7,68 km ; le decoder couvre aussi 15,36 km mais avance de 11,52 km. Le coarse couvre 491,52 km et avance de 368,64 km. Les chevauchements sont une part du coût et de la cohérence. [world_pipeline.py](../terrain-diffusion/terrain_diffusion/inference/world_pipeline.py), lignes 1007–1088, 1150–1303 et 1307–1410.

Le LOD 3 actuel affiche essentiellement la **cinquième composante latente décodée en basse fréquence**. Il ne passe pas les quatre autres composantes au decoder. À LOD 4–6, on interpole le coarse appris sur une grille d'affichage plus fine que 7,68 km. Cela remplit des pixels, sans inventer l'information absente. LOD 0–2 effectue réellement le decoder puis une réduction. [terrain_server.py](../terrain_server.py), lignes 148–206.

Une hiérarchie de générations conditionnées peut rester cohérente sans être une pyramide exacte. Pour affirmer « ce parent est la moyenne des enfants », il faut produire les enfants correspondants ou entraîner une contrainte dédiée. Un simple fondu masque une transition visuelle ; il ne prouve pas la conservation des côtes ou des bassins.

## 4. Ce qui a dégradé le profil `earth`

**Constat :** plusieurs interventions géographiques ont été introduites en même temps : fréquence continentale `0.0012` contre `0.05`, warp du domaine, répartition terre/mer recalibrée mondialement, mélange de plaines pondéré jusqu'à `0.86`, dépressions artificielles, climat dépendant de la latitude, bruit conditionnant réduit de `0.5` à `0.1`, projection positive de trois variables climatiques après le coarse. À LOD ≥7, le rendu évite les réseaux. [terrain_world.py](../terrain_world.py), lignes 20–31, 113–129 et 166–216 ; [terrain_inference.py](../terrain_inference.py), lignes 191–210 ; [terrain_server.py](../terrain_server.py), lignes 213–220.

La formule du conditionnement est `cos(atan(s))·C + sin(atan(s))·N`. Avec `s=0.5`, les coefficients sont environ 0,894 et 0,447 ; avec `s=0.1`, environ 0,995 et 0,100. **Le paramètre appelé SNR se comporte ici comme une amplitude relative du bruit** : le réduire renforce l'emprise de l'ébauche. Cela ne réduit pas les 20 pas et n'est pas une optimisation de performances.

**Inférence causale, à confirmer par ablation :** une entrée beaucoup plus lisse, plus dominée par des plaines et moins bruitée peut contraindre davantage le coarse à reproduire des formes molles. L'affichage direct de cette entrée au dézoom montre en plus les formes procédurales avant leur enrichissement neuronal. Ces mécanismes sont compatibles avec les « patates » et la perte de montagnes signalées. L'audit ne peut pas attribuer tout l'effet au seul paramètre 0,1 sans comparaison isolée.

Les rapports `earth-world-verification.json` et `earth-qa/viewer.json` vérifient des nombres finis, des proportions, des étapes exécutées et l'absence d'erreurs du navigateur. Ils n'établissent pas que les montagnes, côtes et transitions sont meilleures. La fraction terrestre de 27,77 % et l'aperçu 2048×1024 calculé en 1,81 s décrivent **le conditionnement**, pas une planète neuronale finalisée. Les vues du rapport alternent explicitement `conditioning-preview` et `decoder`.

**Proposition :** comparer séparément A0 original, A1 organisation continentale seule, A2 climat seul, A3 procédés régionaux seuls ; conserver `cond_snr=0.5` au départ. Tester ensuite le bruit par canal, sans toucher simultanément la géographie. Les mêmes sites côtiers, montagnards et plats doivent être montrés avec palette fixe, profils de hauteur et masques terre/mer. Tout changement accepté reçoit une nouvelle identité de monde ; l'accélération conserve celle de la référence tant que la fidélité autorisée est respectée.

## 5. Techniques exactes utiles dans `city_generator`

### 5.1 Le chemin du terrain bench

Le fichier [terrainbench.ts](../../city_generator/web/src/ui/terrainbench.ts), lignes 1–10, importe les adaptateurs Rust de terrain, rendu et hydrologie. Les réglages par défaut choisissent l'échantillonnage `gpu-f32` et la génération `gpu-erosion` ; les étapes relief, côtes et hydrologie sont séparées. La caméra demande ensuite des régions d'un terrain préparé, avec cache de détails. [terrainbench.ts](../../city_generator/web/src/ui/terrainbench.ts), lignes 36–38, 299–400 et 410–436.

Ce chemin ne correspond pas aux anciens générateurs TypeScript de `web/src/gen/terrain` : ils restent utiles comme références historiques, mais **le banc demandé utilise le prototype Rust et ses shaders WGSL**. Le renderer actuel reste transitoire ; le domaine permanent du moteur n'est pas fixé par cet audit.

Le GPU est actuellement piloté par les bridges TypeScript/WebGPU. Le crate `core` n'a pas de dépendances déclarées ; le crate WASM dépend du core et de `wasm-bindgen`. Il n'y a donc pas ici de backend Rust `wgpu` natif déjà livré : réutiliser les shaders dans un agent natif demanderait un nouvel hôte, une gestion des buffers et une validation. [Cargo core](../../city_generator/rust/crates/core/Cargo.toml), [Cargo WASM](../../city_generator/rust/crates/wasm/Cargo.toml).

### 5.2 Inventaire technique et usage raisonnable

| Composant réellement présent | Fonctionnement observé | Pour le conditionnement avant NN | Pour le détail après NN |
|---|---|---|---|
| Simplex/fBm filtré, ridged | Gradients seedés, octaves et atténuation selon empreinte sans renormaliser les bandes restantes | Fond de variabilité, warp et rugosité secondaires | Résidu limité en fréquence ; ne pas recréer les grandes montagnes |
| Soulèvement régional | Champs continus, proportions de montagnes, charpente ridged avant érosion | Inspiration forte pour des ceintures de relief structurées | À désactiver si le DEM NN possède déjà la topographie macro |
| Érosion de paysage | Flood, écoulement à deux récepteurs de type D∞, accumulation, incision dépendant pente/surface, relaxation thermique, diffusion | Préparer des relations crêtes/bassins à une échelle résolue | Passe contrainte de versant ou lit ; pas réexécution intégrale du générateur |
| Canyon | Accumulation sélectionnant axes majeurs, élargissement des entailles et érosion des parois | Motif régional possible si suffisamment grand | Dangereux si appliqué automatiquement : modifie fortement la macro |
| Plateau, volcan, caldeira | Formes finies, masques, fondation filtrée, raccords lissés et supports bornés | Placer de rares structures cohérentes dans une géographie globale | Événement explicite, pas détail neutre sous 30 m |
| Côte seedée | Directions terrestres/marines, supports arrondis, axes ramifiés, îlots variés, bruit côtier, transition multiplicative et érosion finale | Vocabulaire de silhouettes à adapter à une composition mondiale | Ne pas redécouper les côtes NN avec ses masques |
| Surface signée + mips | Champ final conservé, niveaux filtrés, échantillonnage bicubique/continu | Utile pour produire une pyramide de champs | Bon précédent pour DEM immuable + résidus |
| Hydrologie | Dépressions, exutoires, accumulation, lacs et réseau vectoriel, méandres contraints | Ébauche de bassins à très grande échelle, sous nouveau contrat | Très utile pour lits continus et petits affluents, avec conditions aux frontières |

Sources : [noise.rs](../../city_generator/rust/crates/core/src/noise.rs), lignes 26–128 ; [erosion.rs](../../city_generator/rust/crates/core/src/erosion.rs), lignes 109–234 et 396–569 ; [coast.rs](../../city_generator/rust/crates/core/src/coast.rs), lignes 30–215 et 297–425 ; [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 463–528 et 839–888 ; [hydrology.rs](../../city_generator/rust/crates/core/src/hydrology.rs), lignes 225–317, 645–855 et 1441–1570.

L'érosion observée est un **modèle géomorphologique simplifié** : soulèvement, incision guidée par pente et aire drainée, relaxation et diffusion. Elle ne résout pas un champ d'eau de pluie avec conservation générale de masse et de sédiments, ni les équations d'un glacier. Le mot « physique » utilisé dans le banc signifie notamment une grille et des longueurs en mètres ; il ne suffit pas à qualifier tous ces phénomènes de simulations complètes.

### 5.3 Ce que WebGPU accélère effectivement

`terrainErosion.ts` orchestre les kernels WGSL de génération, flood, flux, cuvettes, thermique, diffusion, reconstruction, percentiles et normalisation. Les chemins littoraux peuvent partir d'une surface signée et rendre une surface signée ; les chemins de relief ordinaires normalisent l'altitude. La préparation GPU est suivie d'un readback pour le moteur Rust et ses mips. [terrainErosion.ts](../../city_generator/rust/bridge/terrainErosion.ts), lignes 8, 119–207.

L'hydrologie possède une autre chaîne : masque marin connecté au bord, relaxation GPU du remplissage avec contrôles de convergence, récepteurs D8, accumulation, readback groupé, puis finalisation vectorielle Rust. **Ne pas confondre ce drainage D8 avec le flux D∞ de l'érosion de relief.** Les vérifications GPU lisent périodiquement un petit statut ; cela peut limiter les performances malgré le volume réduit des données. [hydrologyGpu.ts](../../city_generator/rust/bridge/hydrologyGpu.ts), lignes 53–76 et 124–202.

Les mesures historiques du banc donnent, pour certains cas régionaux chauds, 198–209 ms sur GPU pour des collines et 404–448 ms pour des montagnes. Le rapport observe aussi 8,46 m RMS et 35,12 m maximum d'écart CPU/GPU sur un cas. Ce sont des variantes de simulation, **pas un remplacement numériquement transparent**. Ces chiffres ne s'extrapolent ni à une planète ni au temps d'inférence neuronal. [PERFORMANCE_AUDIT.md](../../city_generator/rust/docs/PERFORMANCE_AUDIT.md), lignes 9, 61–79.

### 5.4 Limites structurelles pour une utilisation mondiale

Le constructeur borne la carte à 500–100 000 m et le motif à 250–50 000 m. Le champ préparé est en 1024², avec des grilles d'érosion plus petites ou adaptées ; ses bords et supports font partie de la construction. Multiplier la largeur de carte par mille ne produit pas une simulation équivalente : cela change le pas physique, les bassins représentables et le rôle des frontières. [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 140–172 ; [generation_noise.rs](../../city_generator/rust/crates/core/src/generation_noise.rs), ligne 84.

À 100 km de côté, 1024 cellules donnent environ 97,7 m par cellule. À 5 km, environ 4,88 m. Le même moteur peut donc produire une grille sous 30 m sur une région suffisamment petite, mais **son zoom ne relance pas une nouvelle simulation plus fine** : il révèle/interpole le champ conservé. Certaines composantes analytiques filtrées existent selon le relief ; elles ne constituent pas une récupération générale d'information perdue. [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 656–770 et 839–888.

## 6. Un meilleur conditionnement initial : architecture proposée

### 6.1 Ne pas fabriquer un monde de petits bancs juxtaposés

La proposition est un **compositeur macro global déterministe**, qui produit les cinq champs au pas attendu par le coarse. Il possède un système de coordonnées global et des objets persistants indépendants de la caméra : masses continentales, ceintures de haut relief, bassins, plateaux, grandes dépressions et gradients climatiques. Il ne produit pas des carrés autonomes avec chacun son bord marin ou son percentile d'altitude.

Le vocabulaire de `city_generator` sert à trois niveaux : des graphes/axes ramifiés pour les formes, des champs de soulèvement pour les chaînes, et des procédés d'érosion/drainage régionaux pour organiser les bassins. **Le banc ne fournit pas aujourd'hui les quatre champs climatiques nécessaires** ; ceux-ci doivent être construits explicitement à partir des sources WorldClim, corrélés à cette géographie, puis validés conjointement.

### 6.2 Pipeline macro concret

1. **Manifest du monde.** Seed 64 bits, versions des sources et du compositeur, convention des coordonnées, limites du rectangle et niveaux marins. Conserver la seed en chaîne/entier 64 bits : un nombre JavaScript ne représente pas exactement tous les entiers 64 bits.
2. **Organisation continentale.** Générer un petit ensemble global d'objets : supports de continents, arcs/axes de chaînes, bassins et marges. Une tessellation et un graphe de frontières peuvent fournir une organisation inspirée de plaques ; ce serait un nouveau composant, pas de la tectonique déjà livrée. Les côtes de Burgmap fournissent des idées de branches et de baies, sans recopier ses huit boutons directionnels comme modèle du globe.
3. **Relief macro multibande.** Séparer bathymétrie, altitude continentale et relief régional. Distribuer des hautes terres, chaînes et bassins avec continuité et diversité ; conserver une bande régionale que le NN sait enrichir. Éviter qu'une unique fréquence continentale décide à la fois de la silhouette, de l'altitude intérieure et des montagnes.
4. **Préparation géomorphologique facultative.** Sur des régions assez larges et avec conditions aux limites connues, calculer bassins/érosion à une résolution permettant de les voir, puis filtrer vers 7,68 km. Une vallée de 500 m disparaîtra dans ce conditionnement ; la simuler ici n'apporte pas automatiquement un contrôle du NN final.
5. **Climat couplé.** BIO1 : latitude conventionnelle, température de référence et altitude. BIO4 : saisonnalité thermique, continentalité et amortissement maritime. BIO12 : distributions régionales, humidité et ombre pluviométrique simplifiée si un champ de vents est introduit. BIO15 : saisonnalité des pluies cohérente avec BIO12, sans déduire arbitrairement une variable de l'autre. Employer les statistiques des quatre sources, leurs covariances et des bornes par régime climatique.
6. **Adaptateur des cinq champs.** Filtrage, unités, racine signée, normalisation checkpoint ; validation des cinq tableaux avant inférence. Le mode import complet est une frontière possible. Appliquer les transformations climatiques une seule fois.
7. **Coarse appris conservé.** Exécuter le réseau avec le bruit original comme référence. Mesurer combien il déplace les reliefs et rivages. Si un contrôle est insuffisant, ne pas masquer le symptôme en appliquant une silhouette rigide après génération : commencer par l'expérience de conditionnement, puis envisager une extension entraînée si nécessaire.

### 6.3 Ce que signifie « mieux que Perlin »

Le critère n'est pas le nom de l'algorithme. Un compositeur est meilleur s'il produit davantage de structures voulues **et** conserve la variété neuronale et la cohérence au zoom. Il doit être jugé sur :

- Taille et distribution des masses terrestres, détroits, mers intérieures et archipels ; pas seulement pourcentage de terre.
- Hypsométrie, fraction de plaines, hauts plateaux, chaînes et relief local ; ne pas obtenir toutes les montagnes en gonflant l'altitude moyenne.
- Spectre spatial et orientation des crêtes/valleys, hiérarchie de bassins et raccord aux marges continentales.
- Distributions conjointes altitude–BIO1–BIO4–BIO12–BIO15 ; un histogramme correct par canal ne garantit pas une combinaison plausible.
- Conservation des qualités du 30 m à l'arrivée du decoder, dans les mêmes régions et avec plusieurs seeds.

**Expérience discriminante :** garder bruit de diffusion, poids, solveur, positions et rendu identiques ; changer seulement le compositeur. Mesurer entrée → coarse → basse fréquence latente → DEM final. Il faut voir où la montagne ou la côte disparaît. Un beau aperçu ne suffit pas ; un beau crop choisi après coup non plus.

### 6.4 Contraintes que les cinq champs ne peuvent pas exprimer directement

Une altitude coarse permet de suggérer un massif ou un bassin, mais aucun canal ne fournit explicitement orientation tectonique, lithologie, distance à une faille, réseau fluvial imposé, histoire glaciaire ou âge d'érosion. Deux mondes aux cinq champs similaires peuvent légitimement produire des structures locales différentes.

Ajouter un sixième champ « glacier » à l'entrée d'un modèle entraîné pour cinq champs n'est pas une option d'inférence. Les pistes réalistes sont : adapter/réentraîner le coarse ou le base avec un encodeur conditionnel supplémentaire ; apprendre un module de contrôle ; entraîner un raffineur contraint ; ou produire le phénomène par un procédé dédié après le NN, en acceptant et en versionnant la modification du monde.

## 7. Temps réel : budgets et mesures utilisables

### 7.1 Quatre latences distinctes

**Rendu interactif** : déplacer les données déjà disponibles à 60 Hz vise environ 16,7 ms par image. **Premier résultat utile** : délai avant qu'une téléportation montre un terrain pertinent. **Complétion** : délai avant que tout le viewport atteigne la résolution cible. **Débit continu** : surface nouvelle produite par seconde pendant un pan. Une interface fluide peut coexister avec plusieurs secondes de retard de génération.

Il faut encore distinguer téléchargement/initialisation froide du modèle, poids résidents avec monde froid, fenêtres NN chaudes, cache physique disque et textures déjà GPU. Le mot « froid » seul est insuffisant.

### 7.2 Mesures existantes, sans nouvelle exécution

| Observation sur RTX 3090, PyTorch 2.11.0+cu128 | Valeur | Portée réelle |
|---|---:|---|
| Crop 512², référence, poids résidents | 3,440 s | Un crop, pas un p95 ; hors rendu/disque |
| Même type de crop, poids MP et fenêtres GPU | 2,529 s | Écart hauteur maximal observé 0,00269 m |
| Crop déjà couvert par les caches GPU | 8,96 ms | Ne mesure pas le débit de terrain nouveau |
| Graphs, trois autres crops froids de 256² | 1,203–1,461 s | Gain 1,42–1,62 face au profil optimisé sans graphs |
| Premier crop 512² de la série graphs | 1,957 s contre 3,364 s | Rapport 1,72 biaisé par le chauffage des poids du premier passage |
| Pic PyTorch du harness graphs | 1 926 860 288 octets ≈1,79 Gio | Exclut mémoire réservée, navigateur, bureau et autres processus |
| Batch expérimental coarse/decoder 2 ou 4 | Jusqu'à ≈209 m d'écart | Variante rejetée ; causes à isoler par étape |
| Batch decoder seul 2 ou 4 | ≈3,19–3,20 m d'écart maximum | Rejeté par le contrôle numérique actuel |

Sources : [rapport realtime](E:/TerrainDiffusionRuntime/inference-realtime/report.json), [rapport decoder](E:/TerrainDiffusionRuntime/inference-decoder/report.json), [pipeline graphs](E:/TerrainDiffusionRuntime/inference-graph-pipeline.json), [vérification runtime](E:/TerrainDiffusionRuntime/inference-runtime-verification.json). Les gains de séries différentes ne se multiplient pas. L'exactitude entre deux chemins testés sur quelques crops ne prouve pas une exactitude universelle entre GPU, versions de bibliothèques et formes de batch.

Les batches 2/4 ont changé l'algorithme numérique effectif, même si les équations théoriques sont identiques. Le solveur répète les erreurs d'arrondi du coarse et le relief final peut les amplifier. Réouvrir cette piste nécessite une expérience contrôlée avec accumulation/précision et algorithmes de convolution fixés, pas un simple passage à une carte plus puissante.

Le benchmark HTTP actuel [benchmark_navigation.py](../benchmark_navigation.py), lignes 15–38, n'envoie pas `world_profile=natural`. Comme le serveur choisit aujourd'hui `earth` par défaut, **le relancer ne reproduirait pas automatiquement les anciennes mesures naturelles**. Le profil, la version de cache et les sources de LOD doivent être enregistrés dans tout prochain rapport.

### 7.3 Coût d'une planète : calculs de dimensionnement, pas benchmarks

Pour le rectangle demandé, sans assimiler sa surface à celle d'une sphère :

Sa surface plane vaut **800 millions de km²**. Les dimensions choisies évoquent les circonférences terrestres, mais ce rectangle n'a pas l'aire d'une sphère terrestre. Les budgets ci-dessous correspondent exactement à ce besoin plat ; la latitude du climat ne doit pas réduire silencieusement les distances ni changer le pas de 30 m selon la position.

| Grille entière | Nombre approximatif de cellules | Données FP32 seules |
|---|---:|---:|
| 30 m | **888,9 milliards** | Altitude : **3,56 To** ; altitude + cinq sorties climat : **21,33 To** |
| 240 m | 13,89 milliards | Cinq canaux latents : 277,8 Go, sans poids de fusion |
| 7 680 m | 13,56 millions | Cinq champs : 271 Mo ; six champs coarse : 326 Mo |

Ces tailles ignorent halos, contributions chevauchantes, formats de stockage, mips, métadonnées et buffers temporaires. Arrondir les dimensions à des cellules entières ajoute une petite marge. La planète native ne doit pas être générée ou conservée intégralement dans le navigateur.

Le coarse mondial est, lui, assez petit pour un cache persistant et une préparation amortie. Avec stride 48, sa couverture implique de l'ordre de **6 000 fenêtres**, plus le contexte des bords ; avec 20 pas, environ **120 000 forwards coarse**. Multiplier illustrativement ces appels par le forward isolé d'environ 1,85 ms du rapport graphs donne déjà environ **222 s de seul calcul réseau**. Ce n'est **pas** une prédiction de durée mondiale : assemblage, transferts, sérialisation, chevauchements, placement, initialisation et comportement sur une longue série doivent être mesurés. Cela démontre seulement pourquoi « 13,56 millions de cellules » ne signifie pas « aperçu exact instantané ».

### 7.4 Demande de la caméra

À un échantillon par pixel écran, un viewport 1920×1080 au pas 30 m couvre 57,6×32,4 km. Si le pan déplace horizontalement `v` pixels/s, la demande minimale de nouvelle surface est approximativement `1080·v` échantillons/s, avant halo et overlap :

| Pan à l'écran | Nouveaux échantillons/s | Altitude FP32/s | Altitude + cinq champs FP32/s |
|---|---:|---:|---:|
| 100 px/s | 108 000 | 0,432 Mo/s | 2,59 Mo/s |
| 500 px/s | 540 000 | 2,16 Mo/s | 12,96 Mo/s |
| 1 largeur écran/s | 2 073 600 | 8,29 Mo/s | 49,77 Mo/s |

Le dernier cas équivaut à renouveler un écran/s ; **60 images/s n'impose pas de générer 60 écrans/s**. Les mouvements diagonaux, le zoom, la rotation éventuelle, le préchargement et la dépendance des fenêtres augmentent le coût. Inversement le climat peut être conservé à sa résolution utile, sans recopier cinq champs plein écran à chaque arrivée.

Une tuile affichée 256² possède 65 536 points. Un halo de 24 de chaque côté porte le buffer à 304², soit **41 % de points supplémentaires**. Le rapport des surfaces fenêtre/stride vaut environ 1,78 pour coarse et decoder, et 4 pour une passe latente ; il ne faut pas multiplier aveuglément tous ces facteurs car les dépendances sont partagées. Compter les **fenêtres uniques par scénario** est nécessaire.

### 7.5 Téléchargement, mémoire et transit

Les trois fichiers de poids installés totalisent environ **1,138 Go** : base 1 014 772 076 octets, coarse 11 200 936, decoder 111 709 108. Cela représente environ 569 Mo de paramètres en deux octets, avant copies, activations et workspaces. Le cache local de poids MP normalisés mesuré ajoute 568 816 128 octets ; une exportation qui plie cette normalisation peut éviter une duplication, sous contrôle d'équivalence.

À titre de borne de transfert idéale, 1,138 Go nécessitent environ 91 s à 100 Mbit/s, ou 9,1 s à 1 Gbit/s. Les fichiers en précision réduite changeraient ces tailles mais demandent une exportation réellement compatible. La compression réseau, les caches et la latence sont à mesurer. **Un navigateur ouvert pour la première fois et un navigateur aux poids déjà persistés sont deux produits d'expérience très différents.**

Un GPU de 24 Go ne fournit pas 24 Go libres à l'onglet. Il faut budgéter poids, activations de pointe, contextes NN, scratch procédural, textures visibles et mémoire du système. Éviter tout monobuffer géant : vérifier les limites exposées par l'adaptateur, segmenter les poids et les champs, et faire une admission mémoire avec marge. Une perte de device doit reprendre depuis les artefacts persistants, sans changer le monde.

Le navigateur peut garder NN → heightmap → raffinement → rendu sur **un même GPUDevice** si les backends s'intègrent réellement. Le chemin natif demande ordinairement GPU natif → RAM → transport local → RAM navigateur → GPU navigateur. Un service HTTP local ne fournit pas spontanément une texture CUDA au navigateur. L'avantage natif doit donc être évalué **jusqu'à l'image présentée**, et non au seul forward. La conservation des tenseurs sur le device est également recommandée dans la documentation ORT. [ORT device tensors](https://onnxruntime.ai/docs/performance/device-tensor.html).

## 8. Architecture de navigation et de cache proposée

### 8.1 Graphe canonique indépendant de la caméra

```text
WorldManifest + cinq champs globaux + bruit canonique
                         │
             fenêtres coarse apprises
                         │
            latents passe 1 → fusion canonique
                         │
            latents passe 2 → fusion canonique
                         │
               fenêtres decoder → Laplacien
                         │
                DEM 30 m + climat physique
                         │
         résidu local contraint + hydrologie régionale
                         │
                mips / textures / rendu
```

Une demande caméra décrit une zone et une erreur acceptable à l'écran. Elle s'abonne aux nœuds nécessaires. Chaque nœud porte stage, coordonnées entières, paramètres, version, état, dépendances, coût observé, mémoire et nombre de consommateurs. La fusion ne peut être déclarée terminée avant que **toutes** ses fenêtres contributrices nécessaires soient disponibles. Une moyenne partielle publiée comme résultat définitif ferait dépendre le terrain de l'ordre des requêtes.

Identité minimale : hash poids/configuration, seed, données/statistiques, compositeur, bruit, scheduler, tailles/strides, précision et backend validé, stage/passe, coordonnées. Le raffinement ajoute sa version et ses contraintes hydrographiques. Le style de palette n'entre pas dans l'identité de la heightmap. La seed seule n'est pas une identité suffisante d'un monde reproductible.

### 8.2 Priorités et quanta

1. Terrain déjà demandé et nécessaire pour couvrir le viewport.
2. Dépendances partagées débloquant le plus de surface visible.
3. Raffinement proche du centre d'intérêt.
4. Préchargement directionnel selon vitesse caméra et latence observée.
5. Coarse mondial de fond et travaux spéculatifs.

Une téléportation retire les abonnements obsolètes, conserve les nœuds utiles et ne laisse pas un énorme rectangle coarse monopoliser plusieurs secondes. Les tâches sont découpées en fenêtres ou petits groupes compatibles. Pour suspendre le solveur coarse entre pas, conserver son état multistep complet ; le premier port peut simplement terminer une fenêtre. Une annulation n'interrompt pas arbitrairement un kernel.

Le batching est admis seulement parmi nœuds prêts du même contrat numérique. La taille maximale est gouvernée par le **temps d'occupation non interruptible** et la mémoire, pas par le débit maximal isolé. Un petit batch urgent peut être préférable à un grand batch qui retarde la première tuile.

### 8.3 Cache à quatre niveaux

| Niveau | Données | Politique proposée |
|---|---|---|
| GPU | Poids, fenêtres actives, fusion, DEM et textures proches | Épingler les dépendances actives ; évincer selon coût de recalcul et probabilité de réutilisation |
| RAM | Métadonnées, buffers de transit, petits résultats | Borner le nombre de sorties en attente d'encodage/upload |
| Stockage local navigateur ou agent | Coarse mondial, résultats physiques visités, paquets de poids | Quota explicite ; cache versionné ; écritures atomiques |
| Distribution distante facultative | Mondes publiés, coarse et zones de départ préparées | Réduit le démarrage, sans rendre le service distant obligatoire pour les seeds locales |

Pour une seed arbitraire, l'aperçu procédural peut servir à s'orienter mais reste une **prévisualisation de l'entrée**. L'aperçu produit final doit progressivement être remplacé par le coarse appris et porter sa provenance. Pour un monde partagé prépublié, préparer son coarse avant exploration est plus rationnel que payer la même préparation sur chaque machine.

### 8.4 LOD et continuité

- Choisir le LOD à partir du pas projeté à l'écran et ajouter une hystérésis pour éviter le va-et-vient.
- Réutiliser le parent tant que les enfants requis ne sont pas prêts ; ne pas laisser des enfants anciens recouvrir une vue parent sélectionnée après dézoom.
- Construire les mips du DEM final **là où il existe**, avec filtrage physique commun et halo global. Ailleurs, afficher le coarse/latent comme approximation explicitement distincte.
- Pour les transitions, séparer le fondu d'affichage de la hauteur physique utilisée par hydrologie et placement d'objets. Les objets ne doivent pas se déplacer parce qu'une texture parent est arrivée avant un enfant.
- Fixer les bords des tuiles dans les coordonnées globales, et utiliser un échantillonnage identique des deux côtés ; un halo n'efface pas une divergence de données ou d'algorithme.

Le rectangle de 40 000 km n'est pas divisible exactement par 30 m. Conserver le pas natif et gérer une dernière bande partiellement visible est préférable à déformer le pas de tout le monde. La topologie demandée est plate : ne pas introduire un raccord longitude périodique par défaut. S'il est voulu plus tard, il doit concerner tous les champs et le bruit, pas seulement l'image de l'aperçu.

## 9. Exécuter hors Torch : choix réalistes

### 9.1 Premier candidat navigateur : ONNX Runtime Web + WebGPU

**Pourquoi commencer ici :** c'est une voie de test avec des opérateurs existants et un export upstream déjà présent, qui permet d'identifier rapidement les incompatibilités. Cela ne garantit pas qu'elle sera la meilleure voie de production.

L'export [onnx/export.py](../terrain-diffusion/terrain_diffusion/onnx/export.py) remplace certains resamplings, corrige un padding de convolution, aplatit les entrées conditionnelles, exporte en opset 17 et rend le batch dynamique. La vérification n'effectue qu'un forward aléatoire et imprime un écart maximal ; elle ne valide ni les fenêtres assemblées ni le solveur ni un seuil de réussite, et peut ignorer un modèle dont le chargement échoue. Un export réussi n'est donc pas un feu vert.

Travail nécessaire :

1. Inventorier les opérateurs du graphe **réellement exporté** pour les trois formes de production ; fixer les versions ORT/navigateur. Tester convolution, normalisation, attention `einsum`, softmax, interpolation, concaténations, opérations de Fourier et conversions de types.
2. Plier les poids MP et constantes en évaluation, sous comparaison couche par couche. Ne pas remplacer l'attention par SDPA sans reproduire la normalisation q/k/v et son échelle. [terrain_nn_constants.py](../terrain_nn_constants.py), lignes 54–74.
3. Établir un profil FP32 de diagnostic puis un profil FP16/WebGPU si supporté et fidèle. **BF16 CUDA n'est pas automatiquement le type utilisable ou le même calcul en WGSL**. Les sommes de fusion, normalisations sensibles et hauteurs finales restent de préférence FP32.
4. Conserver entrées/sorties sur le device via l'API de tenseurs GPU ; éviter `getData()` entre les pas. Partager le device avec les kernels de bruit/fusion/Laplacien et le renderer, en respectant la propriété et la durée de vie des buffers.
5. Porter l'orchestration InfiniteTensor, le bruit déterministe, le solveur et le climat. Pour le premier port fidèle, générer le bruit portable en WASM/CPU puis l'uploader peut être préférable à une réécriture WGSL précipitée des entiers 64 bits et des fonctions transcendantes.
6. Mesurer compilation, première exécution, mémoire et traces de placement. Tester graph capture seulement lorsque formes statiques et placement WebGPU intégral sont confirmés.

La documentation ORT expose `Tensor.fromGpuBuffer`, `preferredOutputLocation` et un graph capture conditionné par les formes statiques et l'exécution GPU des kernels. Ce sont des capacités à intégrer, pas des performances garanties pour ce terrain. [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html), [diagnostic de performances](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html), [table officielle des opérateurs](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md).

### 9.2 Kernels WGSL spécialisés : ciblés après profilage

Les opérations de rassemblement, fusion pondérée, bruit, interpolation, Laplacien, climat et sous-détail se prêtent à des kernels simples. Écrire toutes les convolutions du base et toutes leurs optimisations avant de mesurer ORT serait un investissement élevé et risqué. Un backend WebGPU spécialisé devient rationnel si les traces montrent quelques opérateurs/allocations dominants que l'on peut remplacer avec un gain bout en bout vérifiable.

La compilation d'un graphe supprime des frais d'interprétation, peut fusionner des opérations et améliorer les layouts. **Elle ne retire pas les 20 évaluations coarse ni les dépendances spatiales.** La part restante du calcul contraint le gain maximal : si une optimisation rend deux fois plus rapide une partie représentant 30 % du délai, le gain total théorique ne vaut que `1/(0.7+0.3/2)≈1.18`.

### 9.3 Agent local open source Linux/Windows

Proposition : cœur d'ordonnancement et protocole portable, intégration Rust ou C++ selon les bibliothèques choisies ; backend ORT natif comme première comparaison, TensorRT optionnel sur NVIDIA. Le moteur procédural existant favorise Rust pour le domaine et WASM, mais ne justifie pas de réimplémenter un runtime d'inférence mature.

Le service doit partager manifest, coordonnées, règles de cache, messages d'annulation et validation avec le navigateur. Il écoute localement, s'apparie au site et transmet des champs physiques. Les builds Linux et Windows doivent être vérifiés séparément. Les plans de compilation dérivés sont cachés par GPU/runtime/options ; la portabilité d'un fichier ONNX n'implique pas celle de tous les plans optimisés. [TensorRT, sujets avancés](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html).

Le code de l'agent peut être open source tout en utilisant des dépendances propriétaires optionnelles. Un backend WebGPU/WASM doit rester une voie indépendante ; ni CUDA ni l'agent ne doivent être requis pour le produit navigateur promis. La redistribution de chaque paquet de poids/runtime est à traiter lors du packaging sur ses licences réelles.

**Go/no-go natif :** le retenir comme accélérateur si son délai caméra → image, son débit durable ou son budget mémoire sont meilleurs sur les mêmes scénarios après prise en compte des copies. Le conserver comme outil de préparation hors ligne reste utile même si le navigateur est plus rapide pour une petite région déjà chaude.

### 9.4 3090, 5090 + 4090 et GPU visiteurs

L'heuristique actuelle multiplie nombre de SM et fréquence annoncée ; elle sélectionne un seul device. Elle n'a pas mesuré les deux cartes de l'autre machine. Le navigateur choisit son adaptateur séparément. [terrain_device.py](../terrain_device.py), lignes 19–51.

Proposition : calibration courte avec corpus fixe, chaque stage, chaque précision validée, mémoires libres et coûts de transit ; puis adaptation de budget et de préchargement selon la charge. **Plus de puissance doit donner moins d'attente, pas changer silencieusement le relief.** Le profil numérique doit être stable pour un monde partagé.

La reproductibilité bit à bit entre CUDA de générations différentes, ORT et WebGPU n'est pas acquise. Pour un monde partagé, choisir un profil canonique et persister ses chunks physiques, ou définir des profils compatibles validés avec tolérances et tests de frontières. Ne pas assembler sans contrôle deux voisins calculés par des backends divergents. Les seeds portables sont nécessaires mais ne rendent pas les convolutions, réductions et fonctions transcendantes automatiquement identiques.

Avec deux GPU natifs, commencer par des branches spatiales complètes : une région froide sur un worker, une autre sur l'autre. Garder les dépendances chaudes sur leur propriétaire. Comparer duplication de petits contextes coarse et transfert explicite ; ne pas découper par défaut coarse sur une carte/base sur l'autre/decoder de retour. Les communications peuvent annuler le bénéfice. La combinaison 5090+4090 n'est pas une seule mémoire unifiée et aucun facteur de vitesse n'est affirmé ici.

## 10. Distillation, compression, MoE : classement sans promesse de facteur

| Piste | Bénéfice possible | Coût / risque | Priorité et décision |
|---|---|---|---|
| Fenêtres partagées, budgets, cache, suppression de copies | Réduit travail dupliqué et attentes ; conserve modèles | Moyen ; bugs de DAG/cache possibles | **P1**, avant nouvel entraînement |
| Export avec constantes/poids pliés, fusion et layouts | Réduit overhead et bande passante | Moyen ; fidélité numérique | **P1**, mesurer par stage |
| Précision FP16 validée pour navigateur | Réduit taille des paramètres et activations | Moyen ; dérive coarse/côtes | **P1 expérimental**, jamais supposé équivalent au BF16 |
| Coarse distillé 20 → quelques évaluations | Réduit préparation macro et contexte froid | Entraînement ; contrôle spatial et diversité | **P2**, si coarse domine les scénarios réels |
| Base plus petit distillé | Réduit poids téléchargés et coût latent | Entraînement plus lourd, perte de relief | **P2**, si base/mémoire/download dominent |
| Decoder plus petit | Réduit coût du renouvellement fin | Peut lisser ravines et côtes | **P2**, si decoder domine le débit |
| Batching supplémentaire | Meilleur débit potentiel | Rejets numériques actuels ; latence de batch | Réouvrir seulement avec une nouvelle validation complète |
| INT8 / quantification mixte | Réduit poids ; accélération dépend des kernels | Calibration rareté/climat/sigmas ; conversions | **P3**, après profil WebGPU et précision réduite |
| Cache approximatif de features entre pas | Réduit forwards partiels | Change résultats ; seulement deux pas base et un decoder | Faible priorité ; coarse seul mérite étude si traces favorables |
| MoE appris | Spécialisation de capacité | Nouveaux modèles, routage, mémoire, transitions | **Recherche**, pas accélération évidente du checkpoint |
| Remplacer tout par GAN/flow/auto-régression | Autre compromis qualité/vitesse | Nouveau système et nouveau risque de coutures | Écarter comme première réponse au problème actuel |

### 10.1 Distiller le coarse de manière pertinente

La distillation peut apprendre à reproduire un enseignant multiétapes en moins d'évaluations. C'est une méthode d'entraînement, pas une option de compilation. [Progressive Distillation](https://arxiv.org/abs/2202.00512), [Consistency Models](https://arxiv.org/abs/2303.01469), [consistance continue / TrigFlow](https://arxiv.org/abs/2410.11081).

Pour ce projet, l'enseignant doit être **le coarse exact de référence**, avec ses cinq champs et son bruit spatial. Commencer avec les mêmes dimensions et sorties ; entraîner sur des conditionnements naturels et les nouveaux conditionnements macro envisagés, sans faire disparaître les cas côtiers, très secs, froids ou montagneux. Tester deux objectifs distincts : fidélité à la sortie de même seed, et qualité statistique globale. Un étudiant peut sembler réaliste tout en déplaçant la carte : il devient alors un nouveau monde, pas une accélération transparente.

L'entraînement sur patches indépendants ne valide pas l'assemblage. Inclure pertes/validation sur chevauchements, champs reconstruits et conséquences sur le base puis le decoder. Le gain total dépend de la fraction du temps occupée par le coarse **après cache et graphs** ; une grosse réduction de ses appels peut avoir peu d'effet sur un pan fin dans un contexte macro déjà chaud.

### 10.2 Base et decoder : déjà peu d'étapes

Le base actuel a deux passes de consistance ; le decoder, une. Passer `T=1` dans le code n'est pas forcément synonyme d'un seul forward : certaines branches regroupent encore deux évaluations dans une même étape externe. `onestep_latent` et les règles de fusion changent un autre aspect du calcul. [world_pipeline.py](../terrain-diffusion/terrain_diffusion/inference/world_pipeline.py), lignes 1248–1303.

Un étudiant base plus étroit, des blocs moins nombreux ou une architecture adaptée aux convolutions WebGPU peuvent réduire le coût et les poids. Les quatre latents et la basse fréquence doivent rester compatibles avec le decoder existant, ou l'ensemble doit être réentraîné. Un nouveau decoder doit préserver spectres, gradients, zéro marin et petites structures, pas seulement l'erreur moyenne de hauteur. Le manuel local décrit déjà les étapes de préparation de dataset, autoencodeur, diffusion puis consistance ; il ne fournit pas un budget mesuré pour ces nouveaux objectifs. [TRAINING.md](../terrain-diffusion/TRAINING.md).

### 10.3 Pourquoi MoE n'est pas le premier choix

Un MoE sélectionne des experts et peut augmenter la capacité totale sans activer tous leurs paramètres à chaque exemple. Ce bénéfice démontré notamment pour des Transformers n'implique pas qu'un U-Net dense existant devienne plus rapide en lui ajoutant un routeur. [Switch Transformers](https://arxiv.org/abs/2101.03961).

Ici, des experts désert/montagne/glacier doivent être entraînés, chargés et raccordés. Les régions mixtes, estuaires et transitions de climat sont précisément les endroits où le routage discret peut créer une rupture. Un mélange doux de plusieurs experts calcule plusieurs réseaux ; un routage dur peut perdre la continuité. Le téléchargement de plusieurs jeux de poids et leur résidence GPU peuvent être défavorables au navigateur.

Alternative moins coûteuse à étudier si la variété le justifie : **un backbone commun avec petits adaptateurs conditionnels** ou un raffineur de phénomène dédié. Cela reste un chantier d'apprentissage ; son objectif premier serait le contrôle et la diversité, pas une promesse de réduction de latence. Le routage procédural de kernels spécialisés selon pente/lithologie serait, lui, une organisation du raffinement et non un MoE neuronal.

### 10.4 Ce qu'un chantier d'entraînement exige réellement

- Dataset et droits d'utilisation identifiés, couverture géographique et verticale, qualité côtière/bathymétrique et manifeste du checkpoint 30 m si disponible.
- Couples enseignant/conditionnement/bruit ou véritables cibles haute résolution selon l'objectif ; corpus séparé par régions pour éviter une validation trop proche du training.
- Échantillonnage équilibré des plaines, montagnes, côtes, déserts, zones froides, bassins fermés, îles et configurations rares.
- Pertes en hauteur transformée et mètres, gradients, spectres, statistiques hydrologiques et coutures ; contrôles de diversité et de conditionnement.
- Mesure du coût d'un step, mémoire, débit de génération des exemples, coût d'évaluation complète et checkpoints. Estimer ensuite `durée ≈ steps×temps_step + données + évaluations`, pas annoncer un nombre de jours à partir de la seule VRAM.
- Pour glaciers/fjords : données de formes glaciaires et contexte régional, pas uniquement une température froide. Pour sous 30 m appris : DEM plus fins, opérateur de dégradation connu, alignement et cohérence avec la macro.

La 3090 et les deux autres GPU sont des ressources utiles pour ces pilotes, mais aucun coût d'entraînement ni succès de distillation n'a été mesuré dans cet audit. Un pilote court doit d'abord démontrer une tendance de qualité et le coût réel avant d'engager un entraînement long.

## 11. Raccorder le DEM 30 m au procédural sous 30 m

### 11.1 Le branchement actuel est incompatible avec une côte NN générale

`new_mixed_with_source` existe, mais `new_prepared` exige une longueur de 1024², parfois le double pour les formes géologiques, un relief parmi une liste donnée, et **toutes les hauteurs ≥0,3 m**. Il rejette donc mer, altitude nulle et dépressions négatives. [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 198–217.

La normalisation historique décale le premier percentile à 1 m et borne à 0,3 m. Cela conserve certaines pentes, mais **déplace le niveau marin** et rend le résultat dépendant du domaine de normalisation. Elle est inacceptable pour importer des tuiles NN signées. [lib.rs](../../city_generator/rust/crates/core/src/lib.rs), lignes 418–428 ; [terrainErosion.wgsl](../../city_generator/rust/bridge/terrainErosion.wgsl), fonction `normalize`.

`set_coast_surface` accepte bien des hauteurs signées et les conserve sans redécoupe, mais exige une côte active et un champ 1024². C'est un précédent utile pour la représentation, **pas le contrat DEM final à détourner**. [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 513–528 et 656–659.

### 11.2 Frontière proposée, sans modifier le domaine du moteur aujourd'hui

Un adaptateur `PreparedElevation` devrait porter :

```text
world_id, source_stage="decoder30m", generation_version
origin_cell_i/j (entiers globaux), cell_size_m=30
width/height, halo, échantillonnage aux centres
heights_m : FP32 signé, sea_level_m=0, valid_mask
climate_grid + unités + transformation thermique
boundary_constraints / hydro_region_id / revision
```

Cette frontière doit accepter des dimensions rectangulaires et un halo, sans translation verticale implicite ni mélange « relief type = montagne » obligatoire. L'orchestrateur détient les données globales ; le moteur régional détient seulement ses buffers et sorties. En GPU, le contrat devrait pouvoir référencer un buffer partagé ; en WASM/CPU, un tableau contigu. Le comportement sans raffinement doit restituer l'entrée après échantillonnage convenu.

Attention au sampler existant `sample_c2` : ses poids sont ceux d'une **B-spline cubique positive non interpolante** appliquée directement au raster. Au nœud, les poids unidimensionnels valent `1/6, 4/6, 1/6, 0`, et ne restituent donc pas en général la valeur centrale. Il évite le dépassement de l'enveloppe locale, mais lisse aussi aux centres. Son commentaire utilise « interpolate » au sens de reconstruction continue ; ce n'est pas une preuve d'interpolation exacte. Importer une heightmap puis la passer par ce sampler modifierait déjà ses échantillons sans ajouter de détail. Il faut choisir consciemment cette reconstruction ou un opérateur reproduisant les samples, puis tester le contrat. [engine.rs](../../city_generator/rust/crates/core/src/engine.rs), lignes 804–834.

### 11.3 Reconstruction d'abord, génération de détail ensuite

Proposition de progression :

1. **Reconstruction continue** du DEM 30 m avec interpolation bornée, normals cohérentes et filtrage de LOD. Tester les dépassements bicubiques près de zéro : une interpolation non bornée peut créer une petite île ou un trou marin absent des samples.
2. **Résidu de détail déterministe** selon coordonnées globales, pente, courbure, rugosité et climat. Les bandes ajoutées disparaissent au dézoom sans renormaliser les autres bandes. Le résidu n'invente pas une chaîne de montagnes à chaque tuile.
3. **Érosion locale contrainte**, appliquée sur supertuiles avec halo et conditions de bord, sous budget d'altération. Conserver crêtes principales, thalwegs, exutoires, niveau marin et statistiques de hauteur à l'échelle de référence.
4. **Hydrologie vectorielle**, puis incision continue du lit dans la surface fine ; pas une simple texture de rivière posée sur des crêtes.

Une formulation utile est `H_f = U(H_30) + r_f`, avec opérateur de réduction `D`. Chercher `D(H_f)≈H_30` constitue un test de conservation de macro. Un résidu projeté comme `r = r0 - U(D(r0))` peut aider, mais **n'impose exactement `D(r)=0` que si les opérateurs sont compatibles**, notamment `D∘U=I` dans le contrat choisi. Il faut le vérifier, et ne pas confondre conservation de moyennes par cellule et conservation des échantillons ponctuels du DEM.

Une érosion qui creuse un véritable lit a nécessairement un effet sur certaines moyennes. Il faut donc définir une tolérance de macro ou des contraintes locales, pas exiger simultanément une érosion arbitraire et une invariance absolue impossible. Les changements plus grands deviennent des opérations géographiques explicites, versionnées et propagées aux LOD concernés.

### 11.4 Coutures et reproductibilité du raffinement

Les simulations ne doivent pas être lancées sur le rectangle courant de caméra. Utiliser des supertuiles canoniques indépendantes du viewport, un contexte déterministe et une propriété unique des bords/cours d'eau. Pour des opérations locales de rayon fini, un halo calculé suffit ; pour drainage et bassins, l'influence peut dépasser tout halo fixé arbitrairement.

Une solution régionale est calculée puis figée ; l'affichage lit ses mips et ses vecteurs. Changer de zoom ne relance pas une érosion différente. Les régions voisines doivent utiliser les mêmes conditions hydrographiques et les mêmes contributions de chevauchement. Le simple fondu de deux solutions incompatibles peut éliminer une marche de hauteur tout en créant une vallée plate ou un cours d'eau incohérent.

Pour les coordonnées mondiales, éviter des mètres absolus en FP32 dans tous les shaders : près de 20 millions de mètres, la résolution de représentation devient de l'ordre du mètre. Utiliser origine entière/haute précision et coordonnées locales relatives, surtout pour un détail métrique ou submétrique. La fonction de bruit doit être indexée canoniquement afin qu'un changement d'origine d'affichage ne change pas les motifs.

## 12. Côtes pixelisées, fjords et hydrologie

### 12.1 Diagnostiquer la côte avant de la modifier

| Aspect observé | Cause possible | Réponse appropriée |
|---|---|---|
| Escalier visible à l'écran sur une côte autrement plausible | Classification par texel, échantillonnage ou couverture sans antialiasing | Isocourbe 0 m interpolée, masque de couverture/supersampling, normales et filtre cohérents |
| Côte douce et arrondie à grande échelle | Conditionnement trop lisse, faible liberté du coarse ou aperçu non neuronal | Ablation de l'entrée et du bruit ; conserver l'enrichissement NN |
| Petite baie absente à 30 m | Structure sous la résolution disponible | Détail contraint, ou données/modèle plus fins |
| Grande vallée noyée absente, pas de fjord | Structure glaciaire régionale absente du paysage | Procédé régional glaciaire ou entraînement/contrôle dédié |
| Côte qui change en zoomant | Sources coarse/latent/finale non identiques et/ou assemblage erroné | Mesurer déplacement de l'isocourbe, provenance, mips finaux lorsqu'ils existent |

Le renderer de `city_generator` possède déjà une approche de masque vectoriel antialiasé tiré de la surface signée. Cette idée se transpose sans imposer ses silhouettes côtières. Appliquer `Coast::apply` sur le DEM NN redéfinirait plages, falaise, datum, îlots et terre/mer ; c'est précisément le risque de détruire ce que l'on cherche à préserver. [coast.rs](../../city_generator/rust/crates/core/src/coast.rs), lignes 388–425 ; [terrainRender.ts](../../city_generator/rust/bridge/terrainRender.ts).

Une petite perturbation littorale peut être bornée autour de l'isocourbe existante, avec une amplitude liée au pas et à la pente. Elle doit préserver détroits, îles et embouchures protégés. Conserver le zéro marin, les altitudes voisines et la topologie importante vaut mieux qu'ajouter un bruit de distance indépendant partout. La longueur de côte varie avec la résolution de mesure : comparer les variantes à la même échelle.

### 12.2 Pourquoi davantage d'érosion fluviale ne garantit pas de fjords

Un fjord associe notamment une vallée glaciaire régionale, un profil transversal élargi, une incision/overdeepening et une connexion marine. Le climat contemporain n'encode pas à lui seul l'histoire des glaciers. Le modèle du banc est principalement fluvial/gravitaire ; ses canyons étroits ne sont pas un substitut universel à ces formes.

La littérature de synthèse de terrain propose des simulations glaciaires qui produisent vallées en U, vallées suspendues, fjords et lacs glaciaires ; cela constitue une **nouvelle famille de processus à intégrer et mesurer**. [Forming Terrains by Glacial Erosion, version auteurs, 2023](https://www-sop.inria.fr/reves/Basilic/2023/CJPBCBGGG23/Sigg23_Glacial_Erosion__author.pdf). Une étude géophysique montre aussi le rôle de l'écoulement glaciaire dirigé par la topographie dans l'insertion des fjords. [Fjord insertion into continental margins driven by topographic steering of ice](https://www.nature.com/articles/ngeo201).

Trois voies, par coût croissant : sélectionner et préserver des régions NN qui possèdent déjà ces formes ; introduire des formes glaciaires contrôlées à l'échelle régionale avec contraintes sur le NN ; entraîner/adapter un modèle avec données et conditionnement glaciaires. Ajouter uniquement du détail sous 30 m peut enrichir les parois d'un fjord existant, pas créer de façon neutre une vallée de plusieurs kilomètres absente de la macro.

### 12.3 Hydrologie : le problème mondial est une contrainte de frontière

Le banc sait dériver réseau, accumulation, cuvettes, lacs, méandres et estuaires d'un DEM régional. Il faut conserver ces diagnostics et leurs unités. Son accumulation part du domaine traité ; un fleuve entrant d'une autre région nécessite un débit/aire contributive externe. Le traitement des mers connectées au **bord de la carte locale** ne suffit pas à classifier correctement une mer intérieure ou une tuile entièrement située dans un océan. [hydrology.rs](../../city_generator/rust/crates/core/src/hydrology.rs), lignes 192–317 ; [hydrologyGpu.ts](../../city_generator/rust/bridge/hydrologyGpu.ts), lignes 53–76.

**Architecture proposée :** un graphe hydrologique régional/macro versionné fournit exutoires, apports entrants, niveaux des grands lacs et connexions. Le solveur local utilise le DEM NN détaillé et cette frontière pour affiner les affluents et les lits. L'eau doit pouvoir entrer et sortir au même endroit pour les deux régions voisines. Les confluences partagent un niveau et une identité ; les lits sont monotones vers l'aval dans la tolérance choisie.

Le coarse 7,68 km peut suggérer de grands bassins, mais il ne révèle pas tous les cols et détroits du DEM 30 m. Une hiérarchie hydrographique doit gérer cette incertitude : analyse régionale suffisamment large avant publication d'un réseau définitif, puis règles de raffinement qui conservent les connexions. Promettre simultanément **hydrologie mondiale exacte**, accès aléatoire totalement local et absence de préparation globale serait injustifié.

Les algorithmes Priority-Flood traitent les dépressions et peuvent être organisés par tuiles avec échange d'informations de bord ; leur existence ne transforme pas le solveur actuel en version mondiale. [Priority-Flood](https://arxiv.org/abs/1511.04463), [version parallèle par tuiles](https://arxiv.org/abs/1606.06204).

Conserver deux surfaces si nécessaire : DEM géographique original et surface hydrologique dérivée. Un comblement de cuvettes pour calculer le drainage n'autorise pas automatiquement à relever des dizaines de mètres de terrain visible. Les bassins endoréiques et lacs naturels peuvent être voulus. Les réglages actuels limitant la quantité de lacs dans un banc urbain ne sont pas une distribution mondiale à reprendre sans examen.

Enfin, BIO12 est une précipitation annuelle et BIO15 une variabilité : leur combinaison peut guider un ruissellement moyen simplifié, mais pas fournir une hydrogramme réaliste sans hypothèses sur évaporation, infiltration et saisonnalité. Nommer explicitement le modèle hydrologique choisi évite de présenter une aire drainée comme un débit mesuré.

## 13. Expériences discriminantes et critères de décision

Les seuils ci-dessous sont des **portes de décision proposées**, à figer avant les essais. Les contrôles automatiques servent à détecter les régressions ; la préférence visuelle de l'utilisateur reste déterminante pour les variantes de géographie.

### 13.1 Corpus commun

Constituer des sites fixes pour : plaine côtière, falaise, îlot, détroit, estuaire, montagne alpine, hauts plateaux, canyon, désert, zone humide, bassin fermé, région froide et candidat glaciaire. Inclure les coordonnées déjà jugées bonnes et mauvaises par l'utilisateur, plusieurs seeds, seed 0, coordonnées négatives et bords de fenêtres. Conserver entrée, coarse, latent basse fréquence, DEM, climat, lignes de côte et profils.

Comparer à mêmes dimensions physiques, palette, exposition, méthode de normales et échelle verticale. Une variante qui améliore seulement le hillshade ne doit pas être créditée d'une amélioration géomorphologique. Séparer cas choisis et échantillon aléatoire stratifié ; publier aussi les échecs.

**Porte visuelle bloquante avant tout changement du profil par défaut :** planches côte + montagne + plaine, au moins aux niveaux continental, régional, 30 m et raffinement éventuel, examinées côte à côte avec la référence. Les tests numériques de la tranche rejetée pouvaient réussir alors que sa géographie ne convenait pas. Aucun résultat `passed=true`, taux terre/mer correct ou absence d'erreur console ne remplace cette acceptation visuelle.

### 13.2 Mesure de performances

Scénarios : premier lancement avec download absent ; lancement poids persistés ; téléportation monde froid ; pan chaud horizontal/diagonal ; zoom rapide aller-retour ; revisite après éviction ; deux vues partageant un contexte ; grand dézoom puis retour au natif. Faire des répétitions et publier médiane/p95, dispersion et température/charge matérielle, sans mélanger réchauffement et régime établi.

Chronométrer séparément : téléchargement, parsing/compilation, upload poids, bruit/conditionnement, attente, coarse/base/decoder, fusion/Laplacien/climat, transferts, encodage, upload final et première présentation. Ajouter nombre de fenêtres uniques, octets transférés, mémoire allouée/réservée/estimée navigateur, travaux abandonnés et retard de couverture.

**Cibles produit initiales proposées, non performances acquises :** frame caméra p95 <16,7 ms à 60 Hz sur la classe desktop ciblée ; première couverture utile après téléportation <250 ms si aperçu/cache disponible ; affinement visible progressif sans blocage ; aucun retard croissant pendant un pan de vitesse définie. La complétion neuronale froide doit être annoncée avec sa mesure réelle, même si elle dépasse la cible souhaitée. Pour le débit, viser une marge d'au moins 20 % sur la demande du scénario afin d'absorber fluctuations et mouvements ; ce seuil est un budget de conception.

### 13.3 Fidélité et coutures

| Test | Mesure | Critère proposé |
|---|---|---|
| Optimisation réputée exacte | Écart tenseurs, DEM et chaque climat | Exact si mêmes opérations/backend ; sinon profil toléré documenté, jamais déclaré exact |
| Port numérique | MAE, RMSE, p99/max hauteur, gradients, erreurs climatiques par canal | Seuils fixés avant essai ; première cible conservatrice ≤1 m max et pas de déplacement structurel, à renforcer aux côtes |
| Terre/mer | Désaccord de masque, distances d'isocourbes, îles/détroits | Aucun changement de topologie des éléments protégés ; examiner même les faibles erreurs près de 0 m |
| Ordre de requêtes | Crop entier vs sous-crops, permutations, retour après éviction | Même résultat selon le contrat du backend ; pas de génération dépendante de la navigation |
| Coutures | Hauteur, pente/normale et climat sur bords partagés | Pas de discontinuité supplémentaire aux coutures par rapport à des lignes intérieures comparables |
| LOD | Parent vs réduction enfants, dérive côte/relief | Résidu mesuré ; distinguer mip exact et approximation apprise |
| Macro nouvelle | Hypsométrie, plaines/montagnes, bassins, spectres et climats conjoints | Amélioration des objectifs sans effondrement de variété ni régression visuelle des cas protégés |
| Raffinement sous 30 m | `D(H_f)-H_30`, relief local, réseaux et bords | Macro dans tolérance fixée ; aucun déplacement implicite de datum |
| Hydrologie | Boucles, continuité confluences/exutoires, monotonie, bilans d'apports | Invariants du modèle satisfaits ; pas d'exutoire artificiel au bord d'une tuile |

Un seuil de 1 m n'est pas suffisant à lui seul pour une côte plate : une faible variation verticale peut déplacer le rivage loin horizontalement. Inversement une variante géomorphologique volontaire peut dépasser ce seuil tout en étant souhaitable ; elle doit alors être classée comme nouveau générateur, pas comme optimisation fidèle.

### 13.4 Expériences et décisions binaires

**E1 — Référence et attribution de la régression.** Rejouer A0–A3 et les variations de bruit sur sites fixes. Go pour une nouvelle macro seulement si ses gains survivent au decoder et si les reliefs/côtes protégés restent satisfaisants. Sinon conserver le conditionnement original et revoir le compositeur.

**E2 — Faisabilité WebGPU.** Exporter puis exécuter les trois forwards et un crop complet avec tous les champs, sans fallback invisible. Go si mémoire compatible, placements maîtrisés, précision et temps bout en bout acceptables. Si un opérateur bloque, corriger cet opérateur avant de réécrire le réseau entier.

**E3 — DAG et navigation.** Même backend/qualité, scénario de pan et téléportation identique. Go si fenêtres dupliquées, retard visible et p95 diminuent sans dépendance à l'ordre. Un gain sur cache chaud seul ne valide pas l'objectif.

**E4 — Coarse mondial.** Générer par budgets de fond une couverture apprise et ses mips, en instrumentant appels/octets/temps. Go si le coût amorti et le stockage répondent au produit. Si la seed arbitraire doit être mondiale et neuronale immédiatement, lancer E5 : le scheduler seul ne résoudra pas cette exigence.

**E5 — Distillation ciblée.** Pilote coarse ou étudiant compact choisi d'après E2–E4. Go uniquement si gain caméra → sortie et qualité/contrôle/coutures atteignent les critères. Une courbe de loss qui baisse ne suffit pas.

**E6 — DEM signé + reconstruction.** Importer sans détail un patch NN contenant mer, côte et relief ; restituer mètres et zero marin. Go si le chemin neutre n'altère pas le terrain et si CPU/GPU suivent le même contrat. Puis seulement ajouter les résidus.

**E7 — Bassin traversant plusieurs régions.** Comparer une grande région à sa subdivision, avec apports et exutoires identiques. Go si rivières/lacs et altitudes de connexion coïncident. Sinon revoir le contrat hydrologique avant déploiement mondial.

**E8 — Fjord.** Séparer interpolation, détail fluvial et nouveau procédé glaciaire. Go pour l'option glaciaire si elle améliore profils et cohérence régionale sans effacer les formes approuvées ; pas de validation par couleur froide seule.

## 14. Lots d'implémentation proposés après décision sur l'audit

| Lot | Livrable concret | Dépendance / preuve attendue | Coût relatif, risque |
|---|---|---|---|
| L0 | Manifest de référence, corpus et export des intermédiaires ; benchmark avec profil explicite | Aucun changement de génération ; E1 prêt | Faible à moyen, faible |
| L1 | POC WebGPU complet sur un crop et rapport placement/mémoire/temps | E2 ; forwards puis pipeline | Moyen, risque technique moyen/fort |
| L2 | Ordonnanceur de fenêtres et cache canonique commun | E3 ; mêmes sorties dans différents ordres | Moyen à fort, risque de cohérence |
| L3 | Préparation coarse mondiale incrémentale + provenance des LOD | E4 ; mesure globale réelle | Moyen, risque de coût/délai |
| L4 | Compositeur macro expérimental à cinq champs, séparé de `natural` | E1 ; ablations et validation utilisateur | Fort, risque esthétique et distribution |
| L5 | Import DEM signé + reconstruction côtière neutre | E6, sans nouveau relief initial | Moyen, risque maîtrisable |
| L6 | Résidu sous 30 m + contraintes hydrographiques régionales | E7 et conservation macro | Fort, risque non local |
| L7 | Distillation du goulot mesuré / étudiant compact | E5 ; données et coût pilote connus | Fort, résultat non garanti |
| L8 | Agent natif optionnel + calibration 3090/4090/5090 | Gain réel face au navigateur, copies incluses | Moyen/fort, maintenance multiplateforme |
| L9 | Contrôle glaciaire et éventuelle extension d'entraînement | E8 ; nouveaux phénomènes assumés | Recherche, risque fort |

Cet ordre protège l'information utile : L0 évite une nouvelle régression non attribuable ; L1 révèle tôt la faisabilité du produit prioritaire ; L2/L3 traitent la quantité de travail réellement nécessaire ; L4 améliore la géographie sans la mêler aux optimisations ; L5/L6 prolongent le terrain au lieu de le remplacer. L7 n'est justifié qu'après identification du goulot. L8 peut avancer plus tôt si WebGPU rencontre un blocage, mais reste optionnel pour le produit visé. L9 ne doit pas être promis comme simple réglage d'érosion.

Les coûts relatifs expriment l'effort et l'incertitude, pas un calendrier contractuel. Aucun runtime ni paramètre n'est changé par ce document. La prochaine étape utile est de choisir le premier lot et ses critères ; il n'y a aucune raison technique de reprendre le profil `earth` actuel comme baseline qualitative.

## 15. Registre des inconnues importantes

1. Débit durable et mémoire de chaque U-Net sur les navigateurs/GPU visés, avec les versions exactes du runtime : **non mesurés**.
2. Précision compatible WebGPU préservant terrain et climat face au BF16 CUDA : **non démontrée**.
3. Temps de couverture coarse mondiale exacte et coût des changements de seed : **dimensionnés mais non mesurés**.
4. Répartition réelle des goulots coarse/base/decoder/assemblage sur des sessions complètes : **quelques crops seulement**.
5. Identité complète du corpus d'entraînement 30 m et couverture glaciaire/polaire : **manifeste manquant**.
6. Paramètres du compositeur qui améliorent la géographie sans écraser la variété NN : **à déterminer par ablations**.
7. Meilleure organisation des bassins mondiaux et compromis d'accès aléatoire : **nouvelle architecture nécessaire**.
8. Gain de l'agent natif et du second GPU sur la machine 5090+4090 : **aucun benchmark physique disponible dans les preuves consultées**.
9. Temps/coût d'un étudiant distillé et données haute résolution pour raffinement appris : **pilote requis**.

Ces inconnues n'empêchent pas de construire le système par étapes. Elles empêchent d'annoncer aujourd'hui une planète neuronale froide instantanée, un port WebGPU terminé, des fjords garantis ou un facteur d'accélération tiré du seul nom d'une technologie.
