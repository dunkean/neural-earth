# Terrain Diffusion : génération et navigation en temps réel

Document d’analyse et de design — 7 octobre 2026 — version 1.1

**Objectif :** naviguer dans un monde naturel varié, généré à la demande à 30 m/pixel, avec un **agent WebGPU dans le navigateur comme produit principal**. Un agent local natif Linux/Windows, distribuable en open source, apporte une option de performance lorsqu’il fait mieux de bout en bout. L’usage du GPU s’adapte au matériel : RTX 3090 sur la machine actuelle ; RTX 5090 et RTX 4090 sur une autre machine ; GPU variés chez les visiteurs. WASM sert au noyau portable et au client de l’agent natif.

**Statut :** architecture proposée, fondée sur l’audit du code installé et sur les observations de la session. Ce document ne constitue pas un benchmark du futur moteur. Les budgets et seuils proposés sont des critères à mesurer. La génération actuelle reste en place.

**Implémentation commencée le 7 octobre 2026 :** le chemin Python/CUDA dispose maintenant d’une file de caméra partagée, de caches physiques, de fenêtres CUDA résidentes et de CUDA Graphs validés. Le navigateur réalise le relief et la palette en WebGPU. Le port des NN vers WebGPU et le moteur natif C++ restent à construire. Les résultats récents et leurs limites sont séparés de l’audit historique dans [l’état de l’implémentation](REALTIME_IMPLEMENTATION.md).

**Navigation mondiale :** le profil `earth` démarre sur une carte plane de 40 000 × 20 000 km, avec un aperçu du conditionnement continental et climatique. Ce même conditionnement alimente les régions neuronales demandées au zoom. Les vues climat/biomes et la sélection uniforme du LOD rendent la diversité visible. Le profil `natural` permet la comparaison avec l'entrée originale. Les causes, les limites géomorphologiques et la distinction entre aperçu et relief appris sont précisées dans [la note de génération mondiale](WORLD_GENERATION.md).

## 1. Décisions recommandées

Construire un **moteur de régions**, dont le travail élémentaire est une fenêtre neuronale réutilisable. Une requête de caméra devient un graphe de dépendances partagé ; elle ne déclenche plus un calcul indépendant pour chaque PNG. Le moteur publie progressivement le contexte continental, les latents, puis le relief final. Le rendu reste fluide pendant ces calculs.

La première voie d’inférence à rendre utilisable est **ONNX Runtime Web/WebGPU**, dans un worker, avec rendu GPU et ordonnancement asynchrones. Le noyau de coordonnées, de bruit déterministe, de planification et de cache reste portable et compilable en WASM. Les kernels WGSL spécialisés viennent après profilage de cette voie principale.

L’agent natif optionnel sera en **C++20**, avec ONNX Runtime CUDA comme référence de portage, puis TensorRT et quelques kernels CUDA pour l’assemblage. Son interface est compatible avec celle de l’agent WebGPU. L’installation native n’est pas nécessaire pour utiliser le site ; on la recommande si des mesures comparables montrent un gain de latence, de débit ou de mémoire.

Publier le site avec **WebGPU local par défaut**, **agent natif local facultatif**, et **service distant facultatif** pour les appareils incapables de faire la génération complète ou pour utiliser l’autre machine. Le site statique et son agent WebGPU constituent un livrable autonome ; héberger un service GPU public n’est pas une condition de réussite. Le site ne peut pas charger CUDA/TensorRT dans le navigateur par une simple compilation WASM.

**Meilleur GPU, plus de vitesse :** adapter batches, caches et préchargement au débit réellement mesuré, en gardant le même terrain et toutes ses données. Sur la machine 5090 + 4090, l’agent natif peut répartir les régions entre deux workers GPU. Le navigateur utilise principalement l’adaptateur qu’il expose ; la sélection explicite des cartes et leur usage simultané sont des fonctions de l’agent natif.

Conserver la totalité du conditionnement du modèle : altitude/bathymétrie, température, saisonnalité thermique, précipitations et saisonnalité des précipitations. Garder également les statistiques de relief, les cinq canaux latents et le post-traitement. La variété relève du contenu généré et de son rendu climatique, pas seulement d’une palette de couleurs.

L’ordre d’investissement est : **mesurer → valider les trois réseaux WebGPU → navigation et graphe partagé → pipeline WebGPU résident → adaptation au matériel → agent natif si son gain est démontré**. Le noyau commun permet de réutiliser les optimisations de planification dans les deux agents.

## 2. Périmètre et définition du temps réel

Le produit cible est une carte procédurale naturelle sans masque d’île imposé. Le monde est décrit par une seed et un profil de génération immuable. Un déplacement revient sur le même terrain ; un changement de zoom affine la représentation de cette région.

Trois délais doivent être distingués :

| Mesure | Définition | Objectif de référence proposé, à mesurer par agent sur la 3090 |
| --- | --- | --- |
| Réactivité de navigation | Temps entre mouvement et image déplacée | Affichage 60 Hz ; aucune attente synchrone d’inférence |
| Première couverture | Temps jusqu’à une représentation disponible de toute la vue | Aperçu depuis cache/procédural en moins de 300 ms, hors téléchargement initial |
| Affinement adjacent | Temps jusqu’au relief final d’une tuile visible de 256², contexte proche déjà calculé | p95 inférieur à 500 ms, à vérifier |
| Téléportation froide | Temps jusqu’au premier relief final, poids résidents, contexte absent | p95 inférieur à 2 s, objectif à vérifier |
| Charge continue | Surface finale nouvelle calculée par seconde | Supérieure à la demande de la caméra, avec 20 % de marge |

Une image à 60 Hz n’implique pas 60 nouvelles inférences par seconde. Le navigateur déplace immédiatement des textures existantes ; les régions nouvelles arrivent de manière asynchrone. Une vue vide doit obtenir un aperçu rapide, puis une couverture de qualité croissante. La télémétrie mesure séparément la couverture finale et la couverture approximative.

La 3090 reste le point de comparaison disponible, pas un plafond de vitesse. Les 4090/5090 doivent convertir leur débit supplémentaire en affinement plus rapide, meilleure couverture finale et davantage de marge de préchargement. Les seuils se déclinent par agent et appareil après calibration ; aucun multiplicateur 3090 → 4090 → 5090 n’est présumé.

Le modèle installé a une résolution native de **30 m/pixel**. Le zoom sous cette échelle utilise l’interpolation du relief ; créer de nouveaux détails sous 30 m demanderait une extension entraînée et versionnée. « Infinite Generation » désigne ici l’extension spatiale à la demande. Une planète sphérique, avec fermeture des longitudes et cohérence des pôles, demanderait aussi un conditionnement et une topologie explicites.

L’article donne, sur une **3090 Ti**, des moyennes de 1,72 s pour la première tuile 512² et de 0,66 s pour la suivante avec T=2. Son terrain principal est à 90 m ; ces nombres ne mesurent ni notre checkpoint 30 m, ni une vue navigateur complète, ni notre 3090. L’accès constant porte sur une région de taille fixe ; le coût augmente avec la surface demandée. [Article, sections 3.5 et 6.3](https://arxiv.org/html/2512.08309v4#S6.SS3)

## 3. État audité et observations

Références de cette analyse :

| Élément | Version ou état |
| --- | --- |
| Code upstream | `e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230`, arbre Git propre lors de l’audit |
| Checkpoint | `xandergos/terrain-diffusion-30m`, révision `9ef8030cb805b433b98ec25c5dddefbac07a9e26` |
| Exécution actuelle | RTX 3090 24 Go ; Python 3.12 ; PyTorch 2.11.0 + CUDA 12.8 ; BF16 |
| Interface | `index.html` ; tuiles 256² ; deux téléchargements simultanés ; délai 180 ms |
| Serveur | `terrain_server.py` ; verrou GPU global ; deux mondes de 512 Mio de cache chacun |
| Modèles | Poids partagés sur CUDA ; batch latent 16 ; coarse et decoder traités fenêtre par fenêtre |
| Cache de rendu | `generated/natural-v1`, PNG et métadonnées |
| Données lourdes | `E:\TerrainDiffusionRuntime`, exposées par des jonctions dans le projet |

Observations disponibles, sans contrôle complet des caches ni de l’activité du bureau :

| Observation | Valeur | Portée de la mesure |
| --- | --- | --- |
| Export rectangulaire enregistré dans `generated/terrain.json` lors de l’audit | 4096 × 2048 ; 36,44 s ; pic alloué PyTorch 1,08 Go | Seed 538474532 ; mesure d’export existante ; compteurs de forwards possiblement cumulés |
| Tuiles naturelles coarse LOD 8 | 14,614 s et 14,984 s | Lignes 748–749 de `server.log` lors de l’audit |
| Tuiles naturelles coarse LOD 9 | 54,286 s et 51,476 s | Lignes 756–757 ; coût du rectangle intermédiaire particulièrement visible |
| Tuiles réutilisant le contexte coarse | Certaines autour de 0,04–0,06 s | Ce délai n’est pas celui d’une génération neuronale froide |
| Échantillonnage GPU pendant une génération | 15 relevés ; utilisation moyenne 44,1 %, min 27 %, max 87 % | Mesure globale courte, bureau inclus ; pas une mesure de l’occupation des Tensor Cores |
| Mémoire GPU totale utilisée pendant ce relevé | Environ 5,1 Gio sur 24 Gio | Inclut les autres usages ; distinct du pic alloué à PyTorch |

Les forwards CUDA ont été vérifiés pendant les tests de génération. Le GPU travaille donc effectivement. En revanche, ces mesures ne prouvent pas une saturation du calcul, et une faible allocation mémoire ne permet pas de déduire la performance. Il faut un profil temporel CPU/GPU pour séparer attente, transferts et kernels.

Les valeurs de session sont archivées dans [audit-session.json](audit-session.json). Les chiffres concernant une ancienne île artificielle sont exclus des comparaisons : cette variante a été retirée et ne représente pas le produit cible.

**Matériel supplémentaire déclaré par l’utilisateur :** une seconde machine avec une RTX 5090 et une RTX 4090. Elle n’a pas été auditée ni benchmarkée dans cette session. Elle devient une cible de validation WebGPU, native mono-GPU et native à deux GPU ; les observations de la machine 3090 ne lui sont pas attribuées.

### 3.1. Principaux problèmes de la chaîne actuelle

| Problème constaté dans le code | Conséquence | Correction proposée |
| --- | --- | --- |
| `sample_field()` matérialise un rectangle dense avant interpolation | À fort dézoom, le calcul croît avec le territoire couvert, même pour 256² pixels affichés | Requêtes bornées, cache macro, aperçu synthétique, calcul progressif |
| Un verrou couvre génération, hillshade, PNG et écriture | CPU et disque retardent la prochaine inférence urgente | GPU worker séparé ; traitements de sortie hors section critique |
| Annulation des fetchs seulement côté navigateur | Une ancienne caméra peut continuer à monopoliser le serveur | Abonnements de vues, epochs, refcounts et annulation dans le graphe |
| Priorité par distance au centre | Une petite tuile coûteuse peut bloquer une couverture plus utile | Priorité tenant compte de couverture, qualité, coût et délai |
| Cache PNG lié surtout à seed/LOD/coordonnées et version manuelle | Les dépendances neuronales restent coûteuses ; profils différents risquent une collision | Caches de fenêtres par stage et clés de contenu complètes |
| `.cpu().float()` après les passes de génération | Synchronisations et aller-retour CPU/GPU | Accumulation et normalisation résidentes sur GPU |
| Batch latent 16, autres stages batch 1 | Mauvaise occupation possible sur les petits réseaux ; décodeur sérialisé | Micro-batches indépendants et calibrés par stage |
| Toutes les sorties visibles sont des PNG hillshade | Recolorer impose de refaire une image ; climat invisible | Tuiles physiques, textures GPU, styles séparés |
| `world.get(..., with_climate=False)` pour le rendu | Les quatre variables climatiques ne sont pas livrées à l’interface | Sortie climatique ou caractéristiques macro réutilisées |

Le dernier point ne veut pas dire que le réseau génère sans climat : le conditionnement climatique reste actif dans la chaîne actuelle. C’est l’exploitation des sorties climatiques qui manque au rendu. Voir [serveur actuel](../terrain_server.py) et [interface actuelle](../index.html).

## 4. Le pipeline exact à préserver

Les valeurs suivantes viennent du checkpoint 30 m installé et de `world_pipeline.py`, et non des dimensions 90 m de l’article. Source de portage : [WorldPipeline au commit audité](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/world_pipeline.py).

| Stage | Grille et fenêtre | Contenu | Exécution à préserver |
| --- | --- | --- | --- |
| Conditionnement synthétique | Grille coarse ; échantillonnage global aligné | Cinq champs procéduraux transformés par statistiques réelles | FastNoiseLite, quantiles, corrélations climatiques et transformation signed-sqrt |
| Coarse | 7 680 m/cellule ; fenêtre 64² ; stride 48 | Six champs : relief moyen, référence p5 et quatre climats ; septième canal de poids | 20 étapes internes EDM/DPM ; chaque fenêtre indépendante |
| Latent | 240 m/cellule ; fenêtre 64² ; stride 32 | Quatre latents résiduels + un relief basse fréquence ; sixième canal de poids | T=2 ; fusion des fenêtres entre les deux passes |
| Decoder | 30 m/cellule ; fenêtre 512² ; stride 384 | Résidu d’altitude ; canal de poids pour fusion | Consistency decoder ; entrée cinq canaux, dont quatre latents interpolés |
| Assemblage final | Grille native 30 m | Basse fréquence débruitée + résidu, puis retour en mètres | Dénormalisation, Laplacian denoising, signed-square |
| Climat final | Grille de sortie choisie | T moyenne, saisonnalité T, pluie annuelle, saisonnalité pluie, gradient thermique beta | Interpolation et correction de T selon l’altitude |

**T=1 pour le coarse décrit une fusion externe, pas un seul forward neuronal.** Le code appelle `scheduler.set_timesteps(20)` puis le réseau à chaque étape. Réduire cette boucle serait une modification de génération à valider, pas une optimisation transparente.

Le latent reçoit un conditionnement de 58 valeurs : 16 moyennes d’altitude, 16 p5, quatre moyennes climatiques, 16 masques, cinq valeurs d’histogramme et un niveau de bruit. Leur ordre, leurs normalisations et les opérations magnitude preserving font partie du modèle.

Les constantes basse fréquence sont `mean=-31.4`, `std=38.6` ; celles du résidu sont `mean=0`, `std=0.7` pour ce checkpoint. La configuration de référence utilise `frequency_mult=[1,1,1,1,1]`, `cond_snr=[0.5]*5`, `drop_water_pct=0.5`, `coarse_pooling=1`. Charger le fichier de configuration réel au lieu de recopier les valeurs par défaut du constructeur.

Il existe déjà un [exporteur ONNX upstream](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/onnx/export.py) : opset 17 par défaut, batch dynamique, dimensions spatiales 64/64/512. Il adapte certaines opérations de resampling et de padding. Il n’exporte ni le graphe de dépendances, ni le RNG, ni le scheduler, ni le post-traitement. Sa vérification affiche un écart numérique mais n’impose pas à elle seule un seuil de réussite ; le projet doit ajouter une validation bloquante.

## 5. Toutes les sources de données et la variété

Il faut distinguer **ce que les poids ont appris**, **les données qui calibrent le conditionnement procédural**, et **les sorties utilisées pour afficher le monde**. Charger un autre DEM en production ne rajoute pas automatiquement sa diversité à des poids déjà entraînés.

### 5.1. Inventaire et rôle des données

| Source ou champ | Rôle dans le système | Action pour l’implémentation |
| --- | --- | --- |
| Copernicus GLO-30 | Apprentissage du checkpoint 30 m selon sa fiche modèle | Conserver les poids complets et leur révision ; documenter cette provenance |
| MERIT DEM 90 m | Jeu de relief terrestre décrit pour l’expérience principale de l’article | Ne pas le confondre avec la provenance du checkpoint 30 m |
| ETOPO / bathymétrie | Relief global, fonds océaniques et distribution terre/mer ; raster runtime `etopo_10m.tif` pour les statistiques | Préserver valeurs négatives, quantiles, fraction d’eau et normalisations |
| WorldClim BIO1 | Température annuelle moyenne | Champ indépendant, transformé et corrigé selon relief/pluie |
| WorldClim BIO4 | Saisonnalité thermique | Garder les unités de la chaîne : écart-type en °C × 100 en interne |
| WorldClim BIO12 | Précipitations annuelles en mm | Conditionnement et rendu humide/aride |
| WorldClim BIO15 | Saisonnalité des précipitations, coefficient de variation | Différencier des climats avec pluie annuelle comparable |
| Référence d’altitude p5 | Information de relief complémentaire à la moyenne | Garder son calcul et son codage ; elle évite de réduire une région à sa hauteur moyenne |
| Quatre canaux résiduels latents et un canal basse fréquence | Variabilité et structure fines | Ne pas éliminer de canaux pour accélérer le port |
| Bruits déterministes par stage | Variations locales cohérentes avec la seed | Streams et offsets exacts, indépendants de l’ordre des demandes |

La [fiche du checkpoint](https://huggingface.co/xandergos/terrain-diffusion-30m) identifie Copernicus GLO-30. Le jeu de l’article associe MERIT, bathymétrie ETOPO1 et climat WorldClim. [Article, section 4](https://arxiv.org/html/2512.08309v4#S4)

Le runtime actuel extrait cinq tables de quantiles à partir d’ETOPO et des quatre fichiers WorldClim, puis génère cinq champs Perlin/FBm. Il applique des relations entre température, altitude, pluie et saisonnalité. Une fois `synthetic_map_stats.json` établi, les gros GeoTIFF ne doivent pas être lus pour chaque tuile. Publier les tables et constantes nécessaires suffit au calcul procédural, tout en conservant un manifeste de provenance des rasters qui les ont produites. [Code des cartes synthétiques](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/synthetic_map.py)

La résolution « 10m » des fichiers WorldClim signifie **10 minutes d’arc**, pas dix mètres. Les statistiques générées ne sont pas une copie géographique de la Terre. Le modèle produit un monde procédural ; il ne reconstruit pas automatiquement les continents terrestres.

### 5.2. Contrat de variété

Le profil de base doit garder les cinq entrées, les transformations climatiques et les paramètres du checkpoint. La baisse de qualité pendant une navigation rapide change le niveau de détail affiché, pas les données du monde. En particulier, elle ne met pas les climats à zéro et ne remplace pas le relief final par un bruit de détail indépendant.

Créer des profils variés seulement après validation : fréquences spatiales des cinq champs, proportion d’eau et intensité du conditionnement. Une modification de ces valeurs définit un autre profil de monde. Les champs partagent les corrélations prévues par le code ; mélanger des climats arbitraires peut pousser le modèle hors de sa distribution. Les paramètres nommés `cond_snr` doivent être interprétés selon leurs formules trigonométriques effectives, plutôt que selon le sens habituel du sigle SNR.

Le rendu doit proposer relief, altitude, température, saisonnalité thermique, pluie, saisonnalité des pluies et biomes dérivés. Les biomes utilisent ensemble altitude, T, saisonnalité T, pluie et saisonnalité pluie. Avec quatre résumés climatiques, une classification est une approximation ; une classification Köppen complète demanderait les séries mensuelles. Les cours d’eau visuellement suggérés par le relief ne constituent pas un réseau hydrologique global calculé.

Stocker les quatre climats à une résolution plus faible que l’altitude est possible, mais il faut garder la reconstruction thermique selon l’altitude et vérifier l’équivalence. `beta`, la cinquième sortie climatique, est un gradient thermique dérivé, pas une cinquième source bioclimatique indépendante. Les normes et les palettes sont des profils de rendu séparés : changer une palette ne relance pas les réseaux.

Validation de la diversité : corpus fixe de 100 seeds et plusieurs régions éloignées par seed ; distributions d’altitude, profondeur, pentes, rugosité, climat et surface terre/mer ; corrélations altitude/T et pluie/saisonnalité ; fréquence des différents groupes de biomes ; atlas visuel par seed. Comparer au pipeline de référence. Le moteur optimisé doit conserver ces distributions dans des tolérances définies avant son activation.

### 5.3. Deux défauts à résoudre avant de figer les mondes

Dans `make_synthetic_map_factory()`, `seed or random.randint(...)` rend la seed 0 non déterministe. La seed 0 doit devenir une valeur valide avec une règle explicite. Conserver les seeds non nulles de la référence ; attribuer une nouvelle version à la correction.

Le fichier de statistiques actuel n’encode pas dans sa clé les paramètres `frequency_mult` et `drop_water_pct` ayant servi à son calcul. Un changement de profil peut donc réutiliser des tables inadaptées. Générer hors ligne un bundle de statistiques par profil et le référencer par hash. Hash SHA-256 du fichier local audité : `4c785eb3be9c0e5e0647d07b5f175ce879f667dd61669e863d596a17e45324dd`.

## 6. Modèle spatial, zoom et aperçu continental

### 6.1. Coordonnées et tuiles canoniques

L’identité d’une position utilise des coordonnées entières signées en pixels natifs de 30 m. Le navigateur transporte ces coordonnées sous forme d’entiers sûrs ou de chaînes ; la seed uint64 est toujours une chaîne décimale, pour éviter les pertes de précision JavaScript. Le rendu emploie une origine locale et des offsets flottants afin de ne pas perdre des mètres loin de l’origine.

Les régions sont demi-ouvertes `[x0,x1) × [y0,y1)`. Pour les coordonnées négatives, la division des indices doit être une division vers moins l’infini ; la division entière C++ vers zéro ne reproduit pas Python. Les centres des pixels, les offsets d’interpolation et `align_corners=False` doivent être explicites.

La tuile de livraison reste initialement 256². Les unités de calcul restent les fenêtres natives 64/64/512 des réseaux. Un décodeur 512² peut alimenter plusieurs tuiles de livraison ; il ne doit pas être exécuté séparément pour chaque enfant. Les fenêtres de fusion et les halos d’assemblage sont déduits du graphe et des kernels, plutôt qu’un halo uniforme de 24 pixels appliqué à tous les LOD.

L’actuel RNG mélange des indices sur 32 bits, et FastNoiseLite échantillonne des coordonnées flottantes. Le domaine numérique valide et les éventuelles répétitions doivent être mesurés. Le stockage int64 ne transforme pas seul ce système en monde mathématiquement infini à précision arbitraire. Un nouveau RNG ou bruit pour coordonnées extrêmes serait une nouvelle version de génération.

### 6.2. Choix du niveau de détail

Calculer le LOD à partir des mètres par **pixel physique de rendu**, avec prise en compte du device pixel ratio et d’une résolution de rendu éventuellement réduite. Base : `lod = floor(log2(mpp / 30))`, borné au minimum à zéro. Employer de l’hystérésis autour des frontières, par exemple 15 %, pour éviter les allers-retours. Ce pourcentage est un paramètre initial à tester.

Déplacer et zoomer immédiatement les textures déjà présentes. Choisir ensuite un niveau disponible : relief final, approximation latente, coarse neuronal, puis aperçu synthétique. Une région remplacée garde son ancienne texture jusqu’à disponibilité de la nouvelle. Un fondu court stabilise la perception ; il ne modifie pas les données physiques sauvegardées.

Un LOD d’affichage n’est pas un autre monde. Les mips finaux proviennent d’une réduction spatialement alignée du même champ d’altitude. Interpoler en signed-sqrt puis remettre en mètres et moyenner des hauteurs en mètres sont deux opérations différentes : les aperçus doivent être explicitement marqués approximatifs et remplacés par la réduction canonique une fois disponible.

### 6.3. Pourquoi le fort dézoom coûte cher

Une tuile 256² au LOD L couvre un côté de `256 × 30 × 2^L` mètres. À LOD 9, elle couvre **3 932 km de côté**. `sample_field()` demande toutes les cellules coarse du rectangle, alors que l’image finale n’a que 256² pixels. Les dimensions du champ coarse croissent avec `2^L`, donc sa surface approximativement avec `4^L`. Le halo augmente encore la région.

Faire une requête sparse aux seuls points visibles évite certains pixels intermédiaires, mais ne suffit pas si ces points touchent presque toutes les fenêtres coarse. Le réseau conserve une fenêtre de contexte fixe. Une couverture neuronale exacte d’une surface continentale froide reste un volume de calcul réel.

**Solution en trois couches :**

1. Aperçu immédiat des cinq champs synthétiques, échantillonnés à la résolution de l’écran. Son coût doit dépendre de l’écran et du budget de travail, pas du nombre de pixels natifs sous-jacents. Le relief affiché est provisoire.
2. Cache neuronal macro de blocs coarse canoniques, calculés en arrière-plan sous quota. Pour le monde publié par défaut, préparer l’aperçu et les zones de départ hors ligne ; servir leurs mips via CDN. Une seed arbitraire obtient d’abord l’aperçu synthétique.
3. Affinement neuronal des zones que la caméra observe, puis des zones prédites. Les requêtes de grande surface sont divisées en tâches bornées ; aucun travail macro ne bloque durablement une tuile détaillée urgente.

Ne jamais recalculer tout le coarse pour une vue générale à chaque zoom. Réutiliser les blocs déjà produits et leur pyramide. Si le besoin devient « une planète froide exacte en quelques centaines de millisecondes pour toute seed », il faudra tester une distillation du coarse ou entraîner un modèle macro supplémentaire. C’est un chantier de modèle, distinct du moteur.

## 7. Ordonnanceur de régions et graphe partagé

### 7.1. Graphe de dépendances

Une vue génère des besoins de sortie. Chaque besoin s’étend en dépendances : tuile finale → assemblage/halo → fenêtres decoder → fenêtres latentes passées 1 et 2 → fenêtres coarse → conditionnement et bruit. Une demande d’aperçu s’arrête plus tôt dans ce graphe.

Une fenêtre est identifiée par `(world_profile, seed, stage, step, wx, wy, numeric_profile)`. Deux tuiles ou deux clients demandant cette fenêtre partagent le même travail. Chaque fenêtre à une étape donnée peut rejoindre un batch compatible dès que ses dépendances sont complètes.

Les barrières restent locales : une fenêtre de la deuxième passe latente attend toutes les contributions de son voisinage requis à la première passe, mais n’attend pas le reste de la vue. Son empreinte de dépendances est calculée avec les strides et l’opération de gather exacts. Cela permet au premier decoder prêt de démarrer pendant que d’autres régions poursuivent le coarse.

Le résultat de fusion est une somme pondérée `A / B`, avec `A = Σ(weight × output)` et `B = Σ(weight)`. Un pixel n’est définitif que lorsque toutes les contributions requises à cette étape sont disponibles. Le moteur doit distinguer accumulation partielle, fenêtre terminée et bloc finalisé. Changer l’ordre des requêtes ne doit pas ajouter deux fois une contribution ou publier trop tôt une frontière.

```text
Caméra et abonnement
        |
        v
Planificateur de couverture -> aperçu synthétique immédiat
        |
        v
DAG partagé : coarse -> latent étape 1 -> latent étape 2 -> decoder
        |                    |                  |              |
        +--------------------+------------------+--------------+
                             |
                     Assemblage + climat
                             |
                     Mips / transport / textures
```

### 7.2. Priorités

Les classes suivantes sont ordonnées ; à l’intérieur d’une classe, on tient compte du délai et de la valeur du résultat :

| Classe | Travail | Règle |
| --- | --- | --- |
| P0 | Couvrir une zone visible sans image | Aperçu ou cache immédiatement ; éviter un trou |
| P1 | Dépendance débloquant du relief final visible | Hérite de la priorité de ses consommateurs |
| P2 | Affiner une zone visible déjà approximative | Centre et surface utile, délai et gain de qualité |
| P3 | Précharger dans la direction probable de la caméra | Budget réduit et annulation agressive |
| P4 | Remplir un cache macro ou final hors vue | Uniquement avec marge ; suspendu dès qu’un besoin visible arrive |

Utilité proposée pour comparer des tâches prêtes d’une même classe : `surface_visible × gain_qualité × probabilité_utilisation / coût_estimé`. Ajouter un âge plafonné pour éviter la famine. Le coût vient de moyennes mesurées par stage, batch et backend ; il doit inclure les dépendances manquantes sur le chemin critique. Une dépendance partagée hérite de la meilleure priorité active, avec prise en compte de ses différents consommateurs.

Les paramètres exacts seront ajustés sur des trajectoires enregistrées. Un score seul ne remplace pas les classes : le moteur doit privilégier la couverture d’un trou visible avant une région spéculative très bon marché.

### 7.3. Préchargement et backpressure

Estimer vitesse et accélération de la caméra par moyenne glissante. Précharger un corridor dans sa direction, et un petit anneau de sécurité autour de la vue. La distance se déduit de `vitesse × latence_p95` avec marge, limitée par le budget mémoire et le volume de travail. Un changement brusque de direction raccourcit le corridor.

Envoyer la caméra au moteur à fréquence bornée, par exemple 20 Hz initialement, plutôt que patienter 180 ms à chaque mouvement. L’ordonnanceur fusionne ces mises à jour. Pendant un zoom rapide, promouvoir la couverture au LOD disponible ; différer les détails qui ne seront visibles qu’un instant. Ne pas déclencher une grille complète de tous les niveaux traversés.

Chaque vue a un budget de travail admis et un nombre maximal de dépendances actives. Si la demande dépasse le débit, réduire l’affinement ou le préchargement et conserver l’aperçu. Une file de demandes infinie ne produit pas du temps réel.

### 7.4. Annulation et résultats tardifs

Une mise à jour de caméra porte un `view_epoch` croissant. Le moteur compare les besoins de la nouvelle vue à l’ancienne, puis ajoute/enlève des références sur les nœuds du graphe. Une fermeture d’onglet ou de session libère les abonnements.

Un nœud en attente sans consommateur est retiré. Une tâche GPU déjà lancée termine son quantum ; son résultat peut être gardé si son coût de réutilisation le justifie, mais il ne doit pas être publié comme résultat de la nouvelle vue. Une tâche coarse peut être suspendue entre étapes internes seulement si son état complet de scheduler est sauvegardé ; le premier port peut simplement finir le petit batch courant. Le GPU n’est pas arbitrairement préempté au milieu d’un kernel.

```text
update_view(view, epoch, camera):
    roots = choose_visible_and_predicted_outputs(camera, budgets)
    reconcile_subscriptions(view, epoch, roots)
    propagate_priority_and_refcounts()
    enqueue_ready_dependencies()

tick():
    retire_completed_gpu_jobs_and_commit_complete_outputs()
    remove_unreferenced_queued_jobs()
    group_ready_jobs_by_backend_model_step_shape_and_precision()
    batch = choose_batch_under_deadline_memory_and_fairness_limits()
    submit_async(batch)
    publish_outputs_to_current_subscribers()
```

Le serveur reste l’autorité d’ordonnancement des travaux partagés. Le client peut prédire la couverture et trier les téléchargements ; il n’impose pas des listes de petits jobs neuronaux qui contourneraient la déduplication globale.

## 8. Cache à plusieurs niveaux

### 8.1. Identité du monde et du calcul

Séparer trois profils :

| Profil | Contenu | Invalidation |
| --- | --- | --- |
| Génération | Poids des trois réseaux, configuration, tables de données, RNG, transformation de conditionnement, schedule, topologie, unités | Nouveau monde logique dès que le contenu change |
| Numérique | Backend, précision, fusions, kernels et règle d’accumulation | Autre variante numérique ; validation d’équivalence obligatoire |
| Rendu | Palette, hillshade, biomes, exposition et style | Recalcul des textures de style ; pas d’inférence neuronale |

Clé physique : `H(generation_profile) / seed_u64 / H(numeric_profile) / stage / step / grid_version / wx / wy`. Clé de sortie : profil physique + coordonnées + résolution + format. Clé d’image stylée : clé de sortie + hash de rendu. Hashes des contenus et versions d’algorithmes, pas uniquement une chaîne `natural-v1` incrémentée manuellement.

### 8.2. Couches et éviction

| Cache | Contenu | Politique |
| --- | --- | --- |
| GPU chaud | Fenêtres coarse/latent, contributions utiles, résidus et textures proches | Épingler dépendances de la vue ; éviction coût/bénéfice sous budget |
| RAM | Blocs finalisés, buffers de transfert, fenêtres proches évincées du GPU | Cache borné en octets ; pools réutilisables |
| Disque natif | Relief final, climats, mips et éventuellement fenêtres chères | Écriture atomique, index, checksum, quota configurable |
| Navigateur | Poids et données réutilisables via Cache Storage/IndexedDB ou OPFS selon support | Quota estimé ; cache opportuniste, reconstructible |
| CDN / service | Contenu immuable de mondes partagés | Déduplication globale et URLs adressées par contenu |

Un cache de fenêtres coûteuses améliore l’exploration contiguë. Un cache de sorties finales améliore les revisites. Il faut les deux. Le cache GPU ne stocke pas le monde entier à 30 m ; il suit le voisinage et ses dépendances.

Évincer une sortie calculée implique de maintenir un état de disponibilité cohérent. Un marqueur « contribution déjà ajoutée » survivant sans la donnée correspondante peut produire un trou ; un marqueur supprimé sans retirer son accumulation peut provoquer un double comptage. La proposition privilégie des fenêtres immuables, avec reconstruction déterministe de blocs fusionnés, et commit atomique de l’état complet. Les contributions d’un bloc finalisé sont additionnées dans un ordre canonique si la stabilité numérique le demande.

Débuter avec une politique simple : vue active épinglée, dépendances coûteuses proches, puis LRU pondéré par coût de régénération et probabilité de retour. Ajouter une politique plus complexe seulement si les traces montrent un problème. Les deux mondes actuels ne justifient pas de dupliquer les poids.

### 8.3. Budget mémoire initial pour la 3090

Les trois fichiers de poids installés totalisent environ **1,14 Go sur disque** : base 1,015 Go, decoder 112 Mo, coarse 11 Mo. Leur taille GPU dépend du type et des transformations du backend. Les activations et le workspace du décodeur peuvent dominer quand le batch augmente.

Budget de départ à mesurer, en Gio :

| Poste | Enveloppe proposée |
| --- | --- |
| Poids et moteurs | Jusqu’à 2 |
| Activations, workspace et buffers de batch | Jusqu’à 6 |
| Fenêtres et champs GPU chauds | Jusqu’à 6 |
| Rendu, sorties et staging GPU | Jusqu’à 1 |
| Marge pour bureau, fragmentation et changements de batch | Au moins 4, en ajustant selon mémoire réellement libre |

Ces enveloppes ne sont pas une allocation obligatoire de 19 Gio. Le moteur détecte la mémoire disponible, mesure ses pics, puis agrandit le cache sans sacrifier la marge. Garder les trois modèles résidents sur la 3090 ; envisager le chargement par stage uniquement sur les petits budgets.

Les enveloppes sont propres à chaque device et backend. En WebGPU, le budget reste conservateur, car les limites exposées ne donnent pas la VRAM réellement libre. Dans l’agent natif, on calibre aussi la 4090 et la 5090 ; leurs caches et workspaces sont indépendants. Deux GPU ne constituent pas un unique pool mémoire pour une allocation de modèle.

À 30 m, une région 1 000 × 500 km contient environ **555,6 millions de pixels**, soit 2,22 Go pour une seule altitude float32. Altitude + cinq climats float32 représenteraient 13,33 Go, avant mips et contexte. À l’inverse, son contexte coarse représente environ 130 × 65 cellules intérieures à 7 680 m/cellule ; les fenêtres et leurs bords ajoutent du calcul. Ce contraste motive un cache hiérarchique et l’affichage progressif.

## 9. Rendu et transport

### 9.1. Rendre des champs physiques

Livrer altitude et caractéristiques climatiques réutilisables. Le navigateur calcule hillshade, couleurs, biomes et transitions de LOD en shaders. Les mips d’altitude sont construits sur une grille mondiale alignée ; un bord d’une tuile reçoit des voisins cohérents, jamais une extrapolation différente pour chaque PNG.

Conserver float32 pour la référence et les blocs nécessaires à la stabilité de l’assemblage. Pour le transport et les textures d’affichage, tester une altitude quantifiée uint16 avec échelle globale par profil. Sur une étendue de 20 000 m, son pas serait environ 0,305 m. Définir les bornes réelles, détecter les dépassements et employer une extension plutôt que clipper silencieusement. Les textures filtrent des mètres reconstitués ; les données de simulation restent séparées si elles demandent une précision supérieure.

Les quatre champs climatiques peuvent être transportés sur une grille macro commune et combinés au relief pour la correction thermique locale. Cela évite cinq rasters natifs redondants. Comparer ce chemin à `world.get(with_climate=True)` avant de l’activer. Le PNG reste utile pour aperçus, export et clients simples ; il ne doit plus être l’unique stockage physique.

### 9.2. Calcul GPU et copie finale

Garder les résultats intermédiaires sur GPU jusqu’à l’assemblage final. Le worker CPU prépare le prochain batch pendant que le GPU exécute le précédent. Deux ou trois slots réutilisables suffisent au départ ; des événements signalent la disponibilité des sorties. Les copies vers la RAM utilisent des buffers épinglés quand le backend natif le permet.

Pour un serveur distant ou local desservant une page web, une sortie GPU doit encore être transférée et transportée : l’interop CUDA n’est pas accessible au navigateur. Pour l’inférence WebGPU locale, viser le partage du même `GPUDevice` et de `GPUBuffer` entre inference et rendu, selon les interfaces effectivement disponibles. Éviter un `getData()` complet à chaque étape. [Sorties et buffers GPU d’ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

### 9.3. Protocole proposé

Le client ouvre un flux WebSocket de contrôle et transmet sa vue. Les métadonnées sont en JSON ; les données volumineuses sont des messages binaires bornés ou des ressources HTTP immuables. Séparer ce choix de transport du moteur de calcul.

```json
{
  "type": "view.update",
  "session": "...",
  "view_id": "main",
  "epoch": 124,
  "world_id": "sha256:...",
  "seed": "42",
  "center_native": ["-1536", "3584"],
  "meters_per_device_pixel": 30,
  "viewport_device_pixels": [1920, 1080],
  "velocity_m_s": [6000, 0],
  "wanted_layers": ["height", "climate", "biome"],
  "max_pending_bytes": 16777216
}
```

Réponses : `world.manifest`, `capabilities`, `tile.preview`, `tile.ready`, `view.progress`, `budget.changed`, `error`. `tile.ready` contient profil numérique, coordonnées, résolution, niveau de qualité, encodage, unités, longueur, checksum, epoch et identifiant de contenu. Le client rejette les résultats d’un ancien monde ; une tuile tardive du même monde peut être gardée sans la rendre si elle est hors vue.

Endpoints immuables proposés : `/models/<hash>/<part>`, `/worlds/<hash>/manifest.json`, `/tiles/<content-key>`. Le téléchargement en HTTP/CDN est indépendant de l’ordre des tâches. Le contrôle de vue permet d’annuler la génération réellement ; fermer une requête HTTP seule ne remplit pas ce rôle.

## 10. Agent local natif optionnel et open source

### 10.1. Choix C++ et Rust

| Critère | C++20 | Rust |
| --- | --- | --- |
| TensorRT et kernels CUDA | API directe et outillage standard | FFI C/C++, wrapper et gestion des streams à maintenir |
| Portage du RNG et FastNoiseLite | Port naturel des types et de la bibliothèque C++ | Faisable, avec validation d’équivalence des implémentations |
| WASM | Sous-ensemble portable via Emscripten | Bon support avec interface JS/WASM explicite |
| Ordonnanceur concurrent | Discipline de propriété et tests nécessaires | Propriété mémoire plus fortement vérifiée par le langage |
| Risque du prototype natif de comparaison | Moins d’interfaces supplémentaires | Plus d’intégration avant de mesurer le backend dominant |

**Décision : C++20 pour le noyau portable et l’agent natif optionnel**, C ABI stable pour les intégrations. La priorité produit reste l’agent WebGPU ; le choix de langage du noyau ne doit pas retarder la preuve des réseaux et du pipeline navigateur. Rust reste une option d’enveloppe ou de client, sans imposer deux noyaux d’ordonnancement. Une implémentation Rust + backend CUDA en C++ est viable, avec une frontière de synchronisation et de propriété supplémentaire.

Le noyau portable ne dépend ni de CUDA, ni de TensorRT, ni du serveur HTTP. Il expose des jobs et reçoit des résultats ; les backends ont leurs implémentations natives ou JavaScript/WebGPU. Compiler le noyau en WASM ne compile pas les bibliothèques GPU natives.

### 10.2. Modules et contrats

```text
engine/
  core/          coordonnées, profils, RNG, planification, DAG, caches
  cpu_ref/       fusion, interpolation, scheduler et climat de référence
  cuda/          kernels d'assemblage et pools CUDA
  backends/      ORT CUDA, TensorRT, futur adaptateur WebGPU
  api/           C ABI, types sérialisables, événements
  service/       vues, sessions, transport, quotas
  tools/         export, engine build, replay, comparaison numérique
web/
  worker/        noyau WASM + adaptateur ORT WebGPU
  viewer/        caméra, atlas de textures, shaders, couches climatiques
```

Contrat principal de backend : `submit(stage, step, shape_bucket, inputs, output_slot) -> completion_event`. Les entrées et sorties portent un emplacement explicite CPU/GPU et un propriétaire. Un résultat devient visible après son événement de fin ; aucune mémoire n’est recyclée avant que le dernier consommateur ait fini.

Interface C illustrative :

```c
td_status td_create(const td_config*, td_engine**);
td_status td_open_world(td_engine*, const td_world_manifest*, td_world**);
td_status td_update_view(td_world*, uint64_t view_id, const td_view*);
td_status td_poll(td_engine*, td_event*, uint32_t capacity, uint32_t* count);
td_status td_copy_tile(td_engine*, td_tile_handle, void* dst, size_t bytes);
void      td_release_tile(td_engine*, td_tile_handle);
void      td_close_view(td_world*, uint64_t view_id);
void      td_destroy(td_engine*);
```

Les handles sont opaques ; versions de structures, tailles, unités et codes d’erreur sont documentés. Une page web utilise le protocole du service natif ou l’adaptateur WebGPU, sans dépendre d’un pointeur CUDA.

### 10.3. Ce qu’on reprend de vLLM

L’organisation en API, moteur d’ordonnancement et workers d’accélérateur est un bon précédent. [Architecture vLLM](https://docs.vllm.ai/en/latest/design/arch_overview/)

L’analogie utile est le **batching continu de travaux compatibles**, l’admission sous budget mémoire, le partage de résultats déjà calculés et la séparation de la boucle de service et du calcul GPU. Ici, les objets réutilisés sont des fenêtres spatiales et des dépendances, pas un KV cache de tokens. L’attention du U-Net n’offre pas le cache autoregressif qui explique une partie des optimisations LLM. Le moteur reste spécialisé pour ce pipeline ; l’extension à deux GPU utilise des workers de régions indépendantes, sans découper les couches d’un réseau entre les cartes.

### 10.4. Backend de référence ONNX Runtime CUDA

Exporter les trois réseaux, valider leurs entrées réelles, puis porter le pipeline autour d’eux. Lier entrées et sorties GPU avec I/O Binding ; instrumenter tout fallback CPU. Garder un stream de calcul contrôlé et des buffers préalloués. Les copies implicites d’ORT faussent facilement l’interprétation d’un benchmark. [I/O Binding officiel](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)

Ce backend fournit une référence C++ et une voie de diagnostic. Il ne garantit pas un gain face à PyTorch ou à WebGPU. Comparer les agents sur les mêmes vues, poids, qualité et états de cache. Inclure les transferts GPU → RAM, le transport local et l’upload des textures dans le temps de l’agent natif : un forward plus rapide ne suffit pas si l’image arrive plus tard à l’écran.

### 10.5. Backend TensorRT et kernels spécialisés

Utiliser TensorRT d’abord comme compilateur de réseaux, pas comme prétexte à réimplémenter les convolutions. Créer des profils de batch compatibles avec la mémoire et les délais. Tester FP16 et BF16 face à la référence BF16 actuelle ; garder les opérations sensibles de normalisation et d’accumulation en FP32 quand nécessaire.

Optimisations candidates :

1. Pré-normaliser les poids magnitude preserving statiques une fois à l’export ; vérifier le pliage des gains. Le code actuel peut recalculer des normalisations de poids immuables pendant les forwards.
2. Fusionner les opérations élémentaires de préconditionnement, concaténation et sortie de scheduler ; éviter les créations CPU et extractions scalaires synchronisantes.
3. Batch coarse : plusieurs fenêtres indépendantes avancent ensemble dans leurs 20 étapes internes. L’état multistep de chaque fenêtre reste distinct.
4. Batch latent : mélanger les fenêtres prêtes d’une même passe, y compris issues de plusieurs sorties, sans franchir une dépendance de fusion.
5. Batch decoder : tester plusieurs fenêtres 512² et mesurer activations, temps d’exécution et délai d’une tuile urgente.
6. Kernels CUDA ciblés : somme pondérée, division, gathers de conditionnement, interpolation, Laplacian denoising, signed-square et correction climatique.
7. Capture CUDA Graph des séquences à formes et adresses stables, après validation des sessions et de leur durée de vie.

Buckets à explorer, et non valeurs imposées : coarse et latent `1,2,4,8,16,32` ; decoder `1,2,4,8`, en arrêtant avant le dépassement mémoire ou de délai. Attente de remplissage initiale 2–5 ms au plus pour un besoin urgent ; batch partiel accepté. Si un batch dure trop longtemps, réduire sa taille, même si le débit maximal baisse. La préemption intervient aux frontières de jobs.

Les CUDA Graphs exigent des conditions propres au backend : formes, allocations et chemins compatibles. ORT CUDA impose notamment des contraintes sur le placement des nœuds, les adresses et la concurrence de session. Ne pas supposer qu’on peut capturer une fonction Python ou un DAG dynamique entier tel quel. [Contraintes du provider CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

### 10.6. Déploiement Linux et Windows

Builds séparés Windows x64/MSVC et Linux x64, avec versions épinglées d’ORT, CUDA, cuDNN et TensorRT. L’indication « CUDA 13.3 » du pilote ne remplace pas la version du runtime utilisé ; l’installation actuelle de PyTorch utilise CUDA 12.8. Les builds natifs doivent valider Ampere/3090, Ada/4090 et Blackwell/5090, dont les capacités CUDA sont respectivement 8.6, 8.9 et 12.0. Ne pas supposer qu’un ancien binaire contenant seulement du code Ampere peut exploiter correctement la 5090. [Architectures CUDA officielles](https://developer.nvidia.com/cuda/gpus)

Un plan TensorRT sérialisé n’est pas automatiquement portable entre Linux et Windows, ni entre toutes les cartes. Le construire et le cacher pour la plateforme, le modèle, la version du runtime, les profils de formes et les options de précision. Les modes de compatibilité ont leurs propres contraintes. [Matrice TensorRT officielle](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

L’export ONNX reste l’artefact de modèle portable. Les plans optimisés sont des artefacts dérivés. Leur construction doit se faire à l’installation ou être fournie pour une combinaison validée, et non bloquer une première navigation urgente.

L’auteur propose aussi un [port Minecraft en Java/ONNX Runtime](https://github.com/xandergos/terrain-diffusion-mc), avec variantes CUDA et DirectML et un explorateur web. C’est une référence à auditer pour le portage du RNG, des fenêtres et des biomes. Ce n’est pas déjà le moteur C++/WASM proposé ; son déchargement des modèles pour réduire la mémoire n’est pas le choix par défaut de la 3090.

### 10.7. Distribution open source de l’agent

Distribuer le code du noyau, du service local, des adaptateurs, des kernels et des outils de validation dans un dépôt public, avec build Linux/Windows reproductible et protocole documenté. Une licence permissive telle que MIT est la recommandation pour le nouveau code ; conserver les notices et obligations des portions upstream et des dépendances. Le choix de licence sera matérialisé lors du packaging, sans modifier les licences des composants tiers.

Le dépôt open source ne rend pas TensorRT ou CUDA open source : ce sont des dépendances optionnelles soumises à leurs conditions. Fournir les sources et scripts de build ; distribuer les runtimes ou télécharger des composants uniquement dans les conditions autorisées. L’agent WebGPU garde sa propre voie de calcul et ne dépend pas de TensorRT pour fonctionner.

L’agent local expose les mêmes profils, données et événements que l’agent navigateur. Par défaut il écoute sur loopback ; un appairage par jeton et origine autorisée permet au site de l’utiliser. Une exposition sur le réseau de la seconde machine est une option configurée explicitement. Les téléchargements de modèles, fichiers de cache et mesures de performance sont visibles et contrôlables ; aucune dépendance à un service privé n’est nécessaire pour exécuter l’agent.

### 10.8. RTX 5090 + RTX 4090 : adaptation et deux workers

Énumérer les GPU natifs avec identifiants stables, capacités, mémoire disponible et latence/débit mesurés par stage. L’objectif est de choisir le device donnant le meilleur délai de sortie, avec les caches et la charge actuels. La 5090 est le premier candidat sur cette machine, puis la calibration tranche. La 4090 peut être préférable pour un job dont toutes les dépendances sont déjà sur elle.

Politiques exposées : `auto`, `best_single`, `balanced_multi` et `device_id` explicite. `auto` commence sur le GPU le plus rapide mesuré, puis active le second seulement si un replay démontre un gain utile sans régression de délai. Chaque GPU a ses modèles résidents, ses plans TensorRT, ses pools et ses caches ; un worker en possède l’exécution.

| Travail | Répartition initiale à tester |
| --- | --- |
| Premier détail visible et dépendances sur son chemin critique | GPU offrant la meilleure date de fin estimée ; 5090 candidate prioritaire |
| Régions visibles indépendantes en attente | Dispatch entre workers, avec affinité de cache et répartition proportionnelle au débit |
| Corridor de préchargement et contexte macro | 4090 tant que cela ne ralentit pas la publication utile ; également 5090 si libre |
| Requête isolée peu coûteuse | Un seul GPU ; éviter le coût de coordination |
| Plusieurs vues ou seeds | Affectation de régions/sessions aux workers, puis équilibrage mesuré |

Privilégier des branches spatiales complètes sur un GPU plutôt que coarse sur une carte et decoder sur l’autre pour chaque tuile. Une dépendance globale reste dédupliquée ; son propriétaire est connu. Un transfert inter-GPU n’est entrepris que s’il coûte moins qu’un autre placement ou une duplication contrôlée. Détecter le support P2P à l’exécution et mesurer les transferts ; le chemin RAM de staging reste possible, sans présumer d’une interconnexion rapide.

Date de fin estimée : `attente_device + dépendances_manquantes + coût_stage_batch + transferts + livraison`. Le scheduler compare ce coût sur chaque device, garde de l’affinité pour les voisins et évite des migrations incessantes. Un changement de device ne change ni seed, ni sources, ni qualité du modèle ; les profils numériques des deux architectures doivent être validés ensemble avant de fusionner leurs résultats dans un même monde.

Le mode double GPU doit battre le meilleur mode mono-GPU sur des scénarios définis. Il vise surtout plus de débit et de couverture ; un seul chemin de dépendances peut rester limité par sa carte. Le gain n’est ni automatiquement ×2, ni proportionnel à la somme des TFLOPS. La mémoire n’est pas additionnée en un pool unique de 56 Go.

## 11. Site public : agent WebGPU prioritaire, WASM et GPU des visiteurs

### 11.1. Trois modes de distribution

| Mode | Où tourne le réseau ? | GPU utilisés | Usage prioritaire |
| --- | --- | --- | --- |
| Agent WebGPU principal | Dans un worker du navigateur | NVIDIA, AMD, Intel, Apple selon navigateur/pilote/capacités | Produit prioritaire, sans installation native |
| Agent natif open source facultatif | Sur l’ordinateur de l’utilisateur, via service local | CUDA/TensorRT ; 3090, 4090, 5090 ; deux GPU si gain mesuré | Gain de performance démontré et contrôle des devices |
| Service distant facultatif | Agent natif sur une autre machine ou service hébergé | Notamment la machine 5090 + 4090 | Déporter le calcul, appareils limités, données canoniques si nécessaire |

Le site publié doit fonctionner avec son agent WebGPU sans installer l’agent natif et sans imposer un serveur de génération. L’agent local est une option explicite. Les accès d’un site HTTPS à un service local subissent les politiques du navigateur ; définir un appairage avec origine autorisée et tester ce chemin sur les navigateurs cibles. Ne pas présumer qu’un `fetch` vers localhost sera accepté partout.

Le premier moteur natif optimisé couvre NVIDIA sous Linux/Windows. Pour le navigateur, WebGPU constitue le chemin multi-fournisseurs. Des backends natifs AMD/Intel peuvent venir ensuite : WinML/ORT sous Windows et une implémentation WebGPU native ou autre provider validé sous Linux. DirectML reste disponible, mais sa documentation indique que le développement de nouvelles fonctions passe vers WinML. [Documentation ORT DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

### 11.2. Architecture WASM/WebGPU

Compiler en WASM la logique portable : profils, coordonnées, DAG, budgets, cache index, RNG de référence et opérations CPU utiles. L’adaptateur JavaScript gère le device, le chargement ONNX, les sessions ORT et les événements GPU. Les kernels d’assemblage pourront être en WGSL. Les convolutions passent initialement par ORT WebGPU, plutôt que par une nouvelle bibliothèque de calcul matriciel.

WASM CPU seul sert au contrôle et au fallback limité ; il n’est pas le chemin de calcul neuronal temps réel visé. WebGPU nécessite un contexte sécurisé et expose des fonctionnalités et limites qu’il faut interroger. FP16 en shaders dépend de `shader-f16` ; la référence BF16 CUDA n’est pas un format WebGPU directement interchangeable. [Spécification WebGPU](https://gpuweb.github.io/gpuweb/) et [WGSL](https://gpuweb.github.io/gpuweb/wgsl/)

Exporter une variante ONNX adaptée, tester chaque opérateur et son placement, les types et les formes réelles. La compatibilité ONNX générale ne suffit pas à prouver que tout le graphe tourne sur WebGPU. Éviter un fallback WASM opérateur par opérateur qui multiplierait les transferts. Accepter le profil FP16/FP32 navigateur uniquement après comparaison physique et visuelle de toute la chaîne.

Partager les sorties GPU avec le rendu quand ORT et l’application permettent un device compatible ; sinon mesurer et borner la copie supplémentaire. Les résultats de fusion, le post-traitement et les textures doivent rester GPU autant que possible. La vie des `GPUBuffer` et des sorties ORT est explicite ; libérer une tuile hors cache, traiter `device.lost`, et recréer les ressources depuis un état portable. [Guide WebGPU d’ORT](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

### 11.3. Détection et adaptation

Au démarrage : obtenir un adapter, ses features et limits ; demander uniquement les capacités nécessaires ; tester la création des sessions, une fenêtre par stage et les buffers de sortie ; mesurer temps et pics estimés. Les limites de buffer ne représentent pas un compteur fiable de VRAM libre. Une tentative d’allocation contrôlée et un budget conservateur sont nécessaires.

| Profil détecté | Politique initiale à valider |
| --- | --- |
| Desktop WebGPU avec FP16, modèles et fenêtres acceptés | Inférence locale complète ; batch selon micro-benchmark ; cache dynamique |
| WebGPU utilisable avec petit budget ou FP32 seulement | Batch 1, rendu plus léger, modèles chargés par stage si nécessaire ; comparer au distant |
| GPU intégré/mobile trop lent ou mémoire insuffisante | Aperçu et rendu locaux ; génération fine distante si disponible |
| Pas de WebGPU | Textures servies par CDN/service et rendu WebGL2/Canvas ; aucun blocage de navigation |

Ce tableau est une politique, pas une liste de cartes garanties compatibles. Valider au minimum des NVIDIA desktop, AMD desktop, Intel intégré, Apple Silicon et un mobile, puis définir les appareils effectivement supportés. Limiter le temps des dispatchs pour préserver le rendu et éviter les jobs monolithiques. Suspendre le travail spéculatif lorsque l’onglet est masqué ; reprendre sans invalider le monde.

Le modèle principal représente déjà environ 1 Go dans son fichier de poids actuel. La taille finale ONNX, sa représentation de précision et les caches runtime doivent être mesurés. Afficher une estimation de téléchargement et stocker les poids versionnés. Précharger un aperçu léger ; ne pas rendre l’arrivée des gros poids nécessaire à l’affichage initial.

### 11.4. Adaptation continue : meilleur GPU, plus de vitesse

Au premier démarrage, calibrer chaque réseau sur quelques buckets avec entrées réalistes. Conserver la matrice `stage × batch × précision → latence, débit, mémoire estimée`, associée au backend et à ses versions. Les identifiants matériels éventuellement exposés aident à retrouver un profil, mais la mesure locale décide ; ne pas imposer un batch 16 universel aux visiteurs.

Pendant la navigation, estimer la durée des tâches par moyenne glissante et surveiller les délais de rendu. Augmenter le batch lorsque cela améliore le débit sans dépasser la durée maximale d’un quantum ; le réduire en cas de contention ou de tâche urgente. Augmenter le cache et le corridor quand la marge mémoire et le débit le permettent. Réduire d’abord la spéculation si la charge monte. Une calibration courte, un budget prudent et une hystérésis empêchent les oscillations du tuner.

| Paramètre adaptatif | GPU plus rapide / marge supérieure | GPU limité ou occupé |
| --- | --- | --- |
| Batch par stage | Bucket plus efficace, sous plafond de délai | Petit bucket ou batch partiel |
| Couverture finale | Davantage de régions affinées par seconde | Aperçu conservé plus longtemps |
| Préchargement | Corridor plus large, revisites mieux anticipées | Corridor court, spéculation suspendue |
| Cache GPU | Plus de fenêtres réutilisables si mémoire disponible | Fenêtres actives épinglées, éviction plus tôt |
| Rendu | Résolution physique cible si budget tenu | Résolution de rendu réduite si nécessaire, données natives inchangées |

Les paramètres de génération restent fixes : toutes les sources, les 20 étapes internes coarse de référence et les deux passes latentes sont conservées. Plus de puissance doit principalement donner moins d’attente et plus de débit, pas imposer un monde différent. La précision du calcul ne change pas pendant une vue sans profil numérique validé.

Demander un adapter WebGPU avec `powerPreference: "high-performance"` dans le mode performance. Cette option reste une indication au navigateur ; elle ne désigne pas une 5090 par UUID. L’API standard ne fournit pas une énumération arbitraire de toutes les cartes pour garantir la répartition 5090 + 4090. Le prototype navigateur utilise un adapter/device ; le contrôle explicite et le double GPU sont assurés par l’agent natif. [Choix d’adaptateur WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/GPU/requestAdapter)

Sur chaque machine, comparer WebGPU et natif avec le même replay et l’image réellement affichée : démarrage, p95 d’affinement, pixels finaux utiles/s, frame time et mémoire. Recommander le natif seulement avec un gain supérieur au bruit de mesure ; un seuil initial de 15 % sur le débit, sans régression sensible de latence, peut servir de porte de décision. Ce seuil est une politique proposée, pas une performance observée.

### 11.5. Utiliser l’autre machine

L’agent installé sur la machine 5090 + 4090 peut être contacté par la page de la machine 3090 via un endpoint configuré. Il expose les mêmes abonnements de vue, epochs et tuiles que le service local. L’accès se fait dans le contexte réseau autorisé, avec appairage, origine autorisée et transport adapté au site HTTPS. Aucun code du navigateur de la première machine ne peut utiliser ces GPU sans cet agent et cette liaison.

Comparer le délai complet à la génération WebGPU locale : réseau, serialisation et upload des textures peuvent absorber le gain de calcul. Les caches des deux machines et les clients partagent les identités de contenu, mais pas des pointeurs GPU. La sélection du mode distant reste explicite ; une déconnexion conserve l’image et revient à la génération locale lorsque disponible, sans mélanger de profils numériques incompatibles.

### 11.6. Hébergement et monde partagé

Servir en HTTPS des assets statiques et poids adressés par hash, avec compression appropriée, CORS, MIME WASM et téléchargement reprenable pour les gros fichiers. Les workers et le cache doivent tolérer fermeture d’onglet et manque de quota. COOP/COEP sont nécessaires si on choisit un WASM multithread avec SharedArrayBuffer ; une première version monothread du noyau évite cette dépendance et simplifie l’intégration.

Si le mode distant est activé, il utilise des sessions de caméra, quotas par client, admission et partage des fenêtres entre clients. Partager les calculs si les profils et seeds correspondent ; limiter les téléportations froides et les grands aperçus pour protéger la latence des autres vues. Garder une part du budget pour chaque session active, puis distribuer le reste selon l’utilité. Le chemin principal WebGPU calcule chez chaque visiteur ; le service distant est un complément à dimensionner séparément.

Pour un même monde publié, des backends différents ne garantissent pas le même dernier bit d’altitude. Choisir un **profil numérique canonique** pour les données sauvegardées, les interactions de jeu et les tuiles partagées. La génération locale est une variante d’affichage validée, ou ses sorties restent dans un namespace numérique distinct. Ne pas mélanger à un bord des variantes non validées et espérer une égalité parfaite. Si l’identité physique stricte est obligatoire, les données canoniques viennent du service/CDN ; le local sert à l’aperçu ou à une expérience individuelle.

## 12. Quantifier le débit et le gain nécessaire

### 12.1. Demande de la caméra

À résolution native, pour une vue de W × H pixels qui se translate de dx, dy en une seconde, sans dépasser ses dimensions, la surface nouvelle vaut approximativement :

`A_new = H·|dx| + W·|dy| − |dx·dy|`.

Pour 1920 × 1080 et un déplacement horizontal de 600 pixels/s, cela donne **648 000 pixels natifs nouveaux/s**, soit environ **9,9 équivalents de tuiles 256²/s**. À 30 m/pixel, le déplacement physique est 18 km/s. Si altitude et cinq sorties climatiques sont toutes livrées en float32 natif, le flux brut serait 15,55 Mo/s, avant compression et protocoles. Le stockage macro des climats réduit ce flux.

Le calcul neuronal doit en plus payer les fenêtres recouvrantes, halos, deux passes latentes et contexte manquant. Mesurer cette amplification par stage : `pixels/fenêtres calculés / pixels finaux utiles`. Pour les dézooms, le zoom et les téléportations, calculer la nouvelle surface à partir de la différence entre la nouvelle vue et les caches pertinents ; la formule de translation ne suffit pas.

Le débit se rapporte au niveau de qualité réellement livré. Ne pas compter une interpolation latente comme des pixels finaux à 30 m. Un aperçu couvre l’écran rapidement, mais ne prouve pas que le moteur suit le déplacement au détail natif.

### 12.2. Calcul du facteur à atteindre

Rejouer une trajectoire fixe, relever `R_baseline`, le débit final utile, puis calculer `gain_requis = 1,2 × R_demandé / R_baseline`. Mesurer aussi p95 du retard de génération et surface finale couverte. Sans `R_baseline` contrôlé, aucun facteur ×5 ou ×10 ne peut être défendu.

Le gain total ne se calcule pas en multipliant des gains présumés indépendants. Moins de travail demandé, meilleur partage, moins de transferts et kernels plus rapides interagissent. Comparer chaque variante par ablation avec une trace identique. La loi d’Amdahl s’applique à la partie remplacée : si 70 % d’un temps était du calcul GPU incompressible et 30 % du reste, supprimer tout le reste ne donnerait que ×1,43. Cet exemple est hypothétique ; le relevé de 44 % d’utilisation GPU ne permet pas d’estimer ces fractions.

| Intervention | Gain attendu qualitativement | Preuve à exiger |
| --- | --- | --- |
| Annulation et admission | Moins de temps perdu sur anciennes vues | Ratio GPU périmé et délai p95 de la vue courante |
| DAG et cache de fenêtres | Moins de fenêtres recalculées | Compteurs de miss et déduplication par stage |
| Aperçu macro borné | Disparition des longues attentes initiales au fort dézoom | Temps de couverture, nombre de fenêtres et qualité indiquée |
| GPU résident et pipeline asynchrone | Moins de sync/copies et d’intervalles vides | Timeline CPU/GPU, octets transférés, durée de chemin critique |
| Batching et TensorRT | Meilleur coût par fenêtre, selon réseau | Matrice batch/latence/débit/mémoire et équivalence |
| Kernels fusionnés / CUDA Graphs | Moins de lancements et d’opérations élémentaires | Part du temps concernée et gain bout en bout |
| Distillation ou quantification | Réduction possible du calcul ou de la mémoire | Nouvelle validation de modèle ; profil distinct |

Quantification INT8, diminution des 20 étapes coarse, T=1 latent et décodeur plus petit sont des pistes ultérieures. Elles changent les résultats ou la qualité et exigent une validation dédiée. Le premier moteur garde le modèle complet T=2 et toutes ses sources.

## 13. Mesures et validation de référence

### 13.1. Corpus et scénarios

Créer un harness headless commun Python/navigateur/natif. Il enregistre manifestes, seeds, caméra, demandes, événements, cache et métriques. Rejouer les mêmes scénarios : démarrage avec poids froids, contexte froid avec poids chauds, voisinage chaud, traversée régulière, diagonale, demi-tour, zoom oscillant, dézoom continental, téléportations et revisite après éviction. Ajouter plusieurs sessions d’un même monde et de mondes différents. Comparer explicitement la 3090, la 4090 seule, la 5090 seule et le mode natif 5090 + 4090 ; tester la liaison entre les deux machines séparément.

Utiliser des seeds 0, 1, 42 et quelques uint64 élevées ; régions positives/négatives, frontières de fenêtres et de tuiles, zones marines et terrestres, relief faible et fort, climats variés. Les comparaisons de backends utilisent une sélection fixe ; le corpus de variété est plus grand et exécuté hors des tests rapides.

Pour chaque résultat, indiquer cache froid/chaud par couche, poids chargés ou non, stage, batch, précision, nombre de fenêtres, surface finale utile et conditions de la machine. Faire plusieurs répétitions et publier p50/p95/p99 ; séparer compilation, téléchargement et génération. Mesurer les temps GPU avec événements et la timeline avec profiler CPU/GPU ou Nsight, pas seulement le temps d’un appel asynchrone.

### 13.2. Métriques indispensables

| Famille | Métriques |
| --- | --- |
| Produit | Temps première couverture, temps premier détail, couverture finale, retard de génération, p95 frame et interaction |
| Travail | Fenêtres uniques par stage, contributions partagées, jobs périmés, volume spéculatif utile, amplification des halos |
| Calcul | Durée kernels, attentes CPU, nombre de lancements, H2D/D2H, coût par bucket, temps de queue, calibration et contention par GPU |
| Mémoire | Pics GPU/RAM, cache en octets, évictions, workspace, fragmentation et recompilations |
| Web | Téléchargement initial, warmup, adapter obtenu, device loss, quota disque, débit transport et placement des opérateurs |
| Multi-GPU / agents | Débit mono/dual, coût des transferts, affinité de cache, charge par worker, délai bout en bout WebGPU/natif/distant |
| Qualité | Écarts physiques, continuité, stabilité par seed, distributions relief/climat/biomes |

### 13.3. Validations bloquantes

1. **RNG et bruit.** Vecteurs de référence PCG et Marsaglia polar, overflow uint64, ordre des canaux, seed/offset et patchs recouvrants. Le RNG Python utilise un stream séquentiel avec rejection sampling ; le remplacer par Philox ne préserve pas le terrain. Conserver d’abord le RNG CPU portable, puis optimiser seulement avec équivalence démontrée. [RNG upstream](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/portable_rng.py)
2. **Conditionnement.** Comparer les cinq champs synthétiques, le p5, les 58 valeurs latentes, unités, quantiles, normalisations, masques et traitement NaN.
3. **Réseaux.** Comparaison par stage avec entrées réalistes, plusieurs batches et amplitudes ; erreur absolue et relative, sortie finie. FP32/FP16/BF16 mesurés séparément.
4. **Schedulers.** Comparer chaque étape et l’état multistep coarse ; préserver les signes, sigmas et préconditionnements. L’export d’un forward seul ne teste pas cela.
5. **Fusion et post-traitement.** Contributions pondérées complètes, denominators, interpolation, blur, Laplacian denoising, signed-square et climat thermique.
6. **Indépendance des demandes.** Même région demandée seule, au sein d’un rectangle, dans un ordre aléatoire, après eviction et après annulation ; batches différents et deux clients. Aucune contribution double ou manquante.
7. **Continuité et précision.** Différences de bords, pentes et hauteur ; dérive terre/mer et températures. Les anciennes comparaisons de crops ont donné des différences submétriques : ne pas annoncer du bit-à-bit avant d’avoir isolé leur cause.
8. **Backends/OS.** Navigateur FP16/FP32 prioritaire, Linux/Windows NVIDIA, 3090/4090/5090 et matériel visiteur représentatif ; erreurs dans un domaine partagé et namespaces numériques explicites. Comparer également ordre et éviction en mode double GPU et vérifier les résultats lors d’une migration de tâches.

Pour le même backend/profil, imposer une stabilité exacte des décisions discrètes, du RNG entier, de la disponibilité et des clés. La tolérance flottante est fixée après mesure de la référence : comparer RMS, p99 et maximum en mètres, particulièrement en haute montagne où signed-square amplifie les écarts. Les tolérances entre backends sont des critères physiques et visuels ; une égalité binaire n’est pas présumée.

Critères de génération : pas de NaN/inf, pas de changement de seed selon ordre/cache, pas de couture nouvelle visible, pas de disparition d’une source climatique, pas de changement silencieux de profil, pas de fallback CPU non observé. Un gain de débit qui échoue à ces critères n’est pas activé par défaut.

## 14. Plan d’implémentation et portes de décision

| Lot | Livrables | Condition de sortie |
| --- | --- | --- |
| P0 — Référence et preuve WebGPU | Manifestes, corpus, replay, audit statistiques/seed 0, exports ONNX, tests des trois sessions WebGPU | Réseaux navigateur utilisables ; débits reproductibles ; qualité comparée à Python |
| P1 — Agent WebGPU complet | Worker, noyau portable/WASM, DAG, epochs, priorités, annulation, aperçu borné, fusion/climat et rendu GPU | Navigation et affinement autonomes dans le navigateur ; toutes les sources conservées |
| P2 — WebGPU adaptatif et site | Auto-tuning batch/cache, pipeline GPU résident, poids versionnés, reprise et device loss, matrice de GPU/browser | Produit principal publiable sans agent natif ni serveur GPU ; gain mesuré sur GPU plus rapides |
| P3 — Agent natif open source de comparaison | Pipeline C++/ORT CUDA, C ABI, appairage local, build Linux/Windows, outils et protocole publics | Comparaison bout en bout WebGPU/natif sur 3090/4090/5090 ; équivalence validée |
| P4 — Natif accéléré et deux GPU | TensorRT, kernels/pools/graphs ciblés, device selection, workers 5090 + 4090 et tuner avec affinité | Gain justifiant l’option native ; mode dual préférable au meilleur mono sur scénarios définis |
| P5 — Options réseau et extension | Endpoint de l’autre machine, service distant facultatif, CDN/cache partagé, profils variés et évolutions de modèle si nécessaires | Modes optionnels validés ; coûts/quotas connus ; changements de génération versionnés |

La preuve WebGPU intervient dans P0 pour détecter immédiatement les incompatibilités de modèle et de mémoire. Le produit WebGPU et son adaptation sont prioritaires dans P1/P2. Des mesures ou corrections du serveur Python peuvent soutenir cette référence, sans en faire le produit final. Le prototype natif P3 sert à décider si sa complexité est justifiée ; ses optimisations P4 sont ciblées sur les gains effectivement observés.

Après P0, calculer le facteur nécessaire à partir de la demande réelle. Après P2, vérifier si l’agent WebGPU suit le scénario cible sur chaque classe de GPU et déterminer le stage dominant. Si le coarse limite la latence froide, tester distillation ou pré-calcul macro. Si le decoder limite le débit, tester kernels/layout/batches avant compression du modèle. Après P3/P4, recommander le natif uniquement pour les configurations où son gain bout en bout est démontré. Sur les appareils limités, proposer l’aperçu, l’affinement plus lent et les modes natif/distant lorsqu’ils sont disponibles.

Un planning calendaire précis dépend du profilage et du résultat ONNX/WebGPU. Ces lots définissent l’ordre, les livrables et les critères ; ils ne supposent pas que l’ensemble est réalisable en quelques jours.

### 14.1. Backlog concret du premier lot

| ID | Travail immédiatement implémentable | Fichiers ou composants concernés |
| --- | --- | --- |
| B01 | Capturer configuration effective et hash des trois poids/tables ; enregistrer toutes les options runtime | `terrain_app.py`, outil de manifeste |
| B02 | Ajouter IDs vue/epoch et métriques de travail périmé ; conserver traces caméra rejouables | `index.html`, `terrain_server.py` |
| B03 | Sortir hillshade/PNG/écriture du verrou GPU après snapshot de sortie immuable | `terrain_server.py` |
| B04 | Limiter la surface coarse par tâche et ajouter aperçu synthétique à budget écran | `sample_field`, API overview, viewer |
| B05 | Mesurer et indexer les fenêtres uniques coarse/latent/decoder plutôt que seulement les tuiles HTTP | instrumentation WorldPipeline / InfiniteTensor |
| B06 | Exploiter climat complet dans un atlas de validation, avec unités contrôlées | `world.get(with_climate=True)`, outils de rendu |
| B07 | Valider l’export ONNX existant avec seuils et entrées de référence | `terrain_diffusion/onnx/export.py`, harness |
| B08 | Tester les trois sessions WebGPU, en priorité sur la 3090 puis sur la machine 5090 + 4090, pour opérateurs, buffers et mémoire | prototype de worker séparé de l’application active |
| B09 | Calibrer les stages, batches et budgets ; définir la comparaison des agents et la sélection GPU | harness navigateur/natif et profil de capacités |

Ces travaux n’exigent aucune modification artistique du monde initial. Ils fournissent les mesures permettant de dimensionner en priorité l’agent WebGPU du site, puis l’option native open source et sa répartition entre cartes.

## 15. Risques techniques et réponses prévues

| Risque | Signal à observer | Réponse prévue |
| --- | --- | --- |
| Le gain C++ seul est faible | Même temps dominé par kernels réseau | Batching, TensorRT et réduction du travail ; mesurer avant spécialisation |
| Fort dézoom toujours lent | Trop de fenêtres coarse nouvelles | Aperçu borné, cache macro, pré-calcul du monde public |
| GPU visiteur trop petit | Échec de buffers/sessions ou latences excessives | Batch 1, budget réduit, distant ; aucune garantie universelle |
| FP16 change relief/climat | Écarts en montagnes et biomes | FP32 ciblé, profil numérique distinct, référence canonique |
| Cache modifie les résultats | Bords instables après éviction | Contributions immuables et commit complet ; tests d’ordre/éviction |
| Variante locale diverge du monde partagé | Coutures ou collisions entre backends | Sorties canoniques du service ; namespace par profil numérique |
| Pipeline numérique peu portable | Opérateur non pris en charge ou fallback CPU | Export adapté et validation WebGPU précoce |
| Multi-GPU plus lent que mono | Transferts, contextes dupliqués ou délai visible accru | Affinité de régions et retour automatique au meilleur GPU seul |
| GPU plus rapide sous-exploité | Même batch, couverture et préchargement que le petit GPU | Calibration par stage et tuner continu sous budget de rendu |
| Plusieurs visiteurs saturent un service distant facultatif | Retard qui augmente sans borne | Admission, fairness, CDN et quotas ; WebGPU reste le chemin principal |
| Variété perdue par optimisation | Champs manquants ou distributions resserrées | Corpus de variété bloquant, préservation de toutes les sources |

Pour la publication, joindre les mentions des sources, licences des poids et dépendances au manifeste de distribution. La fiche modèle annonce MIT ; les gros rasters n’ont pas besoin d’être redistribués au visiteur pour la génération calibrée par tables. Vérifier les conditions des données et composants effectivement livrés pendant le packaging.

## 16. Sources et fichiers de référence

Sources techniques primaires consultées le 7 octobre 2026 ; les documentations `latest` doivent être remplacées par des versions épinglées dans les manifestes de build.

| Source | Utilité |
| --- | --- |
| [Projet Terrain Diffusion](https://xandergos.github.io/terrain-diffusion/) | Présentation et démonstrations |
| [Article v4](https://arxiv.org/html/2512.08309v4) | Propriétés, jeu de données et latences de référence |
| [Checkpoint 30 m](https://huggingface.co/xandergos/terrain-diffusion-30m) | Provenance Copernicus et licence déclarée |
| [WorldPipeline épinglé](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/world_pipeline.py) | Chaîne effective, dimensions, étapes, climat |
| [Synthetic map épinglé](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/synthetic_map.py) | ETOPO, quatre BIO WorldClim et statistiques |
| [RNG épinglé](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/portable_rng.py) | Streams déterministes et bruit |
| [Export ONNX épinglé](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/onnx/export.py) | Point de départ du port des réseaux |
| [Port Minecraft](https://github.com/xandergos/terrain-diffusion-mc) | Référence Java/ORT, exploration et biomes |
| [Architecture vLLM](https://docs.vllm.ai/en/latest/design/arch_overview/) | Séparation serving, ordonnanceur et worker |
| [TensorRT support](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) | Compatibilité et limites de portabilité des plans |
| [ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) | Résidence GPU et transferts explicites |
| [ORT CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | Runtime, streams et CUDA Graphs |
| [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) | Sessions WebGPU et sorties GPU |
| [ORT DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) | Option Windows et évolution vers WinML |
| [WebGPU](https://gpuweb.github.io/gpuweb/) et [WGSL](https://gpuweb.github.io/gpuweb/wgsl/) | Capacités, limites, types et shaders navigateur |
| [Choix d’adaptateur WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/GPU/requestAdapter) | Indication de préférence et sélection par le navigateur |
| [Architectures GPU CUDA](https://developer.nvidia.com/cuda/gpus) | 3090/Ampere, 4090/Ada et 5090/Blackwell |
| [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) et [RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) | Matériel cible de la seconde machine, à benchmarker |

Références locales : [README de l’application](../README.md), [serveur](../terrain_server.py), [viewer](../index.html), [chargement des modèles](../terrain_app.py), [versions installées](../requirements-lock.txt), [mesures archivées](audit-session.json). Version de lecture : [TERRAIN_REALTIME_DESIGN.html](TERRAIN_REALTIME_DESIGN.html).
