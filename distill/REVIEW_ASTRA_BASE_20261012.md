# Revue indépendante — Astra high

12 octobre 2026 · Modèle demandé : `gpt-6-astra` · Effort : `high`.
Rapport examiné : [BASE_EXPERIMENTS_REVIEW_20261012.md](BASE_EXPERIMENTS_REVIEW_20261012.md).
Avis du reviewer ci-dessous ; aucune proposition n'est encore exécutée.

---

Je privilégierais une **compression native BF16 progressive et modérée**, avec supervision de la trajectoire autonome. Le QAT500 mérite dès maintenant une comparaison visuelle : son arrêt ultérieur ne doit pas faire disparaître ce checkpoint intéressant.

J'ai lu intégralement le rapport, inspecté notamment `native_basis_training.py`, `native_basis_train.py`, `native_fp8_training.py`, `native_fp8_train.py`, les pertes et la fusion native, puis confronté les snapshots aux planches QAT500/1000. Aucun calcul GPU effectué.

**Mon diagnostic, par ordre de priorité :**

1. **La compression et les changements de géométrie perturbent une fonction sensible avant que l'apprentissage ne la restaure.** Les oracles situent une grande partie de la perte d'incision dans la première phase, particulièrement son canal0. Le teacher injecté en seconde phase enlève du bruit sans reconstruire les vallées perdues. Les contrôles natifs et SVD écartent une grosse erreur d'enveloppe ou de reparamétrisation. Le QAT, beaucoup plus fidèle initialement, renforce l'intérêt de rester proche de la fonction teacher. Cela ne prouve ni une impossibilité de compression, ni un défaut intrinsèque du canal0, ni une ambiguïté des cibles déterministes.

2. **L'objectif local ne garantit pas la fidélité de la composition T2.** Le QAT utilise effectivement les états actuels de l'élève et interroge le teacher sur les mêmes entrées ; lui reprocher simplement l'absence d'on-policy serait faux. Mais ces états sont détachés : les conséquences finales ne rétroagissent pas jusqu'à la première phase. L'objectif hérite aussi d'une forte asymétrie : `second_height` divise par une variance spatiale H4 de **0,0025867**, contre **1,06067** pour sa variance dans l'opérateur ; les bandes H4 descendent à **4,51×10⁻⁵**. Le « canal0×2 » ne concerne que son terme opérateur. Ces coefficients demandent une justification par leurs effets ; ils ne démontrent pas, seuls, un conflit de gradients.

3. **Le protocole a souvent confondu absence de progrès rapide et impossibilité de récupération.** L'arrêt à1000 exigeant canal0÷2 n'est pas une mesure de convergence. Le QAT conserve un LR constant après100 pas, et une EMA de demi-vie500. À1000, cette EMA garde encore25% de son initialisation. À l'inverse, les longues recettes directes et native128 montrent que prolonger aveuglément ne suffit pas. Les changements simultanés d'architecture, objectifs et données empêchent une explication unique.

Le quart-rang a comprimé42 convolutions en1000 pas, puis disposé de3000 pas de récupération, avec reset Adam/EMA et exposition on-policy tardive. Ce résultat est négatif pour **cette recette**, pas pour toute factorisation. Les scores agrégés, le rappel seul et les contrôles de reconstruction ont parfois reçu trop de poids dans les décisions qualitatives.

Le checkpoint à conserver en priorité est **QAT500**. Sur montagnes/coarse4 LOD0, ses MAE sont **14,9/14,8/36,8m**, contre47,5/28,0/75,2m à1000 ; rappels **0,959/0,937/0,948**. Les planches montrent des formes convaincantes. Mais les anciennes rares limitent ce constat : à500, désert et plaine tempérée LOD3 ont des amplitudes **1,40/1,47** ; la côte humide LOD0 reste à **corr0,873**, malgré8,19m de MAE. Il est visuellement testable, avec bruit et côtes explicitement examinés. Il n'est ni accepté ni accéléré.

**Ma prochaine recette principale serait la suivante :**

- **Architecture :** teacher192×3, mêmes58 conditions, cinq canaux, attention, skips, fenêtres64/pas32, bruits mondiaux et T2. Factoriser uniquement les42 convolutions internes du décodeur déjà étudiées, au **demi-rang**. Garder BF16 et mastersFP32.
- **Initialisation :** teacher original et base SVD FP64 contrôlée ; extinction progressive des directions supprimées sur5000 pas. Conserver la normalisation du poids composé. Compacter par suppression des facteurs éteints, en conservant les états Adam/EMA des coordonnées restantes.
- **Objectif :** opérateur cinq canaux normalisé, phases équilibrées, premier canal0×2 ; ajouter les MSE des **champs fusionnés de première phase et finaux autonomes**, normalisés par canal avec statistiques TRAIN fixes. Une mise à jour régionale sur quatre, avec gradient traversant T2 ; coefficient régional progressif jusqu'à1 sur2000 pas. Ajouter seulement un terme physique faible, coefficient0,1, sur hauteurs et gradients LOD3/0 via decoder original gelé. Pas de KD37 blocs ni d'énergie spectrale supplémentaire initialement.
- **Données :** conserver25% montagnes/15% rares/60% général, les trois coarse et huit mondes distincts par batch effectif. Pour l'opérateur seconde phase,50% d'états actuels de l'élève, teacher évalué sur les mêmes entrées. Préserver les mondes finaux.
- **Optimisation :** Adam β=(0,9;0,99), sans weight decay ; LR facteurs5e-5, autres poids1e-5, warmup500, clip1. Budget **30000 mises à jour**, dont5000 de compression, puis décroissance cosinus jusqu'à un dixième du LR. EMA demi-vie1000 ; conserver brut et EMA, checkpoints tous1000.

**Contrôle préalable falsifiable :** sur16 régions TRAIN fixes couvrant montagnes et rares, comparer à initialisation demi-rang identique l'ancien objectif et le nouvel objectif régional, pendant3000 mises à jour maximum. Mesurer les erreurs des champs autonomes et les images, pas seulement la loss. Si seul le nouvel objectif récupère les incisions, l'hypothèse d'objectif gagne du poids. Si aucun ne réduit fortement l'erreur TRAIN, suspendre le grand run : davantage de mondes ne résout pas ce problème local. Cela ne prouverait pas une incapacité architecturale.

Après compression, arrêter sur divergence ou détérioration persistante pendant trois évaluations ; sinon laisser au moins5000 pas de récupération. Sélectionner sur le compromis par catégorie : position **et profondeur** des vallées, précision/rappel, bruit des plaines, pente, déplacement côtier et biais d'altitude. Les images communes restent décisives. Vérifier ensuite des raccords froids, plusieurs ordres de chargement et les deux axes ; aucune certification héritée d'un autre modèle.

Enfin, le demi-rang n'offre aujourd'hui qu'un indice de gain **1,18× par fenêtre** dans le profil existant. Aucun gain complet n'est acquis. Je viserais d'abord une compression fidèle mesurable ; **10× n'est pas étayé** par ces résultats.
