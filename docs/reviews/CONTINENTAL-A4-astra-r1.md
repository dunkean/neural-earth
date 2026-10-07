# Continental A4 — Astra HIGH, revue r1

Date : 2026-10-07. Revue en lecture seule du candidat A4 terminé côté code
et de son premier corpus CPU/NN. Aucun serveur démarré, aucune nouvelle
inférence CUDA, aucun fichier source modifié. Les sorties de côte recentrée
annoncées en suivi ne sont pas présumées validées par cette revue.

## Verdict : ACCEPT WITH LIMITS

Aucun P0/P1/P2 concret de calcul, transformation, identité ou aiguillage
trouvé dans le périmètre examiné. A4 est acceptable comme candidat
expérimental disponible séparément de `natural`. Cette revue ne valide
ni une promotion automatique du défaut, ni une planète NN complète au
LOD 11, ni des garanties de réalisme côtier/climatique pour toute seed.

Sources examinées : `terrain_macro.py`, `terrain_bootstrap_preview.py`,
`terrain_bootstrap_validate.py`, `test_terrain_bootstrap.py`, les changements
A4 de `terrain_manifest.py`, `terrain_server.py`, `terrain_inference.py`,
`docs/CONTINENTAL_BOOTSTRAP.md`, ainsi que les fonctions de provenance de
`terrain_reference.py`. Sources Rust de côte/bruit lues en référence.

## Contrat de génération

`ContinentalBootstrap` (`terrain_macro.py:296`) reste séparé des versions
A0–A3. Les méthodes partagées ajoutent explicitement A4 aux branches utiles,
sans changer le calcul des anciennes variantes. Les tests protègent A0
contre le factory upstream et les valeurs historiques A1/A2/A3.

La géographie dépend des coordonnées monde et d'un ensemble fini d'objets
seedés. Le seuil de mer est calculé une seule fois sur une grille globale
fixe 512 × 256 (`:345`), jamais sur le crop demandé. Le test de crop confirme
la stabilité des mêmes cellules dans deux fenêtres différentes. Le choix
de cinq supports ramifiés ne garantit pas cinq composantes topologiques :
leurs unions et perturbations peuvent fusionner ou créer des îlots.

L'altitude intérieure est composée de reliefs attachés aux supports,
plateaux et bassins ; seule la bande littorale multiplie cette hauteur par
un smoothstep. Le relief ne dépend plus intégralement du rang de distance
au centre du continent. Le fond marin reste signé ; aucun masque d'altitude
n'est imposé après le réseau. Ce sont des priors géométriques, sans simulation
de tectonique, drainage ou érosion planétaire.

Les cinq champs restent en unités physiques. `MacroConditioning.__call__`
(`:290`) applique une seule racine carrée signée au canal d'altitude, puis
WorldPipeline effectue la normalisation du checkpoint. Les quatre autres
canaux restent inchangés à cette étape. Le SNR reste 0,5 ; le dispatch A4
est explicite dans le serveur et l'inférence.

Les bornes d'altitude viennent de la table source installée ; les bornes
climat viennent du support naturel observé sur une grille fixe. Cela évite
les demandes en dehors de ces bornes, sans constituer une preuve de support
statistique conjoint ni de réalisme polaire. Les diagnostics consignent
environ 20,6 % de BIO1 écrêté avant normalisation, avec minimum −17,5 °C.
Cette limite est correctement documentée. Les sorties NN peuvent dépasser
les bornes de l'entrée : les crops de crête A4 atteignent environ 7 553 m
seed 0 et 6 910 m seed 42 ; aucun écrêtage post-NN n'a été ajouté.

## Rapport au Rust

La projection clampée du point sur le segment, l'interpolation des rayons
et la distance aux extrémités de `capsule_distance` (`:351`) reproduisent
l'algèbre de `Coast::distance`. Le test couvre des points avant/après les
extrémités et les rayons variables. Le produit hauteur indépendante × rampe
cubique reprend le principe algébrique de la branche terrestre de
`Coast::apply` ; les paramètres, le calcul de largeur et le bruit sont propres
au composer planétaire Python.

Le motif `(1-abs(noise))²` est partagé avec `Noise::ridged`, mais la boucle
à rétroaction des octaves Rust n'est pas portée. Aucun binaire Rust/WASM,
RNG Rust ou moteur régional d'érosion n'est exécuté. Les deux empreintes
Rust documentées correspondent aux fichiers relus. La documentation décrit
correctement une adaptation portable, pas une équivalence binaire complète.

## Preuves vérifiées

Commande exécutée sans GPU :
`E:/TerrainDiffusionRuntime/venv/Scripts/python.exe -m unittest test_terrain_macro test_terrain_bootstrap -q`.
**14 tests PASS** en 11,7 s : référence naturelle, anciennes ablations,
algèbre capsule, coordonnées négatives/crops, unités, bornes physiques,
topologie sur trois seeds à la résolution spécifiée et identité distincte.

Artefacts relus dans `continental-bootstrap-v1-r2` et
`continental-bootstrap-nn` :

- Les deux planches mondiales CPU seed 0/42 montrent des continents
  ramifiés, des océans ouverts et des chaînes finies. À 1024 × 512, les
  terres représentent 32,010 % / 31,994 % ; les cinq plus grandes masses
  regroupent 99,912 % / 99,995 % des terres. Les composantes restantes sont
  principalement de petits îlots. Ces nombres décrivent cette grille.
- Douze NPZ natifs 1024², soit quatre sites × A0/A3/A4 : dimensions et
  finitude vérifiées, SHA-256 conformes aux métriques, seed/profil/manifest
  internes cohérents. Les six manifests complets ont un hash canonique
  recalculé correct ; toutes leurs empreintes de sources d'implémentation
  correspondent aux fichiers actuels.
- La comparaison globale NN a 512 vrais points coarse avec contributeurs
  complets. Les NPZ et leurs hashes sont cohérents ; l'accord de signe
  recomputé est 93,75 % pour A0 et 100 % pour A4 à la seed 0.
- Les planches natives de côte seed 0 et de crête seed 42 montrent une
  structure apprise détaillée à partir des champs lisses du composer.
  Les planches comparent les mêmes coordonnées physiques entre variantes,
  avec palette et éclairage fixes.

Empreinte du macro source commun aux preuves CPU et aux manifests NN :

```text
terrain_macro.py
9efe4c353f3647cab4d805d00530f6f79fd7e72b6ec09602eaaa37e39270f109
```

## Limites de qualité et recherche restante

**La sonde mondiale ne démontre pas la topologie NN globale.** Les 32 × 16
points sont séparés d'environ 1 250 km. Les connecter pour compter des
composantes crée de l'aliasing : les trois composantes et fractions d'océan
du tableau sparse ne doivent pas devenir un nombre de continents ou une
mesure de connectivité réelle. Le tableau et son titre disent explicitement
« point samples », pas des moyennes d'aire LOD 11. Le code conserve le statut
`complete_global_nn_not_generated:true`. Une preuve de vue mondiale apprise
nécessite le parcours réel de préparation/réduction, toujours absent ici.

**La côte peut se déplacer fortement.** Au site initial `bootstrap_coast_seed42`,
le conditionnement A4 est terrestre à 50,66 %, mais le DEM atteint 97,53 %
de terre (minimum −7,055 m). Il n'est donc pas littéralement tout terrestre,
mais ce crop cadre mal une côte marine apprise. Le site seed 0 passe de
54,24 % à 72,65 % de terre. Ces déplacements ne sont pas un défaut de signe
ou un double carré détecté : la chaîne de transformation est cohérente.
Ils limitent cependant la qualité démontrée. La comparaison recentrée
annoncée doit conserver les cas initiaux, employer les mêmes nouvelles
coordonnées pour A0/A3/A4 et annoncer son mode de sélection.

**Le corpus n'est pas une comparaison de réalisme universelle.** Les sites
ont été choisis dans le candidat A4 pour une côte pentue et un maximum
d'altitude. Aux mêmes coordonnées, A0/A3 peuvent être océaniques. Cela
vérifie la réponse à la nouvelle géographie, mais n'établit pas qu'A4 gagne
sur une sélection indépendante de côtes ou de montagnes propres à chaque
profil. Les crêtes CPU restent volontairement géométriques et répétitives ;
le détail NN local ne prouve pas une organisation tectonique globale.

Ces limites sont des preuves ou recherches manquantes pour une promotion
globale, pas des P0/P1/P2 de source inventés. Le candidat peut rester
accessible pour comparaison, avec `natural` préservé comme référence.

## Complément — côtes NN recentrées

Relecture de `terrain_bootstrap_shore.py`, de la documentation mise à jour,
de `E:/TerrainDiffusionRuntime/continental-bootstrap-shores/shore-report.json`
et des deux planches associées. Aucun nouveau calcul NN n'a été lancé.

Le replay choisit, dans chaque DEM A4 initial, le pixel terrestre adjacent
à un pixel ≤ 0 le plus proche du centre, puis exporte A0/A3/A4 au même
nouveau focus. J'ai recomputé ce choix depuis les arrays initiaux. Il donne
les offsets (+4 320, +120) m et (−2 760, +6 960) m, soit 4 321,666 m et
7 487,269 m. Les cas initiaux restent présents séparément.

Les six nouveaux NPZ ont été vérifiés : SHA conformes aux métriques,
manifests canoniques valides et identiques à ceux des profils/seed initiaux,
DEM finis de 1024², mêmes bornes entre les trois profils pour chaque site.
Les fractions de terre ont été recomputées depuis les DEM. Les deux crops
A4 ont désormais 41,156 % et 16,847 % d'eau, avec les plages −15,5–1 224,7 m
et −97,0–4 613,5 m indiquées par la documentation.

Les planches montrent une eau ouverte au bord du crop, des rivages
irréguliers et du relief intérieur détaillé. La seed 42 montre une plaine
littorale avec des poches d'eau et une chaîne haute en arrière. Cela complète
utilement les preuves locales. Le focus est sélectionné dans A4 après
observation du NN : il n'est pas un échantillon aléatoire indépendant et ne
rend pas les variantes anciennes comparables à terrain équivalent. Un pixel
adjacent à de l'eau n'est pas, en général, une preuve d'océan connecté ; ici
les planches et les bords du crop appuient l'interprétation littorale locale.
Les limites planétaires et climatiques de r1 restent inchangées.

### P2 de provenance du reçu — signalé au coordinateur

Dans la version relue de `terrain_bootstrap_shore.py:34`, le hash de chaque
NPZ initial est vérifié contre son fichier de métriques, puis n'est pas
persisté dans `shore-report.json`. Ce dernier conserve seulement le hash du
rapport initial et celui de l'exporteur. Le rapport initial contient des
chemins vers les métriques/NPZ, sans leur hash : la phrase documentaire
« binds the original source artifact hash » n'est donc pas encore assurée
par ce reçu seul. Les fichiers actuels sont cohérents, vérifiés ci-dessus.

Correction proposée : ajouter par site le hash du NPZ initial, son manifest
et le record source, en conservant le hash de l'exporteur effectivement
exécuté. Un complément CPU des reçus est suffisant ; aucune régénération
GPU n'est nécessaire. Les hashes indépendamment vérifiés sont :

```text
bootstrap_coast_seed0 / original A4 stages.npz
5258c5c7dccd2322317e1855197f4e5f6cfec56c9469d475023188de6a82ed97
bootstrap_coast_seed42 / original A4 stages.npz
000541c94beea6a7c7119d315c778bcc4defa406089a487caaaade47b9d98a3c
```

Le verdict produit reste **ACCEPT WITH LIMITS** comme candidat expérimental ;
ce P2 concerne la liaison durable du reçu de suivi, pas les calculs NN
inspectés. Sa correction doit être consignée avant de présenter cette
liaison comme complète.
