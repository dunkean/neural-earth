# Revue finale — Astra HIGH r2

7 octobre 2026. Suivi de `RUNTIME-FINAL-astra-r1.md` et des nouvelles remarques
`CONTINENTAL-A4-opus-r2.md`. Lecture seule des sources produit ; aucun calcul
GPU lancé par cette revue.

## Verdict final : ACCEPT WITH LIMITS

Le cache et les corrections des harnesses sont acceptés. A4 peut servir de
défaut d'initialisation expérimental avec `natural` accessible. Les huit sites
NN, les diagnostics côtiers et les parcours navigateur finaux appuient ce
périmètre. Aucun P0/P1 nouveau démontré. Les limites de réalisme, de couverture
planétaire et de latence détaillées ci-dessous restent explicites ; elles ne
sont pas converties en défauts de source ni en exigences de réécriture physique.

## Cache de statistiques, schéma 2 : ACCEPT

Le cache inclut maintenant `source_order` et `rasterio.__gdal_version__` dans
l'identité et utilise le schéma 2. L'ancien cache ne peut donc pas satisfaire
une lecture de cette identité. Les opérations numériques de construction des
statistiques sont conservées.

Les neuf tests ont été relus puis exécutés indépendamment : **PASS**, 0,596 s.
Les nouveaux cas contrôlent un réordonnancement des mêmes rasters, une autre
version GDAL, un changement du texte de calcul et les échecs de mkdir,
création temporaire, fsync et remplacement. Les permissions sont simulées
explicitement pour Windows ; les fichiers temporaires sont vérifiés absents.
L'invalidation du code force cinq lectures de rasters et une nouvelle archive.

Une recomputation sur les vraies sources, cache désactivé, puis une lecture
avec le cache courant ont les mêmes scalaires et exactement les mêmes octets,
dimensions et dtypes d'arrays. GDAL 3.12.2. Empreinte de `terrain_world.py` :
`18f34dfc80a1e83d7acfc3aa20bf8326f59dc8e72b8c23dc0de37ebb81afb45e`.

S1 et S3 sont clos. S2 est documenté : l'identité des terrains coarse et
physiques, y compris `natural`, change avec ce fichier malgré l'égalité des
valeurs ; l'ancienne préparation de 6 160 fenêtres n'est pas réutilisable.
La documentation distingue ce coût du gain limité des statistiques source.

## A4, corpus final : ACCEPT WITH LIMITS

Les sources de calcul macro sont inchangées numériquement depuis v2 ; la
docstring distingue maintenant l'algèbre capsule exacte de la rampe adaptée.
Le garde des sources CPU final passe, avec le nouveau helper de diagnostics
dans l'inventaire. Huit sites sont présents : les quatre originaux conservés,
puis minimum et médiane des sauts côtiers pour chaque seed. Leur sélection
est déterministe avant NN, sans recentrage après observation du résultat.

Les NPZ des paires contiennent 14 983 / 14 461 paires uniques de cellules à
7,68 km. J'ai recalculé leurs deux hauteurs, vérifié les coordonnées, l'unicité,
le tri, les quantiles et le rang des quatre sites sélectionnés. Les dérivées
centrées ±7,68 km et leur norme ont également été recalculées. Les résultats
correspondent au rapport.

| Mesure des entrées | Seed 0 | Seed 42 |
|---|---:|---:|
| Saut côtier médian entre voisins | 182,6 m | 189,1 m |
| Saut maximal inventorié | 4 320,1 m | 4 064,4 m |
| Paires avec terre <20 m et mer >−20 m | 1,175 % | 0,982 % |
| Norme ∇d minimale / médiane | 0,087 / 1,837 | 0,098 / 1,899 |

**A1 est mesuré, pas éliminé universellement.** L'inventaire raffine les côtes
de l'overview 1024 × 512 par des voisinages 3 × 3 ; il n'énumère pas toutes les
cellules côtières de la planète coarse. Le rapport annonce cette limite.
La pente du rampement en distance signée ne constitue pas une pente minimale
garantie dans la carte.

**A2 est couvert selon le critère annoncé.** « Weakest » signifie le minimum
du saut sur une arête x/y, et non le minimum de la norme ∇d. Les deux arêtes
minimales longent presque tangentiellement une côte dont la norme vaut encore
1,79 / 2,55 ; elles ne ciblent donc pas spécifiquement les selles du champ.
Cette nuance interdit de présenter les nouveaux crops comme un test exhaustif
du pire risque géométrique, mais leur sélection n'est pas une sélection
favorable après NN.

Le comparatif ETOPO rapporte médiane 59,6 m et maximum 1 753,0 m, contre les
valeurs macro ci-dessus. Sa source 10 arc-minutes est interpolée sur la carte
plane ; ce n'est ni un DEM natif 30 m ni une référence de pente uniforme sur
la sphère. A3 reste donc une limite de crédibilité mesurée : A4 a des fronts
côtiers beaucoup plus raides dans cet inventaire. Aucun argument ne justifie
de certifier ici des distributions terrestres réalistes.

La compression A5 est quantifiée : environ 2,18 % / 2,20 % des terres ont une
hauteur intérieure brute au-dessus du genou ; la réduction maximale observée
est 1 159 / 1 413 m. Cela apporte plus d'information que l'absence de pixels
au plafond. Un nouveau site de crête sélectionné par relief local, des bornes
climatiques indépendantes du seed et une distribution de basses terres plus
terrestre restent des pistes de qualité, sans défaut d'exécution établi.

Le corpus `continental-bootstrap-v2-final-nn` a été vérifié indépendamment :
huit NPZ, liens metrics/manifests/seed/profil/focus/exporteur, finitude DEM,
climat, coarse et latent, topologies recalculées. Les deux manifests canoniques
correspondent à leurs 107 fichiers d'implémentation actuels. Les poids ne sont
pas rehashés par cette revue. Les quatre sites d'origine reproduisent exactement
les octets DEM/climat/coarse/latent du v2 précédent malgré la nouvelle identité.

| Nouveau site | Eau dans le DEM | Plage DEM | Fraction à moins de 1 m de zéro |
|---|---:|---:|---:|
| Minimum seed 0 | 92,96 % | −456,2 à 18,5 m | 8,91 % |
| Médiane seed 0 | 71,16 % | −560,7 à 420,9 m | 0,76 % |
| Minimum seed 42 | 45,04 % | −549,2 à 533,1 m | 1,52 % |
| Médiane seed 42 | 50,53 % | −792,0 à 642,1 m | 1,02 % |

Les quatre nouvelles planches ont été vues. Elles montrent de l'eau ouverte
et des rivages appris, avec relief côtier bas ou collines selon le site.
Le minimum seed 0 devient majoritairement marin et très bas : il doit rester
visible dans les preuves, sans recentrage qui dissimulerait ce résultat.
Le déplacement de côte et la surface très proche de zéro sont des limites
locales ; ils ne démontrent pas une erreur de transformation ou un masque NN.
Ces résultats soutiennent un initialiseur expérimental, pas une garantie de
relief réussi partout.

Le probe final garde le signe à ses 512 points ; SHA et valeur recomputée
corrects. Il reste une sonde 32 × 16, sans planète apprise complète ni mesure
de topologie continue. A0/A3 restent les baselines historiques explicitement
nommées, pas de nouveaux rejeux à cette identité.

Identités finales NN natives :

```text
seed 0  522aad4559741f2f7ac69bbe9c650c9b72f059c375bb9df876c92cfd540fab94
seed 42 308cb0bf35ace0f7a5a6f87410e3fc4d38f2e6fafff3933f3d697a819c6d4df8
macro   7b008f30876ab8b601bd977e16ea50782d28e7cb1cf5fb804c1466154e0dd863
```

## Harnesses runtime : ACCEPT WITH LIMITS

R1 et R2 sont clos. Le base de test renvoie une prédiction déterministe non
nulle et variable par élément ; la référence conserve le terme complet et
son ordre de calcul. Le test mutant supprime ce terme dans une copie mémoire
du helper et constate l'échec. Chaque fenêtre compare aussi le conditionnement
à la vraie méthode upstream, y compris NaN/inf/poids nuls.

R3 est clos sur le chemin normal et les exceptions : les deux harnesses
sauvegardent `running`, l'échantillon en cours avant vérification, puis
`failed` avec erreur et résultat fautif, ou `complete` après sauvegarde des
arrays. Les garde-fous sont des exceptions explicites, actives avec `-O`.
Un arrêt brutal peut laisser `running`, qui ne certifie jamais la réussite.
Les blocs finally restaurent instrumentation et hooks et ferment le monde.

R4 est corrigé pour les modules demandés, résolus depuis leurs `__file__` :
app, macro, world, manifest, world_pipeline et mp_layers rejoignent les
empreintes précédentes. R5 reste une limite historique correctement dite :
le harness intermédiaire des quatre runs appariés n'est pas archivé et son
protocole ne peut pas être reconstruit exactement depuis les snapshots.

R6 est clos : le supplément annonce 92 comparaisons indépendantes d'arrays et
44 auto-comparaisons, distingue ces dernières, n'annonce pas l'exécution des
tests et utilise `all_array_recomparisons_byte_exact`. J'ai vérifié ses hashes
de reçus, arrays, sources de tests et exporteur. Il ne transforme pas les
anciennes mesures en nouvelles exécutions du harness durci.

Exécutions personnelles, sans CUDA :

- `python -m unittest test_runtime_benchmark test_runtime_latent_batch -q` :
  **7 tests PASS**, 0,842 s.
- `python -O -m unittest test_runtime_benchmark -q` : **5 tests PASS**, 0,110 s.

Les sources NN effectives restent celles des 92 arrays historiques identiques :
`terrain_inference.py` SHA `03fd47dd5c2e6219628b42669865929f58f029350aa3d8cd320d57d83386632c`,
`terrain_nn_constants.py` SHA `2d2465089ba7b0e1634e12f2624c83b6833644594f52d9e1f6c077ee3113b87d`.
Les nouveaux tests renforcent la preuve CPU et les reçus, sans certifier de
nouveaux timings GPU ni d'autres cartes/builds. LOD 3 garde son bénéfice
mesuré de préparation ; l'ampleur du gain mural LOD 2 reste non certifiée.

Harnesses relus après le signal de gel :

```text
benchmark_runtime_lod.py         f8b1ac007fbcb9be7d6c27f66a66a619397a9e23016cc9acde2066f0cd4d858b
benchmark_runtime_progression.py 5cf5af68c2f7c320fb9ccad3416a34078ad67d32e18465514a00e02db8b8eaa0
```

## Navigateur final et portée de la promotion

Les trois reçus `browser-continental-final`, `-final-warm` et
`-final-resident` ont été relus. Leurs hashes de sources et des deux captures
associées correspondaient aux fichiers au moment de leur contrôle. Leur
harness de navigation a ensuite reçu le correctif de provenance ci-dessous ;
les sources produit restent inchangées. La vue mondiale et le rendu
natif résident ont été inspectés : le HUD indique maintenant MACRO-A4 et
l'aperçu mondial demeure explicitement un aperçu des entrées. LOD 11 n'est
pas présenté comme la planète NN entièrement préparée.

| Session | Réponses physiques | Parcours natif complet | Premier LOD 4 |
|---|---:|---:|---:|
| Seed 42 après redémarrage | 25 misses | 13,541 s | 6,188 s |
| Relecture chaude du même site | 25 hits | 0,547 s | 0,059 s |
| Seed 0, modèles résidents, nouveau site | 28 misses | 11,326 s | 1,502 s |

Chaque session suit 4/3/2/1/0 et termine à 15/15 tuiles decoder 30 m, sans
erreur enregistrée. Le parcours résident présente LOD 3 à 3,618 s puis LOD 2
à 7,188 s. Les temps mesurent des draws soumis échantillonnés, pas une mesure
instrumentée de présentation matérielle.

Les premiers aperçus mondiaux prennent respectivement 12,500 / 1,091 / 10,653 s.
Ils ne sont donc pas instantanés, même avec les statistiques persistantes.
Le coordinateur signale que sa suite CPU de 85 tests tournait au démarrage
et au début du premier parcours froid ; ce cas inclut aussi le chargement
des modèles. Il ne sert pas à isoler le coût d'inférence. Le run résident
est distinct, sans cette suite concurrente. Les hits chauds ne doivent pas
être présentés comme de nouvelles sorties NN.

Le changement UI est borné au défaut A4 et au libellé de profil ; une URL
explicite `profile=natural` reste acceptée. SHA UI examiné :
`c8e7a6e54e387587c1048b6fd974f6b4b374da2754be4b67bf1288b452dc992e`.
L'acceptation concerne le défaut d'initialisation **expérimental** avec retour
possible à la référence, pas une promesse de géophysique complète, de climat
universel, de p95 certifié ou de préparation apprise mondiale terminée.

### Provenance du manifest navigateur : limite détectée et corrigée

La représentation `worldManifest` des trois premiers reçus passe par
JSON.stringify, qui normalise les floats Python entiers. Elle ne permet donc
pas de reconstruire seule le hash canonique Python. C'est une lacune du reçu,
pas une incohérence détectée de la génération serveur.

Après signalement, le harness conserve dans `worldResponses` le corps UTF-8
brut de `/api/world` et son SHA avant parsing. Le nouveau reçu
`browser-continental-final-provenance.json` a été vérifié indépendamment :
SHA des bytes, parse Python et `world_identity` passent et correspondent au
hash exposé au client :
`abf0193ffe8f8dffb4b0641130797391c729d2aed81bc360659082db555426e6`.
Ses hashes de sources/images passent aussi. Cette session chaude a 25 hits
et termine en 0,543 s. La capture brute n'est pas attribuée rétroactivement
aux trois anciennes sessions. Les manifests natifs de l'export NN utilisent
un autre descriptif d'exécution, d'où leurs hashes différents déjà cités.
