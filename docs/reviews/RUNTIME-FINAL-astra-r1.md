# Revue finale runtime et initialisation — Astra HIGH r1

Date : 7 octobre 2026. Revue des sources et artefacts existants, sans exécution
GPU et sans modification des sources produit. Ce fichier est le seul livrable
de la revue. Les références antérieures sont `CONTINENTAL-A4-opus-r1.md`,
`CONTINENTAL-A4-astra-r1.md` et `RUNTIME-NN-opus-r1.md`.

## Verdict : ACCEPT WITH LIMITS

Le runtime optimisé, le cache exact de statistiques et A4 v2 sont acceptables
pour le périmètre demandé : initialiser une géographie continentale crédible,
puis naviguer dans les sorties apprises. Aucun nouveau P0/P1 de calcul,
coordonnées, transformation ou identité trouvé. Les deux formules A4 v1
signalées sont corrigées, avec amélioration NN aux coordonnées originales.

La promotion A4 comme **initialiseur expérimental par défaut** est soutenable
par ces preuves, en conservant `natural` accessible et l'étiquette d'aperçu des
entrées. Elle ne signifie pas que la planète NN entière est préparée ou validée.
Le petit P2 d'attribution visuelle ci-dessous a été corrigé pendant la revue.
Il ne remet pas en cause les arrays inspectés. Aucun P0/P1/P2 bloquant ouvert
dans le périmètre final examiné.

## P2 constaté puis corrigé : cartouche A4 attribué au profil naturel

Dans `browser-continental-v2-world.png`, le sélecteur et le statut disent A4,
mais le cartouche inférieur indique « Profil naturel original ». La cause est
`index.html:59` : toute latitude `null` prend ce libellé, alors que le profil
macro actif n'est pas naturel. La capture et le snapshot confirment le défaut.
Signalé au coordinateur pendant la revue, puis correction source relue : seul
`natural` conserve ce texte ; les autres profils affichent leur nom effectif.
Aucune latitude, donnée ou logique de navigation n'a changé. La capture
historique reste attribuée à son ancien hash ; le rendu après correction n'a
pas été recapturé par cette revue.

## A4 v2 : correction et preuves

Sources relues : `terrain_macro.py`, `terrain_bootstrap_preview.py`,
`terrain_bootstrap_validate.py`, `terrain_bootstrap_shore.py`, les parties
d'export/provenance de `terrain_reference.py`, `test_terrain_bootstrap.py` et
`docs/CONTINENTAL_BOOTSTRAP.md`. Le chemin annoncé
`terrain_bootstrap_reference.py` n'existe pas ; le coordinateur a confirmé les
deux helpers réellement utilisés.

La rampe `t(2-t)` donne une pente non nulle à la côte ; les tests couvrent les
distances ±3,84/7,68/15,36 km et plusieurs reliefs/plages de paramètres. La
saturation exponentielle C1 conserve la variation des crêtes. Les anciennes
branches A0–A3 et l'unique racine carrée signée restent séparées. La reprise
Rust exacte se limite désormais à la distance capsule ; la documentation
distingue correctement l'adaptation planétaire de l'ancien rampement cubique.

Le test topologique utilise maintenant une grille 509 × 253 décalée, distincte
de la calibration 512 × 256. Le garde CPU vérifie tout son inventaire de sources,
y compris world, statistiques, helpers et rasters. Le probe mondial ne présente
plus de surfaces ou composantes de « continents » sur une grille de points.
Les planches distinguent le champ analytique 30 m, l'interpolation physique du
vrai treillis d'entrée 7,68 km avec halo, et la sortie NN. L'interpolation est
un affichage ; elle ne prétend pas être un conditionnement réseau à 30 m.

Contrôles indépendants des artefacts `continental-bootstrap-v2` : garde des
sources passé, hash du rapport v1 référencé correct, quatre coordonnées de
sites originales exactement conservées, topologie A4 recalculée depuis les
deux NPZ mondiaux. Les terres occupent 32,010 % / 31,994 %, avec trois/deux
masses d'au moins 1 M km². Aucun pixel n'atteint le cap positif dur. Ces
nombres décrivent exclusivement les entrées sur la grille CPU.

Le corpus `continental-bootstrap-v2-nn` contient **A4 seulement** : quatre
sites natifs et le probe seed 0. A0/A3 restent les références historiques ;
aucun nouveau rejeu GPU de ces deux profils n'est prétendu par cette revue.
Les quatre NPZ correspondent aux SHA des métriques, aux manifests canoniques,
au seed/profil/focus et à l'exporteur internes. DEM, climat, coarse pondéré et
latent pondéré sont finis. Les 107 fichiers d'implémentation de chacun des
deux manifests correspondent aux sources présentes. Les poids n'ont pas été
rehashés indépendamment dans cette revue finale.

| Site original | Eau NN v2 | Plage DEM v2 |
|---|---:|---:|
| Côte seed 0 | 37,80 % | −483,7 à 3 109,7 m |
| Côte seed 42 | 20,02 % | −68,9 à 5 035,3 m |
| Crête seed 0 | 0 % | 4 255,5 à 7 202,2 m |
| Crête seed 42 | 0 % | 4 010,7 à 6 180,7 m |

Ces métriques ont été recalculées. Les quatre planches natives ont été vues :
côtes avec eau ouverte et relief intérieur détaillé ; crêtes et vallées NN
structurées. La côte seed 42 reste décalée par rapport au champ analytique,
mais n'est plus le crop presque entièrement terrestre de v1. Aucun masquage
ou écrêtage du DEM n'est ajouté. La sonde 32 × 16 garde le signe terrestre aux
512 points, avec NPZ/hash/manifest vérifiés ; cela ne démontre pas une
topologie apprise continue entre des points séparés d'environ 1 250 km.

Le supplément `historical-comparison.json` distingue bien les quatre colonnes
historiques/nouvelles, sans annoncer un rejeu A0/A3. Les fractions à ±1 m ont
été recomputées depuis ses NPZ : A4 v2 conserve 1,19 % / 4,06 % de surface
très proche du niveau marin aux deux côtes. La disparition du défaut de formule
ne prouve donc pas l'absence de plats ou de motifs locaux du checkpoint ; cette
qualité reste à explorer sur d'autres sites et seeds.

Les limites climatiques restent explicites : support naturel dépendant de la
seed, clipping surtout polaire mais aussi montagneux, sans preuve de support
joint d'entraînement. Le relief global reste un prior géométrique ; la liaison
au moteur procédural, l'érosion et une tectonique planétaire restent hors scope.

Le P2 de reçu des côtes v1 est traité correctement par un supplément distinct :
le code lie NPZ, métriques, manifest et focus internes, conserve le reçu exécuté
et son exporteur archivé, et expose les différences éventuelles de sources.
Il n'efface pas l'historique et ne prétend pas avoir rerun le GPU.

## Cache exact des statistiques

`terrain_world.py`, les cinq tests du cache et sa documentation ont été relus.
L'identité contient les sources complètes, les versions NumPy/Rasterio et le
calcul des statistiques/lapse rate. Un hit disque conserve le hachage des
sources et évite uniquement leur décodage/calcul. Les tableaux sont validés
par identité, dtype, dimensions, finitude et SHA ; les écritures temporaires
uniques puis remplacement atomique conviennent aux processus concurrents.
Un cache absent, corrompu ou inaccessible ne bloque pas le calcul normal.

Une recomputation réelle, persistence désactivée, a été comparée au chargement
du cache courant : mêmes clés, scalaires, types, dimensions et octets de chaque
array (`source_digest=f6a3c368d63cde65`). Aucune différence détectée. La limite
du cache process à une entrée demeure documentée. Le gain de quelques dixièmes
de seconde de cette fonction ne justifie aucune promesse d'aperçu instantané.

## Runtime, harnesses et navigation

Les deux harnesses enregistrent désormais les chemins des modules réellement
importés, la référence NPZ et son hash/label/reçu. `--require-exact` bloque aussi
les différences de zéros signés. La restauration remet exactement les objets
`_f` sauvés. Le test CPU du helper réel couvre les batches 1/4/9/16, deux passes,
valeurs finies/NaN/±inf/poids nuls, et compare les octets des entrées, conditions,
labels et sorties au chemin historique indépendant.

Les neuf lignes du supplément CPU sont liées aux reçus/NPZ courants par hashes
vérifiés. Les limitations de provenance rétroactive et l'absence de nouveau
run GPU du harness durci sont dites explicitement. Les deux SHA runtime sont
inchangés par rapport à `numerical-summary.json` et son corpus historique de
92 arrays déjà vérifié indépendamment. Le bénéfice LOD 3 reste étayé par environ
0,7 s de préparation économisée ; une taille de gain mural LOD 2 ou un p95
stable n'est pas certifiée.

Exécution CPU personnelle :
`python -m unittest test_source_statistics_cache test_runtime_benchmark test_runtime_latent_batch test_terrain_bootstrap -q` :
**17 tests PASS**, 2,735 s. Aucun checkpoint ni CUDA lancé par la revue.

Les nouvelles assertions UI ont été relues : nouveaux travaux limités au stage,
annulation des vols hors vue, conservation des vols utiles, absence de
redémarrage de leurs chemins et fallback initial non vide. Le succès du test
Chrome mock est rapporté par le coordinateur ; il n'a pas été rerun ici.

Le reçu réel `browser-continental-v2.json` et ses deux captures ont été relus,
avec hashes des images vérifiés. LOD 11 affiche explicitement
`conditioning-preview`, 8/8 tuiles ; le parcours natif suit 4/3/2/1/0, reçoit
25 misses physiques et termine sur 15/15 tuiles decoder 30 m. Temps observés :
premier aperçu 9,377 s, complétion du parcours natif 12,394 s. Le temps 717 ms
de la seconde mise en vue LOD 11 ne doit pas remplacer le premier aperçu à
froid. Ces observations sont une seule session, préparation mondiale et
préchargement désactivés ; elles ne mesurent ni présentation matérielle ni
débit prolongé sur des vues sans cache.
Le premier affichage LOD 4 du parcours v2, à 5,182 s, inclut le premier
chargement des modèles après redémarrage ; il n'est pas comparable au parcours
natural antérieur mesuré avec des modèles déjà résidents.

## Empreintes examinées

Le verdict est lié à cet instantané. Manifests NN natifs validés :
seed 0 `2a5406f7709ead12050aa0cf371d22899369dc0bfa9a3e8623f10707edb254ae`,
seed 42 `503abad85de47e1fc9f61cbf92d520d124cec03c6fbe5426903312b5ef867a8b`.
Le coordinateur annonce ensuite de nouvelles corrections CPU après Opus r2
(identité du cache et diagnostics macro). Elles sont hors de ce snapshot ;
cette acceptation ne prévalide ni leurs hashes ni leurs nouvelles preuves.

```text
terrain_macro.py       20b4627b553aa8ab91671c02dc41045bcd80da29a1d8668607e80f49eebf86e4
terrain_world.py       7e40bd3020dc10ea5cc514c8d97ad00e056f000047547ece4556bcca7f8b2af5
terrain_inference.py   03fd47dd5c2e6219628b42669865929f58f029350aa3d8cd320d57d83386632c
terrain_nn_constants.py 2d2465089ba7b0e1634e12f2624c83b6833644594f52d9e1f6c077ee3113b87d
```
