# Base compact NF4 : bilan pour décision

Le candidat conservé est **EMA70k, environ150Mo** : meilleur score de progression
du run100k, sans promotion automatique. Les100000mises à jour sont terminées.
Raw100k, EMA15k/40k/70k et tous les états A/B restent conservés. Le paquet70k
a174tenseurs vérifiés exactement, dont93kernels NF4. Il réduit le téléchargement
par rapport aux1015Mo FP32 du teacher (507Mo de paramètres seuls enBF16),
mais conserve ses254Mparamètres, sa géométrie et ses deux étapes.

**Qualité physique sur5090**, coarse4 et decoder livré identiques. Les nombres
sont des moyennes des MAE par scène ; FP32 désigne l'écart du base original
FP32 par rapport au base original BF16, dans la même chaîne.

| Banc | LOD | NF4 (m) | Écart BF16/FP32 (m) |
|---|---:|---:|---:|
|12lieux critiques : plaines et côtes|3|4,03|2,37|
|12lieux critiques : plaines et côtes|0|2,37|1,97|
|6montagnes humides réservées|3|22,13|17,17|
|6montagnes humides réservées|0|24,45|25,44|
|7lieux historiques|3|37,26|22,43|
|7lieux historiques|0|36,06|21,10|

Les montagnes réservées donnent12–32m de MAE auLOD0. Les vallées restent
visibles dans les planches ; ces mesures ne prouvent pas une simulation
hydrologique. Deux cas historiques sont plus mauvais : monde202earthlike,
LOD0 **93,85m contre41,92m FP32** ; monde404archipelago, LOD0
**66,93m contre33,69m**. Le grainLOD3 reste trop fort : sur les12cas critiques,
pentes entre1,02× et1,63×, puissance de la dernière bande spectrale entre
1,53× et3,84× la référence. La tolérance proposée de±5% n'est donc pas atteinte.
AuLOD0 des montagnes réservées, les pentes restent entre0,962× et1,012×.
Le banc critique aveccoarse2 sur4090 donne des MAE moyennes4,08m/2,55m
(LOD3/0), résultat distinct des comparaisons5090 ci-dessus.

**Performances** : micro-bench5090, BF16/CUDA Graphs/batch16, original1,43ms
par fenêtre contre1,46ms pourEMA100k. Vue terrain1920×1080 avecEMA70k,
cache neuf, trois répétitions, base sansCUDA Graphs dans les deux variantes :
LOD3 **12,41s original /13,97s NF4** ; LOD0 **2,28s /2,49s**. Le gain actuel
est le stockage ; ces temps excluent HTTP et rendu navigateur.

**Raccords** : sept scènes, deuxLOD, quatre tuiles froides contre grande requête,
deuxaxes et halos. Enbatch16, moyennes proches de l'original mais quelques
maxima aggravés. En modecanonique/batch1, les28lignes ont tous leurs écarts
de partition exactement nuls. Cela ne promet pas tous les mondes ni une
continuité exacte dans le profil usuelbatch16.

**Essai auxiliaire terminé** : A/B ont chacun10kpas depuis40k. B réduit le
grainLOD3 mais aggrave biais et côtesLOD0 ; il est écarté comme candidat,
ses poids sont conservés. Un nouvel avisOpus examine la supervision physique
et le biais avant toute autre recette.

**Pour regarder** : galerie locale `http://localhost:8876`,50vues avec curseur
et tableaux ;14planches512px natifs dans `~/data/distill/_base/base-final-070000-*`.
Le serveur live propose70k comme essai, l'original restant la base par défaut.
Rapports et tableaux physiques : `eval/native-nf4-70k-{rare-final,final-mountains,historical-final}-coarse4`.
Poids : `ckpt/native-weight-nf4-100k/packed/step070000/{weights.bin,manifest.json}`.
Protocole/perfs : [rapport](BASE_NF4_INFERENCE_PERF_20261012.md) ;
manips/limites : [état100k](BASE_NF4_100K_STATUS_20261012.md).
