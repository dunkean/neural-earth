# Base NF4 : première vérification des raccords physiques

12 octobre 2026. Le run100k continue sur5090. Contrôle4090 effectué sur l'EMA20k,
avec **coarse4 et decoder distillés livrés dans Git**, identiques pour le base
original et le candidat. Site `low-plain-sea-202--240-0`, LOD3 et0.

## Adaptateur d'inférence vérifié

`native_weight_runtime.py` fige les kernels NF4 après le cast BF16 puis leur
normalisation MP/Fan-in. Le solveur, fenêtres64/pas32, deux étapes, conditions
58 et attention restent natifs. Il faut préserver ces convolutions : repasser
la préparation générique dessus normaliserait leurs poids une seconde fois.
Les deux outils de comparaison concernés préservent donc ce forward.

Trois vrais batches mixtes sur4090 : **écart maximal exactement0** entre
l'adaptateur, le QAT et le fichier packé20k. Preuve :
`~/data/distill/eval/native-nf4-runtime-20000-proof.json`.
Quinze tests CPU passent, dont chargement EMA seul, identité du forward NF4,
interfaces natives et calcul des raccords sur les deux axes. La préparation
et l'entraînement actifs ne sont pas modifiés.

## Quatre tuiles froides contre une grande requête

Chaque tuile démarre avec un nouveau monde/cache, dans l'ordre inverse. Les
256x256 utiles de quatre requêtes avec halo24 sont comparés aux mêmes pixels
d'une grande requête ; les deux axes et les zones communes des halos sont
mesurés. Le teacher subit exactement la même procédure.

| LOD | Base | MAE toutes zones (m) | MAE raccords (m) | Erreur maximale de saut (m) | Écart maximal dans les halos communs (m) |
|---|---|---:|---:|---:|---:|
|3|Original|0,1130|0,0872|0,8051|0,6110|
|3|NF4 EMA20k|0,1179|0,0830|0,8821|0,4835|
|0|Original|0,000116|0,000402|0,01214|0,03181|
|0|NF4 EMA20k|0,000073|0,000251|0,01529|0,03639|

Ces écarts de partition sont du même ordre que ceux du teacher sur **ce site**.
Ils ne mesurent pas l'erreur du relief NF4 par rapport au relief original et
ne prouvent pas tous les cas de raccords. Les calculs physiques existent aussi
avant le filtre d'affichage ; ce test suit le chemin d'affichage de l'app,
y compris son filtre habituel au LOD3, sans filtre ajouté pour améliorer NF4.

Rapport, CSV et tableaux bruts :
`~/data/distill/eval/native-nf4-cold-seams-20k-coarse4/`.
Job `base-nf4-cold-seams-20k`, terminé normalement, quatre lignes référence /
NF4 × LOD3 /0. Aucune planche finale n'est publiée à ce stade. La côte humide,
les montagnes, d'autres ordres de requête et les candidats finaux restent à
vérifier avant de déclarer le base livré.
