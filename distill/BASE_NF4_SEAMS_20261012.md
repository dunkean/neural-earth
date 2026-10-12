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

## Côte humide sur un monde VAL existant

Le checker accepte désormais `--validation-scenes <capture-native128.npz>`.
Il vérifie splitVAL, révision du teacher et grille32, puis convertit le centre
latent du capture en coordonnées physiques (240m par latent). Cela permet de
réutiliser les cas VAL sans sélectionner les mondes finaux réservés.
Un test CPU couvre les deux axes et rejette les captures TRAIN.

Contrôle terminé sur `val-0696-coarse4.npz`, monde10594, même stack et EMA20k :

| LOD | Base | MAE toutes zones (m) | MAE raccords (m) | Erreur maximale de saut (m) |
|---|---|---:|---:|---:|
|3|Original|1,8353|2,6260|29,7749|
|3|NF4 EMA20k|1,8260|2,6216|16,1162|
|0|Original|0,01447|0,01191|0,62555|
|0|NF4 EMA20k|0,01428|0,01131|0,60449|

Le LOD3 présente des écarts de partition également chez le teacher. Ce n'est
pas une identité numérique ni une preuve de tous les raccords. NF4 n'amplifie
pas les métriques de raccord principales sur ce deuxième cas ; la vérification
visuelle finale et d'autres scènes restent nécessaires.
Rapport et tableaux physiques bruts :
`~/data/distill/eval/native-nf4-wet-coast-seams-20k-coarse4/`.

## Trois hautes montagnes humides VAL

Run4090 terminé normalement sur les trois captures VAL de mondes620001/2/3,
coarse4 + decoder livré, EMA20k, deux LOD :12lignes référence/candidat.

| Monde | LOD | MAE raccords original / NF4 (m) | Saut maximal original / NF4 (m) |
|---|---|---:|---:|
|620001|3|3,9169 /3,6096|34,6821 /60,0610|
|620001|0|0,0229 /0,0218|1,1289 /0,7275|
|620002|3|3,8712 /3,7241|32,4170 /28,3193|
|620002|0|0,0175 /0,0161|0,5525 /0,3901|
|620003|3|2,1808 /2,1408|20,5801 /21,6023|
|620003|0|0,0157 /0,0170|0,7151 /0,6373|

Le premier monde a un maximum NF4 aggravé au LOD3. Les erreurs de saut médianes
restent2,21m pour les deux ; au percentile99 elles sont23,96m original et
24,64m NF4. Le maximum60,06m reste un problème à diagnostiquer, pas un raccord
parfait déclaré. Le contrôle suivant reprend ce monde avec fenêtres base en
batch1 canonique pour distinguer une variation de calcul liée au batch d'une
instabilité de découpage persistante. L'entraînement ne change pas.

Données brutes : `~/data/distill/eval/native-nf4-wet-mountain-seams-20k-coarse4/`.
Le checker consigne désormais le profil d'inférence réel et refuse sa
modification lors d'une reprise d'un même rapport.
