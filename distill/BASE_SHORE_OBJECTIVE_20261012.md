# Pondérer les erreurs terre / mer du base

L'utilisateur propose de compléter l'erreur d'altitude par le fait de rester du
même côté de **0 m** que le teacher. C'est possible avec une perte différentiable.

## Objectif déjà disponible

`decoded_loss.shore_loss` compare `sigmoid(hauteur / 2 m)` de l'élève et du
teacher dans une bande où la hauteur **teacher** est comprise entre -20 et
+20 m. C'est une divergence entre probabilités terre/mer, équilibrée entre les
deux classes et entre les fenêtres actives. Une prédiction exactement égale
au teacher a une perte et un gradient nuls. Les pixels invalides et le relief
éloigné de cette bande ne contribuent pas. Les mauvais signes reçoivent un
gradient qui ramène leur hauteur vers celle du teacher.

Le lissage à 2 m évite d'imposer une marge artificielle de hauteur à chaque
pixel côtier. Un booléen `hauteur > 0` seul n'a pas le gradient requis pour
entraîner le réseau. La température peut être réglée ensuite ; un seuil plus
strict augmente aussi la sensibilité aux très petites variations autour de 0.
La cible de distillation reste la côte du teacher sur le même coarse.

Il faut appliquer cette perte à la **hauteur reconstruite**, après fusion et,
au LOD0, après passage dans le decoder figé. Le zéro d'un canal latent n'est
pas directement le niveau de la mer. La reprise NF4 actuelle utilise les
pertes d'opérateur/latents et KD ; l'altitude et les erreurs de signes sont
mesurées en évaluation. Cette pénalité n'est pas activée dans le run100k.

## Contrôle effectué sans changer les poids

Code : `native_shore_probe.py`. Job4090 `base-nf4-shore-gradient-probe`, terminé
avec code0 ; le trainer5090 et son observer restent actifs. Checkpoint EMA20k,
côte humide et plaine basse/mer, coarse4 et coarse2, LOD3 et LOD0 : **8 cas**.
Les gradients des mauvaises classifications corrigent tous leur hauteur dans
la bonne direction, et des gradients finis/non nuls atteignent les latents du
base. Au LOD0 ils traversent bien le decoder original figé. Aucun poids n'a
reçu de mise à jour. Pic d'allocation PyTorch du probe : environ4,69GiB.
Six tests CPU de la pénalité et des gradients passent aussi.

Exemple de l'évaluation standard EMA20k/coarse4 : sur le patch plaine basse/mer
au LOD0, MAE0,477m mais **14,45% des pixels** ont une classe terre/mer différente.
Le probe reproduit environ0,479m et14,46% ; ses agrégations équilibrées par
classe sont distinctes de la fraction brute de pixels. Ces chiffres portent
sur l'intérieur d'une fenêtre, pas sur une planète ou la continuité des tuiles.

Preuve complète, scènes, SHA du checkpoint/code et normes des gradients :
`~/data/distill/eval/native-nf4-shore-gradient-20000/report.json`.

## Utilisation envisagée

Ajouter cette pénalité comme petit terme complémentaire, avec son coefficient
calibré sur les gradients TRAIN, et conserver les pertes de relief. Un premier
terme LOD3 sur le champ bas est peu coûteux ; il ne couvre pas à lui seul la
côte finale LOD0, qui dépend aussi du decoder et des autres canaux. Vérifier le
LOD0 sur des fenêtres complètes reste nécessaire. Il faudra comparer poids
identiques avec et sans ce terme avant de revendiquer un gain. Le probe actuel
prouve que la supervision est applicable, pas qu'elle améliore les résultats.
La recette du100k reste inchangée et ses checkpoints sont conservés.
