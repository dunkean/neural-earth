# Reprise du base NF4 jusqu'à 100 000 mises à jour

Autorisation utilisateur : « vise 100k passes go », puis « continue sans t'arrêter ».
Objectif : **100 000 pas au total**, soit 94 000 supplémentaires après le pilote.
Ce compteur désigne des mises à jour Adam, pas 100 000 parcours complets du dataset.

## Recette conservée

Le parent est `~/data/distill/ckpt/native-weight-nf4-pilot/latest.pt`, terminé à
6 000 pas. La reprise conserve exactement poids bruts FP32, EMA FP32, moments
et compteurs Adam, RNG CPU/CUDA et compteur global du sampler. Le teacher natif,
la quantification des seuls poids NF4 par groupes de 512, les trois branches
coarse, les pertes et le sampling restent identiques. LR constante héritée du
parent : environ **3e-6**, sans réinitialisation ni saut de taux d'apprentissage.
Ce choix poursuit l'adaptation stable ; il ne garantit pas que la qualité
s'améliorera pendant tous les 94 000 nouveaux pas.

Le smoke vérifie l'égalité des états restaurés, les RNG et deux vrais pas de
backpropagation sur la 5090. Un test CPU compare aussi la prochaine mise à jour
Adam à celle d'un modèle qui n'a pas été interrompu. Aucune reprise depuis les
seuls poids EMA : les snapshots d'évaluation ne sont pas des états de reprise.

## Exécution et sauvegardes

Code : `native_weight_quant_resume.py`. Nouveau dossier indépendant :
`~/data/distill/ckpt/native-weight-nf4-100k/`. Le parent reste inchangé.
Jobs durables : `base-weight-nf4-100k-train` sur 5090 et
`base-weight-nf4-100k-evaluate` sur 4090. Évaluation parallèle uniquement :
aucun arrêt automatique pour un seuil visuel de qualité. Une erreur numérique
ou technique reste signalée et le dernier état est sauvegardé si possible.

Planche initiale au 6 000, puis 7 000, puis tous les 5 000 pas :
`~/data/distill/_base/native-weight-nf4long-<coarse>-<step>.png`, avec JSON/CSV.
Même batterie : plaines désertiques/tempérées, côte humide torturée,
transition plaine basse/mer et trois hautes montagnes humides ; LOD 3 et 0,
coarse 2, coarse 4 et original. Le début de la série reproduit les poids EMA
6 000 du pilote. Les comparaisons finales et les raccords froids complets
restent à effectuer sur les candidats sélectionnés.

Snapshots EMA BF16 et état complet `latest.pt` à 7 000 puis tous les 5 000 pas.
Poids bruts supplémentaires à 20 000, 50 000 et 100 000. Le meilleur score
pratique pointe vers un checkpoint conservé ; il ne vaut pas acceptation
visuelle. Export NF4 tous les 10 000 pas, sans exporter chaque petite amélioration.
Des anciens checkpoints terminés sont archivés sur D: avec vérification SHA256
et liens symboliques ; les données d'entraînement restent dans Linux.

## Point de départ, sans promesse de vitesse

Le fichier NF4 du pilote pèse **149,60 Mo**, environ **149,65 Mo** avec manifeste.
C'est la limite de repli autorisée, pas encore la cible préférée 50–100 Mo.
Sur les trois montagnes humides/coarse4/LOD0, EMA6 000 : MAE 41,0/14,1/29,0 m,
amplitudes TPI8 1,002/0,929/0,920. Le grain des plaines au LOD3 reste amplifié :
ratios RMS haute fréquence désert/tempéré 1,84/2,09. L'amélioration visuelle et
les cas rares motivent la suite. Calcul BF16 simulant les poids NF4 : aucun
gain d'inférence, export WASM ou modèle final accepté n'est revendiqué.

## Candidats conservés à29k du run

Le meilleur score pratique parmi les évaluations disponibles reste l'EMA15k
(score0,73035 ; pas une acceptation visuelle). Son export compact est conservé
à `ckpt/native-weight-nf4-100k/packed/step015000/` :149604746octets de poids,
**149654663octets** avec le manifeste. Contrôle4090 sur trois vrais batches
mixtes : écart QAT/reconstruction exactement0. Preuve et SHA :
`~/data/distill/eval/native-nf4-best15k-packed-proof.json`.
Les checkpoints EMA ultérieurs et les poids bruts20k restent disponibles.

Les poids bruts20k ont aussi été évalués sur les trois coarse, deux LOD et
sept scènes, avec le même decoder original que l'évaluation EMA :
`_base/native-weight-nf4longraw-<coarse>-020000.*`.
Comparaison coarse4 au LOD0 :

| Cas | MAE EMA20k / brut20k (m) |
|---|---:|
|Montagne humide620001|36,55 /33,66|
|Montagne humide620002|15,38 /13,56|
|Montagne humide620003|44,41 /34,82|
|Plaine basse/mer|0,477 /0,556|

Sur le patch plaine basse/mer, la fraction brute de classes terre/mer erronées
passe de14,45% à17,71% avec les poids bruts. Sur la plaine tempérée au LOD3,
le ratio RMS haute fréquence passe de2,12 à2,21. Les poids bruts améliorent
donc certains reliefs sans dominer l'EMA sur les cas côtiers/plats. Les deux
sont conservés ; la sélection finale attend les prochains checkpoints et le
retour visuel. Aucun changement de recette, aucun nouveau base par défaut.

## Audit intégral de l'export30k

`native_weight_artifact_audit.py` compare chaque paramètre reconstruit à la
valeur BF16 effectivement utilisée par le QAT, y compris la normalisation MP,
le fan-in et la quantification des kernels. Il vérifie aussi configuration,
compteur, provenance teacher, inventaire et SHA du fichier packé. L'audit
s'effectue sur le type de device utilisé par l'export : les réductions de
normalisation CPU/CUDA peuvent produire des codes différents.

Checkpoint30k sur4090 : **174tenseurs, dont93kernels quantifiés, tous exactement
égaux**. Le test CPU détecte un mauvais checkpoint à métadonnées identiques
(poison du gain de sortie), ainsi qu'un compteur source/manifest incompatible.
Preuve : `~/data/distill/eval/native-nf4-artifact30k-audit.json`.
Cette égalité des poids complète les contrôles de forwards ; elle ne remplace
pas l'évaluation du relief, des côtes ou des raccords physiques.
