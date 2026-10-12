# Base NF4 : sortir du plateau sur les cas rares

Le100k principal reste en cours, recette inchangée. La baisse de perte TRAIN
n'a pas encore donné de meilleur score de validation après15k. On distingue
le grain **LOD3**, encore autour de×2, du **LOD0**, où les plaines ont surtout
un biais d'altitude négatif et un ratio haute fréquence souvent inférieur à1.
Le score agrégé n'est pas une preuve suffisante de qualité finale.

Revues Opus5.5xhigh complètes, conservées sans réécriture condensée :
[plateau](REVIEW_OPUS_NF4_PLATEAU_20261012.md),
[mesures des gradients](REVIEW_OPUS_NF4_GRADIENTS_20261012.md).
Même session CLI `ed09425a-d4b3-4c47-b681-e26d5d2fe7a0` : première lecture
interrompue à la limite10tours, puis reprise réussie ; addendum après les
mesures nouvelles, sans session concurrente. Modèle réel vérifié5.5.

## Préparation et preuves effectuées

- `native_field_auxiliary.py` : prototype sur champ final fusionné. Pénalité
  côte et erreur passe-haut appariée aux mêmes pixels teacher, sans simple
  cible d'énergie qui récompenserait du bruit. Quatre tests CPU passent.
- `calibrate_native_field_auxiliary.py` :129captures TRAIN, trois coarse,
  puissances physiques fixes par catégorie : désert2,295m², tempérée2,286m²,
  plaine/mer0,05458m². Aucun VAL pour régler ces constantes.
- `native_field_gradient_probe.py` : quatre vrais champs TRAIN, gradients
  à travers les deux étapes complètes NF4 et leurs34fenêtres, sans mise à jour.
  Norme principale0,06–0,18 ; norme passe-haut587–1557 ; norme côte0,01–1,22.
  Une pondération passe-haut0,01 serait beaucoup trop forte. Pic9,19Go alloués,
  hors Adam/EMA. Preuve `eval/native-nf4-field-gradient-probe.json`.
- `native_field_attribution.py` :24captures TRAIN au raw40k, deux scènes par
  catégorie/coarse, modes déterministes. E=élève, T=teacher :

| Mesure LOD3 | EE | TE | ET | TT |
|---|---:|---:|---:|---:|
|Erreur de signe près de la côte,12champs|14,23%|12,62%|3,31%|1,39%|
|Erreur absolue log du ratio HF,18champs plats|1,0085|0,9742|0,0584|0,0114|

Ces oracles remplacent une étape par le teacher ; ce ne sont pas des modèles
entraînés. La deuxième étape porte l'essentiel des défauts visés. TT : MAE
moyenne0,304m, biais absolu moyen0,0078m. Deux forwards identiques donnent
exactement le même champ. Ce contrôle LOD3 ne couvre pas encore toute la
chaîne de décodageLOD0 ni la répétabilité du protocole d'évaluation complet.
Les174tenseurs EMA6k pilote/reprise sont également exactement identiques.

## Comparaison contrôlée à préparer sur4090

1. Source **40k complète choisie à l'avance**, raw/EMA/Adam/RNG conservés :
   `ckpt/native-nf4-objective-source/step040000-full.pt`, SHA en fichier voisin.
   Les deux bras partent de cette source, sans réinitialiser les moments.
2. A garde la recette actuelle. B ajoute une correction de **deuxième étape**
   sur neuf fenêtres fusionnées : teacher et élève reçoivent le même état
   initial de l'élève et le même bruit neuf. La première étape est calculée
   sans gradient pour cet auxiliaire. Cela vise le défaut dominant identifié ;
   les pertes régulières continuent à superviser les deux étapes etC0.
3. Calculer la KL côte en espace racine q, bande√20/température√2, au moins
   16pixels par classe. Le terme passe-haut reste en mètres. Il faut refaire
   sa calibration de gradients sur raw40k avec cette configuration finale.
4. Fixer les coefficients sur TRAIN puis plafonner chaque renfort, en normes
   brutes et préconditionnées par Adam. Un seul pas Adam après accumulation,
   clip habituel, LR≈3e-6 héritée et EMA500. Le primaire5090 n'est pas un
   réplicat statistique : GPU et flags diffèrent.
5. A/B sur la même4090, blocs intercalés, mêmes tirages/seed et mêmes flags.
   Cache CPU borné ; objets auxiliaires séparés pour que les nouveaux bruits
   ne modifient pas les exemples du batch régulier. Smokeλ=0 obligatoire.
6. Évaluer les deux LOD, raw/EMA, davantage de scènes/bruits existants. Vérifier
   côte, grain, biais, canal0 et vallées. Le gain LOD3 ne prouve pas le transfert
   LOD0 ; le proxy ne corrige pas directement le biais des plaines hors côte.

Les nouveaux helpers restent dans des fichiers séparés. Les fichiers dont
les SHA protègent le trainer principal ne sont pas modifiés. Les anciens
checkpoints inactifs sont archivés avec SHA identiques, sans supprimer leurs
poids. Aucune nouvelle base par défaut, aucun poids base ajouté dans Git.
Les planches finales pour décider seront produites après la comparaison.
