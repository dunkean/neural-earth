# QAT500 : examen visuel du checkpoint conservé

12 octobre 2026. Images et mesures déjà calculées ; aucun nouveau calcul GPU.

**Statut après retour utilisateur :** les images QAT avec coarse2/coarse4 conviennent visuellement et les vallées sont jugées bonnes. Le travail restant porte surtout sur la taille embarquée et l'excès de haute fréquence en plaine, avec quelques pixels côtiers dispersés. Le réseau garde254M paramètres et utilise des convolutions BF16 avec arrondi FP8 simulé : aucun gain de vitesse acquis. Le checkpoint contient les poids EMA du pas500 ; ses empreintes correspondent aux trois évaluations. Ce retour ne certifie pas encore un export compact ni les raccords finaux.

## Planches existantes

Chaque planche compare les hauteurs réellement calculées du teacher et de la candidate, avec les mêmes lumières, les sept scènes et les deux LOD. Pas de lissage ajouté.

- [coarse4, QAT500](/home/delete/data/distill/_base/native-fp8-qat-coarse4-000500.png) · [mesures détaillées](/home/delete/data/distill/_base/native-fp8-qat-coarse4-000500.json)
- [original, QAT500](/home/delete/data/distill/_base/native-fp8-qat-original-000500.png) · [mesures détaillées](/home/delete/data/distill/_base/native-fp8-qat-original-000500.json)
- [coarse2, QAT500](/home/delete/data/distill/_base/native-fp8-qat-coarse2-000500.png) · [mesures détaillées](/home/delete/data/distill/_base/native-fp8-qat-coarse2-000500.json)

## Montagnes humides au LOD0, les trois coarse

| Coarse | Monde | MAE (m) | Corrélation TPI8 | Rappel des vallées TPI8 | Amplitude élève/teacher |
|---|---:|---:|---:|---:|---:|
|original|620001|10.551|0.998|0.961|0.983|
|original|620002|24.648|0.990|0.952|1.004|
|original|620003|37.754|0.996|0.944|0.969|
|coarse4|620001|14.894|0.996|0.959|0.978|
|coarse4|620002|14.817|0.988|0.937|0.976|
|coarse4|620003|36.803|0.996|0.948|0.965|
|coarse2|620001|10.834|0.998|0.951|0.972|
|coarse2|620002|14.926|0.984|0.940|0.988|
|coarse2|620003|54.041|0.976|0.921|0.992|

## Autres cas critiques avec coarse4, aux deux LOD

| Cas | LOD | MAE (m) | Corrélation TPI8 | Rappel TPI8 | Amplitude élève/teacher |
|---|---:|---:|---:|---:|---:|
|desert-plain|3|3.817|0.656|0.662|1.398|
|desert-plain|0|1.027|0.975|0.885|1.021|
|temperate-plain|3|7.592|0.626|0.688|1.466|
|temperate-plain|0|1.225|0.996|0.961|0.986|
|wet-complex-coast|3|9.162|0.995|1.000|1.005|
|wet-complex-coast|0|8.186|0.873|0.766|1.119|
|low-plain-sea|3|1.335|0.938|—|1.057|
|low-plain-sea|0|0.174|0.981|0.950|1.121|

Une amplitude de1 correspond au teacher. Corrélation et rappel élevés sont favorables, mais le TPI ne certifie pas un réseau de drainage. Un rappel non disponible est indiqué par un tiret. Les valeurs sont celles de ces scènes, pas une moyenne de tous les mondes.

**Points à préserver et améliorer :** conserver les vallées/affluents jugés bons ; réduire le grain dans les plaines au LOD3 ; examiner la découpe des côtes et la transition basse plaine/mer. Le rendu est jugé utilisable par l'utilisateur sur les images coarse2/coarse4 ; les mesures restent disponibles pour comparer les prochaines compressions.

[Toutes les42 lignes CSV](/home/delete/data/distill/_base/native-fp8-qat500-decision.csv) · [Provenance et JSON](/home/delete/data/distill/_base/native-fp8-qat500-decision.json).

## Conservation

- Poids : `/home/delete/data/distill/ckpt/native-fp8-pilot/step000500.pt`.
- SHA256 : `57f505c10b26e2cbef4dc1ca7b56e64a6e4c59455eaaf9181b0d63945e42f893`.
- Taille du checkpoint EMA FP32 : 1014.859Mo. Ce n’est pas un export FP8 packé.
- Aucun changement du modèle base actif dans le runtime ; aucune mise en Git de ces poids.
- Les nouveaux entraînements restent en attente du top utilisateur sur le [programme centré sur la taille](BASE_SIZE_FIRST_PLAN_20261012.md).
