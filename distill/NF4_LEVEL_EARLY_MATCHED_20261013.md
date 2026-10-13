# Première comparaison A/B à72k, options déterministes

Deux mille mises à jour depuis le même70k : A garde KD seul, B ajoute la
correction basse fréquence. Même4090, mêmes sept capturesVAL, coarse4,
decoder original, même empriseLOD3/0. `cudnn.benchmark=False`,
`cudnn.deterministic=True`, TF32 désactivé. B a été réévalué indépendamment :
les14lignes de métriques sont **exactement identiques**.

| Cas, LOD0 |70k départ, MAE(m)|A72k|B72k|
|---|---:|---:|---:|
| Plaine désertique |2,595|3,173|2,135|
| Plaine tempérée |1,379|2,393|1,263|
| Côte humide complexe |1,750|1,687|1,660|
| Plaine très basse/mer |0,147|0,153|0,140|
| Montagne humide620001 |25,816|21,347|18,494|
| Montagne humide620002 |12,867|12,494|13,129|
| Montagne humide620003 |32,541|26,141|26,296|

B gagne enMAE sur cinq cas, régresse sur deux montagnes. La moyenneLOD0
passe de9,627m (A) à9,017m (B), le biais absolu moyen de5,140 à4,244m.
AuLOD3 : MAE15,583→15,329m ; biais absolu moyen4,529→3,661m.
Ce sont des résultats **descriptifs précoces**, pas une sélection : les critères
de la paire se lisent aux jalons prévus et sur les bancs complets. Le grain
LOD3 reste visible. Les valeurs de côte ne sont pas toutes meilleures :
le désaccordLOD0 de la côte humide passe de0,9048% à0,9293%, celui de la
plaine/mer de7,0114% à6,9107%.

Les poids ont été figés depuis les sauvegardes complètes par liens d'inode,
puis exportés en snapshotsEMA avec gainFP32. Le run actif continue ; ses
poids, son étatAdam et sa recette n'ont pas été changés.

## Planches et preuves

Sous `/home/delete/data/distill/_base/`, chaque planche possèdeCSV/JSON/TXT :

- `native-weight-nf4levelstartdet-coarse4-070000.*`
- `native-weight-nf4levelaearlydet-coarse4-072000.*`
- `native-weight-nf4levelbearlydet-coarse4-072000.*`
- `native-weight-nf4levelbearlyrepeatdet-coarse4-072000.*`

Rapport apparié :
`eval/native-nf4-level-72000-matched-deterministic-4090.json`.
Script : `native_level_early_compare.py`. Ces images de progression agrandissent
les champs natifs32/256px ; elles ne remplacent pas les planches finales512px.

## Budget d'arrondi, sans modifier le modèle

Sur cinq capturesTRAIN, mêmes entrées de deuxième étape pour toutes les
variantes, repeats exacts des réseauxBF16 etFP32. Le dernier arrondi de sortie
seul représente1,5 à4,7% de l'énergieHF de la référenceF32 conservant les
mêmes constantes trigonométriques. L'ensemble des opérationsBF16 représente
6,7 à12,3% ; les effets des trigonométries sont rapportés séparément.

Cela ne soutient pas une explication du gros excèsNF4 par le dernier arrondi
seul sur ces cas. Le réseauFP32/BF16 produit aussi un écart mesurable :12 à25%
d'énergieHF d'erreur à solverBF16 commun. Ces composantes ne sont pas supposées
indépendantes et ne sont pas additionnées comme un plancher de bruit.

La comparaison change uniquement l'inférence de copies teacher, aucune mise
à jour. Preuve : `eval/native-nf4-rounding-budget-4090.json`. Script :
`native_rounding_budget.py`. Les calculs distinguent mêmes trigonométries,
arrondi final, intermédiaires, trigonométrie et précision du réseau. Deux tests
CPU vérifient l'identité à angle zéro et la symétrie de signe.
