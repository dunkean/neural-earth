# NF4 : entraînement 100k terminé, sélection et validation en cours

Le job principal `base-weight-nf4-100k-train` est terminé à 100 000 mises à
jour, code 0. L'entraînement de reprise a duré 32 439 secondes, soit environ
9 heures. L'état complet raw/EMA/Adam/RNG, tous les snapshots EMA et les raw
20k/50k/100k sont conservés. Aucune acceptation du dernier checkpoint n'est
déduite de la fin du training.

## Qualité sur les scènes de progression

Le meilleur score du moniteur parmi les points évalués jusqu'à95k est
**70k : 0,64844**, contre 0,72970 à40k. Ce score est un outil de classement,
pas un seuil d'acceptation. Les six évaluations 100k (raw/EMA × coarse4/2/original)
ont terminé normalement sur5090 ; le dernier EMA est moins bon que70k.

| Scène, LOD0, coarse4 | EMA70k MAE (m) | EMA100k MAE (m) | Raw100k MAE (m) |
|---|---:|---:|---:|
| Plaine désertique10294 | 2,554 | 6,632 | 4,817 |
| Plaine tempérée10296 | 1,392 | 9,732 | 5,563 |
| Côte humide10594 | 1,714 | 6,736 | 4,408 |
| Plaine-mer10296 | 0,145 | 0,375 | 0,144 |
| Montagne humide620001 | 26,128 | 56,929 | 36,530 |
| Montagne humide620002 | 13,113 | 32,296 | 16,141 |
| Montagne humide620003 | 33,278 | 50,374 | 32,557 |

Le biais négatif explique une grande partie de la régression EMA100k ; dans
la première montagne, il vaut −56,921m. Le raw100k est meilleur que son EMA
sur ces cas, sans surpasser systématiquement70k. Les vallées, le grainLOD3,
les côtes, les biais et les raccords doivent rester des critères séparés.
Les chiffres ci-dessus couvrent sept scènes de progression et ne constituent
pas encore le banc étendu de validation finale.

Rapports complets et planches :
`~/data/distill/_base/native-weight-nf4long-<coarse>-100000.*`,
`native-weight-nf4longraw-<coarse>-100000.*`,
`native-weight-nf4long-<coarse>-070000.*`.

## Poids compacts et performances

Audits5090 des paquets70k et100k terminés : **174tenseurs, dont93kernels
quantifiés, tous exactement identiques en valeurs BF16 effectives** à leur
sourceQAT. Les fichiers poids font149604746octets ; le manifeste ajoute
environ50Ko. Preuves :
`eval/native-nf4-artifact70000-5090-audit.json` et
`eval/native-nf4-artifact100000-5090-audit.json`.

Benchmark EMA100k5090 : à batch16 / CUDA Graphs, original1,425ms/fenêtre,
distillé1,461ms. Le gain actuel est la compression de stockage, pas le
calcul. [Rapport et protocole](BASE_NF4_INFERENCE_PERF_20261012.md).
Les paquets restent horsGit.

## Travail restant en cours

- B termine ses10000mises à jour auxiliaires sur4090 ; A est terminé.
  Les trois points intermédiaires montrent moins de grainLOD3 mais davantage
  de biaisLOD0 pour B. Tous les raw/EMA sont conservés.
- Le contrôle des raccords physiques70k tourne sur5090 : sept scènesVAL,
  deuxLOD, coarse4 et decoder livré communs aux deux bases, grande requête
  contre quatre tuiles froides en ordre inverse, raccords et halos sur les
  deuxaxes. Sortie : `eval/native-nf4-70k-cold-seams-5090`.
- Comparaison finale des candidats retenus sur des scènes contrastées,
  contrôle de généralisation, planches finales de bonne qualité et temps sur
  une vue complète restent à produire. Aucun poids base n'est promu par défaut.
