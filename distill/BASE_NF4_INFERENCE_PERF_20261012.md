# Original / NF4 : première mesure d'inférence

Mesure du 12 octobre 2026, RTX 4090, PyTorch BF16, fenêtres 64×64,
format contigu, embeddings précalculés et CUDA Graphs pour les deux modèles.
Les deux étapes natives sont chronométrées séparément puis moyennées.
Le candidat utilise l'EMA à 15 000 étapes, chargé par l'adaptateur
`native_weight_runtime` avec ses convolutions NF4 déjà déquantifiées en BF16.
La quantification simulée d'entraînement n'est pas chronométrée.

| Batch | Original, ms / fenêtre | Distillé, ms / fenêtre | Rapport distillé / original |
|---|---:|---:|---:|
| 1 | 4,976 | 6,041 | 1,214 |
| 16 | 2,016 | 2,037 | 1,011 |

Chaque temps est la médiane de cinq séries de dix forwards, après chauffe.
Les échantillons individuels, les mesures eager et les pics mémoire sont dans
`/home/delete/data/distill/eval/native-nf4-inference-4090-preliminary.json`.
Le job `base-nf4-inference-benchmark-4090` est terminé avec code 0.

## Interprétation et limites

À batch 16, ce candidat a pratiquement le même temps de calcul que l'original.
À batch 1, l'adaptateur actuel est environ 21 % plus lent. Les deux réseaux
conservent 253 688 037 paramètres, les convolutions et les deux étapes natives.
Le stockage NF4 n'implique donc pas un calcul quatre bits : les convolutions
de cette implémentation utilisent BF16. Aucune accélération WASM / WebGPU
n'est mesurée ici.

Le fichier teacher actuel fait 1 014 772 076 octets (FP32). Une représentation
BF16 des paramètres seuls ferait 507 376 074 octets. Le paquet NF4 vérifié
fait environ 149,65 Mo, manifeste compris. La compression du téléchargement
est distincte de l'occupation GPU après déquantification.

Il s'agit d'une première mesure : l'affichage Windows est actif sur la 4090
et l'entraînement principal continue sur la 5090. Aucun entraînement ne
tournait sur la 4090 pendant ce micro-benchmark. L'attention reste la référence,
les opérations ponctuelles utilisent PyTorch sous Linux, sans kernels exacts
Windows. Ces résultats ne sont pas des temps de génération d'une vue complète.

À la fin, répéter la comparaison sur la 5090 sans calcul concurrent sur cette
carte, avec les poids retenus, puis mesurer une vue complète à coarse et decoder
identiques. Conserver les temps du base isolé et ceux de la chaîne complète
afin de rendre visible le coût du decoder, des caches et de l'orchestration.

## Reproduction

```bash
source distill/env.sh
CUDA_VISIBLE_DEVICES=0 python -m distill.benchmark_native_runtime \
  --checkpoint /home/delete/data/distill/ckpt/native-weight-nf4-100k/step015000.pt \
  --output /home/delete/data/distill/eval/native-nf4-inference-4090-new.json \
  --iterations 10 \
  --note 'Décrire les charges concurrentes et le GPU physique'
```

Le script refuse d'écraser une mesure existante. Le rapport ne marque
`completed: true` qu'après les huit mesures et les contrôles de sorties finies.

## Mesure 5090 après les 100 000 étapes

Le trainer principal est terminé avec code 0. Avant le benchmark, la 5090
était à 0 % d'utilisation et 54 Mio occupés. B continue sur la 4090 ; aucun
training ne partage la 5090 pendant cette mesure. Même protocole que ci-dessus,
avec 30 forwards par série et le checkpoint EMA100k.

| Batch | Original, ms / fenêtre | Distillé EMA100k, ms / fenêtre | Rapport distillé / original |
|---|---:|---:|---:|
| 1 | 4,459 | 5,068 | 1,137 |
| 16 | 1,425 | 1,461 | 1,025 |

Rapport complet : `eval/native-nf4-inference-5090-100k.json`.
Job `base-nf4-inference-benchmark-5090-100k`, terminé avec code 0 et toutes
les sorties finies. À batch16, le candidat est environ 2,5 % plus lent dans
ce runtime. La compression des poids reste son gain actuel ; aucune réduction
du nombre de fenêtres ou des deux étapes n'est réalisée. La vue complète,
les métriques finales et l'audit du paquet100k restent à faire.

## Vue complète : terrain d'un viewport1920×1080, candidat70k

Benchmark terminé avec code0, après la fin des trainings et évaluations A/B.
Les deuxGPU étaient libres de calcul avant lancement. Mesure5090, trois
répétitions par variante/LOD après un passage de chauffe ; ordre intercalé,
monde neuf par mesure, poids résidents. Coarse4 et decoder livré identiques.

| LOD | Original (s) | NF4 EMA70k (s) | Fenêtres base calculées |
|---|---:|---:|---:|
| 3 | 12,408 | 13,971 | 5986 |
| 0 | 2,276 | 2,487 | 310 |

Ce sont les médianes des requêtes **terrain backend** des40tuiles256px
nécessaires à ce viewport, avec un cache de monde neuf, pas un temps HTTP
ou de rendu navigateur. AuLOD3, la zone visible représente environ119439km² ;
les40tuiles couvrent un rectangle un peu plus grand. Les fenêtres calculées
sont strictement identiques pour original et candidat sur toutes les répétitions.
Les sorties sont finies et les tableaux physiques d'une répétition sont conservés.

Les graphesCUDA **du base** sont désactivés pour les deux variantes dans ce
chemin Linux, comme dans le banc physique. Le profil exact est enregistré :
decoder/coarse gardent leurs options communes. La différence avec le micro-bench
CUDA Graphs ci-dessus n'est donc pas une contradiction : cette mesure inclut
les requêtes, transferts, construction des champs et l'adaptateur d'inférence
actuel. Le candidat est environ12,6% plus lent auLOD3 et9,3% auLOD0.
Ces résultats n'anticipent aucune optimisation future ni performanceWASM.

Code : `python -m distill.benchmark_native_view`. Données :
`eval/native-nf4-70k-full-view-5090/report.json` et ses tableaux `.npz`.
Job : `base-nf4-70k-full-view-benchmark-5090`. Un GPU occupé fait refuser
le lancement, et le script conserve les mesures existantes sans les écraser.
