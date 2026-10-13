# Base : reprendre la géométrie native et entraîner la précision FP8

**État terminal :** le pilote s'est arrêté au vrai pas1057 sur3000 prévus,
après le garde1000. Les jobs5090/4090 sont terminés. Sur les montagnes/coarse4
au LOD0, checkpoint500 :14,9/14,8/36,8m ; checkpoint1000 :47,5/28,0/75,2m.
Le500 est conservé pour examen visuel, sans acceptation ni accélération.
[Images et métriques](BASE_QAT500_VISUAL_20261012.md).
La [revue globale](BASE_EXPERIMENTS_REVIEW_20261012.md) et les deux avis sont
terminés ; [choix utilisateur en attente](BASE_REVIEW_SYNTHESIS_20261012.md).
Le texte ci-dessous documente la recette exécutée, pas un training actif.

Le pilote à quatre sorties s'est arrêté au1051, sur son critère1000 : baisse
du canal0 seulement4–5%, contre15% demandés. Montagne/coarse4 :159/123/89m,
corrélations0,838/0,858/0,921. Les poids, états Adam/RNG et photos restent
conservés. Les essais de champ sont archivés à froid avec symlinks et SHA
vérifiés (`preparation/field-heads-cold-archive.json`), sans toucher au teacher
ni aux données et checkpoint actifs sous Linux.

La revue réelle Opus5.5 (`review/opus-field-heads-stopped-20261012`) recommande
de suspendre cette famille et de distiller autour de la quantification. Cette
reprise conserve le réseau original192×3, ses fenêtres64/pas32, les deux
phases, les58 scalaires de conditionnement et l'attention native. Les échecs
des têtes ne prouvent pas que les fenêtres soient sans effet ; ils montrent
que cette modification limitée ne suffit pas. Aucun gain dû à la suppression
des recouvrements n'est acquis ni revendiqué.

Le contrôle natif BF16 sans quantification reproduit le terrain enregistré à
0,012–0,17m sur les anciennes scènes rares et0,88–1,80m sur les trois montagnes
humides/coarse4. La quantification simulée, sans entraînement, donne sur ces
montagnes27/20/52m et des corrélations0,991/0,988/0,997. Sur les neuf branches
montagneuses original/coarse4/coarse2,15,5–81,8m ; le rappel des vallées et
certaines formes de côtes ne passent pas encore. C'est un meilleur point de
départ, pas une acceptation. Photos réelles et alternatives CSV/JSON/TXT :
`_base/native-fp8-ptq-<coarse>-000000.*` et contrôle
`_base/native-fp8-bf16-control-coarse4-000000.*`.

`NativeFP8Base` arrondit les poids effectifs après normalisation MP globale en
E4M3, avec une échelle par canal de sortie. Les entrées des93 convolutions
internes ont une échelle statique par couche. Celle-ci est calculée exactement
sur le percentile99,99% des activations BF16 de64 batches TRAIN : histogrammes
de32768 motifs binaires, sans sous-échantillonnage ni VAL. Entrée/sortie,
embeddings, FiLM, softmax/logits, skips et decoder externe restent natifs.
Les calculs de normalisation des poids gardent le FP32 prévu par l'amont.
La simulation arrondit exactement, même sur valeurs BF16 écrêtées, avec gradient
straight-through et masters FP32. Les échelles d'activation restent figées.

Départ du teacher original, Adam neuf1e-5, warmup100, clip1, EMA demi-vie500,
budget3000 pas. Batch8 mondes TRAIN distincts,4 fenêtres par phase,
25% montagne/15% anciens cas rares/60% général, les trois coarse par phase.
Loss native des cinq canaux, premier canal0 pondéré×2, gradients/bandes,
KD sur37 blocs calibré à0,5 de l'opérateur pondéré (poids0,35330).
Dans exactement50% des batches, les quatre appels de seconde phase prennent
les premières sorties **actuelles** de l'élève : neuf fenêtres natives
fusionnées par exemple, détachées. Le teacher résident fournit sa cible sur
ces mêmes entrées. Aucun buffer périmé ou état VAL n'alimente le TRAIN.

GPU smoke réel sur5090 : gradients finis, teacher figé, échelles inchangées,
deux batches on-policy et un batch teacher-state,37 comparaisons KD ; pic7,93GiB.
Trois tests CPU vérifient arrondi/gradient, identité sans quantification,
préservation des couches sensibles et poids×2 du premier canal. Jobs
`base-native-fp8-train`5090, `base-native-fp8-evaluate`4090 ; photos tous500
dans `_base/native-fp8-qat-<coarse>-<step>.*`. Critère1000 : erreur du canal0
divisée par deux et corrélation montagne≥0,97 sur chaque branche.

**Performance : simulation en convolutions BF16, aucune accélération acquise.**
Les sondes locales4090/5090 montrent que `torch._scaled_mm` accepte E4M3,
mais `torch.nn.functional.conv2d` le refuse, y compris channels_last.
Rapports `eval/fp8-native-ops4090.json` et `fp8-native-ops5090.json`. L'entraînement
prépare des poids et échelles ; un chemin d'exécution FP8 réel devra être
mesuré avant toute annonce de gain. L'architecture conserve tous les paramètres
du teacher ; l'économie de stockage dépend d'un export quantifié ultérieur,
les checkpoints d'entraînement restent en FP32.

Acceptation toujours en attente : vallées/incisions visibles, métriques et
autres cas rares, mondes finaux, chaîne app, vrais raccords froids, export
équivalent et timings réels. Aucun poids base expérimental dans Git ; base
original toujours actif par défaut dans l'application.
