# Base : poids seuls NF4, adaptation réellement lancée

12 octobre 2026, après le top utilisateur. ValléesFP8QAT500 jugées bonnes ;
priorité50–100Mo, plafond150Mo, grain des plaines et côte à suivre.

## Format et contrôles

Le réseau natif254M, fenêtres64/pas32, deux phases,58conditions et attention
reste inchangé.93 convolutions internes enNF4 ; activationsBF16 sans arrondi
FP8. Groupes512 en ordreC, échellesFP32, autres tenseursBF16. NF4 utilise une
table16 valeurs issue des quantiles gaussiens ; pas un calcul flottant4bits
matériel. Valeurs : QLoRA / table publique bitsandbytes (MIT), provenance dans
`native_weight_quant.py`.

Fichier `weights.bin` réel : **149 604 746octets**, échelles incluses. Le manifest
ajoute formes/offsets/codebook ; ni accélération ni port navigateur annoncé.
Les kernels quantifiés contiennent la normalisationMP globale et le fan-in :
la reconstruction DOIT les utiliser sans nouvelle normalisation.

Le premier export normalisé surCPU a échoué au contrôleGPU : max~0,10 latent.
Ces résultats restent conservés comme diagnostic, pas livrés. Exportv2 sur
le même deviceCUDA que la simulation : **max0 exact sur3batches réels**,
teacher sur5090 et QAT500 sur4090, les trois coarse et deux phases. Les sept
testsCPU passent : codes/zeros/padding/STE, identité BF16 sans quantification,
gradients, pack/reconstruction/corruption, cache, EMA/snapshotBF16 et objectifs
d'entraînement. Les hashes des sources du training actif restent inchangés.

Départs conservés :
`DATA/eval/weight-nf4-{teacher,qat500}-start-v2/`, avec `initial.pt`,
`packed/{weights.bin,manifest.json}` et `status.json`.
DATA=`/home/delete/data/distill`.

## État physique avant adaptation

Les deux départsNF4 sont nettement moins bons que le FP8QAT500 jugé utilisable.
Montagnes/coarse4 LOD0, teacher : MAE245/106/193m, amplitudes0,829/0,781/0,658.
DépartQAT500 :231/96/178m, amplitudes0,824/0,784/0,652. Le petit gain de MAE
ne constitue pas une récupération des formes. En plaineLOD3, ratios RMS haute
fréquence~4,5–6,6. Ce sont des candidats à adapter, pas des poids acceptés.

Images déjà publiées dans `_base` :
`native-weight-teacher-nf4-<coarse>-000000.png` et
`native-weight-qat500-nf4-<coarse>-000000.png`, CSV/JSON/TXT voisins.
Le teacher original est choisi pour l'adaptation, sans modifier QAT500.

## Entraînement et observation

Job `base-weight-nf4-train` sur5090 ;
`ckpt/native-weight-nf4-pilot/`, budget6000 pas, départ teacher BF16 exact.
Adam3e-5, β=(0,9;0,99), warmup200, cosinus vers0,1LR, clip1, EMA demi-vie500.
Batch8,4par phase,8mondes distincts ;25%montagnes/15%rares/60%général et les
trois coarse par phase. Une moitié des batches utilise la première sortie
ACTUELLE de l'élève pour la seconde phase, fusion/détachement ; teacher
résident évalué sur les MÊMES entrées.

Loss : opérateur5canaux avec varianceTRAIN, canal0 première phase×2,
gradients0,05, x0 des5canaux0,25 avec varianceTRAIN globale ; KD37blocs,
coefficient0,202752 calibré sur16batchesTRAIN à0,25de l'opérateur pondéré.
Pas de loss d'énergie spectrale isolée qui pourrait récompenser du grain
décorrélé. La géométrie et les bruits restent natifs.

Smoke5090 :3vrais backprops, gradients finis, teacher sans gradients,
états on-policy4/0/4, EMA deepcopy avec owners corrects ; pic7,64GiB.
Le training avecEMA atteint~8,61GiB. Ces preuves ne garantissent pas la qualité.

Job `base-weight-nf4-evaluate` sur4090, comparaison tous500pas,
coarse4/coarse2/original, sept scènes et deuxLOD. Métriques ajoutées : RMS
haute fréquenceσ1, biais d'altitude, désaccord terre/mer global et près de±20m.
Photos : `_base/native-weight-nf4qat-<coarse>-<step>.png`.
`_base/image-publications.jsonl` enregistre chaque nouvelle image ; l'assistant
doit prévenir l'utilisateur lorsqu'une nouvelle série est disponible.

Pas de garde canal0÷2 à1000. Conserver le meilleur score pratique comme
candidate, sans en faire une acceptation. Arrêt qualitatif seulement sur
dégradation catastrophique persistante des six montagnes/coarse2/4 après3000,
trois contrôles consécutifs ; divergence/nonfinite restent des erreurs.
Les snapshotsEMA BF16 reproduisent le cast fonctionnel d'évaluation ;
`latest.pt` garde mastersFP32/EMA/Adam/RNG pour reprise. Aucun checkpoint
important supprimé. Export packé des meilleurs jalons ; pas de poids dansGit.

Anciennes poidsFP8 terminaux archivés à froid, avec SHA et symlinks conservés,
QAT500 reste sur le filesystemLinux. Preuve :
`preparation/fp8-for-weight-qat-cold-archive.json`. Données actives Linux.

## Premiers résultats de progression, pas une acceptation

Coarse4, trois montagnes humides au LOD0 :

| Pas | MAE des trois scènes (m) | Amplitudes TPI8 |
|---|---|---|
|0|244,6 /106,8 /191,3|0,828 /0,781 /0,658|
|500|199,1 /69,4 /140,4|0,884 /0,836 /0,713|
|1000|142,8 /37,2 /87,2|0,938 /0,874 /0,777|

Plaines désertique/tempérée au LOD3, ratios RMS haute fréquenceσ1 :
4,54/6,59→4,29/6,13→3,08/4,23 aux mêmes paliers. Le fitting progresse mais
le grain reste excessif et la qualitéFP8QAT500 n'est pas encore récupérée.
Les PNG500/1000 et les fichiers numériques associés sont publiés dans `_base`.
Au1000, paquet poids+manifest :149 654 662octets, sans accélération acquise.

Revue complémentaire Opus5.5 xhigh lancée en lecture seule sur ce changement
de stratégie et les preuves, sous `review/opus-weight-nf4-start-20261012/`.
Ne pas modifier silencieusement la recette d'un run actif sur un avis : vérifier
ses premiers résultats et conserver sa provenance.

## Suite et limites

Le premier pilote doit montrer si le4bits à topologie intacte récupère une
qualité utile≤150Mo. Si ce n'est pas le cas, la réduction progressive vers
128×3 et des poids8bits fournit un palier de secours estimé125Mo, avant la
cible50–100Mo. Rien n'est déclaré acquis sur les nouvelles architectures.

Sur une candidate visuellement retenue : validation plus large, mondes
réservés, vraie chaîne app et raccords froids sur les deux axes/ordres de
chargement, export et reconstruction contrôlés, puis timings au repos.
La présence des deux phases et du blending ne certifie pas les seams.
Le runtimeWASM/WebGPU et ses optimisations restent à l'utilisateur à ce stade.
