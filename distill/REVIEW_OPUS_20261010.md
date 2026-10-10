# Revue critique de la distillation — 10 octobre 2026

## Décision demandée

Objectif utilisateur : **trois modèles distillés de qualité**, avec des choix
pratiques de look & feel plutôt qu'une perfection numérique obligatoire.
Les seuils initiaux de `TASKS.md` restent documentés ; aucun candidat ne les
satisfait tous. Les données existantes ne prouvent donc pas la réussite du projet.

**Retour live utilisateur, qui prime sur les scores proxy :** coarse8 et decoder
distillés fonctionnent bien à ses yeux. Les bases distillés, quelle que soit leur
largeur testée, sont défectueux : lignes horizontales/verticales et formes
d'érosion mal reproduites. Le base128 initial est le moins mauvais candidat,
mais ce n'est pas un modèle acceptable. Le problème dépasse le seul bruit de grille.
Ce retour n'est pas encore une mesure quantitative de morphologie ou de drainage.

L'utilisateur veut essayer coarse4 lui-même et décider s'il le garde. Il demande
un avis indépendant d'Opus 5.5 avant le prochain entraînement du base. **Aucun
nouvel entraînement n'est lancé.** Les deux essais de loss aux frontières évoqués
avant ce retour n'ont ni été codés ni démarrés.

## Environnement et état live

- Dépôt : `/home/delete/self/neural-earth`, branche `distill`.
- Données/checkpoints : `/home/delete/data/distill`, hors Git, filesystem Linux.
- WSL2, Python3.12, torch2.11/CUDA12.8, GPU0 RTX4090, GPU1 RTX5090.
- Viewer : <http://localhost:8765/?seed=42&profile=orogen>, job durable
  `live-model-review` sur GPU1. Ne pas le modifier pendant cette consultation.
- Sélection conservée au rechargement pour ajouter coarse4 : coarse8, base
  original, decoder distillé. Le choix est ensuite contrôlé par l'utilisateur.
- Reset après Orogen crée un namespace neuronal neuf, conserve seed/caméra/
  paramètres et bootstrap Orogen. Poids et code contribuent aux identités des
  caches. Le serveur permet de remplacer chaque NN indépendamment.

## Teacher, entrées, élèves

Teacher épinglé : `xandergos/terrain-diffusion-30m`, révision
`9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Le base original représente
environ 254M paramètres. Il calcule des fenêtres64² latentes au pas32,
avec deux étapes et blending. Son conditionnement historique est un vecteur58
issu d'un voisinage coarse4×4, une cellule coarse couvrant32 pixels latents.
Les essais historiques one-step sans réentraînement et FP16 base étaient rejetés.

L'élève base régresse directement le **champ final à deux étapes, après blending**.
Entrées32 canaux : deux champs déterministes de bruit5 canaux, coarse6 canaux
normalisés et suréchantillonnés en nearest, masque constant, phases sin/cos64,
histogramme global et padding nul. Sortie5 canaux : quatre latents pour le
decoder et un canal d'altitude basse fréquence. Même construction en train/runtime.

Architecture : U-Net convolutif local, PixelNorm par pixel, width128/192,
depth4, dilations1/2/4 au milieu, sous-échantillonnage stride2, remontée nearest,
heads1×1 FP32 (résiduel CNN + bypass linéaire des entrées). Pas d'attention.
Halo conservateur192 ; tuiles globales d'inférence512² fixées pour éviter les
différences BF16 entre tailles de convolution. Paramètres width128 :13,33M ;
width192 :29,96M. Cette architecture et cette représentation sont à remettre
en question si elles empêchent la reproduction des formes du teacher.

Coarse8/coarse4 : architecture teacher conservée, solver différentiable raccourci
depuis20 évaluations, puis entraîné sur les sorties finales. Ce n'est pas une
compression de paramètres. Decoder : U-Net local width64, fenêtres512 pas384
et blending d'origine conservés. Tous les poids d'inférence exportés sont FP32 ;
calcul principalement BF16. Pas de QAT/FP8 ni de GAN mis en œuvre.

Code à lire en priorité : [student.py](student.py), [features.py](features.py),
[teacher.py](teacher.py), [dataset.py](dataset.py), [train.py](train.py),
[decoded_loss.py](decoded_loss.py), [inference.py](inference.py),
[coarse_solver.py](coarse_solver.py), puis l'upstream
`terrain-diffusion/terrain_diffusion/inference/world_pipeline.py` et
`backend/terrain_inference.py`. Ne pas déduire de cette description que
l'alignement, l'information disponible ou les pertes sont forcément adéquats.

## Données et protocole

Jeu principal : `/home/delete/data/distill/crops/main/`.
Manifest : `manifest.json`; fichiers NPZ et métadonnées dans `base/`, `coarse/`,
`decoder/`. **19 616 fichiers train, 384 val**, validation par mondes/seeds
indépendants. Seeds42/101/202/303/404 réservées aux évaluations ; pas de labels
supplémentaires générés sur ces seeds. La base utilise des cibles256², avec
coarse couvrant le halo et translations de crop alignées32.

Sampler du parent base : 75% uniforme, 25% catégories rares issues du train.
Rapport classifiant : `/home/delete/data/distill/eval/rare-training-coverage.json`.
Cas critiques utilisateur : plaines désertiques, côte humide torturée,
transition plaine très basse/mer, vastes plaines tempérées. Les groupes val
classifiés comptent respectivement30,5,3,17 fenêtres, se recouvrent et ne
constituent pas un échantillon large de toutes les côtes.

Banque physique : sept sites historiques × LOD3/0, plus douze sites rares ×
LOD3/0 ; 38 vues/candidat. Référence BF16 et base FP32 servent à mesurer la
tolérance naturelle **par site**, avec MAE, pentes, spectres et côte. Les
comparaisons n'ont pas encore de métrique spécifique de fidélité des formes
d'érosion. Pentes et puissance spectrale seules ne prouvent pas leur conservation.

## Résultats conservés et performances

Poids EMA immuables : `/home/delete/data/distill/ckpt/candidates/`.

| Base | Checkpoint |
|---|---|
| 128 initial,100k | `base-step100000-f25dbf12474a.pt` |
| 128 couplé,20k | `base-step20000-ce40c7f92b00.pt` |
| 128 affiné,20k | `base-step20000-91abfda4b2b6.pt` |
| 192 affiné,20k | `base-step20000-a93f7b67bdd3.pt` |

Le parent128 était issu d'un élargissement du base96 entraîné295k étapes,
puis100k étapes avec cible256, LR2e-4, batch1, spectral0,02,
bandes spectrales0,05, gradients0,05, poids altitude8, MAE altitude0,02.
Ce n'est donc pas un essai non entraîné. Les passes couplées ont utilisé
des paires base/decoder parfaitement alignées, mais seulement des crops base64².
Les essais128/192 supplémentaires n'ont pas amélioré régulièrement les cas rares.

Mesures pipeline **par étape**, champs neufs, entrées/transferts inclus,
poids résidents, médiane3 répétitions, machine entière au repos :

| Étape/surface | RTX4090 teacher→élève | Gain | RTX5090 teacher→élève | Gain |
|---|---:|---:|---:|---:|
| Base128 initial/2048² |16,272→2,217 s|7,34×|12,728→1,842 s|6,91×|
| Base128 couplé/2048² |16,282→2,250 s|7,24×|12,769→1,848 s|6,91×|
| Base128 affiné/2048² |16,277→2,248 s|7,24×|12,813→1,849 s|6,93×|
| Base192 affiné/2048² |16,230→3,134 s|5,18×|12,842→2,328 s|5,52×|
| Coarse8/128² |0,430→0,216 s|1,99×|0,279→0,195 s|1,43×|
| Decoder/1024² |0,332→0,244 s|1,36×|0,256→0,219 s|1,17×|

Ces gains ne sont ni des FPS ni des temps de vue complète. Le seuil initial
base10× n'est pas atteint. FP32 export : base12853,349Mo, base192119,874Mo,
coarse11,218Mo, decoder13,360Mo. Teacher base environ1014,77Mo, decoder111,71Mo.
Le 192 coûte davantage sans résoudre les défauts : plaine tempérée très plate
pente×5,04 et bande fine×15,47 ; seconde côte humide MAE71,95m et désaccord
terre/mer12,04%. Contrôle128 :×5,12/×16,04,69,84m/11,80%. Ce sont des cas
mesurés spécifiques, pas les moyennes du corpus.

Raccords physiques froids des candidats retenus :72 contrôles, écart maximal
zéro sur joints et halos avec tuiles globales512 fixées. Deux diagnostics
base à tailles de convolution variables restent des rejets BF16 archivés.
**La cohérence du découpage n'est pas une preuve d'absence de lignes visuelles
à l'intérieur des tuiles, ni de fidélité du relief.**

## Livraison existante et liens vers les preuves

Pack figé : `/home/delete/data/distill/final/index.html` ; galerie interactive152
paires,18 planches PNG2688×1895 et PDF18 pages, CSV de qualité/performances/tailles,
12 exports et18 checkpoints archivés aux SHA vérifiés. Cette livraison sert
à la décision ; le nom `final` ne signifie pas acceptation. Ne pas l'écraser.

Les rapports physiques isolant le base128 initial se trouvent dans
`/home/delete/data/distill/eval/base128-pilot{,-rare}/` (`report.json`,
`arrays.npz`, `physical-audit.json`). Base128 affiné :
`base128-control-pilot{,-rare}/`; base192 : `base192-decoded-pilot{,-rare}/`.
Chaînes optimisées : `optimized-parent-physical{,-rare}/`,
`optimized-bundle-physical{,-rare}/`. Chaque rapport conserve les digests des poids
et du code. Les données physiques correspondent aux vues exactes enregistrées,
et ne couvrent pas tous les lieux du test interactif utilisateur.

Chronométrages : `/home/delete/data/distill/bench/` ; pack rassemble les CSV et
les JSON sources sous `/home/delete/data/distill/final/evidence/`.
Live : `/home/delete/data/distill/live/`, instructions [LIVE.md](LIVE.md).
Code livré initialement : commit34b253e, bundle sous `final/code/` ; les nouveaux
diagnostics sont dans le worktree courant et ne modifient pas les poids figés.

## Essais récents écartés

Deux essais couplés2000 étapes, même EMA parent, optimiseur réinitialisé,
LR1e-4, seed/budget/sampler identiques. Paires utilisables :2129 train/41 val.
Dans ±20m de la mer :169 fenêtres train,113 mixtes ; val5 mixtes.

| Proxy sur41 fenêtres val | Parent | Témoin | KL côte0,1 | Gradient physique0,01 |
|---|---:|---:|---:|---:|
| MAE globale intérieure,m |23,56|25,11|26,09|24,70|
| Désaccord équilibré près de la mer |35,25%|36,05%|37,52%|36,74%|
| Erreur relative de gradient |0,625|0,675|—|0,659|
| Composante axiale d'erreur,m RMS |5,11|5,55|—|5,49|

**Tous rejetés vs parent.** Le proxy reconstruit l'intérieur via decoder gelé ;
il ne remplace pas l'inférence physique complète. Décisions et contrats SHA :
`/home/delete/data/distill/shore-trial/decision.json` et
`/home/delete/data/distill/gradient-trial/decision.json`. Recalcul parent/témoin :
métriques communes exactement identiques, preuve `control-reprobe-equivalence.json`.
Le suivi auxiliaire Autoresearch a été arrêté pour une erreur de résolution de
baseline ; aucun keep ni promotion. Les rapports complets font foi.

Diagnostic Fourier CPU sur les champs existants :
`/home/delete/data/distill/eval/aliasing-frequency-diagnostic.json`.
Une plaine désertique montre période8latents0,077m RMS vs teacher0,020m.
Cela n'établit pas une cause ni une correction.

Audits sans entraînement, cibles base complètes :

- Biais périodique moyen train seulement, période8/32 :
  `eval/lattice-error/`, `eval/lattice-error-32/`. À période32, variance d'erreur
  altitude max/min1,80 train/2,33 val ; pics aux frontières coarse.
  Soustraire un biais moyen ne change la MAE val que de23,7911 à23,7896m,
  et dégrade légèrement la MSE latente. Correction rejetée.
- Ablation causale du bypass coarse direct, nearest→bilinéaire,
  CNN inchangé : `eval/coarse-skip-audit/`. Coefficient optimal sur500 train
  seulement0,0271. Sur384 val, MAE23,7911→23,7898m ; gradient relatif
  MSE0,42817→0,42787. Lissage complet :35,6376m/0,71080. Rejeté.
- Incréments normalisés par le gradient teacher de chaque fenêtre :
  `eval/boundary-gradient/report.json`,500 train/384 val, aucun seed partagé.
  Canal altitude val : erreur/teacher aux frontières0,57764 vs intérieur0,41778
  (**+38,3%**) ; train0,68403 vs0,49327 (**+38,7%**).
  Le teacher lui-même varie peu entre phases : énergie normalisée aux
  frontières1,0246 vs intérieur0,9831. Les quatre autres canaux présentent
  aussi un surcroît plus modéré. Cela confirme un défaut de gradients aux
  frontières, mais n'explique pas à lui seul la perte d'érosion observée.

## Coarse4 prêt pour décision utilisateur

Checkpoint immuable : `ckpt/candidates/coarse-step50000-4e5f9a331777.pt`,
50k étapes terminées. Rapports38 vues dans `eval/coarse-solver4-pilot{,-rare}/`,
seuils stricts non passés. MAE proxy coarse de validation enregistrée38,03m.
Pas encore de chronométrage pipeline final coarse4 sur les deux GPU au repos.

Export vérifié : `/home/delete/data/distill/live/weights/coarse4/` ; sortie
du checkpoint et de l'export exactement identique sur une fenêtre GPU0,
preuve `export-equivalence.json`. Le menu live propose `solver4`,
label « Distillé · 4 étapes · essai ». Le serveur a été rechargé avec sélection
et nonce de reset conservés, preuve `catalog-verification.json`. Aucune
promotion automatique : l'utilisateur donnera son verdict après le test.

## Questions précises pour la revue indépendante

1. Pourquoi l'élève base peut-il manquer les formes d'érosion malgré l'effort
   d'entraînement et la largeur192 ? Distinguer preuve, hypothèse, bug de
   conditionnement/phase/bruit, limite d'architecture, manque de données et
   objectif mal adapté. Chercher une erreur fondamentale avant plus de training.
2. La régression entièrement convolutive sans attention est-elle un objectif
   raisonnable pour ce teacher déterministe à attention et conditioning par
   fenêtres ? Faut-il conserver la géométrie/attention du teacher, utiliser un
   élève avec attention locale, garder deux étapes plus petites ou une véritable
   distillation de solver ? Quels compromis qualité/vitesse et raccords ?
3. Quels diagnostics de formes appariées ajouter pour empêcher une puissance
   spectrale correcte mais des vallées/crêtes/érosion incorrectes ? Ne pas
   assimiler un score de spectre ou des raccords exacts à la bonne morphologie.
4. Proposer au plus trois expériences **décisives et séquentielles**, avec
   contrôle, coût estimé prudent, seuils de poursuite/abandon et réutilisation
   des données/checkpoints existants. Éviter une nouvelle série de losses à
   l'aveugle, une prolongation192 ou un GAN sans vérifier la cause.
5. Recommander une stratégie praticable : coarse/decoder stabilisés, coarse4
   décidé par l'utilisateur, base reconçu si nécessaire. L'objectif est trois
   modèles de qualité ; on peut revoir le gain10× si sa poursuite détruit le
   relief. Préciser ce qu'il faut mesurer avant de promettre un gain.

Consultation en lecture seule : ne pas modifier le dépôt, les poids, les données,
le serveur, les choix de l'utilisateur ; ne lancer ni entraînement ni génération.
Répondre en français. On attend un avis critique fondé sur les sources et les
rapports, avec désaccords explicites si nécessaire, puis un plan concret.
