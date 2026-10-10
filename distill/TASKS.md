# Neural Earth — distillation des trois modèles

2026-10-10 · Grégory Beurier

## Livraison finale demandée et comparaison en cours

L'utilisateur demande les planches **à la fin** : paysages différents et
contrastés, bonne qualité d'image, métriques permettant de choisir. Ne pas
publier de planches intermédiaires. `final_plates.py` prépare huit planches
PNG/PDF à partir des champs physiques **512² natifs**, quatre catégories
critiques × LOD 3/0, mêmes cadrages/lumière, détails centraux et erreurs
signées avec une échelle commune par vue. Deux lieux historiques supplémentaires,
montagnes enneigées et relief aride, sont désormais ordonnancés après les
chronométrages, sur les mêmes poids. Trois cas difficiles déjà présents dans
la banque figée sont ajoutés : seconde côte humide, seconde plaine basse/mer,
plaine tempérée très plate au grain excessif. **18 planches** au total. Ces
vues supplémentaires servent à montrer les limites mesurées ; elles ne
remplacent aucun cas d'acceptation et leur sélection est explicitement motivée
par les erreurs observées. Leur capture
utilise les coordonnées et le seed42 réservés de `screenshots.json`, sans
changer ni requalifier la banque rare. Les CSV constituent l'alternative
textuelle. `decision_pack.py` réunit aussi les 38 vues de chaque candidat,
un comparateur interactif, les exports EMA et les performances par étape.
Il refuse d'associer des images, joints, exports ou chronométrages portant
des poids ou du code différents. Les gains par étape ne sont pas des temps
d'affichage complet.

Les deux essais 128/192 ont fini leurs **20 000 étapes supplémentaires**.
La sélection `best.pt` conservait dans les deux essais le checkpoint du
smoke à 32 étapes ; leur loss de validation n'a pas été améliorée. Pour
mesurer réellement la capacité à budget égal, les inspections utilisent
maintenant `--checkpoint latest` et enregistrent `candidate_step` ; les
preuves du contrôle initial à 32 étapes restent archivées sous
`eval/*-initial32`. Rien n'est promu sur la seule loss. Les 38 tests dédiés
passent, dont le refus d'un checkpoint smoke dans une comparaison `latest`
à budget égal et le refus de planches utilisant des références différentes.
Le contrôle des banques supplémentaires porte ce total à **39 tests** et
refuse des poids différents ou une vue dupliquée. Le dossier final conserve
également une copie des rapports JSON complets sous `evidence/`.

La chaîne durable vérifie successivement contrôle 128, équivalence physique
des optimisations sur les deux anciens bases, essai 192, puis chronométrages
sur les deux cartes au repos. `final-review-pack` attend leur achèvement,
mesure également le base 128 initial avec ses propres poids, puis assemble
`~/data/distill/final/`. Ce dossier reste **à vérifier et à commenter après
la fin des jobs** ; sa préparation n'est pas une acceptation de qualité.

Résultats désormais acquis : contrôle128 mesuré sur les 14 vues historiques
et 24 vues rares, avec huit raccords physiques froids **exactement identiques**
(joints et halos compris). Le contrôle améliore certains lieux, mais sa plaine
tempérée très plate présente encore davantage de grain : il n'est pas promu
sur la seule durée d'entraînement. Le bundle128 couplé sous l'inférence
optimisée est **identique octet pour octet sur ses 38 vues** à la version
antérieure (`optimized-bundle-physical{,-rare}/optimization-equivalence.json`) ;
ses huit contrôles de raccords sont également à écart maximal zéro. La preuve
du parent128 initial est maintenant acquise également : ses **38 vues** sont
identiques à celles d'avant optimisation et ses **huit raccords froids** ont
un écart maximal nul, halos compris. Le candidat192 a terminé son inspection
individuelle et celle du bundle (14 + 24 vues) ; ses huit raccords physiques
froids sont également exacts. Les deux diagnostics base128/192 à tailles de
convolution différentes restent des rejets numériques BF16 documentés, sans
invalider l'identité des tuiles physiques à découpage global fixe.

Le passage 192 n'apporte pas de gain de qualité régulier sur les cas critiques
face au contrôle128 à budget égal : la plaine tempérée très plate reste trop
granuleuse (pente ×5,04 et bande fine ×15,47 contre ×5,12/×16,04 au contrôle),
la seconde côte humide atteint 71,95 m de MAE et 12,04 % de désaccord terre/mer
(contrôle : 69,84 m / 11,80 %). Certaines plaines désertiques s'améliorent, mais
cela ne justifie pas de prolonger aveuglément ni de passer à 256 canaux. Le
parent128 initial demeure un compromis plus lisse à examiner. Aucun de ces
candidats n'est présenté comme ayant satisfait tous les seuils stricts.
Les **18 checkpoints immuables** sont vérifiés dans l'inventaire, dont les
15 anciens aux SHA inchangés. Les chronométrages finaux sont en cours.

Le microbenchmark CUDA Graph du coarse a rencontré une copie CPU→GPU interdite :
son helper de capture était hors du mode d'inférence et contournait les caches
exacts du runtime. `bench_student.capture` couvre maintenant warmup et capture
par `torch.inference_mode()`. Les rapports réseau enregistrent ce mode et le
SHA du script ; le coordinateur recalcule les anciens rapports réseau, tandis
que les mesures de pipeline complètes et les preuves physiques restent
valides (leurs chemins utilisent déjà le mode d'inférence). Aucun poids ni
code d'inférence n'a changé pour cette correction de mesure.

Le benchmark192 est complet sur les deux GPU. Les huit rapports réseau/pipeline
coarse/decoder sont réutilisés pour le bundle128 optimisé : mêmes checkpoints
figés, GPU et SHA des scripts, système au repos, rapports complets. Le sidecar
`bench/optimized-bundle/reused-component-measurements.json` conserve les sources
et empreintes ; ce ne sont **pas** des répétitions indépendantes. Chaque base
continue d'être mesuré avec ses propres poids. Une archive Git vérifiée du code
est préparée sous `delivery/distill-code.bundle`, à rafraîchir après les derniers
commits ; elle constitue une livraison locale, sans masquer le push403 restant.

Les chronométrages sont maintenant complets pour les quatre bases sur les
deux GPU. Poids résidents, champs neufs, construction des entrées et transferts
inclus, dépendances préchargées hors chronomètre, médiane de trois répétitions :

| Étape / surface | 4090 teacher → élève | Gain | 5090 teacher → élève | Gain |
| --- | --- | --- | --- | --- |
| Base128 initial / 2048² | 16,272 → 2,217 s | ×7,34 | 12,728 → 1,842 s | ×6,91 |
| Base128 couplé / 2048² | 16,282 → 2,250 s | ×7,24 | 12,769 → 1,848 s | ×6,91 |
| Base128 +20k / 2048² | 16,277 → 2,248 s | ×7,24 | 12,813 → 1,849 s | ×6,93 |
| Base192 +20k / 2048² | 16,230 → 3,134 s | ×5,18 | 12,842 → 2,328 s | ×5,52 |
| Coarse8 / 128² | 0,430 → 0,216 s | ×1,99 | 0,279 → 0,195 s | ×1,43 |
| Decoder / 1024² | 0,332 → 0,244 s | ×1,36 | 0,256 → 0,219 s | ×1,17 |

Ce sont des gains **par étape**, pas un chronométrage d'affichage complet ; le
seuil initial ×10 n'est pas atteint. Les secondes des évaluations de qualité
ne remplacent pas ces mesures. Les exports d'inférence FP32 vérifiés font
53,349 Mo (base128), 119,874 Mo (base192), 11,218 Mo (coarse8) et 13,360 Mo
(decoder). Le 192 coûte davantage sans amélioration régulière sur les cas rares.
Les captures physiques supplémentaires et l'assemblage des 18 planches restent
en cours ; aucune acceptation stricte n'est revendiquée.

### Dossier de décision terminé

`~/data/distill/final/index.html` rassemble **18 planches PNG 2688×1895**, un PDF
de 18 pages (JPEG qualité100 sans sous-échantillonnage couleur ; PNG sans perte),
**152 comparaisons interactives**, 152 lignes de qualité détaillée, 32 synthèses
des cas critiques, 48 lignes de performance, 72 mesures des grandes vues natives
et 48 diagnostics des structures axiales. Les **12 exports** d'inférence sont
revérifiés octet pour octet contre leurs manifests. Les 72 vues physiques élèves
(quatre candidats, cas initiaux et supplémentaires) ont des joints, halos et
différences de partition exactement nuls.

Contrôle Chromium : les 152 paires d'images chargent, le curseur fonctionne,
pas d'erreur JavaScript ni de débordement mobile. `pdfinfo` confirme 18 pages.
Les **40 tests dédiés** passent. La revue des planches confirme le grain ajouté
par les passages couplés sur la plaine très plate, le lissage du parent128 et
les changements importants du trait de côte dans les transitions très basses.
Le parent128 est le compromis visuel conseillé parmi les variantes examinées,
avec coarse8 et decoder200k ; il n'est pas déclaré conforme aux seuils stricts.
Le choix pratique reste ouvert à l'utilisateur, comme demandé.

La livraison Git reste incomplète : l'API GitHub confirme encore
`permissions.push=false` sur `dunkean/neural-earth` pour le compte configuré.
L'archive Git locale sera rafraîchie sur le dernier commit. Aucun nouveau fork,
changement de remote ni publication de poids n'est effectué sans destination
autorisée. Ce blocage de publication ne change pas les résultats locaux.

Objectif initial : remplacer le base model à 2 étapes (254 M paramètres) par un élève une passe ×10 plus rapide, dans la tolérance BF16/FP32.

Objectif étendu par l'utilisateur le 2026-10-10 : **trois modèles distillés de
qualité : coarse, base, decoder**. Adapter les tâches contre-productives, éviter
la surveillance coûteuse, anticiper les interruptions et les reprises. Aucun
checkpoint n'est accepté sur la seule base d'une loss d'entraînement.

Précision utilisateur en cours de travail : conserver plusieurs bons compromis,
les erreurs peuvent être acceptables selon le **look and feel** ; vérifier
soigneusement la continuité entre tuiles. Les seuils initiaux restent des
diagnostics publiés, sans présenter leur échec comme une interdiction de montrer
ou de choisir un candidat. Conserver les checkpoints figés et les mesures à
côté des images ; distinguer sélection visuelle/pratique et acceptation stricte.

## Contexte

Lis d'abord le doc de passation : [Neural Earth — handoff accélération du base model](https://claude.ai/code/artifact/0bd29cbe-5009-4628-845e-4fed483653ba). Il contient les mesures, les variantes déjà rejetées et le raisonnement ; ce doc-ci n'est que la liste d'exécution.

- **Repo :** `/home/delete/self/neural-earth` (WSL2 Ubuntu, système de fichiers Linux), sous-module `terrain-diffusion` initialisé. Push via le credential helper déjà configuré.
- **GPU :** `0` = RTX 4090 24 Go, `1` = RTX 5090 32 Go. Pas de P2P : 4090 = génération teacher et évaluation, 5090 = entraînement. Toujours `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES`.
- **Le teacher est déjà un modèle de consistance** (`terrain-diffusion/configs/diffusion_base/consistency_base_192-3.cfg`) exécuté en 2 étapes : `t_init`, puis `t = atan(0.35/0.5)`, avec blending des fenêtres 64 au pas de 32. Code : `_build_latent_stage` et `_latent_inference` dans `terrain-diffusion/terrain_diffusion/inference/world_pipeline.py`.
- **Cible de l'élève :** la sortie latente finale à 2 étapes, blending compris. Pas la sortie onestep, pas une seule fenêtre.
- **Règles :** ne pas modifier les poids ni le code amont de `terrain-diffusion` ; tout le nouveau code va dans `distill/` à la racine du repo. Données et checkpoints dans `~/data/distill/`, jamais sous `/mnt/*`. Jobs longs dans `tmux`.

## Phase 0 — environnement

Installation vérifiée : venv et checkpoint épinglé présents. Les commandes
ci-dessous servent à réinstaller sur une machine neuve.

```bash
cd ~/self/neural-earth
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
huggingface-cli download xandergos/terrain-diffusion-30m --revision 9ef8030cb805b433b98ec25c5dddefbac07a9e26
python -c "import torch; print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

- [x] venv créé, torch voit les deux cartes
- [x] Compatibilité du lock Linux vérifiée : `uv pip install --dry-run` audite 118 paquets, aucune modification ni exception au lock nécessaire.
- [x] Checkpoint téléchargé ; `resolve_model_source()` trouve le snapshot épinglé sous Linux, avec les sept fichiers requis présents.
- [x] `python -m pytest tests/python -x -q` passe : **353 tests et 333 sous-tests passent, 22 tests ignorés**, Node 22, GPU 0 visible, serveur arrêté (log `~/data/distill/preparation/python-tests-standard.log`).

**Pièges Linux déjà repérés :**

- `RUNTIME = Path('E:/TerrainDiffusionRuntime')` est codé en dur (`backend/terrain_app.py:8`, plus `terrain_world.py`, `terrain_manifest.py`, `terrain_backend_proxy.py`) ; sous Linux cela crée un dossier `E:` relatif. Les rendre configurables par une variable d'environnement `TERRAIN_RUNTIME` (défaut inchangé sous Windows), pointer sur `~/data/runtime`.
- Node système = v12.22, trop vieux pour Orogen (requis par les sites `orogen` des scripts de mesure). Installer Node 22 LTS via `nvm`.

## Phase 1 — mesures de référence

Avant tout code de distillation : mesurer le vrai débit des deux cartes et confirmer la tolérance BF16/FP32 sur Blackwell. Serveur Neural Earth arrêté, rien d'autre sur la carte mesurée.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0 python tools/benchmarks/benchmark_base_model.py --output ~/data/distill/bench/base-4090.json
CUDA_VISIBLE_DEVICES=1 python tools/benchmarks/benchmark_base_model.py --output ~/data/distill/bench/base-5090.json
CUDA_VISIBLE_DEVICES=1 python tools/verification/compare_base_variants.py run ~/data/distill/eval/fp32-5090 --variants reference fp32base
python tools/verification/compare_base_variants.py sheet ~/data/distill/eval/fp32-5090
```

- [x] ms par fenêtre et TFLOPS pour chaque carte, reportés dans le tableau ci-dessous (repère 3090 : 4,96 ms, 39,1 TFLOPS)
- [x] Borne de gain FP8 brut mesurée séparément avec `torch._scaled_mm`, via `distill/bench_student.py --fp8-only`. Une convolution BF16 n'est pas présentée comme une convolution FP8.
- [x] Comparaison BF16/FP32 sur les 14 cas de la 5090 et investigation des deux écarts >36 m : répétitions identiques octet pour octet, 41,38 m et 36,32 m aux deux holdouts terrestres LOD 0. Décision : utiliser le seuil MAE **mesuré par site et par carte**, sans relever les seuils ±5 % de pente/spectre de l'élève.

| Carte | ms / fenêtre BF16 (batch 16, graph) | TFLOPS | Erreur FP32 vs BF16 |
| --- | --- | --- | --- |
| RTX 3090 (ancienne machine) | 4,96 | 39,1 | 7–36 m |
| RTX 4090 | 1,758 (contiguous) | 110,2 | LOD 3 : 7,10–36,23 m ; LOD 0 : 7,25–38,79 m |
| RTX 5090 | 1,295 (channels_last ; contiguous : 1,324) | 149,6 | LOD 3 : 7,21–35,36 m ; LOD 0 : 9,10–41,38 m |

Attention : le FP32 lui-même diffère du BF16 jusqu'à +12,1 % en pente et +42,9 %
dans une bande spectrale, sur certains holdouts terrestres. Ces observations
ne relâchent pas les critères de distribution de l'élève par rapport au teacher
BF16. Résultats complets : `~/data/distill/eval/fp32-5090/report.json` ; répétition
des deux LOD 0 : `fp32-recheck-5090/arrays.npz` (différence max 0).
L'étalon 4090 comporte aussi les 14 cas dans `eval/fp32-4090/report.json`.
La comparaison physique a partagé la carte avec la génération teacher ; ses
secondes ne sont pas utilisées comme mesures de débit.

Gain FP8/BF16 GEMM brut, trois dimensions représentatives : 4090 ~1,56–1,92×,
5090 ~1,27–2,57×. Ces chiffres excluent conversion et abaissement des convolutions
et ne démontrent pas un gain FP8 du réseau complet. Le backend FP8 complet/QAT
reste après la validation de qualité ; priorité aux trois élèves BF16.

## Phase 2 — scripts à écrire

Quatre scripts dans `distill/`, plus une variante dans l'outil de comparaison existant. Branche `distill`.

**Faits utiles sur le teacher** (`world_pipeline.py`) :

- Le bruit est un champ global déterministe : `gaussian_noise_patch(seed + 5819, y, x, …, channels=5)` pour l'étape 1, `seed + 5820` pour l'étape 2. On peut le recalculer à la volée : inutile de le stocker.
- Le conditionnement par fenêtre (`_process_latent_conditioning`) se déduit de champs coarse denses : moyenne et p5 d'élévation, 4 canaux climat, masque, plus le scalaire `histogram_raw`. Le bruit de conditionnement est désactivé (`COND_MAX_NOISE = 0`).
- La sortie du teacher n'est invariante par translation que modulo la grille de 32 latents (fenêtres et cellules coarse alignées dessus). Aligner les crops sur cette grille et donner à l'élève la phase de position modulo 64 (sin/cos).

| Script | Rôle | Points clés |
| --- | --- | --- |
| `distill/teacher.py` | Génère les paires (entrées denses → latent final à 2 étapes) | Réutiliser la construction de monde de `compare_base_variants.py`. Crops de 256×256 latents alignés sur 32, avec un halo suffisant pour le blending. Stocker la cible (5 canaux, FP16) + les champs coarse + (seed, x, y) ; recalculer le bruit. Profils variés (`orogen`, `natural`, `terrestrial-*`), équilibre terre/mer. **Exclure** les seeds 42, 101, 202, 303, 404 (sites d'évaluation). Reprise sur interruption. |
| `distill/student.py` | Modèle élève | U-Net entièrement convolutif, 96–128 canaux, sans attention globale ni embedding de temps. Entrée : bruit étape 1 + bruit étape 2 + champs coarse sur-échantillonnés + phase de position + scalaires diffusés. Canaux multiples de 16 (FP8). Exposer le champ récepteur pour fixer le halo. |
| `distill/train.py` | Entraînement sur la 5090 | BF16 autocast, MSE sur les latents + loss spectrale multi-échelle contre le lissage (adversarial seulement si le lissage persiste). Checkpoints réguliers, reprise, logs de loss et de pente/spectre sur un petit jeu de validation. |
| `distill/bench_student.py` | Vitesse de l'élève | Temps par surface équivalente à une fenêtre 64×64, même méthode que `benchmark_base_model.py` (CUDA Graphs, BF16, puis FP8). |
| `compare_base_variants.py` (variante `student`) | Évaluation physique | Remplacer l'étage latent par l'élève (tuiles larges + halo, sans blending) via un hook sur `_build_latent_stage`, sans toucher au code amont. Mêmes sites, mêmes métriques, même planche. |

## Phase 3 — lancements et acceptation

1. **Test rapide :** 50 crops sur la 4090, sur-apprentissage d'un petit élève (64 canaux) sur ces crops : la loss doit tomber près de zéro. Mesurer le temps par crop pour dimensionner la suite.
2. **Jeu de données :** ~20 000 crops sur la 4090 dans `tmux` (cible FP16 seule ≈ 640 Ko par crop, ≈ 13 Go). Ajuster le nombre selon le temps mesuré en 1 ; garder 2 % en validation.
3. **Entraînement :** sur la 5090 dans `tmux`, d'abord 96 canaux. Passer à 128 seulement si les critères ne sont pas atteints.
4. **Évaluation :** `compare_base_variants.py run <dossier> --variants reference fp32base student` puis `sheet`, sur la 4090.
5. **Vitesse :** `bench_student.py` sur les deux cartes, BF16 puis FP8.

| Critère (LOD 3 et LOD 0, 7 sites) | Seuil |
| --- | --- |
| Erreur moyenne élève vs `reference` | dans la plage `fp32base` vs `reference` du même site (~10–35 m) |
| `slope_mean`, `slope_p90` | ±5 % de `reference` |
| Bandes du spectre (`psd`) | ratio 0,95–1,05 vs `reference` |
| Trait de côte (`coast`) | ≤ valeur de `fp32base` |
| Jointures entre tuiles sur la planche | aucune visible |
| Vitesse, même carte, BF16 | ≥ ×10 vs `reference` |

**Livrables :** branche `distill` poussée, checkpoint retenu sous `~/data/distill/ckpt/`, chiffres reportés dans les tableaux de ce doc, planches jointes.

Hors périmètre tant que les élèves ne sont pas acceptés : quantization FP8 en
entraînement (QAT) et intégration dans le runtime. La distillation du coarse et
du decoder fait désormais partie du périmètre demandé.

## Exécution active du 2026-10-10

Branche : `distill`. Code amont et poids teacher inchangés. Les sections de
préparation ci-dessous sont conservées comme historique ; les jobs ont reçu le top.

### Implémentation et choix

- [x] `teacher.py` : triples base/coarse/decoder, écriture NPZ atomique, reprise
  par index, 256² latents et halo 192, conditionnement dense, seeds réservées
  exclues, mélange de six profils et propositions terre/mer équilibrées.
- [x] `student.py` : U-Net local, largeur 96 pour base, 64 pour coarse/decoder,
  sans normalisation spatiale ni attention globale. Halo exposé et validé.
- [x] `train.py` : autocast BF16, MSE, gradient/spectre multi-échelle, EMA,
  checkpoints atomiques toutes les deux minutes, états optimiseur/RNG repris,
  batch/crop déterministes par numéro de pas, diagnostics par canal et composante
  d'altitude. Pondérer davantage les canaux d'altitude évite de sélectionner
  un modèle sur les seuls canaux latents du decoder.
  Après le rejet physique du base initial : canal hauteur pondéré 32 plutôt
  que 4 et MAE de la contribution d'altitude basse fréquence en mètres
  (poids 0,02, unité de loss = 100 m) pour les prochains passages principaux.
  L'objectif et le score de sélection changent ensemble ; optimiser/RNG/EMA
  sont conservés, mais le meilleur score est réinitialisé lors du changement.
- [x] Têtes de sortie 1×1 en FP32, reste du réseau sous autocast BF16 : éviter
  un plancher de quantification sur les champs fusionnés. Loss supplémentaire
  sur moyenne–p5 du coarse, normalisée à l'échelle de cette différence ; arrêt
  propre et reprise du coarse au pas 27518 pour appliquer cette recette.
- [x] `bench_student.py` : eager/CUDA Graphs BF16, coût du halo inclus, borne
  GEMM FP8 séparée. Pas de prétention de convolution FP8 sans backend dédié.
- [x] `bench_pipeline.py` : champs latents neufs 1024² et 2048², features CPU,
  transferts et fusion inclus ; mêmes poids teacher résidents, coarse préchargé
  des deux côtés, warmup exclu et trois répétitions. Refus explicite d'une carte
  occupée, vérifié sur la 4090 pendant la génération. L'acceptation exige aussi
  ≥10× sur cette mesure complète du base, pas seulement une borne de convolutions.
  L'inventaire est pris avant l'initialisation CUDA : ici WSL attribue le même
  PID à des workers différents sur les deux cartes. Une mesure provisoire avec
  l'autre GPU actif est permise explicitement, signalée dans le rapport, et ne
  peut pas accepter le modèle (CPU partagé). Les mesures finales exigent les
  deux cartes libres.
- [x] `inference.py` et variantes `student`, `student_coarse`, `student_decoder`,
  `student_all` dans l'outil existant. Base sans blending ; coarse remplace les
  20 pas du solveur d'une fenêtre ; decoder remplace sa fenêtre 512/384.
  Le blending coarse/decoder est conservé dans ce premier prototype pour
  préserver le comportement aux bords. Les checkpoints évalués sont figés en
  mémoire, empreintes des mêmes octets, pour éviter des sites évalués sur des
  poids différents pendant un entraînement concurrent.
- [x] Comparaisons physiques reprenables par (variante, site, LOD), fichiers
  atomiques, erreurs non silencieuses, planches à colonnes réellement présentes
  et table `sheet-summary.csv` comme équivalent textuel.
- [x] `jobs.py` : tmux, PID et identité `/proc` vérifiés, état terminal/code retour
  durable. `run_training.py` : coordination par minute, logs de transition,
  arrêt sur erreur. `check_dataset.py` : audit complet des triples et de la
  séparation par monde. `evaluate.py` : refuse une acceptation sans les 14 cas,
  les seuils physiques, la vitesse sur même carte et la vérification des joints.
- [x] `check_seams.py` : champs calculés en une grande tuile et plusieurs
  petites tuiles sur les mêmes features, mesure des écarts aux joints et à
  l'intérieur. La revue des planches physiques reste explicitement en attente.
  Benchmarks, joints et revue visuelle doivent porter les mêmes empreintes de
  poids et de code que les évaluations physiques ; un benchmark au pas 0
  ne peut pas accepter un checkpoint entraîné.
- [x] Quatorze tests dédiés : bruit/crops négatifs, features globales, invariance
  avec halo, échantillonnage repris, losses masquées/gradients finis, écriture
  interrompue, identité des processus, précision des sorties sous autocast,
  supervision du relief coarse, refus d'une validation incomplète et rejet
  des preuves issues d'autres poids ou sans revue des joints, refus des mesures
  de débit sur champs en cache/partiels ou répétitions manquantes ; conversion
  de hauteur connue 100→121 m donnant une MAE 21 m et gradients masqués finis.

Le masque du conditionnement base upstream est constant (ones), ce n'est pas
un masque terre/mer ; `histogram_raw` est un vecteur de cinq valeurs. Les données
de validation viennent de mondes distincts, répartis entre les six profils
(384 crops sur 20 000, 1,92 %), et sont produits en premier pour permettre
l'entraînement pendant que le reste du dataset est généré.

### Preuves et état des résultats

- [x] 50 triples natural (deux mondes, seeds 10000000–10000001), audit complet
  réussi ; ~0,9 s par triple après chargement, ~51 s pour le test initial.
- [x] Sur-apprentissage coarse 5000 pas : MSE validation sur les exemples
  d'entraînement ~0,00199, ratio pente ~0,971, ratio puissance ~0,998.
  Ce n'est pas une acceptation physique. Le coarse principal a un champ
  récepteur agrandi pour couvrir toute la fenêtre 64².
- [x] Arrêt/reprise réel base au pas 9092, sauvegarde des 57 états optimiseur
  et RNG, reprise confirmée au pas suivant. Smoke base terminé à 20000 pas,
  MSE ~0,0191 : le sur-apprentissage n'a pas atteint une erreur proche de zéro.
  Ce résultat impose une investigation de capacité/apprentissage si les
  entraînements principaux échouent ; il ne constitue pas une validation.
- [x] Smoke decoder terminé à 10000 pas : MSE ~0,0113, ratio pente ~1,120,
  ratio puissance ~1,001. Les pentes restent hors tolérance : non accepté.
- [x] Arrêt/reprise réel teacher principal après 673 triples, conservation
  vérifiée par SHA-256 sur des cibles existantes. Aucun crop durable ne doit
  être régénéré à la reprise.
- [x] Hooks évalués sur un holdout natural LOD 3 : aucun appel teacher base
  dans `student`, appels coarse remplacés dans `student_coarse`. Les checkpoints
  smoke testés sont **rejetés pour qualité** : MAE ~95 m et ~223 m,
  vs étalon ~7 m. Résultats dans `~/data/distill/eval/hook-check/`.
- [x] Diagnostic coarse principal vers 50000 pas sur ce même holdout :
  MAE **168,54 m**, contre 7,21 m pour FP32/BF16, côte modifiée sur 12,50 %
  des pixels. Ce candidat reste rejeté malgré sa petite loss latente. Le
  checkpoint évalué est figé par empreinte dans
  `~/data/distill/eval/coarse-pilot/report.json`. Le test partageait la 5090
  avec l'entraînement : ses temps ne sont pas des benchmarks. La génération
  continue ; réévaluer après exposition au dataset complet avant d'accepter
  la capacité ou de prolonger aveuglément cette recette.
- [x] Base initial terminé à 100000 pas sur 2690 crops figés : proxy MAE
  d'altitude ~46,99 m. Diagnostic physique au même holdout natural LOD 3 :
  **MAE 50,16 m**, côte différente sur 4,17 % des pixels ; pentes proches
  (+1,3 % moyenne, +0,06 % p90), mais les trois bandes PSD hautes valent
  **1,74 / 1,87 / 1,61×** la référence. Candidat rejeté contre l'étalon 7,21 m.
  Rapport et empreinte figée : `~/data/distill/eval/base-pilot/report.json`.
  Carte partagée avec le decoder pendant ce diagnostic : aucun chiffre de
  débit ne doit être tiré de ses secondes. La recette suivante priorise la
  hauteur et sa conversion physique ; validation complète toujours requise.
- [x] Decoder initial terminé à 100000 pas : MSE de validation 0,001276,
  pente 0,9588× et spectre agrégé 0,9761×. Diagnostic physique des sept sites
  LOD 0 : MAE 0,35–4,29 m, sous les étalons locaux, mais PSD hors ±5 %
  (jusqu'à 1,55×) : candidat rejeté. Rapport et SHA figée dans
  `~/data/distill/eval/decoder-pilot/report.json`. La prochaine reprise complète
  ajoute une loss de puissance relative dans les cinq bandes radiales,
  `--spectral-band-weight .05`, avec réinitialisation du meilleur score.
  Les 16 tests passent, dont une perturbation haute fréquence de petite
  amplitude cachée par une basse fréquence dominante et les gradients à zéro.
  Essai CUDA de reprise de 32 pas réussi dans `ckpt/decoder-band-smoke`,
  avec optimiseur/EMA conservés et correspondance des seeds attachée ; les
  checkpoints principaux n'ont pas été modifiés par cet essai.
- [x] Reprise après rejet topologique du monde continents seed 10322, avant
  tout crop de ce monde : `resume_teacher.py` remplace cette seed par 1010322
  du même profil. La correspondance est atomiquement enregistrée dans
  `seed-replacements.json`, auditée et attachée aux prochains checkpoints.
  Les six fichiers témoins gardent leurs SHA ; `teacher.py` et sa provenance
  restent identiques. Une substitution ne peut modifier un exemple existant.
  Les 15 tests de distillation passent. Le coordinateur arrêté à cet incident
  a été inspecté puis relancé, sans refaire les trois passages initiaux.
- [x] Coarse élargi terminé à 200000 pas sur 10484 crops. Le meilleur score
  de cette recette choisit l'EMA du pas **135000**, conservée sous
  `ckpt/candidates/coarse-expanded-step135000-2796a5901b85.pt` (SHA identique
  au rapport `eval/coarse-expanded-pilot/report.json`). Évaluation des 14 cas
  sur 4090 : encore 31–247 m sur natural/orogen ; candidat rejeté.
  La différence mean/p5 ne voit pas un décalage commun de ces deux hauteurs.
  Le passage final ajoute donc `--height-mae-weight .05` en mètres, utilisant
  les échelles enregistrées par le teacher, contrôlées par l'audit dataset.
  Essai CUDA de reprise de 32 pas réussi dans `ckpt/coarse-height-smoke` ;
  proxy MAE de validation ~49,7 m, aucun effet sur les checkpoints principaux.
  Les 17 tests passent, dont l'offset commun invisible à la loss de relief.
- [x] Borne architecture base96, **poids non entraînés, mesure de coût seulement** :
  512² utiles + halo 192 → entrée 896² ; 47,244 ms (graph, channels_last) sur
  4090, 0,738 ms par surface 64², ~19,05× vs huit forwards teacher BF16.
  Entrées CPU et I/O exclus. Cela ne prouve ni la qualité ni le débit de la
  pipeline réelle ; la mesure finale avec les checkpoints retenus reste requise.
  Mesure antérieure à l'ajout des têtes FP32, donc à refaire avec le modèle final.
- [x] Dataset principal : 20000 triples générés, 19616 train / 384 val,
  613 mondes train et 12 mondes val. Audit complet des 60000 NPZ réussi :
  formes, valeurs finies, profils, séparation des mondes, seeds réservées,
  correspondance de remplacement et échelles coarse invariantes.
  Rapport : `~/data/distill/crops/main/dataset-audit.json`.
- [x] Base élargi terminé à 200000 pas sur 12512 crops ; proxy MAE ~28,81 m.
  Diagnostic des 14 cas sur 4090, `eval/base-expanded-pilot/` : MAE
  **15,4–125,05 m**, 14 cas refusés. Sur natural LOD 3, MAE 35,51 m et les
  anciennes grilles sont moins visibles, mais les reliefs orogen LOD 0 sont
  trop lissés (bandes PSD jusqu'à 0,18×). EMA du pas 200000 conservée sous
  `ckpt/candidates/base-expanded-step200000-e145587ec614.pt`.
  Prochaine reprise : poids hauteur 8 plutôt que 32, MAE mètres .02 conservée,
  bandes PSD .05 appliquées aussi au base, pour restaurer les gradients des
  quatre canaux de détail. Essai CUDA de 32 pas réussi dans
  `ckpt/base-balance-smoke`, aucun checkpoint principal modifié.
- [x] Trois familles entraînées et reprises sur le jeu complet audité. Le
  calendrier initial de passages automatiques est remplacé par les essais
  mesurés décrits ci-dessus : coarse à huit étapes, base128 initial/couplé et
  contrôle128/192 à budget égal, decoder200k. Les variantes antérieures restent
  figées ; les prolongations sans bénéfice de qualité sont évitées.
- [x] Évaluation des trois modèles séparément et ensemble, sur 7 sites × LOD3/0
  et la banque rare complète, avec étalon BF16/FP32 mesuré sur la 4090.
  **Mesurer n'est pas accepter** : les seuils stricts restent non satisfaits.
- [x] Joints physiques vérifiés sur les modèles entraînés : 72 vues froides,
  deux axes et halos partagés, écart nul à découpage global512 fixe. Les rejets
  numériques BF16 lors d'un changement de taille de convolution sont conservés.
- [x] Benchmarks BF16 complets sur les deux cartes, construction des entrées
  et transferts inclus, mêmes poids que les évaluations, système au repos.
- [x] Checkpoints et exports immuables, tableaux, 18 planches et comparateur
  final vérifiés dans `~/data/distill/final/`.
- [ ] Choix pratique de l'utilisateur sur le look and feel et les erreurs des
  côtes basses ; aucune acceptation stricte ni intégration par défaut au runtime.
- [ ] Branche poussée à jour : accès GitHub en écriture manquant, archive Git
  locale fournie pour la livraison. Le but reste actif tant que la publication
  demandée et le choix de qualité restent à résoudre.

Commit local initial : `0c6215f`. Push tenté vers `dunkean/neural-earth` :
GitHub renvoie 403, compte `GBeurier` sans accès en écriture. Aucun fork ni
changement de dépôt distant effectué ; destination autorisée demandée à
l'utilisateur. Cette limitation de livraison n'arrête pas les calculs locaux.

### Commandes de reprise et inspection espacée

Les trois passages initiaux sont terminés à 100000 pas chacun. Les probes
`architecture-pipeline` et `base-initial-pipeline` sont terminaux en échec :
`nvidia-smi` attribue le PID teacher 49270 aux deux cartes, y compris à la
transition après fin du trainer, donc la garde GPU refuse la mesure. Déplacer
la garde avant l'initialisation CUDA n'a pas résolu l'ambiguïté. Aucun timing
produit ; ne pas relancer ces probes à chaque transition. La mesure finale
reste requise quand teacher et trainers sont tous terminés.

Le coordinateur a bien été rétabli après l'échec du probe, puis le decoder a
démarré. Il a ensuite été rechargé sans interrompre le decoder afin d'appliquer
la nouvelle recette du base à son passage sur le dataset complet. Le wrapper
`idle_probe.py` n'interrompt pas le trainer et ne redémarre pas un trainer
échoué/interrompu sans inspection.

```bash
source distill/env.sh
# Etat vivant vérifié (pas seulement un fichier de verrou).
python -m distill.jobs status teacher-main --compact
python -m distill.jobs status training-workflow --compact
cat ~/data/distill/training-workflow.json

# Après avoir constaté que le handle teacher est terminal/manquant :
python -m distill.jobs start teacher-main --gpu 0 -- python -m distill.resume_teacher \
  --output ~/data/distill/crops/main --count 20000 \
  --profiles natural orogen terrestrial-earthlike terrestrial-archipelago \
  terrestrial-continents terrestrial-gondwana

# Après avoir constaté que le coordinateur ET le probe sont terminaux/manquants :
python -m distill.jobs start training-workflow --gpu cpu -- python -m distill.run_training

# Après sélection d'un candidat base immuable, GPU 0 libre :
CUDA_VISIBLE_DEVICES=0 python -m distill.check_seams \
  --checkpoint ~/data/distill/ckpt/base/best.pt \
  --output ~/data/distill/eval/base-seams.json
# Ce rapport reste passed=false jusqu'à la revue des planches physiques.
# La revue doit enregistrer leurs checkpoint_digests exacts sous visual_review.

# Après sélection d'un candidat immuable, sur une carte libre :
CUDA_VISIBLE_DEVICES=0 python -m distill.bench_pipeline \
  --checkpoint ~/data/distill/ckpt/base/best.pt \
  --output ~/data/distill/bench/pipeline-4090.json
# Fournir ce résultat avec --pipeline-benchmark à distill.evaluate.
```

Les logs sont sous `~/data/distill/jobs/<nom>/output.log`, les checkpoints sous
`~/data/distill/ckpt/<stage>/`. Le coordinateur réutilise les checkpoints existants
et n'arrête pas les étapes sur la seule présence d'un fichier d'état. Les plans
de pas sont des points d'inspection, pas des preuves de qualité : adapter ou
prolonger après les mesures physiques. Un arrêt du coordinateur laisse les jobs
teacher/trainer indépendants en tmux ; inspecter et arrêter aussi leurs handles
si une pause globale est demandée.

## Cas rares ajoutés le 2026-10-10

À la demande de l'utilisateur, l'acceptation exige désormais aussi une banque
supplémentaire : plaine désertique, côte humide très découpée, transition plaine
très basse/mer et grande plaine tempérée. Les sept lieux historiques et leurs
14 mesures restent obligatoires. Plusieurs holdouts historiques sont en mer,
et le site « désert » comporte des montagnes : ils ne suffisent pas à couvrir
ces cas particuliers.

`distill/rare_cases.py` propose des lieux à partir du conditionnement physique
de cinq mondes **déjà exclus de l'entraînement**, puis vérifie le relief du
teacher aux deux LODs avant de figer les coordonnées. Aucun élève n'intervient
dans la sélection. Une plaine doit avoir une dispersion des hauteurs terrestres
<100 m et une pente terrestre p90 <5°, une côte doit contenir terre et mer aux
deux échelles ; terre et mer doivent chacune avoir au moins 75 % de leurs pixels
dans une composante connexe pour la transition très basse (un semis de flaques
près de zéro ne suffit pas). Les précipitations et la complexité du rivage distinguent les
côtes humides, et le p90 terrestre <100 m distingue la transition très basse.
Des lieux proches sont écartés pendant la proposition. Si un type manque, la
sélection échoue et doit être élargie ; aucun cas n'est rebaptisé pour passer.

La comparaison accepte `--site-manifest` dans un dossier séparé, refuse les
élèves sur une banque non figée et refuse de reprendre sur des lieux différents.
Les planches utilisent les lieux enregistrés dans le rapport, avec leur table
CSV. L'audit supplémentaire reprend les seuils physiques historiques et mesure
aussi MAE/p99 et inversions terre/mer sur les terres 0–20 m, 0–100 m et une
bande de 300 m autour du rivage. La dispersion des erreurs moyennes par ligne
et colonne sert de diagnostic du quadrillage signalé par l'utilisateur.
Ces erreurs locales sont comparées au FP32 base du **même lieu et même GPU**.
L'acceptation finale exige cette preuve avec les mêmes poids, sources et GPU
que la banque historique : `--rare-manifest ... --rare-evaluation ...`.

96 mesures teacher ont qualifié les propositions sans erreur. La banque finale
`eval/rare-sites-coherent.json` contient deux lieux par type, soit 16 vues.
Plaines : p90 des pentes ~0,27–1,28°, dispersion terrestre ~2,3–11,8 m.
Transition mer : p90 de hauteur terrestre <22 m aux deux échelles. Une première
banque `rare-sites.json` reste archivée comme diagnostic : l'un de ses rivages
très proches de zéro était un semis de flaques au LOD 0. Il a été écarté du
cas « transition » sur la géométrie du teacher uniquement, avant toute analyse
des erreurs élèves. Le rapport provisoire est conservé sous `rare-student-pilot`.
La banque finale ne sera plus modifiée en fonction des résultats élèves.

Complément climatique figé : `eval/rare-sites-complete.json` conserve **les huit
lieux ci-dessus**, et ajoute deux plaines arides chaudes et deux plaines tempérées
douces, soit **12 lieux / 24 vues**. Les premiers déserts avaient une température
de -1,9 à 4,9 °C et les plaines tempérées de 3 à 4,9 °C. Une seconde prospection
teacher seule (`warm-plain-proposals.json`, 96 vues dans `warm-teacher-survey`)
a qualifié les ajouts aux deux LODs, sans regarder les erreurs des élèves :

| Ajout | Température | Précipitations | Écart-type terrestre LOD 3 / 0 | Pente p90 LOD 3 / 0 |
| --- | --- | --- | --- | --- |
| Désert seed 202, (28, 63) | 15,9 °C | 121 mm | 11,5 / 4,6 m | 1,28 / 1,28° |
| Désert seed 202, (-275, 55) | 24,1 °C | 211 mm | 11,9 / 5,0 m | 1,29 / 4,32° |
| Plaine tempérée seed 42, (37, -78) | 15,2 °C | 570 mm | 8,4 / 4,3 m | 1,03 / 1,64° |
| Plaine tempérée seed 42, (27, -86) | 11,9 °C | 512 mm | 11,8 / 4,3 m | 1,19 / 0,70° |

Les coordonnées sont celles des tuiles LOD 3. Les parents figés et leurs SHA
sont enregistrés dans la banque augmentée ; l'acceptation vérifie aussi leur
intégrité et la conservation de chaque ancien lieu. L'inspecteur utilise cette
banque complète par défaut. Les planches et leur CSV indiquent température,
précipitations et variante climatique. Aucun de ces mondes réservés ne sert à
l'entraînement. **25 tests passent**, dont conservation des anciens lieux et
rejet d'un parent modifié.

Premier audit de cette banque avec les trois candidats précédents : aucun base
ou coarse ne passe les 16 vues. Le base lisse particulièrement les plaines et
les côtes au LOD 0 ; le coarse peut déplacer des rivages entiers. Le decoder
reste proche en hauteur (MAE LOD 0 : 0,122–2,350 m), mais ses huit LOD 0 ont
encore au moins un défaut de pente/spectre ; ses LOD 3 sont identiques au teacher.
Rapports, audits locaux, quatre planches par catégorie et CSV :
`eval/rare-coherent-pilot/`. Ces résultats complètent les échecs historiques,
ils ne valident aucun candidat.

Audit de couverture des 20000 cibles : sur 19616 crops train, 1059 plaines
désertiques, 983 plaines tempérées, 227 côtes humides et 281 transitions très
basses. Proxy de hauteur basse fréquence à 240 m/pixel, sans prétendre mesurer
le terrain décodé ; les mondes réservés ne sont pas utilisés. Dans la validation
séparée : respectivement 30, 17, 5 et 3. Fichier :
`eval/rare-training-coverage.json` (reprendre avec `rare_cases coverage`).

La prochaine passe base conserve **75 % d'échantillonnage uniforme** et ajoute
25 % répartis également entre les quatre catégories, sur les cibles train
existantes. Environ ×5–6 d'exposition aux deux catégories côtières, sans retirer
les exemples ordinaires ni ajouter les lieux d'évaluation. La policy JSON
`eval/rare-base-sampling.json` fige fichiers, probabilités et empreintes ; elle
est aussi enregistrée intégralement dans chaque checkpoint. Un changement
explicite exige `--allow-sampling-change` ; une reprise sans fichier policy
réutilise les probabilités sauvegardées. Les crops restent déterministes par
numéro de pas. Des exemples rares des **mondes de validation** complètent le
diagnostic réparti sur les six profils, pour vérifier les compromis au cours
de cette passe. Le score best est réinitialisé quand la policy change.
Le coarse en cours n'est pas interrompu. Le decoder conserve sa recette actuelle
avant décision sur ses propres mesures ; sa géométrie de fenêtre diffère du base.

Le coarse 64 a ensuite terminé 300000 pas. Son meilleur score choisit le pas
260000 (`ckpt/candidates/coarse-step260000-9c78ae036a21.pt`). Inspection complète :
1/14 vues historiques et 0/16 rares passent, MAE historique jusqu'à 306 m,
contre 11,64–41,89 m pour le diagnostic FP32 coarse. La supervision en mètres
ne suffit donc pas. Rapports et planches : `eval/coarse-full-pilot{,-rare}`.

Essai de capacité lancé séparément, **coarse128-pilot** sur la 4090 : largeur
128, 9641318 paramètres, batch 4, objectif 100000 pas, même jeu complet et
supervision hauteur/delta. Son dossier `ckpt/coarse128` conserve ses états/RNG ;
il ne remplace aucun checkpoint précédent. Bien que plus gros que le teacher
coarse (2797960 paramètres), il vise une seule passe contre 20, dont le coût
doit être mesuré après validation. La 5090 poursuit base 200000→400000 avec
la policy rare figée ; le coordinateur passera ensuite au decoder 200000.

`distill/inspect_candidate.py` fige les octets du meilleur checkpoint d'une
passe terminée, puis produit banque historique, banque rare, audits et planches
avec la même empreinte. Exemple pour cet essai après sa fin :

```bash
python -m distill.inspect_candidate --stage coarse --minimum-step 100000 \
  --checkpoint-dir ~/data/distill/ckpt/coarse128 --tag coarse128-pilot
```

22 tests de régression passent. Tests GPU séparés : 32 pas de sampling rare,
puis reprise de 8 pas sans fichier policy (probabilités restaurées du checkpoint),
puis 8 pas avec les 14 exemples de validation rares supplémentaires ; losses
finies et états optimiseur/RNG conservés, sans modifier le checkpoint base principal.

### Coarse : diagnostic de généralisation et essai à quatre étapes

Coarse 128 a terminé ses 100000 pas ; EMA retenue au pas 90000,
`coarse-step90000-829e2f9847b1.pt`. La MAE proxy passe de 39,2 m (coarse 64)
à 32,8 m. L'inspection physique `eval/coarse128-pilot{,-rare}` rejette encore
les 14 vues historiques et les 16 rares : améliorations locales, mais côtes
humides jusqu'à 114 m au LOD 0. Prolonger la même recette seule n'est pas
considéré comme un résultat.

Diagnostic indépendant sur **un exemple train natural** : coarse 64, 5000 pas,
LR .001, validation sur ce même exemple uniquement. MAE proxy 1,705 m,
MSE 0,000048, pentes/spectre ~1. Le réseau peut ajuster précisément une cible ;
ce test ne prouve aucune généralisation. Dossiers `crops/coarse-single-diagnostic`
et `ckpt/coarse-single-diagnostic`, sans modification du jeu principal.

Nouvel essai : `distill/coarse_solver.py`, teacher coarse initialisé puis
**20→4 évaluations** du solveur, optimisé par régression sur les cibles finales
existantes, à travers les quatre appels. Architecture et taille du teacher
coarse conservées (2797960 paramètres) ; ce n'est pas une compression des poids
ni une passe unique. Gain réel à mesurer, pas de promesse ×5 du pipeline.
Les poids teacher sur disque et le sous-module restent inchangés. Les masters
optimiseur sont FP32, les paramètres/buffers de chaque forward sont convertis
fonctionnellement en BF16 ; le mode eval des couches MP conserve les gradients
et évite de muter les poids entre plusieurs appels avant backward.

Contrôle du solveur au réglage **20 étapes**, sur la cible sauvegardée du même
exemple : MSE 9,72e-9 et MAE proxy 0,213 m, compatible avec la cible FP16.
L'autocast global créait ~16 m d'écart en modifiant des opérations scalaires/
embeddings ; il est désactivé à l'intérieur de ce solveur, dont les kernels
utilisent explicitement BF16. Métadonnées : `preparation/coarse-solver-parity.json`.
Un test CPU vérifie backward fini, absence de mutations et invariance à
l'autocast englobant ; **23 tests passent**. Smoke GPU 32 pas réussi après cette
correction, sans changer les checkpoints de candidats précédents.

Job `coarse-solver4-pilot` sur la 4090 : 50000 pas, batch 4, LR 2e-5,
checkpoints complets/EMA/RNG dans `ckpt/coarse-solver4`. Les seuils physiques et
la banque rare restent inchangés. La 5090 poursuit la passe du base en parallèle.
Empreintes du nouveau module ajoutées aux rapports physiques, benchmarks et
preuves de joints pour empêcher l'acceptation de résultats issus d'un autre
code de solveur.

### Base 128 initialisé depuis les poids appris

Inspection intermédiaire du base, EMA du pas **265000** : **0/14 historiques et 0/16
rares**, dans `eval/base-rare-interim{,-rare}`. Plusieurs bandes LOD 3 se
rapprochent de la référence, mais les plaines et rivages LOD 0 restent très
lissés (certaines bandes à 0,01–0,09×). Prolonger seul le 96 à 400000 pas est
remplacé par une fin de passe à 300000, puis un essai **base 128**, 100000 pas,
dans `ckpt/base128`. Les poids précédents restent conservés.

`distill/widen.py` initialise ce nouveau modèle depuis l'EMA : canaux appris
copiés, nouveaux canaux activés par de petits poids, skips concaténés remappés,
variance/epsilon PixelNorm compensés. Même champ récepteur/halo, sans filtrage
du terrain. Test FP32 sur un vrai réseau à sorties non nulles : champ préservé
à 1e-5. Test BF16 4090 sur trois crops val : MAE latente 0,0011–0,0017 et écart
proxy hauteur 2,18–3,58 m, enregistré dans `preparation/base-widening-check.json`.
Taille base 128 : **13331237 paramètres**, ~53,3 Mo de poids FP32, contre
7507077 / ~30,0 Mo pour le 96. Le gain d'inférence de cette version reste à mesurer.

L'élargissement est une **nouvelle architecture avec un nouvel optimiseur**,
pas une reprise prétendant conserver des moments de formes différentes.
Le checkpoint source intégral est figé dans `source.pt`, avec son SHA, seed et
recette dans `warm-start.json` et les checkpoints suivants. Les probabilités
train/val et le plan de seeds sont repris ; les futurs arrêts/reprises de ce
nouvel essai conserveront son propre optimiseur/RNG. **24 tests passent**,
dont le remappage des skips et la préservation du champ.
Smoke GPU élargi : 32 pas finis, gradients/losses finies, policy rare restaurée
du checkpoint, provenance de warm start conservée. Dossier `ckpt/base128-smoke`
distinct de l'essai principal ; il ne sert pas de preuve physique d'acceptation.

Le coordinateur attend la fin base 96→300000, initialise base128 une fois,
puis entraîne base128 et passe au decoder→200000. Le solveur coarse 4 en
parallèle a reçu la loss des cinq bandes (.05), après un diagnostic de pente
proxy +32 % au pas 10000 : reprise propre, optimiseur/RNG conservés, meilleur
score réinitialisé pour le nouvel objectif. Les seuils d'acceptation physiques
restent inchangés.

État après élargissement : le base 96 a terminé ses 300000 pas ; le base 128
reprend son EMA du **pas 295000**, SHA source
`1dee59a036d36fd1f297f3cc08906b0b730cfd68baa48139da372592c5b220b8`.
Les premiers checkpoints du nouveau run contiennent sa propre state AdamW et
la policy rare, avec epsilon compensé 7,5e-7. Les inspections suivantes sont
enchaînées par des jobs tmux durables : `coarse-solver4-inspection`, puis
`decoder-followup` (reprise decoder 150000→200000 sur la **4090** libérée),
`decoder-full-inspection`, et `base128-inspection` après la fin du base sur 5090.
Le coordinateur principal attend le même handle decoder et ne crée pas un
second entraînement. Chaque inspection utilise 14 vues historiques + les
24 vues de la banque rare complète ; un succès de processus ne vaut toujours
pas acceptation physique.

### Continuité du terrain final et conservation des alternatives

`distill/check_physical_seams.py` complète le test local du halo du base :
quatre tuiles viewer voisines (256 + halo 24), demandées en ordre inversé, sont
comparées à une grande requête du même terrain dans un **autre monde neuf**.
Les trois étages peuvent être remplacés ensemble. Le test mesure aussi les
pixels communs des halos, les erreurs de saut aux raccords horizontaux et
verticaux et les différences de signe terre/mer. Il produit les hauteurs brutes,
les vues grande requête / mosaïque / erreur en mètres et un tableau CSV.

Smoke teacher sur la transition très basse seed 202, (-240, 0), réussi aux
LOD 3/0 (`eval/physical-seam-teacher-smoke`) : **halos voisins identiques** aux
deux échelles. LOD 0 : différence max avec la grande requête 3,58e-5 m.
LOD 3 : MAE 0,108 m, max 4,29 m et écart de saut max 0,498 m avec la grande
requête ; ce comportement du teacher est mesuré séparément des joints des
élèves, plutôt que de supposer une identité entre toutes les tailles de requête.
**26 tests passent**, dont détection de fissures synthétiques sur les deux axes.
L'audit des artefacts attend maintenant les quatre sources student/solveur,
comme les empreintes des rapports, au lieu du compte historique de trois.

Huit alternatives sont déjà figées sous `ckpt/candidates/`, avec inventaire
SHA/config/pas dans `inventory.json`. Ajouts conservés pour comparaison :
base 96 EMA 295000 et base 128 EMA 20000. Cela ne les déclare pas acceptés ;
les futurs meilleurs checkpoints ne peuvent pas écraser ces copies.

Coarse 4 a terminé : candidat du pas **50000**, SHA
`4e5f9a33177779dd23b65a7e3ec46422bcb237a6f69b82e5d47a7994d0ea8a3c`,
conservé dans `ckpt/candidates/`. Inspection complète : **0/14 + 0/24** passent
tous les seuils stricts, MAE historique 9,68–275,25 m ; plaines rares LOD 0
0,61–10,71 m, côtes humides 11,01–72,69 m. Planches sous
`eval/coarse-solver4-pilot{,-rare}`. Il améliore le solveur 4 non entraîné
(snow LOD 3 : 368→149 m ; coast LOD 0 : 192→96 m), mais pas assez pour
conclure à un bon compromis sur tous les reliefs.

Contrôles séparés, **pas 0 / non distillés**, conservés dans `ckpt/controls` :
le hook à 20 étapes reproduit exactement le teacher sur snow/coast aux deux
LODs (MAE et max **0**). À huit étapes sans entraînement, les mêmes LOD 0
ont une MAE de 51,85 / 51,28 m, contre 275,25 / 95,53 m pour le 4 entraîné ;
snow LOD 3 reste à 233 m. Cela motive un essai **coarse-solver8-pilot**,
25000 pas, batch 4, LR 1e-5, mêmes losses/données, après smoke 32 pas réussi.
Initialisation depuis le teacher BF16, nouveau dossier/optimiseur/RNG ; aucune
prétention de compression des paramètres ou d'un gain mesuré ×2,5.
Son inspection sera produite automatiquement sous `coarse-solver8-pilot`.

Les jobs `candidate-bundle-seams`, `base128-cuda-seams` et
`candidate-bundle-physical` attendent les inspections des candidats figés.
La dernière produit les images des **trois élèves ensemble**, sur les lieux
historiques et rares, avec les mêmes SHA que la preuve de continuité.
La variante coarse 4 de ce premier assemblage reste explicitement un
compromis à comparer, pas un modèle accepté par les seuils stricts.

Le contrôle de continuité final utilise désormais un **monde neuf pour chaque
tuile viewer**, pour éviter qu'un cache voisin ne masque les écarts. Smoke
teacher plus sévère `eval/physical-seam-cold-teacher-smoke` : erreur des halos
max ~0,444 m au LOD 3, ~0,442 m au LOD 0 ; MAE LOD 0 0,000148 m, max du saut
0,0447 m. Ces petits écarts du teacher seront publiés à côté des mesures élèves.
L'ancien smoke avec cache partagé reste conservé et n'est pas présenté comme
une preuve à cache froid.

`distill/export.py` prépare des bundles d'inférence immuables avec l'EMA exacte,
sans optimiseur ni seconde copie des poids. Test : sorties non nulles conservées
à l'identique pour les trois étages, EMA distincte des poids courants, refus de
réécrire un bundle avec un autre checkpoint ; **27 tests passent**.
Premier export **provisoire**, sans acceptation :
`exports/preview-base128-45000-coarse4-50000-decoder145000/`.
Poids FP32 : base 53,35 Mo, coarse 11,22 Mo, decoder 13,36 Mo ; manifest avec
SHA de chaque source et fichier. Les checkpoints complets/RNG restent conservés.

### Supervision du base à travers le decoder

Inspection base 128 intermédiaire, EMA **45000**, SHA
`d2a91c5167fe0bfab21c4c35351248fbfc4b76f852af7077e38c9ec5968c07ca` :
les détails des côtes rares LOD 0 restent trop faibles (~0,08–0,18× de puissance
sur la première côte humide). L'élargissement seul apporte peu de progrès
visible à ce stade ; l'entraînement latent ne suffit pas à garantir le rendu.

Audit en lecture seule : **2129 crops train et 41 val** contiennent exactement
la fenêtre decoder 512² dans le début du champ base 256² (origines alignées
sur 48 latents, donc aussi 96 compte tenu du pas 32). Les quatre latents stockés
dans les deux fichiers sont **identiques sur les 2170 paires**, sans nouveau
target teacher et sans les cinq seeds réservées. Rapport
`eval/paired-decoder-coverage.json`. Couverture proxy du sous-ensemble train :
130 plaines arides, 97 tempérées, 33 côtes humides, 35 transitions basses.

`distill/decoded_loss.py` ajoute une phase séparée : base sur un cœur 64 + halo,
decoder élève figé, loss sur son résiduel et sa reconstruction intérieure en
mètres (Laplacien du teacher, bord de 64 pixels exclu). La MAE relative utilise
une échelle minimale de 5 m pour ne pas écraser les plaines derrière les reliefs
montagneux ; les cinq bandes physiques sont aussi supervisées. Cette hauteur
reste un **proxy de fenêtre non blendée**, pas la mesure finale du pipeline.
Le coût d'inférence et le champ récepteur du base sont inchangés.

Smoke GPU 32 pas puis reprise de 8 pas réussis dans `ckpt/base-decoded-smoke` :
gradients/losses finis, état AdamW de 57 paramètres restauré, decoder/audit/loss
restaurés sans répéter leurs flags. **30 tests passent**, dont gradients vers
les latents sans modifier le decoder, gradients du canal LF en mètres,
coordonnées négatives et rejet d'une paire teacher modifiée. Ce smoke utilise
base 45000 / decoder 145000 ; il n'est pas un candidat validé.

Jobs durables : `base-decoded-followup` attend l'inspection base 128 complète,
puis initialise un nouveau trial depuis les **EMA figées du base 128 et du decoder 200000**
et entraîne `train-base-decoded` sur 20000 pas (LR 1e-4, batch 1). Sampling rare
75/25 recalculé sur les paires, les 41 val restent séparées. Source, decoder,
audit et provenance sont copiés ; nouvel optimiseur explicite, futures reprises
avec ses propres états/RNG. `base-decoded-inspection` produira les 38 mesures
physiques usuelles avant de choisir un meilleur compromis visuel.

Decoder 200000 terminé, candidat du **pas 200000**, SHA
`3a6303b22b45af0c62f9ea4bd81957699823d245caa77f62c4895d8c4bf57e10`.
MAE LOD 0 historique 0,449–3,872 m, rare 0,119–2,287 m ; 2/7 historiques
passent tous les seuils stricts, 0/12 rares (au moins un défaut de distribution
ou local par cas). Les deux versions 145000/200000 restent conservées pour
comparaison du rendu. Rapports `eval/decoder-full-pilot{,-rare}`.

```bash
python -m distill.rare_cases survey ~/data/distill/eval/rare-proposals.json --count 12
python tools/verification/compare_base_variants.py run ~/data/distill/eval/rare-teacher-survey \
  --variants reference --site-manifest ~/data/distill/eval/rare-proposals.json
python -m distill.rare_cases freeze ~/data/distill/eval/rare-proposals.json \
  ~/data/distill/eval/rare-teacher-survey ~/data/distill/eval/rare-sites-coherent.json --per-kind 2
# Évaluer ensuite reference/fp32base/student_* sur rare-sites-coherent.json dans un autre dossier.
```

Decoder élargi à 150000 pas : les 14 mesures physiques sont complètes dans
`eval/decoder-expanded-pilot`. L'EMA retenue est celle du **pas 145000**, figée
dans `ckpt/candidates/decoder-expanded-step145000-b855d345a369.pt`.
Aux sept LOD 0 : MAE 0,476–4,115 m, pentes moyenne/p90 dans ±5 % partout ;
deux lieux passent aussi les cinq bandes spectrales, cinq restent hors tolérance
(notamment côte ~0,76× et snow ~1,12×). Les sept LOD 3 conservent le teacher.
Le decoder n'est donc pas accepté. Planches et CSV mis à jour dans ce dossier.

Diagnostic FP32 **coarse** sur la 4090, 14 lieux : MAE 11,64–41,89 m dans
`eval/precision-coarse-4090`. Cela confirme une sensibilité numérique de ce
teacher à 20 étapes ; les erreurs du coarse distillé précédent (jusqu'à 247 m)
restent trop élevées. Cette mesure ne modifie aucun seuil d'acceptation.

### Conservation et mesures finales des candidats

Onze checkpoints immuables sont conservés dans `ckpt/candidates/`, avec
`inventory.json` (SHA complet, architecture, pas, taille du checkpoint). Les
fichiers d'entraînement contiennent optimiseur et EMA en double ; les tailles
de distribution viennent uniquement des exports contenant les poids retenus.
Les prochains candidats seront conservés séparément, sans écraser ces alternatives.

`benchmark_candidates.py` attend la fin des évaluations physiques et des joints,
puis exporte le bundle et mesure les trois réseaux et les trois étages sur les
deux cartes, séquentiellement. Job `decoded-bundle-benchmark`, sortie prévue
`bench/decoded-bundle/`. Les deux GPU doivent être libres avant chaque mesure.
`bench_pipeline.py` précharge les dépendances communes hors chronomètre : coarse
pour base, latents teacher pour decoder, aucune pour coarse ; il refuse une
mesure où une dépendance manquante serait calculée dans la zone chronométrée.
Features, transferts et blending de l'étage restent inclus. Ces chiffres ne sont
pas encore disponibles et ne représentent pas le temps d'une vue complète.
La mesure n'enregistre aucun graphe d'autograd.
Un contrôle CUDA des joints terminé avec un rejet numérique conserve son
rapport et permet quand même les mesures de vitesse : ce rejet ne devient pas
une validation. Un crash, un rapport incomplet ou antérieur au job bloque la
coordination. Ce comportement est testé ; 31 tests de distillation passent.

Le bundle provisoire base128 **100000** (`f25dbf12474a…`), coarse4 **50000**
(`4e5f9a331777…`), decoder **200000** (`3a6303b22b45…`) a terminé le contrôle
physique à froid `eval/candidate-bundle-seams` : quatre catégories rares,
LOD 3 et 0, quatre tuiles indépendantes contre une grande requête. Pour les
huit vues de ce bundle, différence maximale, erreur des sauts aux joints et
désaccord des halos **exactement nuls**. Cela ne valide pas sa fidélité au teacher.
Le contrôle latent 128² contre 256², avec BF16 et des formes de convolution
différentes, dépasse le seuil initial 1e-4 ; son rapport est conservé dans
`eval/base128-cuda-seams.json`, sans transformer ce rejet en réussite.

`review_gallery.py` crée des comparaisons hors ligne avec curseur référence/élève
et les métriques physiques en tableau. `eval/review-current/index.html` contient
72 vues rares (base, decoder, bundle provisoire). Vérification dans Chromium :
chargement des 48 vues initiales, extrémités du curseur, écran mobile, aucune
erreur JavaScript ; la galerie a ensuite été étendue au bundle. Le job
`decoded-bundle-gallery` produira la galerie du prochain bundle après son
évaluation. Le base couplé au decoder est en entraînement sur les mêmes cibles
teacher alignées ; son checkpoint du pas 5000 est déjà conservé séparément.

Capacité : 128/192/256 canaux donnent respectivement **13 331 237 / 29 962 341 /
53 237 157 paramètres**, soit **53,32 / 119,85 / 212,95 Mo** de poids FP32 seuls.
192 et 256 sont des options mesurées par comptage, pas des modèles entraînés.
53 Mo ne constitue pas une contrainte : comparer un élève plus large si la
supervision du relief décodé ne préserve pas suffisamment les détails visibles.

Le coarse8 termine son passage à 25000 pas ; EMA retenue **22500**,
`10010316d2d1…`. MAE sur les 14 vues : **17,08–113,90 m** ; sur les 24 rares :
**0,224–48,896 m**. Aux deux côtes humides LOD 0 : **4,094 et 16,490 m**,
contre 11,006 et 72,688 m pour coarse4. Aux transitions plaine basse/mer :
0,224 et 1,288 m. Il reste hors acceptation stricte, mais est conservé comme
candidat principal pour les comparaisons combinées.

La première passe base après décodage termine ses **20000** pas. Le bundle
coarse8 + base couplé + decoder200k a lui aussi des raccords physiques
**exactement nuls sur les huit vues rares testées**, dans
`eval/decoded-bundle-seams`. Sa comparaison physique complète est en cours.
À 5000 pas, la nouvelle loss restaurait davantage de relief sur certains cas
mais en exagérait d'autres ; les checkpoints parent100k et couplé5000 sont
conservés. `eval/review-matched-bundle8/` compare ces deux bases avec exactement
le même coarse8 et decoder200k. La galerie actuelle a été vérifiée dans
Chromium sur **72 vues**, y compris mobile ; les tableaux comportent désormais
les erreurs sur les terres, les plaines 0–20 m et la bande côtière de 300 m.

Après les benchmarks du bundle 128, deux passages comparables sont préparés :
base **192 canaux / ~120 Mo** sur la 5090 et contrôle **128 / ~53 Mo** sur la
4090. Même EMA source de la première passe couplée, données alignées, loss,
sampling rare, nouvel optimiseur et **20000 pas supplémentaires** chacun,
avec un smoke de 32 pas et reprise de ses états. Cela distingue l'effet d'une
capacité accrue d'un entraînement prolongé. Les contrôles physiques du contrôle
128 se terminent avant ceux du 192 ; les mesures de vitesse sont séquentielles,
avec les deux GPU libres. Jobs `base192-decoded-followup` et
`base128-control-followup` ; aucune nouvelle génération teacher ni intégration
dans le serveur par défaut.

Le compteur du benchmark préserve désormais les méthodes d'embeddings des
modèles de production. Les captures CUDA du teacher remplacé sont libérées
avant le warmup de l'élève, et les allocations/réservations CUDA sont consignées
par mesure. Une première mesure a saturé la VRAM de la 4090 et produit des
temps instables ; elle est conservée comme diagnostic, sans servir de résultat
de performance. Le benchmark corrigé et ses jobs dépendants ont été relancés.
Les **32 tests de distillation passent** ; les mesures GPU restent en cours.

Le contrôle de mémoire a identifié le problème restant : le compteur devait
être partagé entre mondes frais, comme les poids résidents, pour partager aussi
le cache du solver coarse. Avec ce partage, les réservations restent à
**9,38 Gio** pendant les quatre répétitions élève au lieu d'augmenter de plusieurs
Gio par monde. Un garde refuse toute mesure réservant plus que la VRAM physique.
Le contrôle 512×512 latents sur 4090 donne une médiane teacher **1,249 s**, élève
**0,164 s**, soit **×7,59**, halo et construction CPU des features compris,
coarse pré-calculé hors chronomètre (`bench/base-memory-probe-v3.json`). Les
anciens rapports contaminés par la pagination sont conservés comme diagnostics.
Le coarse8 actuel construit son scheduler sur CPU : son micro-benchmark publie
le temps eager et indique explicitement CUDA Graphs indisponible. Les mesures
physiques des poids n'ont pas changé. **33 tests de distillation passent.**

### Mesures complètes et optimisation d'inférence

Le benchmark précédent est maintenant complet sur les deux GPU
(`bench/decoded-bundle/summary.json`). Médianes, mondes frais, dépendances hors
chronomètre, features/transferts compris :

| Étape / surface | 4090 teacher → élève | Gain | 5090 teacher → élève | Gain |
| --- | --- | --- | --- | --- |
| Base / 2048² latents | 16,226 → 2,539 s | ×6,39 | 12,839 → 2,119 s | ×6,06 |
| Coarse / 128² cellules | 0,304 → 1,264 s | ×0,24 | 0,285 → 1,231 s | ×0,23 |
| Decoder / 1024² pixels | 0,337 → 0,294 s | ×1,15 | 0,245 → 0,290 s | ×0,85 |

Ces chiffres montrent des limites réelles : le coarse8 eager est plus lent que
le teacher à 20 étapes capturées, et la construction CPU des entrées pénalise
le decoder. La taille réduite des poids n'assure pas à elle seule l'accélération.
Les essais base192 et contrôle128 ont démarré après ces mesures ; leurs 20000
pas sont en cours, sans génération de nouveaux targets.

Optimisation désormais en validation : assemblage des entrées directement sur
le GPU en conservant le bruit et la trigonométrie CPU identiques ; nearest
upsampling et remplissage des canaux constants sur GPU. Le coarse précharge
ses seuls sigma d'entrée sur GPU, conserve les coefficients du scheduler sur
CPU et utilise les caches de constantes exacts du runtime. Sa capture complète
est partagée entre mondes empruntant les mêmes poids.

`eval/coarse-capture-exact.json` vérifie **six fenêtres de validation**, eager
optimisé et graph replay contre l'ancienne implémentation archivée : **écart
maximal 0**. Les entrées base et decoder CPU/GPU sont également identiques sur
ces six cas. **35 tests CPU passent.** Le code teacher et `features.py` sont
inchangés ; données et états d'entraînement restent compatibles.

La preuve physique complète est ordonnancée dans `optimized-baseline-physical`,
après les deux entraînements et les diagnostics du contrôle128 : **14 + 24
vues** comparées à celles du bundle précédent, puis **huit vues de raccords**.
`verify_equivalence.py` refuse les différences de poids, de référence ou de
cas couverts. L'inspection du 192 attend cette preuve ; benchmarks du 192,
du bundle128 optimisé et du contrôle128 s'enchaînent ensuite avec les deux GPU
libres. Les anciens chiffres restent des mesures de la version précédente.

Galerie des deux bases128, avec coarse8 et decoder200k constants :
`eval/review-base-before-width/index.html`, **48 vues**. Le passage couplé20000
réintroduit du relief mais amplifie trop le grain sur certaines plaines
(pente ×4,56, bande la plus fine ×12,63 sur `temperate-plain-42-159--112` LOD0).
Le parent100k reste préservé comme alternative plus lisse. **15 checkpoints
figés** sont inventoriés dans `ckpt/candidates/inventory.json`.

Régression complète après l'optimisation : **377 tests passent, 33 ignorés,
291 sous-tests passent** (`tests/python` + `distill/test_distill.py`, GPU masqués,
mock de `torch.cuda.current_device()` pour les tests scheduler CPU). Log :
`logs/inference-optimization-cpu-tests.log`.

Tailles vérifiées des fichiers de poids **FP32**, hors états d'optimiseur :

| Modèle | Teacher épinglé | Export élève actuel |
| --- | --- | --- |
| Base | 1014,77 Mo | 53,35 Mo (128) ; ~120 Mo (192 en cours) |
| Coarse | 11,20 Mo | 11,22 Mo (même architecture, 20 → 8 étapes) |
| Decoder | 111,71 Mo | 13,36 Mo |

Le teacher est bien stocké en FP32, vérifié dans les en-têtes safetensors ; les
tailles élève viennent des exports d'inférence, sans optimizer ni EMA dupliquée.

Les assemblages GPU des **trois** types d'entrées passent désormais leurs six
comparaisons exactes ; le coarse évite aussi le retour CPU des conditions et des
cinq scalaires. **36 tests de distillation passent.** Le diagnostic
`grid_artifacts.py` mesure les projections cohérentes horizontales/verticales
aux périodes **32 et 64 latents** en LOD3, après retrait du halo et des variations
larges. CSV et JSON portent les empreintes des vues et des modèles ; ce n'est
pas un seuil d'acceptation ni une preuve de cause. Il sera calculé sur les quatre
bases comparées après les dernières mesures.

La preuve d'équivalence physique couvrira aussi le **parent128100k**, dans
`optimized-parent-physical{-rare}` et `optimized-parent-seams`. Les références
BF16/FP32 déjà mesurées sont réutilisées après vérification de leurs identités ;
les vues élève sont recalculées sous la nouvelle empreinte d'inférence.

## Préparation du 2026-10-10 — en attente du top

La passation a été reçue en texte dans la conversation. Les performances de la
3090 restent des références historiques ; les gains des nouvelles cartes ne
sont pas encore mesurés. La cible reste le latent final à deux étapes, blending
compris. `onestep` et FP16 base ont déjà été rejetés pour leur qualité.

- Python 3.12.12, Torch 2.11.0+cu128, Triton 3.6.0 ; CUDA voit la RTX 4090
  (sm_89) et la RTX 5090 (sm_120). `torch._scaled_mm` est disponible ; ses
  performances FP8 ne sont pas encore mesurées.
- Node 22.21.1 est déjà installé via nvm. `source distill/env.sh` sélectionne
  ce Node et le venv, sans changer la configuration globale du shell.
- Le runtime existant accepte déjà `TERRAIN_RUNTIME_ROOT`. L'environnement
  de distillation le fixe à `~/data/runtime` et fournit aussi `TERRAIN_RUNTIME`.
  Le cache HF existant sous `~/.cache/neural-earth/huggingface` est réutilisé.
- Snapshot : `9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Les rasters ETOPO et
  WorldClim requis sont présents. Les répertoires `bench`, `eval`, `crops`,
  `ckpt`, `logs` et `preparation` sont créés sous `~/data/distill/`.
- `pytest` 9.1.1 et ses dépendances ont été ajoutés au venv ; aucun paquet du
  lock n'a été remplacé.
- Vérification sans GPU : **332 tests passent, 33 sont ignorés, 291 sous-tests
  passent**. Commande ci-dessous. Avec les GPU masqués, un test du scheduler
  demande `torch.cuda.current_device()` malgré ses mocks de capacité/mémoire ;
  le mock du périphérique ci-dessous permet de finir la vérification CPU.
  La commande standard sans ce mock et les vérifications CUDA dédiées restent
  à exécuter lorsque les cartes seront libres.
- `uv pip check` confirme la compatibilité des 147 paquets installés.
- Vérification navigateur sous Node 22 : 12 fichiers de tests passent, deux
  échouent aussi en exécution séquentielle (`test_terrain_coarse_gpu.cjs:38`,
  option coarse GPU non cochée, et `test_terrain_tile_continuity.cjs:68`,
  rétention du parent). Aucun fichier du viewer n'a été modifié pour cette
  préparation. Logs : `~/data/distill/preparation/browser-tests.log` et
  `browser-recheck.log`. Ces échecs du viewer sont séparés des futurs critères
  physiques de distillation.
- Les scripts teacher/student/train/bench_student et la variante `student`
  n'existent pas encore. Les deux outils de référence sont présents ; le
  benchmark base n'implémente pas encore le FP8. Rien n'a été mesuré ou entraîné.
- Le serveur Neural Earth est actif et occupe les cartes. Il n'a pas été arrêté.
  Les modifications déjà présentes dans le dépôt sont conservées ; aucun
  changement de branche, commit ou push n'a été effectué.

Contrôles déjà exécutés :

```bash
source distill/env.sh
CUDA_VISIBLE_DEVICES='' python - <<'PY'
from unittest.mock import patch
import pytest
with patch('torch.cuda.current_device', return_value=0):
    raise SystemExit(pytest.main(['tests/python', '-x', '-q', '--disable-warnings']))
PY
bash -n distill/env.sh distill/phase1.sh
bash distill/phase1.sh --dry-run
```

Au top : arrêter le serveur et libérer les GPU, puis lancer la première phase :

```bash
bash distill/phase1.sh --run
tmux attach -t neural-earth-distill-reference
```

Ce lanceur mesure les deux cartes puis compare `reference`/`fp32base` sur les
sept sites aux deux LODs. Il écrit `logs/phase1.log` et `logs/phase1.exit` et
s'arrête sur une mesure manquante, en erreur, ou une MAE BF16/FP32 supérieure à
36 m. Une MAE inférieure à 7 m est acceptable. La borne de gain FP8 brut reste
à mesurer séparément avant de conclure la phase 1. Le lanceur ne démarre aucune
génération de dataset ni aucun entraînement.
