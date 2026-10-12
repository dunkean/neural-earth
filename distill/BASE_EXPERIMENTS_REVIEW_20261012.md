# Distillation du base : bilan complet et demande de revue indépendante

État arrêté au 12 octobre 2026. Dépôt : `/home/delete/self/neural-earth`, branche `distill`.
Toutes les données et les poids expérimentaux sont sous `/home/delete/data/distill` (abrégé **DATA** ci-dessous).
Les calculs GPU sont suspendus à la demande de l'utilisateur ; aucun nouveau training ne doit être lancé pendant cette revue.

## 1. Décision à prendre

L'objectif est un **base distillé qui conserve les formes du teacher**, notamment les vallées et affluents d'érosion, avec un gain d'inférence utile. Aucun base testé n'est accepté. Le runtime conserve le base original ; coarse4 et decoder200k distillés sont les choix usuels, jugés bons visuellement par l'utilisateur. Coarse1 et coarse2 ont fini leurs 200 000 pas et sont livrés ; leur optimisation d'inférence est laissée à l'utilisateur.

Retour live de l'utilisateur : les premiers élèves base ressemblent à des reliefs, mais suppriment ou modifient les vallées fluviales. Il voit aussi du bruit et des lignes horizontales/verticales. Passer de 128 à 192 canaux dans la première famille ne résout pas ce défaut. Ce n'est pas seulement un problème de petite erreur d'altitude ni de seams entre tuiles.

Cas essentiels : plaines désertiques ; côtes humides très découpées ; transition plaine très basse/mer ; vastes plaines tempérées ; **haute montagne humide, fortement incisée**. L'utilisateur peut accepter un écart au teacher si le look & feel convient. La morphologie doit cependant être préservée. Des checkpoints doivent rester disponibles pour comparaison visuelle.

Le premier objectif était une seule passe, conditionnement dense, gain ≥10× et erreur comparable à BF16/FP32 (environ 10–35 m selon les lieux), pentes/spectres proches et côtes stables. Ce gain n'a pas été obtenu avec une qualité acceptable. Les pistes suivantes ont gardé deux passes, voire toute la largeur du teacher. Les seuils intermédiaires de chaque pilote sont des décisions expérimentales, **pas une loi démontrant un plafond de capacité**.

## 2. Teacher, géométrie et conventions

Checkpoint `xandergos/terrain-diffusion-30m`, révision `9ef8030cb805b433b98ec25c5dddefbac07a9e26`, poids SHA256 `351d1b1a77cf32e15adc4f72ed5fd26d317340598329d35f5dce2ff6dbcce735`.

- Base : **253 688 037 paramètres**, U-Net MP/EDM, largeur192, trois blocs par niveau, canaux192/384/576/768, cinq entrées et cinq sorties, connexions de saut et attention du teacher.
- Inférence : fenêtres64×64 latentes, pas32, **deux phases**, mélange pondéré. Bruits mondiaux déterministes, seed+5819 et seed+5820.
- Condition : **58 scalaires par fenêtre**, construits à partir du coarse dans une géométrie4×4 ; une cellule coarse couvre32 latents. Une condition dense interpolée n'est donc pas une identité automatique avec cette entrée scalaire.
- Sur les entrées64, l'attention effectivement active est celle du bloc central, avec grille profonde8×8. L'architecture déclarée en résolution512 ne signifie pas que toutes les attentions configurées s'activent sur64.
- `MPConv` normalise le poids **globalement**, en FP32, puis applique gain/facteur de fan-in. Une factorisation normalisée séparément, ou canal par canal, ne reproduirait pas le teacher.
- Runtime principalement BF16. Le canal4 donne l'altitude coarse au LOD3 ; le decoder utilise les autres features pour le relief fin. Les erreurs d'altitude LOD3 et d'incision LOD0 doivent être distinguées.

Code source : `terrain-diffusion/terrain_diffusion/models/mp_layers.py`, modèles EDM du submodule, `terrain-diffusion/terrain_diffusion/inference/world_pipeline.py`, `backend/terrain_inference.py` ; adaptateurs expérimentaux dans `distill/`.

Le handoff initial rapporte le rejet du one-step teacher sans réentraînement (relief changé et granuleux) et du base FP16. Ces essais sont antérieurs à cette série ; la norme FP16 débordante est une hypothèse non vérifiée. Le FP16 du decoder ne change pratiquement pas le terrain. BF16/FP32 produit lui-même un écart dépendant du lieu : ne pas exiger une identité octet par octet entre GPU différents.

## 3. Données, séparation et évaluation

### Première famille à champ direct

`DATA/crops/main/` : 20 000 triples coarse/base/decoder, **19 616 train et384 val**,613 mondes TRAIN et12 VAL. Cibles base256² après les deux phases/blending ; entrée avec halo. Audit des60 000 NPZ, échelles, profils, seeds remplacées et séparation des mondes conservé dans `dataset-audit.json`. Seeds42/101/202/303/404 réservées aux mesures finales.

Les affinages via decoder utilisaient2129 paires TRAIN et41 VAL intérieures. Seulement cinq fenêtres VAL mixtes proches de la mer : proxy limité, à distinguer des vues finales. La banque physique historique est7 sites×LOD3/0, complétée par12 sites rares×LOD3/0 :38 vues par candidat.

### Familles natives puis champ pleine largeur / QAT

- Première banque native :624 scènes,576 TRAIN/48 VAL ; fenêtres, entrées des deux phases, conditions et cibles conservées.
- `DATA/crops/native-mixed-scenes-v2/` :700 positions×3 branches coarse =2100 scènes, **1896 TRAIN/204 VAL**. Branches coarse original / coarse4 livré / coarse2 EMA200k. Chaque cible est calculée par le **base teacher original sur SON propre coarse**. SHA du manifest : `1881c44cd6424f9098da9607f109f0e3ffba45cc37c5086f68ace8326ebbce83`.
- Montagnes : `native-wet-mountains-v1/` et `native-wet-mountain-train-extension-v1/`, quinze mondes TRAIN au total. L'extension ajoute douze mondes TRAIN,36/36 branches qualifiées. TRAIN récent :1947 scènes. VAL montagne :620001/620002/620003 ; mondes finaux101/303/404 réservés.
- Qualification montagne sur le teacher : pluie≥1200, terre≥95%, altitudep90≥2000m, reliefp90−p10≥500m, pentep90≥15° et TPI8 RMS≥10m. Ces proxies ne prouvent pas un réseau hydrologique correct ; les images teacher ont été examinées.
- Sampling récent :25% montagne /15% anciennes catégories rares /60% général ; trois coarse ; statistiques de variance/bandes **TRAIN figées**.

Évaluation : hauteur en mètres, pentes, spectres, côte, position topographique multi-échelle (**TPI**), corrélation, RMS relatif, rappel/IoU des positions de vallées teacher, direction des gradients. Le rappel TPI n'est pas une vraie analyse de drainage. Un champ bruité peut obtenir du rappel sans reproduire des incisions. Les pertes TRAIN, les moyennes physiques, les coordonnées de vallées et le rendu peuvent diverger.

Images de progression à plat dans `DATA/_base/`, mêmes lumières et hauteurs réellement calculées, sans lissage ajouté ; CSV/JSON/TXT voisins. Le dossier historique `DATA/final/` contient18 planches et une galerie ; son nom **ne signifie pas acceptation**. Ne pas l'écraser.

## 4. Inventaire des entraînements du base

Les pas ci-dessous sont les pas réellement effectués, pas les budgets annoncés. Les reprises d'une recette sont regroupées mais identifiées. Les smokes et changements sans training sont détaillés plus loin. **Tous les résultats sont négatifs pour une livraison base**, avec des différences importantes de qualité.

| Famille / intervention | Durée réelle | Résultat et raison de non-sélection | Poids / preuves sous DATA |
|---|---:|---|---|
| U-Net local direct96, régression du champ final T2 | Jalon100k,200k puis300k |100k :50,16m sur un holdout, PSD haute fréquence1,61–1,87× ;200k :15,4–125m sur14 vues, certaines bandes jusqu'à0,18×. Grain puis lissage ; érosion non fidèle. | `ckpt/base/`, `ckpt/candidates/`, `eval/base-pilot/`, `eval/base-expanded-pilot/` |
| U-Net local direct128, élargi depuis le96 appris |100k supplémentaires |13,33M paramètres, export53,349Mo. Compromis visuel plus lisse que certaines reprises, mais lignes et pertes d'érosion signalées en live. | `ckpt/base128/`, `ckpt/candidates/base-step100000-f25dbf12474a.pt`, `eval/base128-pilot{,-rare}/` |
| Base128 couplé au decoder200k gelé |20k | Cible physique via decoder et latent ; des écarts baissent, les formes ne deviennent pas acceptables. | `ckpt/base-decoded/`, `eval/base-decoded-pilot/` |
| Base128 contrôle prolongé à budget comparable |20k | Pas de progrès régulier ; plaine très plate pente×5,12 et bande fine×16,04. | `ckpt/base128-control/`, `eval/base128-control-pilot{,-rare}/` |
| Base192 direct à budget comparable |20k |29,96M paramètres,119,874Mo. Même plaine pente×5,04/PSD×15,47 ; côte humide71,95m et12,04% terre/mer différent. Plus gros sans gain fiable. | `ckpt/base192-decoded/`, `eval/base192-decoded-pilot{,-rare}/` |
| Reprises côtières128 : témoin / KL terre-mer0,1 |2k chacun | Proxy41 VAL : MAE23,56→25,11/26,09m ; désaccord équilibré35,25→36,05/37,52%. Régression. | `shore-trial/{control,shore}/`, `shore-trial/decision.json` |
| Gradients physiques relatifs128, poids0,01 |2k | MAE24,70m, erreur de gradient0,659 contre0,625 parent ; lignes/côtes aussi pires. | `gradient-trial/`, `gradient-trial/decision.json` |
| Premier prototype natif |1k | Pilote technique complet mais non accepté ; ne pas le confondre avec les runs opérateur suivants. | `ckpt/base-native-pilot/` |
| Native128×2, largeur/blocs réduits depuis teacher |5k→15k→50k |90,84M paramètres. À50k, MAE VAL intérieures médianes47,83/30,66m LOD3/0 ; corrélation fine rare0,136/0,617 sur branche originale. Bonnes moyennes insuffisantes pour la morphologie. | `ckpt/native128x2-{pilot,operator-long}/`, `eval/native-operator5000-matched-validation.json`, validations50k |
| Native128×2, T2/blending différentiable régional |2k après5k |7 sites : MAE médianes196,69/185,93→195,94/167,24m LOD3/0 ; formes toujours très différentes. | `ckpt/native128x2-fusion-pilot/`, `eval/native128x2-fusion-pilot/` |
| Native128×2, loss physique via decoder original |Arrêt1k | Canaux0–3 dégradés >10% ; best750 : proxy fin71,07m contre101,81m départ, sans fidélité suffisante. | `ckpt/native128x2-decoded-pilot/` |
| Native128×2, pertes multi-LOD original/coarse4/coarse2 depuis5k puis50k |Deux pilotes arrêtés1k chacun | Coarse4 en app, premier pilote500 :263/304m LOD3/0. Scores physiques baissent mais features/morphologie régressent. | `ckpt/native128x2-multilod{,50k}-mixed-pilot/` |
| Native192×2, canaux conservés, blocs réduits |15k |198,76M paramètres. MAE intérieures51,16/41,40m ; corrélation vallée désert LOD3~0,293. Reste incorrect. | `ckpt/native192x2-operator-long/` |
| Native192×2, pertes physiques multi-LOD |Arrêt1k | Rétention fine branche originale0,127→0,025→0,005 (source/500/1000). Decoder original confirme la régression. | `ckpt/native192x2-multilod-mixed-pilot/`, `eval/native192-physical-val-amplitude.json` |
| Native192×2, amplitude/profondeur des vallées ajoutée |Arrêt500 | Score physique7,640→6,551, mais rappel fin plaines tempérées18,7→15,0%, côtes37,2→33,4%. | `ckpt/native192x2-valley-mixed-pilot/` |
| Native192×2, distillation de8 sorties internes |Arrêt500 | Score agrégé légèrement meilleur mais rappel tempéré18,6→16,2%, corrélation désert0,328→0,226. | `ckpt/native192x2-feature-mixed-pilot/` |
| Native160×3, tous blocs/skips conservés, canaux calibrés TRAIN |15k puis continuation arrêtée28 579 |178,14M paramètres.15k : rappel fin~0,417/amplitude0,683.25k :~0,441/0,681, bruit natif encore excessif. La durée seule n'a pas corrigé ce run. | `ckpt/native160x3-{operator-pilot,operator-long}/`, rapports/app15k et25k |
| Native160×3, pertes distinctes par phase |Arrêt500 | Améliore un score physique mais affaiblit davantage les vallées ; brut et EMA tous deux négatifs. | `ckpt/native160x3-phase-mixed-pilot/` |
| Native160×3, premier canal0 renforcé×16 et H4 seconde réduit |Arrêt500 | MSE canal0 presque fixe,0,142170→0,142014 EMA/0,141989 brut ; amélioration brute locale limitée, vallées toujours faibles. | `ckpt/native160x3-channel0-mixed-pilot/`, audits opérateur TRAIN/VAL |
| Native160×3, LR1e-4 et5e-5, AdamW/batches diversifiés/audits gradients |Deux pilotes arrêtés500, puis deux reprises bornées jusqu'à1k | Réduit une partie du bruit ; canal0 au1k0,13778/0,13988. Rappel rare médian0,386/0,411 contre~0,425 source ; régressions fines Orogen. | `ckpt/native160x3-opt{1e4,5e5}-{pilot,diagnostic}/` |
| Teacher192×3,42 convolutions internes du décodeur au quart-rang |4k |137,83M paramètres. Extinction directions sur1k, compaction/reset Adam/EMA,3k supplémentaires. Montagnes/coarse4 :98/75/68m, corr0,881/0,869/0,923, rappel0,600/0,758/0,569. Affluents/incisions atténués. | `ckpt/native-basis-decoder-quarter-pilot/`, `eval/native-basis-quarter-final-cached/` |
| Champ pleine largeur192×3/T2, condition dense, attention locale |Arrêt2 037 sur6k prévus |254M paramètres. Au2k montagne/coarse4 :146/125/147m, amplitudes~0,80/0,82/0,67. Adaptation des grands champs non fidèle. | `ckpt/field-full-width-pilot/`, `_base/field-fullwidth-*` |
| Même champ, coordonnées périodiques64 et premier champ davantage supervisé |2 037 nouveaux pas sur4k prévus | Départ brut/EMA/Adam/RNG exact parent. Au2k :154/128/89m, amplitudes0,848/0,876/0,753. Canal0 seulement10–13% mieux en fin de run. | `ckpt/field-phase-pilot/`, `_base/field-phase-*` |
| Même champ, quatre sorties de première phase/tronc partagé |1 051 nouveaux pas sur3k prévus | Départ équivalent parent, supervision49 fenêtres non fusionnées, classes de parité et blending après arrondi BF16. Au1k :159/123/89m ; canal0 seulement4–5% mieux. | `ckpt/field-heads-pilot/`, `_base/field-heads-*` |
| Teacher natif192×3, QAT FP8 E4M3 simulé |Arrêt1 057 sur3k prévus | Topologie originale. Au1k montagne/coarse4 :47/28/75m, corr>0,99, rappel~0,93. Canal0 amélioré~20–26%, pas50% requis ; MAE souvent pire que le PTQ de départ. Pas accepté, pas d'accélération réelle. | `ckpt/native-fp8-pilot/`, `_base/native-fp8-*` |

### Première architecture directe et pertes

U-Net local avec PixelNorm par pixel, strides2, remontée nearest, dilations1/2/4 au milieu, têtes1×1 FP32 et bypass linéaire. Pas d'attention. Entrées32 canaux : deux bruits5 canaux, coarse6 canaux suréchantillonnés, masque, phases sin/cos64, statistiques globales/padding. Sortie champ final5 canaux. Halo192 ; tuiles globales512 pour figer les arrondis BF16 selon la taille de convolution.

Les différentes reprises ont utilisé MSE latente, gradients, spectre global et bandes, poids de hauteur élevé puis réduit, MAE mètres, sampling rare, loss différentiable à travers le decoder. Ces recettes et contrats sont dans `train.py`, `decoded_loss.py`, `shore_trial.py`, `gradient_trial.py` et leurs dossiers. Le parent128 provient du96 déjà appris (~295k lors de l'élargissement), puis100k supplémentaires : ce n'est pas un essai à peine entraîné.

### Compression native et apprentissage opérateur

`native_base.py`/`native_compression.py` reprennent l'architecture teacher ; réduction calibrée des canaux et/ou du nombre de blocs. Le réseau apprend les vitesses teacher aux deux labels (environ1,5625 et0,609375), puis les essais régionaux reconstruisent le solveur T2/blending BF16. `native_fusion.py`, `native_decoded.py`, `native_multilod.py` et les scripts de training décrivent les variantes. Plusieurs essais réévaluent le teacher sur **les mêmes entrées de seconde phase générées par l'élève** ; l'erreur d'état de seconde phase n'a donc pas été ignorée dans toute la série.

## 5. Manipulations sans nouvel entraînement complet

| Diagnostic / intervention | Résultat et limite | Preuve |
|---|---|---|
| Biais moyen périodique8/32 appris sur TRAIN | Période32 : proxy MAE23,7911→23,7896m, MSE légèrement pire. Variance max/min1,80 TRAIN/2,33 VAL. Soustraction du biais rejetée. | `DATA/eval/lattice-error{,-32}/` |
| Ablation bypass coarse nearest→bilinéaire | Coefficient TRAIN optimal0,0271 ; MAE VAL23,7911→23,7898m. Remplacement complet35,64m, rejeté. | `DATA/eval/coarse-skip-audit/` |
| Gradients selon frontières coarse | Erreur normalisée altitude VAL0,578 frontière vs0,418 intérieur (+38%) ; difficulté teacher seule ne l'explique pas. Les deux losses spécifiques envisagées n'ont PAS été lancées. | `DATA/eval/boundary-gradient/` |
| Ridge de sortie simple et multiscale sur features160 figées | Deux fits TRAIN avec régularisation par mondes. Contrôle auto-reconstruction~3e-6 MSE relative ; aucune récupération canal0/vallées. Ceci n'exclut pas une information non linéaire. | `DATA/eval/native160-{readout,multiscale-readout}-*` |
| Audits gradients/déplacements/arrondi BF16 | Les poids bougent réellement. Le cosinus tête agrégé était dominé par `out_gain` ; séparation corrige ce diagnostic. Pas de conflit fortement négatif persistant démontré dans les blocs partagés. | Audits des dossiers `native160x3-opt*` |
| SVD globale72 convolutions, grilles de rang1/2 à5/6, sans training | Première préparation FP32 échoue au contrôle max0,002 ; FP64 corrige le contrôle. Troncature introduit trop de bruit/amplitude. Plein rang274,7M : contrôle plus gros que teacher, pas une compression réussie. | `DATA/eval/native-lowrank-{fp64-basis.pt,rankgrid-main-validation.json,rankgrid-secondary-validation.json}` |
| Profils42 convolutions de décodeur puis quart-rang sélectif | Le vrai training quart-rang est dans le tableau ; les autres rangs sont des sondes de coût, pas des élèves appris. | `DATA/eval/native-lowrank-{cached,quarter-cached}-profile.json` |
| Diagnostic lignes d'attention internes64 sur champs | Pas de flag fort persistant sur les trois montagnes ; un500 suspect seulement. Ne prouve pas l'absence d'artefacts de grille sur tous les terrains. | `_base/field-attention-lines-*`, `distill/field_attention_diagnostic.py` |
| Calibration et primitives FP8 |93 couches internes, percentile99,99% TRAIN sur64 batches de8 fenêtres. GEMM E4M3 supporté sur les deux GPU ; `F.conv2d(float8)` refusé. Ne dit pas que toute bibliothèque NVIDIA refuse la convolution FP8. | `DATA/eval/native-fp8-calibration-v2/`, `DATA/eval/fp8-native-ops{4090,5090}.json` |

## 6. Contrôles de reconstruction, précision et raccords

Plusieurs contrôles **réussissent** :

1. Teacher non comprimé dans le wrapper natif : mêmes entrées/chemin contrôlé, exact ou au plancher BF16 selon le cas. En morphologie fine rare/coarse4 : corr0,99936, amplitude1,00015, rappel0,97879. L'enveloppe seule n'explique pas la disparition des vallées160.
2. Teacher capturé4090 vs rejoué5090 : plancher BF16 inter-GPU mesuré, inférieur au plancher FP32/BF16 sur ces entrées. Une comparaison physique ponctuelle17m peut venir de ce drift ; proxy→runtime sur les MÊMES latents~0,04–0,06m. Ne pas l'attribuer au decoder.
3. SVD FP64 plein rang : max FP32~0,000202 factorisé,~0,000054 recomposé ; MSE opérateur BF16 relative4,85e-5. Le changement d'ordre des arrondis BF16 reste distinct de l'identité FP32.
4. Compaction quart-rang : avant/après RAW, FP32 MSE relative~3e-12 ; BF16~4–5e-5. Ce n'est pas une énorme erreur de reparamétrisation cachée.
5. FieldBase **initial** sur entrée64 et condition constante : exactement teacher. Cela ne prouve ni l'identité sur champ256 ni la fidélité après training.
6. Ajout périodique à zéro : identité bit à bit parent, six contrôles brut/EMA/coarse. Quatre têtes : final initial exactement parent ; premier mélange ne change que de~1e-6 par ordre FP32.
7. QAT sans quantification : anciennes rares/coarse4 LOD0~0,012–0,17m, montagnes~0,88–1,80m. Le plancher n'est pas zéro ; il est très inférieur aux erreurs QAT.

Raccords : les premières familles à champ direct ont plusieurs comparaisons physiques froides, quatre tuiles, axes/halos, **écart maximal zéro avec découpage global fixe**. Cela ne certifie pas l'absence de lignes à l'intérieur des tuiles. Les champs pleine largeur ont des contrôles de recadrage décalé32 sur les deux axes et une loss associée, mais halo64 n'est PAS une borne mathématique prouvée du champ récepteur. Les dernières familles ne disposent pas d'une validation finale app/cold-seams/final worlds : ne pas réutiliser une preuve d'un autre modèle pour les certifier.

Références : `DATA/eval/native-full-wrapper-control-validation.json`, `native-fusion-startup-gpu1-floor.json`, `native-basis-quarter-cut-check.json`, dossiers `*seams*` ; `distill/check_physical_seams.py`, `distill/test_native_physical_seams.py`.

## 7. Diagnostics causaux les plus informatifs

**Ces expériences injectent du teacher : elles ne sont pas des élèves autonomes utilisables.** Elles isolent des défauts sans identifier à elles seules leur cause d'apprentissage.

### Native160×3

- Teacher première phase→élève seconde : récupère les vallées fines (rappel~0,838 vs0,400) mais garde le bruit natif (amplitude~3,49 vs3,35).
- Élève première phase→teacher seconde : retire le bruit natif (amplitude~0,948), mais ne récupère pas les vallées fines (rappel~0,412).
- Remplacer **seulement le canal0 de première phase** : rappel fin médian0,397→0,788, amplitude0,628→0,923 ; remplacer0–3 :0,830/0,937 ; H4 seul :0,408/0,638.
- Remplacer0–3 finaux teacher dans un192×2 : coast~94–96% de rétention, contre~15–37% selon l'élève ; remplacer seulement H4 ne suffit pas au LOD0.

Preuves : `DATA/eval/native160x3-step15000-phase-oracle.json`, `native160-first-channel-oracle.json`, dossiers `native192-operator15000-matched-oracles/` et `native192-physical500-matched-oracles/`.

### Champ pleine largeur

- Au1500, champ autonome montagne/coarse4 :138/126/164m ; **premier champ teacher fourni** au même réseau seconde phase :36/29/41m, corr~0,997.
- La feature0 représente environ80% de la MSE normalisée des premières features0–3 ; autres features bien moins erronées.
- Au pilote périodique1000 : autonome150/129/107m, corr0,848/0,854/0,897. Remplacer seulement le premier canal0 :56/28/66m, corr~0,994/0,993/0,993. Remplacer les autres canaux/H4 ne rétablit pas les formes de façon comparable. Tout premier champ teacher :38/23/39m.
- La capture fournit49 sorties natives64 avec pas32 : champ initial192 central complètement couvert ; bord externe du256 seulement partiellement couvert. Les oracles ne doivent pas traiter ce bord comme une vérité planète complète.

Preuves : `DATA/eval/field-oracles1500/`, `field-first-features500.json`, `field-first-features-final.json`, `field-channel-oracles1000.json` ; `distill/field_oracles.py`, `field_channel_oracles.py`.

Conclusion empirique limitée : un défaut du premier canal0 explique beaucoup de la perte d'incision **dans ces familles et ces cas** ; le canal hauteur/seconde phase ajoute un problème distinct de bruit. Cela ne démontre pas une ambiguïté fondamentale de la cible déterministe, ni que les58 conditions/recouvrements sont seuls responsables, ni une incapacité des architectures à toute durée.

## 8. Dernier essai : QAT native, désormais arrêté

`NativeFP8Base` conserve toute la topologie, les conditions58 scalaires,64/32 etT2.93 convolutions4D internes : poids effectifs globalement normalisés arrondis E4M3 avec échelle par canal de sortie ; entrées arrondies E4M3 avec échelle statique par couche. Entrée/sortie, embeddings/FiLM, normalisations, softmax/logits et decoder externe conservés. STE correct vérifié sur BF16 et clipping ; le défaut d'une première expression STE a été corrigé **avant training**. Calibrationv2 figée.

Adam1e-5, warmup100, clip1, EMA demi-vie500, mastersFP32 et convolutionBF16 simulant l'arrondi ; budget3k, batch8 mondes distincts,4 par phase. Loss cinq canaux normalisée + premier canal0×2 + gradients/x0/bandes/énergies + KD37 blocs, coefficient0,35330 calibré sur16 batches TRAIN. Exactement50% des batches utilisent, pour les quatre entrées de seconde phase, les neuf fenêtres de première phase de l'élève **actuel**, fusionnées puis détachées. Teacher évalué sur les MÊMES entrées, sans états périmés. Pas de différentiation à travers ce premier état on-policy.

GPU smoke5090 : trois vrais backprops, gradients finis, teacher figé, échelles inchangées,37 taps, pic7,93GiB. Les3 tests CPU dédiés passent. Les preuves techniques ne prouvent pas que l'objectif optimisé soit approprié.

Résultats **autonomes LOD0, montagnes/coarse4**, même protocole :

| Monde VAL | PTQ avant training : MAE / corr / rappel / amplitude | QAT1000 : MAE / corr / rappel / amplitude | MSE canal0 première phase, départ→1000 |
|---|---|---|---|
|620001|27,10m /0,99095 /0,93386 /0,9646|47,47m /0,99816 /0,93662 /0,9515|0,038075→0,028049|
|620002|20,28m /0,98839 /0,92826 /0,9649|28,01m /0,99042 /0,92888 /0,9670|0,061052→0,048944|
|620003|52,18m /0,99683 /0,94839 /0,9595|75,20m /0,99489 /0,92651 /0,9432|0,053368→0,041760|

Le garde1000 exigeait canal0÷2 et corr≥0,97 sur neuf branches montagneuses. Corr passe, canal0 n'est pas divisé par deux : arrêt réel1057,10,4min, `status=guarded`, pas une panne. Il faut critiquer cette règle si nécessaire ; elle ne prouve pas qu'un run plus long échouerait. En revanche la MAE autonome se dégrade sur ces trois mondes malgré une baisse de MSE canal0.

Les anciennes rares sont aussi évaluées aux deux LOD : **ne pas conclure sur les seules montagnes**. Exemple désert/coarse4 au1000 : LOD0 MAE0,77m/corr0,980, mais LOD3 MAE3,65m/corr0,663/amplitude1,42. La bonne restitution fine ne garantit pas celle du canal4 coarse. JSON complets : `DATA/_base/native-fp8-{ptq,qat}-{original,coarse4,coarse2}-{000000,000500,001000}.json` selon les étapes disponibles.

Code actuel non encore committé : `native_fp8.py`, `native_fp8_calibrate.py`, `native_fp8_training.py`, `native_fp8_train.py`, `native_fp8_evaluate.py`, `native_fp8_watch.py`, `test_native_fp8.py`. Les hashes du run sont dans `ckpt/native-fp8-pilot/contract.json`. Les anciennes introductions de `TASKS.md`/`BASE_NATIVE_FP8_20261012.md` mentionnaient le lancement ; le statut terminal ci-dessus fait foi.

## 9. Performance : ce qui est et n'est pas mesuré

### Poids appris, ancienne famille directe ; champs2048², entrées/transferts inclus

| Base |4090 teacher→élève|5090 teacher→élève|Gain5090|
|---|---:|---:|---:|
|128 initial|16,272→2,217s|12,728→1,842s|6,91×|
|128 couplé|16,282→2,250s|12,769→1,848s|6,91×|
|128 contrôle+20k|16,277→2,248s|12,813→1,849s|6,93×|
|192+20k|16,230→3,134s|12,842→2,328s|5,52×|

Poids résidents, GPU au repos, mondes neufs, médiane3 répétitions. Ce sont des temps **de l'étape base**, pas d'affichage complet. Optimisations exactes d'inférence historiques vérifiées sur38 vues et seams ; elles ne réparent pas la morphologie.

### Coûts d'architecture / contrôles sans garantie de poids finaux

- Native5090, champ1024², préparation/fusion/transferts inclus, poids aléatoires candidats : teacher3,462s ;192×2=2,797s ;128×2=1,726s ;96×2=1,540s ;128×1=1,541s ;96×1=1,481s. **Coûts seuls**, pas benchmarks de bons modèles appris.
- Profil lowrank caché BF16/CUDA Graphs,5090,batch16, **ms par fenêtre64** : teacher1,400 ; demi-rang sélectif1,182 ; quart-rang1,001 ; huitième0,904. Pas un temps de chaîne complète. Le quart-rang a été entraîné, mais ces timings précèdent les poids finaux.
- Ancien wrapper lowrank eager :3,974ms teacher et3,57–3,84ms tronqués ; casts/normalisations/lancements Python inclus. Ne pas prendre cela pour la borne du déploiement caché.
- Champs pleine largeur : **pas de benchmark de déploiement validé**. Ils visent la suppression du recouvrement, mais aucun gain utile ne doit être annoncé avec ces poids.
- QAT native : **convolutions BF16 avec arrondi FP8 simulé ; aucune accélération FP8 obtenue**. Pas d'export packé FP8 ni de timings d'une convolution FP8 réelle. Le réseau conserve254M paramètres. Ne pas présenter cela comme une distillation compacte terminée ou un gain×2 acquis.

Preuves : `DATA/bench/{optimized-parent,optimized-bundle,decoded-bundle,wide-bundle}/`, `bench/native-architectures/gpu1.json`, `eval/native-lowrank-{cached,quarter-cached}-profile.json`, `eval/fp8-native-ops{4090,5090}.json`, pack `final/evidence/`.

## 10. Limites de notre conduite expérimentale à examiner

- Beaucoup de recettes ont été arrêtées à500/1000 pas sur des critères parfois très ambitieux ; elles n'ont PAS toutes été entraînées jusqu'à convergence. Les longueurs/rates/schedulers/EMA doivent être critiqués, pas présumés suffisants.
- Le choix de métrique a plusieurs fois favorisé un score agrégé tout en diminuant les incisions. Les garde-fous protègent d'une régression mais peuvent aussi interrompre une amélioration transitoire avant récupération. Un rappel sur champ déjà bruité peut être trompeur.
- Nombreuses variables changées au fil des familles : capacité, entrées, géométrie, losses, sampling, états autonomes. Les comparaisons appariées existent pour certains essais, pas pour toutes les transitions ; on ne dispose pas d'une ablation causale complète.
- La distillation directe finale est déterministe si toutes les entrées/state/contexte sont corrects. La moyenne/lissage de régression peut révéler des entrées insuffisantes, un objectif mal conditionné ou un échec d'optimisation ; elle ne prouve pas une ambiguïté intrinsèque du teacher.
- Le champ fullwidth adapte conditionnement/attention/padding et une géométrie de fenêtres à un grand champ. Conserver les poids/paramètres n'est pas conserver la fonction sur ces entrées.
- KD des blocs teacher sur fenêtres64 contre les grands champs peut entrer en conflit avec la cible blended/contextuelle. Ce conflit est plausible, pas établi comme cause unique.
- Le quart-rang a une exposition on-policy tardive via4090 : les logs échantillonnés ne prouvent pas la fraction prévue sur tout le run. QAT corrige ce point par états actuels et cadence déterministe.
- Répétition d'analyses sur les mêmes204 scènes et trois mondes montagne VAL : risque d'adapter les stratégies à VAL. Les mondes finaux sont réservés mais la dernière candidate n'a pas été évaluée dessus.
- Spectre et TPI ne remplacent ni les images, ni un test de drainage, ni les raccords réels froids. Pas de preuve finale pour les familles récentes.
- La piste QAT a une meilleure fidélité initiale car elle garde la fonction native, mais son objectif quantification ne suffit pas à atteindre l'objectif initial de forte accélération/compactage. Le poids stocké, la précision calcul et la distillation de fonction sont trois choses distinctes.
- Tous les poids/Adam/RNG significatifs sont conservés hors Git. Quelques anciens checkpoints sont archivés sur `/mnt/d/workspace/distill-archive` avec SHA et symlinks Linux ; données actives Linux. Aucune suppression de poids uniques pour cette revue.

## 11. Dossier de preuve et code à lire

Rapport initial : [REVIEW_OPUS_20261010.md](REVIEW_OPUS_20261010.md). Plans et recettes : [STRATEGY_20261010.md](STRATEGY_20261010.md), [BASE_PLAN_20261011.md](BASE_PLAN_20261011.md), [BASE_FIELD_PHASE_20261012.md](BASE_FIELD_PHASE_20261012.md), [BASE_WINDOW_HEADS_20261012.md](BASE_WINDOW_HEADS_20261012.md), [BASE_NATIVE_FP8_20261012.md](BASE_NATIVE_FP8_20261012.md). Journal long : [TASKS.md](TASKS.md), avec descriptions historiques parfois rédigées au lancement ; **statuts et contrats des dossiers sont prioritaires**.

Code par famille :

- Champ direct : `student.py`, `features.py`, `teacher.py`, `dataset.py`, `train.py`, `decoded_loss.py`, `inference.py`.
- Native : `native_base.py`, `native_compression.py`, `native_bank.py`, `native_fusion.py`, `native_decoded.py`, `native_multilod_train.py`, `native_phase_train.py`, `native_optimization_train.py`.
- Compression/KD : `native_lowrank.py`, `native_basis_training.py`, `native_basis_train.py`, `native_feature_kd.py`.
- Champ pleine largeur : `field_base.py`, `field_train.py`, `field_loss.py`, `field_phase_train.py`, `field_heads_train.py`, `field_oracles.py`.
- QAT : `native_fp8.py`, `native_fp8_training.py`, `native_fp8_train.py`, `native_fp8_evaluate.py`, `native_fp8_watch.py`.

Les anciennes revues Opus sont sous `DATA/review/opus-*/` (rapport, prompt, transcript, modèle réel, verdict). Elles ont suggéré les pistes successives ; leurs hypothèses ne sont pas toutes confirmées. Pour une **nouvelle revue indépendante**, partir de ce rapport et des sources, sans reprendre aveuglément leur conclusion.

Un snapshot léger des statuts, inventaire de chemins et métriques QAT est préparé sous `DATA/review/base-comprehensive-20261012/` ; aucune nouvelle inférence n'est nécessaire pour le lire. Le rapport est fourni identique à Astra high et Opus5.5 xhigh. Les réponses seront conservées en Markdown séparé, avec identification du modèle et effort.

## 12. Questions aux deux reviewers — réponse de deux pages maximum

Répondre en français, **≤1000 mots**, soit deux pages environ. Lecture seule : ne pas modifier les sources, lancer d'inférence, de génération de données, de benchmark ou de training. Examiner les preuves et le code utiles ; signaler les éléments impossibles à confirmer.

1. Quel est selon vous le problème principal ? Classer2–3 hypothèses, avec faits favorables/contraires, et distinguer architecture/entrées, apprentissage/objectifs, protocole/arrêts, précision. Critiquer également les erreurs de méthode dans les décisions précédentes.
2. Que montrent vraiment les oracles canal0/première phase, les contrôles teacher exacts et le QAT initial relativement fidèle ? Que **ne** montrent-ils pas ?
3. Quelle **une prochaine recette principale** choisir pour distiller correctement le base ? Donner architecture, entrée/cible, géométrie/T2 ou1 passe, initialization, losses, données, LR/scheduler, durée/budget, EMA et critères d'arrêt praticables. Justifier la voie choisie plutôt qu'une liste de nouvelles pistes non priorisées.
4. Quel contrôle minimal falsifierait votre hypothèse avant de brûler un nouveau gros run ? Un surapprentissage exact d'un petit jeu ou une ablation n'est utile que s'il départage une cause concrète. Éviter de remettre en tête des contrôles déjà concluants sans raison nouvelle.
5. Comment juger simultanément vallées et bruit/côtes/plaines, et préserver les vrais raccords de tuiles ? Quand considérer une candidate visuellement testable malgré des erreurs ? Quelle accélération peut-on raisonnablement viser, sans transformer des spéculations en mesures ?

Contraintes : deux GPU (5090 principale,4090 teacher/validation utile), poids prioritaires, pas de développement de kernels d'inférence à ce stade ; comparer avec teacher original conditionné par le même coarse ; préserver les mondes finaux. Aucune nouvelle stratégie ne sera lancée automatiquement sur la seule réponse d'un reviewer.
