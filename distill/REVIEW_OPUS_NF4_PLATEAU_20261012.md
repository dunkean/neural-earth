# Revue du plateau NF4 natif : analyse complète et protocole contrôlé sur la 4090

Je n'ai rien écrit, modifié ni lancé. Je me suis limité à des lectures : le rapport figé, les huit fichiers demandés, les deux nouveaux prototypes et les quelques dépendances nécessaires pour conclure (`native_weight_quant_evaluate.py`, `native_weight_quant_watch.py`, `native_fusion.py`, `x0_from_velocity`, `representation_loss`, `laplacian_encoder.py`, `mp_layers.normalize`). J'ai aussi lu cinq CSV d'évaluation coarse4 dans `~/data/distill/_base`. Ces CSV changent le diagnostic. Le run principal sur la 5090 n'est pas concerné.

---

## 0. Verdict

1. **Le rapport mélange les niveaux de détail.** Ses chiffres de plaine (MAE 4,8 à 8,8 m, ratio de hautes fréquences 1,7 à 2,1) sont mesurés au **LOD3**. Ses chiffres de montagne et de côte (plaine basse/mer) sont mesurés au **LOD0**. Au LOD0, le ratio de hautes fréquences des plaines est **inférieur à 1** (0,72 à 0,93) : il n'y a pas d'excès de grain fin au sens de la métrique σ=1.
2. **Au LOD0, l'erreur dominante est un décalage vertical négatif systématique**, pas du bruit. On a `height_bias_m = −mae_m` en désert et en plaine tempérée : le candidat est sous le teacher sur pratiquement tout l'intérieur 256². Ce biais vient de la quantification NF4 brute (step 0 : −17,6 m désert, −20,8 m tempérée, −1,90 m plaine/mer, −245 m montagne 620001). La QAT en a absorbé l'essentiel en 6 000 pas, puis il stagne ou fluctue. Sur la plaine/mer, le désaccord de signe au LOD0 suit ce biais.
3. Pour le plateau, je retiens surtout deux causes :
   - un **objectif** qui ne voit presque pas ce biais ni ce grain ;
   - des **preuves insuffisantes** : une seule scène et un seul bruit par catégorie, une évaluation non déterministe, et un score qui suit surtout l'offset.
   
   Le LR et le plancher du quantifieur sont plausibles mais non mesurés. L'EMA n'est pas une cause démontrée. Rien dans la recette ne prédit un « grokking ».
4. **Le proxy LOD3 sur le core final fusionné est le bon premier essai** pour le grain LOD3 et la côte LOD3. Il ne peut pas corriger le biais du désert et de la plaine tempérée (hors bande ±20 m, et un passe-haut n'a pas de composante continue). Rien ne garantit qu'il se transfère à la côte LOD0 : cela doit être un critère mesuré, pas une hypothèse.
5. Deux corrections sont nécessaires avant de figer le prototype :
   - calculer la KL de côte dans l'espace racine `q`, et non dans `h = q|q|` ;
   - imposer un nombre minimum de pixels par classe.
6. Protocole : une phase 0 courte (forward et CPU seulement), puis deux bras A et B partant de la **même copie complète** d'un checkpoint du primaire, entrelacés par blocs de 2 500 pas, 10 000 pas par bras.

---

## 1. Faits vérifiés dans le code

### 1.1 L'objectif actuel (`native_weight_quant_train.py:26-51`)

- **`operator`** : MSE des vitesses, normalisée par la variance globale de chaque canal, sur les 8 fenêtres.
- **`+.1*channel0`** : ce terme double le poids du canal C0 de la première phase (commentaire explicite dans `native_fp8_training.py:59-60`).
- **`+.05*gradient`** : différences finies du résidu, à l'échelle latente.
- **`+.25*x0`** : ce terme n'apporte pas d'information nouvelle. `x0_from_velocity = cos·inputs + sin·v` (`native_phase_loss.py:19-23`), donc `actual − reference = sin(t)·(v_s − v_t)`. C'est la même erreur, repondérée par sin²(t).
- **KD sur 37 blocs**, normalisée par l'énergie du teacher sur le batch.

**Conclusion.** Toute la supervision est une MSE latente « même entrée », locale à chaque fenêtre, pondérée par une variance globale. Aucun terme ne porte sur le champ fusionné, le signe près de 0, un décalage basse fréquence, ou une normalisation par scène. Le premier étage ne reçoit jamais de signal sur ses conséquences en aval : les états on-policy sont construits sous `no_grad` (`native_fp8_training.py:35-46`).

Ordre de grandeur (estimation) : avec LOD3 `h = q|q|` et `q = 38,6·C4 − 31,4` :
- −4 m sur une plaine à environ 300 m correspond à ΔC4 ≈ −0,003 ;
- −0,33 m autour de 0,3 m correspond à ΔC4 ≈ −0,02.

Ces écarts sont cohérents spatialement mais petits dans une MSE moyenne dominée par le relief.

### 1.2 La continuation (`native_weight_quant_resume.py`)

- **LR** : constant, hérité des `param_groups` et jamais réassigné (l.65, 106-122), soit 3e-6.
- **Batches** : ils dépendent seulement de `(seed, step)` via `SeedSequence` (l.107). Deux bras au même step voient les mêmes scènes, fenêtres et variantes de coarse. Seuls les états on-policy divergent quand les modèles divergent. C'est un atout pour un appariement exact.
- **Sauvegardes** :
  - snapshots EMA en BF16 tous les 5 000 pas (l.138, 157) ;
  - `latest.pt` complet **écrasé** à chaque fois (l.158) ;
  - poids raw en BF16 seulement à 20k, 50k et à la cible (l.159).
  
  Donc l'EMA15k n'a ni état Adam ni poids maîtres FP32.
- **Contrat** : il vérifie le SHA de 8 fichiers sources (l.61-64). Modifier l'un d'eux empêcherait tout redémarrage futur du primaire. Le prototype est dans des fichiers séparés, c'est correct ; l'ablation doit continuer ainsi.

### 1.3 La quantification (`native_weight_quant.py`)

La chaîne est : maître FP32 → cast BF16 (l.122) → `normalize` → division par √fan_in (l.77) → NF4 absmax par bloc de 512 (l.30-45) → STE (l.48-50).

- Ici, `normalize` utilise `dim=None`, donc une **RMS globale du tenseur** (`mp_layers.py:9-12`).
- Les échelles ne sont pas apprises.

Conséquence (estimation, non mesurée) :
- Le forward ne change que si un code franchit un milieu NF4, ou si le maximum d'un bloc change de valeur BF16.
- Hypothèses : RMS ≈ 1 grâce à la normalisation forcée pendant l'entraînement du teacher (`mp_layers.py:206-208`), et absmax ≈ 3σ pour 512 valeurs.
- Le demi-intervalle central vaut alors environ 0,04 × 3 ≈ 0,12. Avec au plus 3e-6 par pas, il faut des milliers à des dizaines de milliers de pas de gradient de signe constant pour un seul franchissement.

La dynamique est donc lente et procède par sauts. C'est compatible avec les fluctuations d'offset entre checkpoints, sans le prouver.

### 1.4 L'évaluation

- **Échantillon** : une scène VAL par catégorie (la première par seed) et par coarse (`evaluate.py:33-37`), plus 3 montagnes. Un seul bruit, celui stocké.
- **LOD3** : `q|q|` du C4 sur le core fusionné de 64×64.
- **LOD0** : résidu du décodeur ×0,7 + basse fréquence ré-encodée par `laplacian_denoise(σ=5)` (`decoded_loss.py:76-80`), puis recadrée deux fois jusqu'au 256² central (`decoded_loss.py:79` puis `evaluate.py:51`). **Le σ=5 filtre les hautes fréquences de C4 au LOD0.** Le grain LOD3 n'atteint donc pas le LOD0 ; seule sa basse fréquence passe, y compris le biais.
- **Score** (`watch.py:42-45`) : il mélange LOD3 et LOD0, coarse4 et coarse2, MAE/35 ou /10, |log HF| et désaccord moyen. Comme les MAE sont dominées par l'offset, le score suit surtout l'offset.
- **Non-déterminisme** : `cudnn.benchmark=True` (`evaluate.py:110`, `watch.py:27`). `nf4qat-006000` et `nf4long-006000` devraient être les mêmes poids (même EMA FP32 convertie en BF16), à vérifier par égalité de tenseurs. Ils donnent pourtant :
  - C0 MSE : 0,031397 contre 0,031448 ;
  - côte LOD0 : 0,2834 contre 0,2901 m ;
  - désaccord : 10,93 % contre 10,75 % ;
  - montagne 620002 au LOD0 : 14,10 contre 14,56 m.
  
  Si les poids sont bien identiques, c'est du bruit d'évaluation pur, de l'ordre de 2 à 3 % relatif.

### 1.5 La sonde de côte (`native_shore_probe.py`)

- Le gradient est pris par rapport à un **core détaché** (l.45). La sonde ne dit donc rien sur les poids ni sur les étages.
- Le contrôle « correctif » (l.57-59) est vrai par construction, puisque la KL est monotone en p.
- Les « 8 cas » sont en réalité 4 régions × 2 LOD.

### 1.6 Le prototype (`native_field_auxiliary.py` et sa calibration)

- **`low_field_terms`** combine deux termes :
  - `shore_loss` sur `h = q|q|` : bande 20 m, T = 2 m, masque plein ;
  - un passe-haut apparié σ=1 sur les pixels de terre du teacher, bord de 3 px exclu, normalisé par une puissance TRAIN fixe par catégorie.
- Ce passe-haut correspond exactement à la définition de la métrique (`evaluate.py:60-61`).
- **Calibration** : moyenne des puissances par scène (pas la médiane), plancher 0,1 m RMS. Valeurs obtenues :

  | Catégorie | Puissance | RMS équivalente |
  |---|---|---|
  | Désert | 2,295 m² | 1,51 m |
  | Tempérée | 2,286 m² | 1,51 m |
  | Plaine/mer | 0,0546 m² | 0,234 m |

  Les scènes VAL sont plus rugueuses (référence HF LOD3 de 2,36 / 4,23 / 0,44 m RMS). Ce n'est pas un problème, puisque le dénominateur est fixe et non bruité ; la moyenne donne simplement plus de poids aux scènes rugueuses de chaque catégorie.
- La catégorie `wet-complex-coast` n'a pas de puissance : elle ne doit recevoir que le terme de côte.

---

## 2. Ce que disent les CSV coarse4

Colonnes : step 0 = `nf4qat` ; 6k = `nf4long`, qui correspond exactement au rapport ; 20k = **raw** ; 30k = EMA.

| Métrique | 0 (NF4 brut) | 6k | 20k raw | 30k EMA |
|---|---|---|---|---|
| Plaine/mer LOD0 : MAE / biais (m) | 1,91 / −1,90 | 0,290 / −0,268 | 0,556 / −0,555 | 0,336 / −0,331 |
| Plaine/mer LOD0 : désaccord de signe | 36,6 % | 10,75 % | 17,7 % | 11,6 % |
| Plaine/mer LOD3 : désaccord près de la côte | 29,8 % | 13,9 % | 14,3 % | 12,5 % |
| Plaine/mer LOD3 : ratio HF | 9,51 | 2,26 | 2,26 | 2,24 |
| Désert LOD0 : MAE = \|biais\| (m) | 17,61 | 4,66 | 4,80 | 3,99 |
| Désert : ratio HF LOD3 / LOD0 | 4,54 / 0,60 | 1,84 / 1,04 | 1,78 / 1,08 | 1,70 / 0,89 |
| Tempérée LOD0 : biais (m) | −20,83 | −3,53 | −5,21 | −5,38 |
| Tempérée : ratio HF LOD3 | 6,59 | 2,09 | 2,21 | 2,08 |
| Montagne 620001 LOD0 : MAE (biais) | 244,6 (−244,6) | 41,1 (−40,0) | 33,7 (−21,2) | 33,5 (−31,5) |
| Côte découpée LOD0 : désaccord | 2,04 % | 1,11 % | 1,15 % | 0,74 % |

Lecture :
- **Plaine/mer LOD0** : le désaccord de signe suit |biais| (0,27 → 10,75 % ; 0,33 → 11,6 % ; 0,56 → 17,7 %). La « dispersion » de côte visible est d'abord un décalage vers le bas d'une plaine sous-métrique.
- **Hautes fréquences LOD3** : le ratio tombe de 4,5–9,5 à environ 2 en 6 000 pas, puis reste plat. Au LOD0, il est inférieur à 1.
- **Plaine tempérée** : le biais LOD0 s'aggrave entre 6k et 30k.
- **Montagnes** : la MAE LOD0 est elle aussi surtout un offset, alors que la forme est bonne (corrélation TPI 0,99). Le « meilleur 15k » est probablement une oscillation d'offset choisie après coup sur la VAL : à 15k, 620001 était à 16,1 m mais 620002 à son pire, 21,7 m. Je n'ai pas lu le CSV de 15k pour le confirmer.
- **Côte découpée** : faible et en baisse ; ce n'est pas le problème.

---

## 3. Diagnostic par hypothèse

| Hypothèse | Verdict | Fondement |
|---|---|---|
| **Objectif** | Cause principale probable (confiance moyenne) | La loss ne contient rien pour l'offset du champ final, le signe près de 0 ou la HF relative en scène plate. La loss TRAIN baisse encore (−13 %, `operator` −16 %) sans gain VAL sur ces axes. |
| **Échantillonnage** | Secondaire | La plaine/mer n'apparaît qu'environ 1 fois sur 4 dans les 15 % de créneaux « rares » (`choose`, `native_basis_training.py:92-116`), plus une part du « général ». À ne pas changer dans l'ablation : cela confondrait les effets. |
| **LR / schedule** | Plausible, non mesuré | Le réseau bouge encore (la loss TRAIN baisse), mais avec une dynamique NF4 + BF16 très lente (§1.3). Changer le LR serait une autre ablation. |
| **Plancher du quantifieur** | Indécidable aujourd'hui | La quantification NF4 brute crée le biais ; la QAT en retire 75 à 98 %. Le reste peut venir des blocs de 512, des échelles non apprises ou de l'objectif. |
| **EMA** | Non démontré | Le raw à 20k est pire sur la côte parce que son offset est plus grand (−0,555 m). C'est cohérent avec les oscillations du STE. Il faut évaluer raw et EMA, et décider sur l'EMA fixée à l'avance. |
| **Preuves insuffisantes** | Certain | Une seule scène et un seul bruit, du bruit d'évaluation, un score dominé par l'offset, un 15k choisi après coup sur la VAL. |

**Sur le grokking :** à LR constant de 3e-6, avec un objectif fixe et un STE, rien ne prédit un saut tardif. Continuer le primaire jusqu'à 100k ne coûte rien, mais ce n'est pas un plan.

---

## 4. Quel proxy choisir, et les corrections à apporter

**Le proxy LOD3 fusionné plutôt qu'une supervision LOD0 décodée en premier.**
- La HF au LOD0 n'est pas excessive.
- L'erreur de côte LOD0 sur la plaine/mer est dominée par l'offset, qui est porté par la basse fréquence de C4 — le même levier que la côte LOD3 près de 0.
- Le transfert vers le LOD0 reste à mesurer.
- **Ce proxy ne couvre pas** l'offset du désert et de la plaine tempérée : la MAE des plaines n'est donc pas un critère de réussite de cette ablation.

**Le champ fusionné plutôt que des fenêtres individuelles.**
- La fusion moyenne 2×2 fenêtres par pixel, ce qui atténue le grain non corrélé.
- Le signe d'une moyenne n'est pas la moyenne des signes.
- Les bords de fenêtre ont un poids de fusion proche de 0, mais pèseraient autant dans une loss par fenêtre.

**Corrections avant de figer le prototype :**

1. **Calculer la KL de côte dans l'espace `q`.**
   - Appel : `shore_loss(q_s, q_t, mask, band_m=√20≈4,472, temperature_m=√2≈1,414)`.
   - La bande, les classes et le zéro à cible exacte restent identiques.
   - Dans l'espace `h`, le gradient est multiplié par dh/dq = 2|q|, qui **s'annule sur le trait de côte**. Exemple : q_s = +0,3, q_t = −0,7 donne un gradient par rapport à q de 0,022 dans l'espace h contre 0,123 dans l'espace q, soit environ 6× plus faible.
   - Le passe-haut, lui, reste dans l'espace h, comme la métrique.
2. **Imposer au moins 16 pixels LOD3 par classe**, en mettant à zéro le masque des classes sous ce seuil. Aujourd'hui, `shore_loss` active toute classe dès qu'elle a un pixel (`decoded_loss.py:106`) : un pixel seul pèse alors autant que 2 000.
3. **Utiliser une cible « même entrée »** :
   - le teacher applique l'étage 2 sur les `inputs1` de l'élève ;
   - l'étage 1 de l'élève est calculé sans gradient ;
   - le bruit est **neuf à chaque pas auxiliaire**. C'est gratuit : les entrées de l'étage 1 sont `zeros + noise`, et le conditionnement ne dépend pas du bruit (`native_fusion.py:131-137`).
   - Pas de cible de trajectoire stockée.
4. **Échantillonnage auxiliaire stratifié** : 4 catégories × 3 variantes de coarse, TRAIN seulement, avec le contrôle `HOLDOUT_SEEDS`. `flat_scene=True` pour les 3 catégories plates, avec leur puissance. Terme de côte seul pour `wet-complex-coast`.

---

## 5. Les risques demandés

- **Cible « même entrée » ou cible de trajectoire.**
  - La même entrée est cohérente avec `operator` : la seule optimale reste le teacher, la loss change seulement quelles erreurs comptent. Mais elle ne peut pas corriger une erreur née à l'étage 1.
  - La cible de trajectoire (`region.target`) demande à l'étage 2 de compenser l'étage 1 : c'est une autre fonction que le teacher. Elle est figée sur un seul bruit par scène, donc on risque d'apprendre des corrections propres à ce bruit, et elle pousse l'étage 1 vers des états qui ne ressemblent pas à ceux du teacher.
  - La phase 0.3 dit lequel des deux étages porte l'erreur.
- **Appariement du bruit, du conditionnement et de la variante de coarse.** Le teacher et l'élève doivent recevoir les mêmes `inputs1`, le même bruit d'étage 2, la même `condition`, les mêmes labels et la même variante de coarse. Sinon, le passe-haut apparié pénalise des différences irréductibles dues au bruit, ce qui pousse vers la moyenne conditionnelle : lissage (ratio HF < 1) et biais.
- **Proxys de signe par fenêtre** : refusés pour les raisons du §4.
- **Biais et amplitude.**
  - La KL de côte corrigera l'offset près de 0 : c'est voulu, et c'est borné par le teacher.
  - Si le mécanisme de l'offset est global, l'effet peut déborder sur les montagnes et les plaines. Comme le biais y est aussi négatif, ce pourrait être favorable, mais il faut le mesurer.
  - Le passe-haut peut réduire l'amplitude en cas d'incertitude : il faut surveiller un ratio HF LOD3 inférieur à 0,8, le `rms` TPI des montagnes et le ratio HF au LOD0.
- **Raw ou EMA.** L'EMA des poids maîtres, une fois quantifiée, n'est pas la moyenne des réseaux quantifiés. Il faut sauvegarder et évaluer les deux, et décider sur l'EMA, choisie à l'avance.
- **Protection du primaire.**
  - Ne toucher à aucun des 8 fichiers dont le SHA est vérifié.
  - Disque : le primaire a besoin d'espace pour ses écritures atomiques.
  - RAM : `QATBank` met en cache toutes les régions chargées (`native_fp8_training.py:16-25`). Un second processus double ce cache, et un OOM sous WSL2 pourrait tuer le primaire.
  - `evaluate()` ajoute une entrée à `_base/image-publications.jsonl` (l.98) : utiliser des labels distincts (`abl-A` / `abl-B`).

---

## 6. Le protocole

### 6.0 Pré-enregistrement
Un `contract.json` fixe, avant tout résultat, tous les éléments ci-dessous : source, λ, τ, bande, seuils, panels, bruits, critères, marges et règle de décision. Tout le code nouveau va dans de nouveaux fichiers.

### 6.1 Phase 0 : forward et CPU uniquement, sur la 4090
1. **Contrôle BF16.** Lancer `evaluate --no-quantization` sur `step000000.pt` (poids teacher passés par le pipeline natif), en coarse4 et coarse2, avec des flags déterministes. Biais et désaccord attendus proches de 0. **Si ce n'est pas le cas, une partie de « l'offset » vient de l'évaluation : il faut s'arrêter et la corriger d'abord.**
2. **Déterminisme de l'évaluation.**
   - Vérifier sur CPU l'égalité des tenseurs entre le `step006000.pt` du pilote et celui de la continuation.
   - Réévaluer deux fois le même fichier avec `cudnn.benchmark=False` et `cudnn.deterministic=True` : les métriques doivent être identiques.
3. **Attribution par étage**, au checkpoint source, sur un panel TRAIN fixe de 24 captures (6 par catégorie, variantes de coarse équilibrées, bruit stocké). Trois configurations, comparées à `region.target` :
   - **ÉÉ** : élève sur les deux étages ;
   - **TÉ** : étage 1 teacher (entrées stockées `second['inputs']`), étage 2 élève ;
   - **ÉT** : étage 1 élève, étage 2 teacher.
   
   Métriques : désaccord LOD3 près de la côte, |log HF| au LOD3, biais.
   
   Règle :
   - si TÉ conserve au moins 50 % de l'erreur ÉÉ sur ces deux métriques, on lance le bras **B** (étage 2) ;
   - si TÉ en conserve moins de 25 % sur les deux, on lance plutôt **B′** : gradient complet sur les 34 fenêtres, avec checkpointing et cible `region.target` (risque de conflit assumé) ;
   - entre les deux, on lance B.
4. **Changements de codes NF4** entre les snapshots EMA successifs du primaire, et entre raw20k et EMA20k, couche par couche (CPU). Cela répond à la question « le réseau bouge-t-il à 3e-6 ? ».
5. **Calibration de λ**, détaillée en 6.4.

### 6.2 Source commune
- **Recommandé : copie du prochain `latest.pt` complet du primaire**, à un multiple de 5 000 pas, idéalement **40k** : ainsi S+10k = 50k, où le primaire sauve aussi son raw. Vérifier le SHA de la copie. Restaurer avec `restore_training` et `restore_rng`, avec les contrôles d'égalité de `resume.py:83-102`.
  - Cela donne les mêmes poids maîtres FP32, la même EMA, les mêmes moments Adam et le même RNG pour A et B.
  - Le choix n'est pas fait sur la VAL.
  - Le primaire devient un réplicat gratuit de A.
- **Si vous tenez à l'EMA15k** : poids raw = EMA15k BF16 convertie en FP32, Adam neuf identique (bêtas 0,9/0,99), warmup de 200 pas jusqu'à 3e-6, même seed. Inconvénients : pas de réplicat, biais de sélection sur la VAL, et positions STE perdues.

### 6.3 Les deux bras
- **A** : recette primaire inchangée.
- **B** : A plus le terme auxiliaire.

Paramètres communs :
- même seed de sampler (2026101217), donc mêmes batches au même step ;
- LR hérité constant de 3e-6 ;
- clip global à 1,0 ;
- EMA de demi-vie 500 ;
- flags cudnn déterministes dans les deux bras et pour toutes les évaluations.

Mise en œuvre de l'auxiliaire dans B :
- seulement sur les **pas impairs** (batches principaux off-policy), soit 5 000 pas auxiliaires sur 10 000 ;
- backward séparé, qui **accumule** dans les gradients du backward principal ;
- puis un seul clip et une seule étape d'optimisation ;
- RNG auxiliaire séparé : `SeedSequence([seed, step, 0xA0])`, que A ne consomme pas.

### 6.4 Terme auxiliaire exact
1. Étage 1 élève sur 25 fenêtres, sans gradient, avec `cached_weights`, sur un bruit neuf.
2. Construction de `inputs1` avec un nouveau bruit d'étage 2.
3. Le teacher (BF16, sans gradient) applique l'étage 2 sur ces 9 fenêtres, puis `core_crop` donne `core_t`.
4. L'élève applique l'étage 2 sur les 9 fenêtres, avec gradient, puis `core_crop` donne `core_s`.
5. `aux = λ_c·KL_q(core_s, core_t) + λ_f·flat_hp` :
   - KL en espace q, bande √20, τ = √2, minimum 16 pixels par classe ;
   - passe-haut du prototype, avec la puissance de la catégorie.

**Calibration** : 16 pas TRAIN au checkpoint source, sans mise à jour.
- `λ_c = 0,25 · médiane(‖∇main‖ / ‖∇côte‖)`, sur les pas où la côte est active ;
- `λ_f = 0,25 · médiane(‖∇main‖ / ‖∇hp‖)`, sur les scènes plates ;
- normes mesurées sur les poids maîtres, à travers le STE, avant clip.

Conséquence : l'auxiliaire vaut au plus environ la moitié du gradient principal sur un pas auxiliaire. Journaliser le taux de clip dans les deux bras.

### 6.5 Budget réaliste (estimations à confirmer au smoke)

| Poste | Estimation |
|---|---|
| Poids maîtres + gradients + Adam m,v + EMA (FP32, 253 688 037 paramètres) | ≈ 5,1 Go |
| Teacher BF16 | ≈ 0,5 Go |
| Graphe des poids par forward avec gradient (upcast FP32 sauvegardé par `normalize` + poids de conv BF16) | ≈ 1,5 à 2,5 Go |
| Activations (8 fenêtres + KD) | **à lire dans `peak_bytes` du `train.jsonl` du primaire** |
| Mémoire disponible sur la 4090 | ≈ 19,5 Go (24 Go − ~4,5 Go pour Windows) |

- Règle : si le pic mémoire du primaire est d'au plus ~17 Go, la recette tient sur la 4090.
- Le pic de l'étape auxiliaire (9 fenêtres avec gradient, sans KD, backward séparé) est à peu près égal au pic principal. En secours : `forward_batches(checkpointed=True, batch=3)` pour l'auxiliaire. Pour l'étape principale, la seule option est de placer l'EMA sur CPU, **dans les deux bras**.
- **Disque** : environ 25 Go (checkpoint complet ≈ 4,1 Go ×2 bras, plus les écritures atomiques temporaires, plus la copie source, plus 16 snapshots BF16). Prévoir au moins 40 Go libres, ou un autre volume.
- **Temps** : B coûte environ 1,6× A par pas (estimation en comptant les passes, non mesurée). Le débit réel se mesure au smoke.

### 6.6 Déroulé
1. **Smoke sur la 4090** :
   - restauration exacte ;
   - 3 pas de A, puis 3 pas de B avec λ=0 : pertes, normes de gradient et état RNG torch doivent être **identiques** ;
   - 3 pas de B calibré : valeurs finies, teacher sans gradient, pic mémoire et secondes par pas.
2. **Blocs entrelacés de 2 500 pas** : A puis B de S à S+2,5k, puis A puis B de S+2,5k à S+5k, et ainsi de suite jusqu'à S+10k. À chaque fin de bloc : checkpoint complet, plus EMA et raw en BF16. Ainsi, une paire complète existe toujours si on arrête la 4090.
3. **Aucun arrêt de qualité précoce**, ni règle de division par deux de C0. Seuls arrêtent le run : NaN, gradient sur le teacher, mémoire insuffisante, ou régression catastrophique de type watch. Ne jamais prolonger un bras seul. Prolongation à 16k pour les deux bras seulement si le panel TRAIN sépare A et B mais que la VAL reste ambiguë.

### 6.7 Mesures
- **Panel VAL** :
  - toutes les scènes VAL existantes des 4 catégories, en coarse4 et coarse2, plus les 3 montagnes ;
  - le bruit stocké plus 3 bruits fixes via `DecodedRegion.refresh_noise` (le teacher est recalculé, en forward seulement) ;
  - les deux LOD, EMA et raw ;
  - les mondes 101, 303 et 404 ne sont jamais utilisés.
- **Panel TRAIN** fixe : termes auxiliaires, désaccord LOD3, HF LOD3 et biais, pour A et B. Il sert à vérifier que B optimise bien ce qu'il vise.
- **Changements de codes NF4** pour A, B et le primaire.

### 6.8 Règle de décision
On calcule la différence appariée d = B − A par échantillon (scène × coarse × bruit), sur l'EMA, en moyenne sur S+7,5k et S+10k.

Critères :
- **E1** : |log ratio HF| au LOD3 sur les plaines (visé directement).
- **E2** : désaccord LOD3 près de la côte (plaine/mer + côte découpée), visé directement.
- **E3** (transfert, secondaire, sans présumer) : désaccord LOD0 et biais LOD0 de la plaine/mer.
- **E4** : biais par catégorie, à titre descriptif.

Un gain n'est accepté que si toutes ces conditions sont remplies :
- |moyenne de d| dépasse 2× l'écart-type du réplicat (A contre primaire à S+5k et S+10k, plus le raw à 50k si S = 40k), aux deux points d'évaluation ;
- le signe de d est cohérent sur au moins 70 % des échantillons ;
- les garde-fous tiennent :
  - corrélation TPI des montagnes ≥ A − 0,005 ;
  - rétention des vallées ≥ A − 0,02 ;
  - |rms TPI − A| ≤ 0,05 ;
  - HF LOD0 des plaines ≥ 0,9 × A ;
  - HF LOD3 ≥ 0,8.

Effet minimal à pré-enregistrer avant de vous montrer des planches (ce sont mes propositions) : ratio HF LOD3 de 2,0 vers ≤ 1,7, ou E2 en baisse d'au moins 2 points.

Ensuite, des planches A / B / teacher sur les mêmes scènes. Votre verdict visuel reste nécessaire. La taille du paquet ne change pas (même quantifieur, 149,65 Mo).

### 6.9 Comment lire le résultat

| Résultat | Interprétation | Suite |
|---|---|---|
| Panel TRAIN : B ≈ A | L'objectif ne peut pas s'exprimer à 3e-6 avec un STE NF4 | Ablation LR ou quantifieur, séparée |
| TRAIN : B < A ; E1/E2 améliorés ; E3 non | Le proxy marche sans transférer à la côte LOD0 | Terme basse fréquence / offset, ou côte LOD0 décodée |
| TRAIN : B < A ; VAL inchangée | Généralisation insuffisante ou VAL trop bruitée | — |
| Garde-fous violés | λ trop fort, ou fuite vers les montagnes | Réduire λ |

---

## 7. Limites de cette revue

- Je n'ai lu que des CSV coarse4, pas coarse2 ni « original », et pas le CSV de 15k.
- Je n'ai pas lu `train.jsonl` (normes de gradient, `peak_bytes`) ni les lignes du JSON de calibration.
- L'identité des poids des deux évaluations à 6k n'est pas vérifiée.
- L'origine exacte du biais NF4 n'est pas établie.
- Les estimations de dynamique NF4, de mémoire et de temps viennent de la lecture du code, pas de mesures.
- Je suppose, d'après les docstrings, que `reconstructed_height` reproduit le pipeline de production.
- Je n'ai vérifié aucune exactitude des cibles `DecodedRegion` pour le core 64 : c'est l'objet de la phase 0.1.