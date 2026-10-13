# Revue : grain NF4 en racine q, sonde INT8 et suite après la paire A/B

Revue en lecture seule : je n'ai rien lancé ni modifié, et je n'ai fait aucun accès réseau. Je n'ai lu ni les poids, ni `arrays.npz` (binaire), ni aucun secret.

---

## 0. Périmètre, statuts et limites de lecture

**Rapports lus**
- `REPORT.md`
- `distill/BASE_NF4_LEVEL_PRECISION_20261013.md`
- `distill/NF4_GRAIN_DIAGNOSTIC_20261013.md`
- `distill/NF4_SCALAR_PACKING.md`
- `distill/BASE_NF4_NEXT_STRATEGY_20261013.md`
- mon avis précédent `distill/REVIEW_OPUS_NF4_FINAL_BIAS_20261012.md`
- des extraits de `REVIEW_OPUS_NF4_PLATEAU_20261012.md`

**Code lu**
- Objectif et run de niveau : `native_level_objective.py`, `native_level_train.py`
- Précision et quantification : `native_scalar_precision.py`, `native_weight_quant.py`
- Pertes : `native_weight_quant_train.py` (`loss_terms`), `native_phase_loss.py`
- Chaîne : `native_fusion.py`, `native_decoded.py`
- Diagnostics et évaluation : `native_level_diagnostic.py`, `native_last_layer_probe.py`, `native_weight_quant_evaluate.py`, `terrain_shape.py`
- Amont : `world_pipeline.py:1242` (`T_INTER`), `unet_block.py:72,130`, `edm_unet.py:103,179`

**Données lues** (texte : JSON, CSV, JSONL), sous `/home/delete/data/distill/` :
- `eval/native-nf4-root-highpass-train-diagnostic.json` (D3)
- `eval/native-nf4-last-int8-train-probe-4090.json`
- `eval/native-nf4-level-train-floor-4090.json`
- `eval/native-nf4-level-gradient-calibration-5090.json`
- `eval/native-nf4-level-a-early-matched-comparison.json`
- `_base/native-weight-nf4levelstart-coarse4-070000.csv` et `…aearly-coarse4-072000.csv`
- `ckpt/native-nf4-level-70k-80k/b/train.jsonl` et `controller-status.json`
- `eval/native-nf4-scalar-audit-20261013.csv` (premières lignes)
- le manifeste du paquet prototype (tailles seulement)
- `eval/native-nf4-level-train-attribution-5090/report.json` (pentes de régression par canal)

**Non lu ou introuvable**
- `arrays.npz`, les poids, les résultats de B, la décomposition des mondes 202/404.
- **Le script qui a produit D3 n'est pas dans le dépôt.** Aucun fichier ne contient `height_highpass_loss_dc_derivative` ni `root_highpass_loss_dc_delta`. Le JSON porte les SHA de ses entrées, mais pas celui de son code. C'est une lacune de reproductibilité ; je me fie à la description et au JSON.

**Statuts utilisés**
- **[Constat]** : lu dans les données ou les rapports.
- **[Démontré-code]** : découle du code ou d'un calcul exact.
- **[Calcul]** : arithmétique sur des chiffres publiés ; les approximations sont signalées.
- **[Hypothèse]** : compatible avec les données, non prouvée.
- **[Test]** : expérience proposée pour trancher.

---

## 1. Verdict

1. **D3 précise l'explication du grain sans la renverser.**
   - Le grain est un excès de haute fréquence du canal C4, produit par l'étage 2 élève. Il existe en q et ne dépend pas du niveau.
   - Environ 66 à 85 % de l'énergie HF de l'élève est **orthogonale** à celle du teacher. La part corrélée est **atténuée** : pente de 0,75 à 0,84.
   - La cible même entrée (ST) ne change presque rien. Le choix de cible n'est donc pas le levier.
2. **L'échappatoire de niveau du passe-haut en h est démontrée numériquement**, avec le signe que prévoit la dérivée :
   - les découpages dominés par la terre sont abaissés ;
   - ceux dominés par la mer sont relevés ;
   - tout est donc tiré vers le niveau 0, c'est-à-dire vers la côte.
3. **Je corrige ma recommandation R2 précédente.** La pénalité unilatérale de déficit d'énergie ne protège pas du lissage quand R ≈ 1,9 : elle laisse diviser l'amplitude corrélée par R avant de s'activer. Il faut un **plancher sur la pente corrélée**.
4. **Cinq mécanismes restent ouverts** : objectif aveugle, annulation imparfaite du bruit injecté, limite de représentation NF4 en blocs de 512, plancher d'arrondi BF16, non-conservation de la norme par le NF4. Ils se départagent **sans training** (G0, §6).
5. **Sonde INT8.** Le rejet est justifié comme correctif après entraînement. Il ne dit rien d'un QAT mixte entraîné.
   - Le JSON montre aussi un fait que le résumé omet : le remplacement **augmente le relief LOD0 des plaines** (TPI, rétention des vallées), surtout pour la plaine basse/mer.
   - Le même remplacement double ou triple le désaccord côtier LOD0 de la plaine-mer et de la côte.
   - C'est un indice de co-adaptation et de sensibilité d'amplitude, pas d'innocuité.
6. **Un QAT à 150 Mo est plausible à tester, pas démontré.**
   - 27,4 Mo du paquet sont des `emb_linear` en BF16.
   - Les passer en INT8 financerait des blocs NF4 huit fois plus fins (64), à taille quasi constante.
   - Viser 50 à 100 Mo par simple réduction de bits n'est soutenu par aucune donnée actuelle.
7. **Risques de lecture de la paire A/B.**
   - Les 36 `emb_gain` restent quantifiés en BF16 dans le forward.
   - Le terme LF en montagne est dominé par sa partie C0–C3.
   - Le « plancher » montagne de 26 m est surtout un **biais systématique** BF16/FP32, pas un bruit.
   - Les rappels de vallées LOD3 portent sur 18 à 125 pixels.
   - Le journal de B ne montre que les pas auxiliaires de montagne.
   - L'évaluateur n'est pas déterministe.
8. **Prochaine expérience minimale : G0, « budget du grain ».** D'abord la partie CPU sur les données existantes (G0a), puis des forwards sur la 4090 sans aucune mise à jour (G0b). La paire en cours n'est pas touchée.

---

## 2. Question 1 : le diagnostic change-t-il l'explication du grain et les priorités ?

### 2.1 Ce que D3 établit

**Données** : `eval/native-nf4-root-highpass-train-diagnostic.json`.

**Mesures** [Constat] : médianes par catégorie du rapport RMS HF élève/teacher (R) et de leur corrélation (c), en q, passe-haut σ = 1 latent, core central 32×32.

**Colonnes dérivées** [Calcul] :
- pente corrélée : a = c·R ;
- RMS orthogonal relatif au teacher : R·√(1−c²) ;
- part orthogonale de l'énergie HF élève : 1 − c² ;
- énergie à l'optimum « gain seul » d'une MSE appariée : c².

Ces valeurs sont approximatives, car les médianes de R et de c sont prises séparément.

| Catégorie | Cible | R | c | a = cR | RMS orth./teacher | Part orth. | Énergie optimum c² |
|---|---|---:|---:|---:|---:|---:|---:|
| Désert | TT | 1,895 | 0,359 | 0,68 | 1,77 | 0,87 | 0,13 |
| | ST | 1,919 | 0,391 | 0,75 | 1,77 | 0,85 | 0,15 |
| Tempérée | TT | 1,933 | 0,391 | 0,76 | 1,78 | 0,85 | 0,15 |
| | ST | 1,965 | 0,429 | 0,84 | 1,78 | 0,82 | 0,18 |
| Côte | TT | 1,646 | 0,421 | 0,69 | 1,49 | 0,82 | 0,18 |
| | ST | 1,666 | 0,475 | 0,79 | 1,47 | 0,77 | 0,23 |
| Plaine basse/mer | TT | 1,552 | 0,421 | 0,65 | 1,41 | 0,82 | 0,18 |
| | ST | 1,586 | 0,476 | 0,76 | 1,40 | 0,77 | 0,23 |
| Montagne | TT | 1,298 | 0,538 | 0,70 | 1,09 | 0,71 | 0,29 |
| | ST | 1,320 | 0,586 | 0,77 | 1,07 | 0,66 | 0,34 |

**Lecture**
- **[Calcul]** L'élève reproduit environ 75 à 84 % du détail HF du teacher (pente a, cible ST). Il y ajoute une composante orthogonale de 1,07 à 1,78 fois le RMS HF du teacher.
- **[Constat]** Passer de TT à ST augmente c de 0,03 à 0,05, et R ne change pas. La décorrélation vient donc de l'étage 2 lui-même, pas des écarts d'état de l'étage 1. C'est cohérent avec le niveau de bruit de l'étage 2 : `T_INTER = arctan(0,35/0,5)` (`world_pipeline.py:1242`), soit t₂ ≈ 0,609 rad. À ce niveau, le bruit injecté masque la HF de l'état d'entrée.
- **[Constat]** La variabilité intra-catégorie est forte. En montagne, R va de 1,09 (c = 0,77, monde 10060, coarse4) à 1,63 (c = 0,38, monde 10366, original). Le défaut dépend du terrain local, pas seulement de la catégorie.
- **[Calcul]** Même à corrélation inchangée, ramener l'amplitude HF à celle du teacher diviserait l'énergie orthogonale par R² : de 1,7 (montagne) à 3,9 (tempérée). Pour le rendu, l'excès d'amplitude est donc un problème au moins aussi important que la décorrélation. C'est utile, puisque l'utilisateur accepte des erreurs numériques si le rendu convient.

### 2.2 Le décalage constant démontre l'échappatoire de niveau, avec son signe

**[Démontré-code + numérique]** Un passe-haut est linéaire et son noyau a une somme de 1. Un décalage constant de q ne change donc pas la perte passe-haut en q : D3 mesure au plus 5,83·10⁻¹⁶.

**[Constat]** La dérivée de la perte en h selon le décalage vaut :
- désert et tempérée : +1,67 à +5,18 ;
- montagnes : +25,8 à +54,0 ;
- plaine basse/mer : −0,65 à −0,05 ;
- côte : −5,48 à +0,90, selon la part de terre.

**[Démontré-code]** C'est le signe que prévoit la dérivée de mon avis précédent (§3.1).
- Sur la terre (q > 0), HP(2|q|) ≈ 2r : un excès d'énergie HF pousse q vers le bas.
- Sur la mer (q < 0), HP(2|q|) ≈ −2r : le signe s'inverse, et q est poussé vers le haut.
- Une perte passe-haut en h tire donc |q| vers 0 des deux côtés. Elle aplatit vers le niveau de la mer et déplace les côtes.

Cela explique aussi le doublement du désaccord côtier dans l'ancien bras B à 50k. Une perte en q n'a pas ce défaut, et au trait de côte son poids ne s'annule pas, contrairement à 2|q|. **Le signe de la côte se décide en q.**

### 2.3 Ce qui change dans mon explication : cinq hypothèses, chacune testable

Avant D3, mon explication principale était : un objectif presque aveugle à la bande HF du C4 de l'étage 2. Elle reste plausible, mais elle n'est plus la seule.

**H-obj : objectif mal pondéré** [Hypothèse, calcul indicatif]
- `loss_terms` (`native_weight_quant_train.py:26-40`) normalise l'erreur de vitesse par la variance de chaque canal. Celle de C4 vaut 1,06 (`eval/native-decoded-probe/channel-statistics.json`), et elle est dominée par la basse fréquence de l'altitude.
- À même entrée, l'erreur de C4 en q vaut ΔQ = 38,6·sin t₂·Δv₄ ≈ 22,1·Δv₄.
- Dans un lot de 8 fenêtres (4 par étage), la partie « C4 de l'étage 2 » ne représente que 1/10 des éléments de l'opérateur.
- La contribution du grain orthogonal à l'opérateur vaut donc environ ρ²·E_t / 5 200, où E_t est l'énergie HF du teacher en q² et ρ² l'énergie orthogonale relative.
- Avec E_t entre 0,05² et 0,3² q² (valeur **non mesurée** : G0a la fournit) et ρ² entre 1,1 et 3,1, on obtient 0,1 à 1 % de l'opérateur. Les journaux de B donnent un opérateur de 0,0009 à 0,0023.
- Le grain pèserait donc environ 0,1 à 2 % du signal d'entraînement. C'est compatible avec un objectif qui ne le corrige jamais.

**H-cancel : annulation imparfaite du bruit injecté à t₂** [Hypothèse, mécanisme démontré par le code, ampleur non mesurée]
- Pour l'étage 2, x̂ = cos t·x_in + sin t·v (`native_fusion.py:22-28`, `native_phase_loss.py:19-23`).
- En haute fréquence, x_in est dominé par le bruit injecté. Le teacher doit produire v_HF ≈ −(cos t / sin t)·x_in,HF : deux grands termes qui s'annulent presque.
- Une erreur relative ρ de l'élève sur v_HF laisse environ 0,45ρ en latent normalisé, soit environ 17ρ en q.
- Une erreur de 1 % sur la vitesse HF, banale avec des poids NF4, suffit donc à créer un grain de l'ordre de la HF du teacher en plaine.
- **Prédictions** :
  1. un résidu HP(q_SS − q_ST) corrélé au bruit injecté de l'étage 2, une fois fusionné ;
  2. une amplitude orthogonale **absolue** en q à peu près constante d'une catégorie à l'autre. Le ratio serait alors plus grand là où la HF du teacher est faible : plaines 1,9, montagnes 1,3. C'est ce qu'on observe, sans le prouver.

**H-rep : limite de représentation NF4 en blocs de 512** [Hypothèse]
- `encode` (`native_weight_quant.py:30-45`) utilise un maximum absolu par bloc de 512 valeurs consécutives, soit environ un tiers de filtre de sortie pour une convolution 192×192×3×3.
- Les valeurs aberrantes fixent l'échelle, et beaucoup de poids tombent dans les niveaux centraux ou à zéro.
- La précision HF exigée par H-cancel pourrait ne pas être atteignable avec ce quantifieur, quel que soit l'objectif.

**H-BF16 : plancher d'arrondi de la sortie de fenêtre** [Hypothèse ; ordre de grandeur calculé, ampleur réelle non mesurée]
- `window_outputs` calcule x̂ **en BF16** avant la fusion (`native_fusion.py:24-28`).
- Pour C4 dans [1, 2), soit h entre environ 52 m et 2 100 m, l'écart entre valeurs BF16 vaut 2⁻⁷ en latent normalisé, soit **0,30 en q**. Il vaut 0,15 sous 52 m et 0,60 au-dessus d'environ 2 100 m.
- En hauteur LOD3, cet écart vaut 2|q|·Δq : environ 10 m à 300 m d'altitude, 27 m à 2 000 m, 0,4 m à 2 m.
- Or la HF LOD3 de référence du teacher vaut 2,39 m (désert), 3,97 m (tempérée), 0,20 m (plaine basse) et 17 à 26 m (montagnes) (CSV 70k, colonne `high_frequency_reference_rms_m`).
- La partie σ = 1 de l'erreur d'arrondi dépend de la structure du champ : un champ lisse donne des marches plutôt qu'un bruit blanc, et la fusion des fenêtres peut moyenner les arrondis.
- Je ne peux donc pas conclure. Mais l'ordre de grandeur est le même que la HF du teacher en plaine. Si c'est le cas :
  - la corrélation c est plafonnée en plaine ;
  - la cible de la bande σ = 1 contient du bruit irréductible ;
  - les critères doivent être rapportés à ce plancher.
- Le teacher et l'élève passent tous deux par cette expression. H-BF16 n'explique donc pas l'excès R ≈ 1,9 ; il en fixe le plancher.

**H-norm : le NF4 ne conserve pas la norme des filtres** [Hypothèse, née de la sonde INT8, §4.4]
- Dans `weight_conv` (`native_weight_quant.py:77-78`), le noyau effectif est normalisé, puis arrondi en NF4, puis utilisé tel quel. La norme d'un filtre arrondi n'est plus exactement celle voulue par l'architecture à magnitude préservée.
- Cela crée des erreurs de gain par couche, compensées par le QAT. C'est un levier de niveau et d'amplitude plus que de grain.

**Correspondance hypothèses → tests**
- H-cancel et H-BF16 : se départagent sur CPU, avec les données existantes (G0a).
- H-rep et H-norm : forwards de quantification post-entraînement (PTQ) du teacher (G0b).
- H-obj : test de premier ordre sans mise à jour, puis sonde de capacité (§4.6).

### 2.4 Ce qui ne change pas

- **[Constat]** La source est l'étage 2 et le canal C4.
  - D3 donne SS/ST ≈ SS/TT.
  - Dans `report.json` de l'attribution, la pente de régression C4 de TS (élève en étage 2 sur l'état du teacher) suit celle de SS : 0,76 à 1,17 contre 0,71 à 1,16 en plaine, sur 32×32. En montagne, les deux restent dans 0,96–1,02.
  - L'étage 2 élève dérègle donc aussi le **contraste à moyenne échelle** du C4 en plaine, pas seulement la bande σ = 1.
  - Réserve : en plaine, la variance du C4 est faible. Si le C4 du teacher contient de l'arrondi (H-BF16), la pente est biaisée vers 0 par dilution.
- La correction du niveau avant le grain reste raisonnable. La paire en cours ne doit pas être touchée.

### 2.5 Un second défaut, distinct : les incisions LOD0 de la côte et de la plaine basse

**[Constat]** VAL, coarse4, `_base/*070000.csv` et `*072000.csv` :

| Cas | 70k | 72k (A) |
|---|---:|---:|
| Rappel des vallées LOD0 (σ = 16), côte | 0,477 | 0,472 |
| Rappel des vallées LOD0 (σ = 16), plaine basse | 0,450 | 0,391 |
| Amplitude TPI LOD0, plaine basse | 0,82 | 0,79 |
| Rappel des vallées LOD0, désert | 0,81 | 0,80 |
| Rappel des vallées LOD0, tempérée | 0,91 | 0,91 |
| Rappel des vallées LOD0, montagnes | 0,89 à 0,94 | 0,90 à 0,95 |

Le désert, la tempérée et les montagnes se situent donc entre 0,80 et 0,95.

- **[Constat]** Sonde INT8 (TRAIN) : la plaine-mer 10020 en NF4 a une amplitude TPI LOD0 de 0,44 à 0,53 et une rétention des vallées de 0,00 à 0,09.
- **[Constat]** Ce sont précisément deux des cas que l'utilisateur veut conserver : « côte humide tortueuse » et « plaine très basse/mer ».
- **[Démontré-code]** Le LOD0 dépend de C0–C3 à travers le decoder ; la HF du C4 est filtrée par `laplacian_denoise(σ=5)`. C'est donc un autre canal que celui du grain LOD3.
- **[Constat]** Pentes C0–C3 sur 32×32 (`report.json`) : SS vaut 0,94 à 1,00, ST ≈ SS, et TS vaut 0,99 à 1,00. La légère perte de contraste C0–C3 (1 à 6 %, surtout sur C0) vient donc de l'**étage 1** de l'élève.
- **[Hypothèse]** L'aplatissement LOD0 de la plaine basse et de la côte vient d'une atténuation de C0–C3 produite à l'étage 1. Un objectif HF limité au C4 de l'étage 2 ne le corrigera pas. Le test est en G0a-5.

### 2.6 Priorités révisées

1. **Ne pas toucher à la paire A/B.**
   - Décider sur le niveau, par catégorie, en unités de plancher, mondes 202/404 compris.
   - Ajouter des métriques HF en q, descriptives : B ne devrait pas changer la HF, et toute variation serait une fuite.
2. **G0a** (CPU, données existantes) : dès maintenant.
3. **G0b** (forwards 4090, sans training) : entre deux évaluations de la paire.
4. **Après A/B et G0** :
   - G1, une paire « contrôle / objectif HF » (§3) ;
   - en parallèle, la décision sur le quantifieur (§4.5), seulement si G0b la justifie.
5. **Incisions LOD0 de la côte et de la plaine basse** : critère co-principal, avec un objectif propre (C0–C3, probablement l'étage 1), testé séparément de l'objectif du grain C4.

---

## 3. Question 2 : objectif HF concret après la décision A/B

### 3.1 Où l'appliquer, et « en q ou en vitesse »

**[Démontré-code]** À entrée identique, sur une fenêtre de l'étage 2 : Δx̂ = sin t·Δv. La perte en vitesse et la perte en q sont donc **la même fonctionnelle**, à la constante 38,6·sin t₂ ≈ 22,1 près.

Le vrai choix est ailleurs :
- **une perte par fenêtre** sur les fenêtres de l'étage 2 du lot KD (labels ≈ 0,609, `PHASE_LABELS`, `native_phase_loss.py:11`) ;
- **plutôt qu'une perte sur le cœur fusionné**.

La perte par fenêtre a trois avantages :
- sa cible teacher à même entrée est **déjà calculée** dans `step_loss` (`native_weight_quant_train.py:43-51`) : aucun forward supplémentaire ;
- l'entrée est l'état élève dans les lots on-policy (cible ST) et l'état stocké du teacher dans les lots off-policy (cible TS) ;
- c'est exactement le cadre qu'isole l'attribution.

Je l'exprime en unités q pour l'interprétation, et je calcule x̂ en FP32 comme `x0_from_velocity`.

**Ce que j'écarte**
- **h** : échappatoire de niveau démontrée (§2.2).
- **Le poids jacobien 2|m_t|** : le défaut est relatif (R = 1,3 à 1,9 partout). En mètres, la plaine basse (J ≈ 2) deviendrait invisible et les montagnes (J ≈ 90) domineraient. La normalisation par fenêtre, définie plus bas, rend l'objectif indépendant de l'altitude.
- **TT pour la HF** : D3 montre qu'une petite part de la HF de TT est imprévisible depuis l'entrée de l'élève. Une MSE appariée la transformerait en pression de lissage, sans rien gagner.
- **La cible TT reste en revanche la bonne pour le niveau** (terme LF, si B est retenu) et pour l'évaluation.

### 3.2 Formules

**Notation, pour chaque fenêtre w de l'étage 2**
- x : l'entrée BF16, commune à l'élève et au teacher ;
- t = 0,609375, la valeur stockée du label ;
- v_t = sg(teacher(x)), cible détachée.

```text
X_s = cos t·x + sin t·v_s          X_t = cos t·x + sin t·v_t     (FP32)
Q_s = 38,6·X_s[4]                  Q_t = 38,6·X_t[4]             (l'offset −31,4 est sans effet ici)
ΔQ  = Q_s − Q_t = 22,1·(v_s − v_t)[4]
```

**Bandes** : différences de gaussiennes, avec réflexion aux bords (`blur`, `native_decoded.py:145-150`).

```text
B1 = I − G1    (σ = 1 latent, environ 240 m : la bande du grain)
B2 = G1 − G2   (environ 240–480 m)
B3 = G2 − G4   (environ 480–960 m : échelle des vallées LOD3 ; protection seulement)
```

**Poids par pixel**

```text
ω_w(p) = π(p) · 1[ Q_t(p) > −δ_c ]
```

- π est la fenêtre de fusion `linear_weight_window(64)`, normalisée à une moyenne de 1. Elle reflète la contribution réelle du pixel au champ fusionné et efface les bords réfléchis.
- On garde la terre et une bande côtière en q (δ_c ≈ 2, à fixer dans le contrat). Le grain y décide du signe de la côte.

**Statistiques par fenêtre et par bande** (toutes les quantités teacher sont détachées)

```text
E^t_{b,w} = Σ ω (B_b Q_t)²      E^s_{b,w} = Σ ω (B_b Q_s)²
C_{b,w}   = Σ ω (B_b Q_s)(B_b Q_t)
D_{b,w}   = Σ ω (B_b ΔQ)²
a_{b,w}   = C_{b,w} / (E^t_{b,w} + ε_b)          (pente corrélée)
```

- ε_b est le plancher de la bande b : énergie de la différence teacher BF16/FP32, plus l'arrondi de la sortie, mesurés en G0b-1. C'est une constante TRAIN figée.

**Termes de perte**

```text
L_pos   = Σ_{b∈{1,2}} moy_w [ D_{b,w} / (E^t_{b,w} + ε_b) ]          erreur appariée, relative
L_slope = Σ_{b∈{1,2,3}} moy_w [ (α_{b,k(w)} − a_{b,w})₊² ]            plancher de pente corrélée
L_HF    = L_pos + μ·L_slope
L       = L_KD + κ·L_HF
```

- Le gradient de κ·L_HF est mis à zéro sur les 37 scalaires, comme `protect_scalar_gradients` (`native_level_objective.py:40-47`).
- On garde le plafond global de 0,5 de `combine_gradients`.
- k(w) est un **seau d'énergie teacher** : quantiles de E^t_{1,w} calculés sur TRAIN. C'est calculable pour n'importe quelle fenêtre, y compris les scènes générales qui n'ont pas d'étiquette de catégorie.

### 3.3 Protection contre le lissage de Wiener

**[Démontré-code]** Dans l'espace de sortie, le gradient de L_pos vaut 2(B s − B t) = 2[(a − 1)·B t + e]. Une descente augmente donc a et réduit e : il n'y a **pas** de raccourci dans cet espace.

Le lissage naît dans l'espace des **paramètres** : si l'élève ne sait pas réduire e directement, le plus simple est un gain global g sur sa bande. Le long de cette direction, en unités de E_t :

```text
L_pos(g) = g²R² − 2ga + 1        optimum g* = a / R²        énergie finale = c²
```

C'est 15 à 34 % de l'énergie du teacher selon la catégorie (cible ST) : un objectif de lissage.

**[Démontré-code] La pénalité de déficit que je proposais est insuffisante.** (1 − E_s/E_t)₊² ne s'active qu'à g = 1/R, quand a est déjà tombé à a/R : 0,75 → 0,39 dans le désert, 0,77 → 0,59 en montagne. L'énergie est alors celle du teacher, mais 85 % de cette énergie est du bruit. **Je retire cette recommandation.**

**Le plancher de pente bloque le raccourci.** Avec α = a_départ, toute baisse de g est pénalisée dès le départ, alors que réduire e à a constant ne coûte rien. Le long du gain seul, l'équilibre g = 1 − δ exige :

```text
μ ≥ ((1−δ)·R² − a) / (a²·δ)
```

| Cas (cible ST) | δ = 2 % | δ = 5 % |
|---|---:|---:|
| Désert (R² = 3,68 ; a = 0,75) | 254 | 98 |
| Montagne (R² = 1,74 ; a = 0,77) | 78 | 30 |

C'est une borne de pire cas. La valeur retenue vient de la calibration au premier ordre (§3.6, C6). Comme la pénalité est une charnière, elle est nulle tant que a ≥ α : elle ne déforme pas le reste de l'objectif.

### 3.4 Énergie inventée

- **[Démontré-code]** L_pos pénalise directement l'énergie orthogonale.
- L_slope ne récompense que le contenu **corrélé** au teacher : a ne peut augmenter que par ajout du motif B t.
- **Aucun terme ne récompense une énergie non corrélée.**
- C'est pourquoi j'écarte la perte symétrique `log(E_s/E_t)²` de `band_terms` (`native_phase_loss.py:26-50`). Là où l'élève est sous l'énergie du teacher, elle pousse l'énergie vers le haut quelle que soit la corrélation.
- **Garde-fou mesuré** : ρ²_b (énergie orthogonale relative) ne doit augmenter dans aucune bande ni aucun seau.

### 3.5 Préserver les incisions

1. **C4 seulement.** Aucun gradient direct sur C0–C3, qui portent le détail LOD0.
2. **Planchers de pente sur B2 et B3**, aux échelles des vallées LOD3.
3. **Niveau neutre par construction.** Le passe-haut en q ne voit pas le niveau (démontré) ; le niveau reste géré par le terme LF si B est retenu.
4. **Les 37 scalaires sont exclus.** Ce sont les leviers globaux d'amplitude.
5. **Garde-fous mesurés au LOD0 et au LOD3** (§3.7). Si G0a-5 confirme que l'étage 1 atténue C0–C3, il faudra une seconde paire avec un plancher de pente C0–C3 (α → 1, pour restaurer). Ce sujet n'a rien à faire dans la première paire du grain.

### 3.6 Protocole de calibration, sur TRAIN uniquement

**C0. Contrat** figé avant tout résultat :
- SHA du code (nouveaux fichiers, par exemple `distill/native_grain_objective.py`) et du checkpoint de départ, choisi après A/B ;
- graines, ensembles, bandes, δ_c et seaux ;
- assertion `split == train` et contrôle `HOLDOUT_SEEDS` ;
- ensemble de calibration : les 32 régions du diagnostic de niveau, plus 64 lots KD tirés de l'échantillonneur TRAIN, sur graines dédiées.

**C1. Planchers ε_b** (G0b-1) :
- étage 2 du teacher en FP32 contre BF16, sur des entrées identiques ;
- `window_outputs` en FP32 contre BF16, à vitesse identique ;
- répétabilité : même forward deux fois avec les options déterministes, résultat attendu nul.

**C2. Statistiques de départ**, par bande et par seau : R, c, a, ρ².
- Deux versions : par fenêtre (lots KD) et sur le cœur fusionné (SS contre ST, 32 régions).
- Corrélation de rang entre les deux versions de ρ² sur les régions. Je propose ≥ 0,7 : c'est la validité du proxy par fenêtre.

**C3.** α_{b,k} = a_{b,k}(départ), figé. On protège sans chercher à restaurer.

**C4. Moments Adam** : 500 pas à LR nul avec l'objectif complet, par bras, et contrôle d'identité des poids (déjà fait pour la paire de niveau).

**C5. κ** : r̃ = max(rapport brut, rapport préconditionné) de ‖∇L_HF‖ / ‖∇L_KD‖ par lot ; κ = 0,25 / médiane(r̃) ; plafond 0,5.

**C6. μ par alignement au premier ordre**, sans mise à jour :
- Pour μ ∈ {0, 10, 30, 100, 300}, on calcule la direction préconditionnée d = −P(∇L_KD + κ∇L_HF) sur l'ensemble de calibration.
- On calcule les variations prédites par produits jacobien-vecteur : Δa_{b,k}, ΔE^orth_{b,k}, Δbiais_LF_k, ΔL_KD.
- On retient le plus petit μ qui vérifie :
  - Δa ≥ 0 dans au moins 90 % des lots, pour chaque bande ;
  - ΔE^orth_1 < 0 dans au moins 75 % des lots ;
  - |Δbiais| ≤ 0,1·τ_k.
- **Si aucun μ n'y parvient**, même dans l'espace continu du STE, aucune direction ne retire le bruit orthogonal sans réduire la part corrélée. Il faut alors s'arrêter avant tout training et passer à la piste du quantifieur.

**C7. Smoke** : avec λ = 0, le bras HF doit être identique au contrôle au bit près.

### 3.7 Critères de décision pour la paire G1, à pré-enregistrer

**Dispositif**
- Bras C (recette retenue après A/B) contre bras H (C + κ·L_HF).
- Même départ, mêmes tirages, blocs entrelacés, même GPU.
- Décision sur l'EMA, à deux points fixés à l'avance.

**Panel**
- 7 captures de progression, 12 lieux critiques, 6 montagnes réservées ;
- **mondes 202 et 404** ;
- coarse4 et coarse2 ;
- bruit stocké plus 3 bruits.
- Les 32 régions TRAIN servent au suivi, pas à la décision.

Les seuils ci-dessous sont des **propositions à figer avant le premier résultat**, pas des prédictions.

**Principal (grain, q, même emprise)**
- ρ²₁ de H inférieur d'au moins 30 % à celui de C ;
- R₁ réduit dans toutes les catégories ;
- intervalle de confiance par bootstrap apparié (régions × bruits) qui exclut 0 dans au moins 4 catégories sur 5, aux deux points.

**Anti-lissage**
- a_b(H) ≥ a_b(C) − 0,02 pour b = 1, 2, 3, dans chaque catégorie ;
- c₁(H) ≥ c₁(C) ;
- R₂ et R₃ ≥ 0,95 × C.

**Énergie inventée**
- ρ²_b(H) ≤ ρ²_b(C) pour b = 2 et 3, et ρ²₁ en baisse.

**Incisions LOD3 (q)**
- Corrélation TPI à σ = 2 et 4 latents ≥ C − 0,01, sur un découpage de 48×48 plutôt que 32×32 (§5.3).
- **Rapport de profondeur des vallées**, une métrique continue :

```text
Σ_{p∈V_t}(−TPI_s) / Σ_{p∈V_t}(−TPI_t)     ≥ C − 0,02
```

**Incisions LOD0**
- Amplitude TPI à σ = 2..16 ≥ C − 0,02 ;
- même rapport de profondeur ≥ C − 0,02 ;
- côte et plaine basse rapportées séparément, nombre de pixels inclus ;
- désaccord côtier LOD3 et LOD0 ≤ 1,1 × C.

**Niveau** : |biais(H) − biais(C)| ≤ 0,5·τ_k par catégorie.

**Opérateur** : opérateur KD ≤ 1,05 × C ; `channel0_mse` de l'étage 1 ≤ 1,10 × C.

**Mondes 202, 404 et montagnes réservées** : aucune régression supérieure au plancher.

**Raccords** : batch 1 canonique exactement nul.

**Enfin** : planches teacher / C / H, mondes difficiles compris, et verdict visuel du propriétaire.

| Résultat | Lecture | Suite |
|---|---|---|
| Critère principal et garde-fous tenus | Objectif mal pondéré, corrigé | Candidat ; graines neuves, une seule fois |
| ρ²₁ inchangé, a inchangé, r̃ conforme | Représentation ou LR trop faible pour déplacer les codes NF4 | Sonde de capacité (§4.6), piste du quantifieur |
| R₁ en baisse mais a en baisse | Fuite de Wiener | Augmenter μ via C6 |
| Grain amélioré, LOD0 dégradé | Fuite vers C0–C3 par les paramètres partagés | Plancher de pente C0–C3 |
| Niveau déplacé | Couplage par les paramètres partagés | Lire le journal par tenseur ; garder le terme LF |

**Coût structurel** [Démontré-code] : aucun forward réseau supplémentaire, seulement des flous sur 4 fenêtres 64×64 par lot. Je n'avance aucune durée : le smoke la mesurera.

---

## 4. Question 3 : le remplacement INT8, 150 Mo, représentation ou objectif

### 4.1 Faits

**Source** : `eval/native-nf4-last-int8-train-probe-4090.json`, maîtres EMA70k, TRAIN, coarse4, une scène et un bruit par cas.

| Cas | HF q (R) NF4 → INT8 | corr. q | biais LOD0 (m) | TPI LOD0 σ2 / σ16 | Rétention vallées LOD0 σ2 / σ16 | Désaccord côtier LOD0 |
|---|---|---|---|---|---|---|
| Désert 10047 | 1,662 → 1,743 | 0,412 → 0,360 | +0,70 → +0,99 | 0,794 → 0,815 / 0,774 → 0,797 | 0,781 → 0,802 / 0,465 → 0,478 | 0 → 0 |
| Tempérée 10041 | 1,922 → 2,005 | 0,356 → 0,346 | −0,65 → −0,19 | 0,790 → 0,815 / 0,792 → 0,813 | 0,667 → 0,707 / 0,509 → 0,556 | 0 → 0 |
| Côte 10032 | 1,782 → 1,962 | 0,424 → 0,390 | +0,15 → +0,26 | 0,959 → 0,991 / 1,094 → 1,134 | 0,887 → 0,905 / 0,889 → 0,893 | 0,020 → 0,037 |
| Plaine-mer 10020 | 1,603 → 1,726 | 0,442 → 0,390 | −0,10 → +0,16 | **0,444 → 0,822 / 0,531 → 1,179** | **0,093 → 0,634 / 0,000 → 0,493** | **0,045 → 0,128** |
| Montagne 10042 | 1,340 → 1,341 | 0,488 → 0,465 | −6,0 → +17,3 | 0,966 → 0,977 / 0,967 → 0,979 | 0,937 → 0,943 / 0,947 → 0,961 | 0 |
| Montagne 10060 | 1,101 → 1,111 | 0,765 → 0,739 | −5,9 → +33,3 | 0,985 → 0,998 / 1,007 → 1,023 | 0,953 → 0,961 / 0,978 → 0,984 | 0 |
| Montagne 10066 | 1,372 → 1,371 | 0,530 → 0,490 | −19,9 → +9,3 | 0,967 → 0,977 / 0,964 → 0,974 | 0,935 → 0,943 / 0,954 → 0,961 | 0 |

**Portée de la sonde** [Démontré-code], `native_last_layer_probe.py:70-78` : une seule convolution, `dec.512x512_block3.conv_res1` (192×192×3×3, la dernière du decoder à pleine résolution). Elle est encodée en INT8 avec le même bloc de 512, sans aucun ajustement.

### 4.2 Ce qu'on peut conclure

1. **[Constat]** Comme correctif immédiat, la substitution aggrave :
   - le grain LOD3 en q dans les quatre plaines et côtes (+5 à +10 %, corrélation en baisse) ;
   - le niveau en montagne (+23 à +39 m) ;
   - le désaccord côtier LOD0 (×1,9 pour la côte, ×2,9 pour la plaine-mer).

   Le rejet est fondé.
2. **[Démontré-code]** Les maîtres ont été entraînés **à travers** l'arrondi NF4 de cette couche (STE, `native_weight_quant.py:48-50`). Le reste du réseau, `out_gain` compris, s'est adapté à NF4(w), pas à w. INT8(w) ≈ w est donc une **perturbation** de taille ‖NF4(w) − w‖, pas une correction.
3. **[Constat]** Le niveau est un mode collectif fragile : une seule couche porte 23 à 39 m en montagne. Toute modification du quantifieur (taille de bloc, précision mixte) doit être jugée **après réentraînement**, avec le terme de niveau et le gain FP32 dès le départ.
4. **[Constat, absent du résumé]** Le relief LOD0 augmente **partout** : +1 à +4 % d'amplitude TPI dans le désert, la tempérée et les montagnes, et un saut pour la plaine-mer. L'arrondi NF4 de cette couche agit donc comme un **atténuateur d'amplitude**, partiellement compensé. Le relief LOD0 des plaines est sensible à la précision des couches à pleine résolution.

### 4.3 Ce qu'on ne peut pas conclure

- Qu'un QAT mixte entraîné, avec cette couche ou d'autres en INT8, échouerait.
- Que cette couche, ou le NF4 en général, est innocente du grain.
- Que le gain LOD0 de la plaine-mer est réel. Une seule scène, un seul bruit, un niveau déplacé de +0,26 m et une côte qui bouge (×2,9) : sur une plaine très basse, un déplacement de niveau suffit à changer le TPI autour du rivage.
- Quoi que ce soit pour VAL, pour les mondes 202/404 ou pour coarse2.

### 4.4 Mécanisme probable de l'effet d'amplitude

**[Hypothèse H-norm]** `weight_conv` normalise le noyau, puis l'arrondit en NF4, puis s'en sert sans renormaliser (`native_weight_quant.py:77-78` ; avertissement dans `export_packed`, l. 149-150). La norme par filtre de sortie dérive donc de celle que suppose l'architecture à magnitude préservée :
- l'erreur d'arrondi ajoute de la variance ;
- les petits poids passent à zéro.

L'INT8, qui retire environ 99 % de l'énergie d'erreur de la couche, retrouve la norme. D'où l'amplitude en hausse partout, le niveau en hausse en montagne, et un grain LOD3 plus fort (l'amplitude du bruit orthogonal monte aussi).

**[Test, G0b-2]**
- PTQ du teacher en NF4 b512, avec et sans **renormalisation de chaque filtre déquantifié**.
- Mesurer le biais de niveau, la pente C4 et ρ²₁.
- Une renormalisation au chargement ne coûte **aucun octet** : la norme cible est connue. En revanche, elle change le contrat du runtime et impose un réentraînement. C'est une piste de quantifieur pour la suite, pas un correctif du 70k.

### 4.5 150 Mo : plausible à tester, pas démontré

**[Calcul]** Budget, d'après le manifeste du paquet prototype :
- 36 tenseurs `emb_linear` en BF16 : **27 426 816 octets**, soit 18,3 % des 149,6 Mo.
- Autres tenseurs BF16 : environ 0,42 Mo.
- NF4 et échelles FP32 : environ 121,8 Mo, pour environ 239,8 M de paramètres quantifiés. Les échelles coûtent 4/512 octet par poids.

| Option (même architecture) | Variation estimée | Taille estimée |
|---|---:|---:|
| `emb_linear` en INT8 avec échelle FP32 par ligne | −13,7 Mo | 135,9 Mo |
| Blocs NF4 de 512 → 128, échelles FP32 | +5,6 Mo | |
| Blocs NF4 de 512 → 64, échelles FP32 | +13,1 Mo | |
| Blocs de 64, échelles 8 bits + second niveau FP32 | +1,9 Mo | |
| **`emb_linear` INT8 + blocs de 64 FP32** | | **≈ 149,0 Mo** |
| `emb_linear` INT8 + blocs de 64, échelles 8 bits | | ≈ 137,8 Mo |
| Quelques convolutions 192 canaux en INT8 | +0,166 Mo par 192×192×3×3 | |

- Pour l'INT8 partiel, il faut calculer le total exact depuis le manifeste.
- **Risque des `emb_linear`** : ces poids modulent chaque fenêtre selon le conditionnement. Leur précision est un levier de niveau qui dépend du type de terrain. Il faut donc du INT8 avec échelle par ligne, pas du NF4, et tester leur effet sur le niveau.

**Mon jugement**
- Un QAT de cette architecture vers environ 150 Mo, avec des blocs plus fins financés par les `emb_linear`, est **plausible et testable**. Rien ne montre encore qu'il retirerait le grain.
- La réponse dépend de G0b-2 : la PTQ en blocs de 64 produit-elle nettement moins de grain qu'en blocs de 512 ?
- **50 à 100 Mo** par réduction de bits (environ 3 bits en moyenne) n'est pas soutenu : le 4 bits peine déjà en HF. Le chemin restant est un élève plus étroit, initialisé depuis le teacher, à envisager **après** validation de l'objectif HF.

### 4.6 L'expérience qui départage représentation et objectif

1. **Sans mise à jour (C6, §3.6)**, test d'existence au premier ordre.
   - Si aucune direction préconditionnée ne réduit l'énergie orthogonale sans réduire a, on a une limite **architecturale ou d'optimiseur**, avant même la quantification.
   - Ce test est imparfait : le STE prédit des effets continus, alors que les codes NF4 changent par sauts.
2. **Contexte (G0b-2/3/4)** : grain de la PTQ du teacher en blocs de 512, 128 et 64, par groupe de couches, et le long de l'historique QAT.
   - Si le QAT70k fait aussi mal que la PTQ b512, le training n'a jamais réduit le grain.
3. **Sonde de capacité, la plus discriminante** (training, hors paire, après A/B) :
   - **Données** : 16 régions TRAIN « ajustées », entrées et bruits fixes, et 16 régions TRAIN disjointes.
   - **Objectif** : KD + κ·L_HF calibré.
   - **Deux LR** : 3·10⁻⁶, celui de la paire, et le pic d'origine 3·10⁻⁵ (contrat de `native_weight_quant_train.py`).
   - **Mesures** : ρ²₁ et a sur les deux ensembles, plus la **fraction de codes NF4 changés** par groupe de couches.

**[Calcul indicatif]** Pourquoi deux LR. Avec Adam, un maître avance d'au plus environ lr par pas. Une case NF4 mesure de l'ordre de 0,1 à 0,3 en unités de maître, selon le bloc. À 3·10⁻⁶, seuls les codes déjà proches d'une frontière peuvent changer en 10k pas. Les tenseurs BF16 (`emb_linear`, etc.) bougent, eux, en continu.

| Sonde | Lecture |
|---|---|
| ρ²₁ baisse d'au moins 50 % sur l'ajusté et sur le disjoint, garde-fous tenus | Objectif mal pondéré : G1 a un sens |
| Baisse sur l'ajusté seulement | Apprenable, mais généralise mal : données ou pas, pas la représentation |
| Pas de baisse alors que les codes à pleine résolution changent | Limite de représentation en b512 : piste du quantifieur (§4.5) |
| Pas de baisse et presque aucun code ne change | LR trop faible pour juger : sonde non concluante à ce LR |

---

## 5. Question 4 : risques

### 5.1 `out_gain` et `emb_gain` encore en BF16

**Faits**
- **[Démontré-code]** Seul `out_gain` est en FP32 (`native_scalar_precision.py:53-58`).
  - Le noyau de `out_conv`, une fois mis à l'échelle, est **réarrondi en BF16 élément par élément** (l. 25). Le gain réalisé est donc une fonction tramée du gain FP32 : par incrément de 1/16 de cran, 1,1 à 5,0 m en montagne (rapport de niveau). C'est une résolution acceptable.
- **[Démontré-code]** Les 36 `emb_gain` passent toujours en BF16 dans le forward.
  - Chacun module tout un bloc par fenêtre : `c = emb_linear(emb, gain=emb_gain) + 1` (`unet_block.py:130`).
  - Ce sont des leviers spatialement uniformes, qui dépendent du conditionnement.
- **[Constat]** Valeurs dans les lignes lues de l'audit : de −1,37 à +10,44. Un cran BF16 vaut 0,4 à 0,8 % relatif ; à 10,44, il vaut 0,0625.
- **[Constat]** A72k contre 70k : les plaines descendent (−0,5 et −1,5 m) et les montagnes montent (+6 à +10 m). **Aucun gain global ne produit des signes opposés** : un gain de sortie déplace tout le C4 dans le même sens.

**Risques**
- **[Hypothèse]** Le levier le plus probable de cette dérive dépendante du terrain est la modulation par le conditionnement : `emb_linear` (BF16, continu) ou `emb_gain` (sauts d'un cran).
- Dans la paire, les scalaires suivent le gradient principal dans **les deux** bras. Leurs trajectoires peuvent diverger et brouiller l'attribution d'un écart de niveau B − A au terme LF.
- Adam normalise chaque coordonnée : un gradient faible mais de signe stable avance à pleine vitesse. Ce n'est pas constaté sur `out_gain` (un cran entre 42,5k et 50k), mais c'est à vérifier.

**[Test]** Lecture seule, sur CPU, sans toucher au run :
- aux états 70k, 72,5k, 75k… de A et de B, relever `out_gain` (FP32), chaque `emb_gain` (maître et valeur BF16 effective), les changements de cran, et la variation relative de chaque `emb_linear` ;
- pour tout changement de cran, forward causal en échangeant ce scalaire, comme la sonde `out_gain`, et mesure du niveau sur les 32 régions ;
- produire un tableau « mètres par cran » pour les 37 scalaires.

**Pour les recettes suivantes, pas pour la paire en cours**
- Ancrer en FP32 les 37 scalaires : 148 octets, même mécanisme que `anchored_output_conv`.
- Ou les geler dans la paire HF pour isoler l'effet des poids, avec un contrôle λ = 0 exact.

### 5.2 Un auxiliaire LF muet en montagne, parce que le plancher est grand

**Faits**
- **[Constat]** Calibration, 8 montagnes (`native-nf4-level-gradient-calibration-5090.json`) :
  - `height_lf` vaut 0,115 à 1,646, soit un RMS LF de 8,9 à 34 m une fois remultiplié par 26,33 ;
  - 0,1·`features_lf` vaut 1,0 à 4,1 ;
  - **la partie hauteur ne fait que 4 à 38 % de la perte de niveau en montagne** (médiane environ 15 %) ;
  - en plaine basse, elle en fait 76 à 92 % ; dans le désert et la tempérée, 14 à 87 %.
- **[Calcul]** Le gradient par unité d'erreur de q est proportionnel à J²/τ². Il vaut environ 11,5 en montagne (q ≈ 44,7, τ = 26,3) contre environ 1 300 dans un désert à 300 m (τ = 0,95). Par mètre, l'écart est d'environ 770 fois : **10 m d'erreur en montagne coûtent autant que 0,36 m dans le désert.**
- **[Constat]** Le plancher n'est pas un bruit. Dans `native-nf4-level-train-floor-4090.json`, le biais BF16/FP32 a **le même signe dans 31 cas sur 32**. En montagne, il vaut +10 à +46 m au LOD0, alors que l'erreur centrée n'est que de 3 à 12 m. Le « plancher » de 26 m est surtout un **écart déterministe** entre deux précisions du teacher. La cible de l'élève est TT en BF16, déterministe : rien ne l'oblige à rester à 26 m.
- **[Constat]** La partie C0–C3 domine en montagne. Le terme agit donc d'abord sur le niveau LOD0 par le decoder, ce qui peut être utile, et peu sur la hauteur LOD3.

**Risques de lecture de la paire**
- « B ≈ A en montagne » est **attendu par construction**. Ce ne sera pas un échec de la méthode.
- La décision doit se prendre par catégorie.
- Le témoin A72k déplace les plaines de 0,5 à 1,9 fois leur plancher, et les montagnes de 0,2 à 0,4 fois le leur. Ces écarts restent à l'échelle des planchers.

**[Test]**, sans toucher au run :
- normes de gradient préconditionnées **par terme** (hauteur, puis C0–C3) sur les 8 montagnes de calibration ;
- puis, aux checkpoints A et B, l'erreur LF de hauteur par catégorie, à la fois en unités de plancher et en mètres.

**Pour la suite**
- Séparer le plancher en deux : une partie biais (systématique) et une partie centrée (aléatoire).
- Normaliser l'erreur de niveau de l'élève par la partie centrée, ou par une tolérance de rendu fixée sur TRAIN.
- En montagne, le niveau absolu compte surtout à travers les **raccords** et la cohérence entre LOD : ce sont eux qu'il faut mesurer.

### 5.3 Métriques sur 32 pixels

**Faits** [Démontré-code], `native_weight_quant_evaluate.py:59` et `terrain_shape.py:30-31` :
- Le LOD3 est évalué sur 32×32 latents.
- `compare_shape` exclut une marge de ⌈2σ⌉ :
  - σ = 2 : 24×24 = 576 pixels au plus (terre seulement) ;
  - σ = 4 : 256 pixels ;
  - **σ = 8 et 16 : indisponibles**.
- Le rapport HF en h porte sur au plus 26×26 = 676 pixels.

**[Calcul]** Les rappels de vallées LOD3 des CSV ont 18 (côte), 26 (plaine basse), 57, 99, 120 et 125 pixels de vallée. **Un pixel vaut 0,8 à 5,6 points.** Exemple : côte 11/18 → 14/18 entre 70k et 72k. Les seuils « ≥ A − 0,01/0,02 » sont **en dessous de la résolution** pour la côte et la plaine basse.

**Autres points**
- **[Démontré-code]** Une basse fréquence σ = 8 lue sur 32 pixels se réduit à la moyenne et à une pente. C'est voulu pour le niveau, mais aveugle aux structures de 0,5 à 2 km.
- **[Démontré-code]** Le rapport HF du LOD3 est calculé en h : il est confondu par le niveau, or B déplace justement le niveau.
- **[Constat]** Un seul monde VAL et un seul bruit par catégorie dans la planche de progression.

**Corrections, pour l'évaluation et la suite**
- Le LOD3 sur un découpage de 48×48 latents, apparié à un LOD0 de même emprise.
- Toutes les métriques HF et TPI en q, plus la version « à niveau égalisé ».
- Rapport de profondeur des vallées (continu) en plus du rappel.
- Intervalles de confiance par bootstrap sur régions × bruits.
- Effets minimaux détectables pré-enregistrés à partir de la répétabilité : même checkpoint, autres bruits.

### 5.4 Autres risques de lecture de la paire

- **Le journal de B ne montre que les montagnes** [Démontré-code + Constat].
  - Une ligne est journalisée quand `engine.step % 100 == 0` (`native_level_train.py:239`), donc pour `step = 100k − 1`.
  - Alors `index = 50k − 1 ≡ 4 (mod 5)`, et `kinds[4]` est la montagne humide (l. 83-86).
  - Dans `b/train.jsonl`, toutes les lignes auxiliaires sont en effet `wet-high-mountain`.
  - L'échantillonnage réel reste équilibré. Mais l'usage du plafond sur les plaines et la côte (la côte 10096 a un `height_lf` de 343,8 en calibration) est **invisible** dans les journaux. Il faudra le recalculer hors ligne.
- **Évaluateur non déterministe** : `native_weight_quant_evaluate.py:123` met `cudnn.benchmark=True`, un point déjà signalé dans la revue « plateau ». Il faut vérifier la répétabilité (même checkpoint évalué deux fois), ou évaluer avec les options déterministes.
- **Où la paire agit vraiment** [Hypothèse]. À 3·10⁻⁶, la correction de niveau de B passe probablement par les tenseurs BF16 (`emb_linear`) et peu par les codes NF4. **[Test]** : fraction de codes changés par groupe de couches, de 70k à 72,5k et 75k, pour A et B (CPU, lecture des états).

---

## 6. Question 5 : prochaine expérience minimale et mesurable

### G0 : « budget du grain », sans training

**Contraintes**
- Aucune mise à jour de poids ; aucune modification de la paire, de ses fichiers ou de ses checkpoints.
- Données TRAIN uniquement, avec contrôle `HOLDOUT_SEEDS`.
- Sorties dans de nouveaux chemins, sans écrasement, avec les SHA des entrées **et du code**.

**Ce que G0 ne fait pas**
- Il ne promet aucune amélioration.
- Il n'annonce aucune durée : les coûts seront mesurés.
- Il ne se substitue pas au jugement visuel.

#### G0a : CPU seulement, sur les données existantes

**Entrées**
- `eval/native-nf4-level-train-attribution-5090/{contract.json, report.json, arrays.npz}`, avec vérification des SHA ;
- les captures TRAIN qu'elles désignent, pour la géométrie des fenêtres ;
- `gaussian_noise_patch`, pour régénérer le bruit de l'étage 2 à partir du `noise_seed` de chaque ligne.

**Calculs**
1. **Composante orthogonale absolue.**
   - Pour les paires SS/TT, SS/ST, TS/TT et ST/TT, par bande B1/B2/B3, en q.
   - Mesures : E_t, R, c, a, ρ², et RMS orthogonal **en unités q**.
   - Sur 32×32 comme D3, puis sur 48×48 pour vérifier la robustesse.
2. **Passage du bruit injecté (H-cancel).**
   - Fusionner les 9 bruits de l'étage 2 (canal 4) avec les mêmes poids que `core_crop`.
   - Régresser B_b(Q_SS − Q_ST) sur B_b(z₁ fusionné) et sur B_b(Q_ST) (erreur de gain). Même chose pour TS − TT.
   - Mesure : R² partiel.
3. **Signature d'arrondi BF16 (H-BF16).**
   - Aux pixels où une fenêtre porte au moins 90 % du poids de fusion, mesurer la distance de 0,5·C4 à la valeur BF16 la plus proche, en fraction de cran.
   - Une accumulation près de 0 signe l'arrondi.
   - Monte-Carlo : bruit d'arrondi par fenêtre d'après la carte des crans, fusionné puis filtré par bande, rapporté à E_t.
4. **Localisation par étage pour C0–C3.** Pentes par bande et par canal pour SS/TT, ST/TT et TS/TT. Si ST ≈ SS < TS, l'atténuation vient de l'étage 1.
5. **Part du grain dans l'opérateur KD** (H-obj), d'après E_t et ρ² mesurés.

**Sortie** : `eval/native-nf4-grain-budget-cpu.json`, avec lignes par région, résumés par catégorie, SHA des entrées et du code, et `updates: 0`.

#### G0b : forwards sur la 4090, entre deux évaluations de la paire

**Options déterministes.** Même protocole que TS : l'état du teacher en entrée de l'étage 2, donc des entrées identiques pour toutes les variantes.

**Mesures**
1. **Planchers.**
   - Étage 2 du teacher en FP32 contre BF16.
   - `window_outputs` en FP32 contre BF16, à vitesse identique : c'est l'arrondi **mesuré**, plus seulement prédit.
2. **PTQ du teacher, sans training.**
   - NF4 en blocs de 512, 128 et 64.
   - Blocs de 512 avec renormalisation de chaque filtre (H-norm).
   - Mesures : ρ², a, R par bande ; biais LF sur la chaîne complète en PTQ.
3. **PTQ par groupe de couches** (niveaux 192, 384, 576 et 768 canaux ; encodeur et decoder), en blocs de 512 : part de chaque groupe dans ρ²₁.
4. **Historique QAT.** EMA 15k, 30k, 50k, 70k et 100k, puis les états A/B quand ils sont publiés : trajectoire de ρ²₁.
5. **Lecture des états, sur CPU.** Fraction de codes changés par groupe de 70k à 72,5k et 75k (A et B), variation des `emb_linear`, et scalaires (§5.1).

#### Règles de lecture, à figer avant les résultats

| Observation | Conclusion | Conséquence |
|---|---|---|
| Arrondi mesuré ≥ 0,3·E_t,1 en plaine | La HF σ = 1 du teacher en plaine est en grande partie de l'arrondi | ε₁ fixé à ce plancher ; critères B1 relatifs au plancher ; l'effort porte sur B2 |
| RMS orthogonal absolu en q à peu près constant entre catégories (coefficient de variation < 0,3) et R² partiel sur z₁ ≥ 0,3 | Annulation imparfaite, erreur de faible dimension | G1 a de bonnes chances d'être utile ; cela reste à démontrer |
| R² partiel sur z₁ < 0,1 et RMS orthogonal proportionnel à E_t | Erreur de détail propre au terrain | La sonde de capacité devient prioritaire |
| PTQ b512 : ρ²₁ ≈ celui du QAT70k (rapport ≥ 0,8) | Le QAT n'a jamais réduit le grain | Objectif aveugle, ou LR trop faible ; voir le taux de codes changés |
| PTQ b64 : ρ²₁ ≤ 0,5 × PTQ b512 | La taille de bloc est un levier fort | Branche quantifieur à environ 149 Mo (`emb_linear` INT8 + b64) à tester en QAT |
| Les niveaux à 192 canaux portent ≥ 60 % du ρ²₁ de la PTQ | Grain localisé à pleine résolution | Précision mixte ciblée (quelques Mo) |
| La renormalisation réduit d'au moins 50 % le biais de niveau en PTQ | La non-conservation de la norme contribue au niveau | Piste quantifieur sans octet (réentraînement nécessaire) |
| Presque aucun code changé à pleine résolution à 3·10⁻⁶ | G1 à ce LR ne peut pas trancher la représentation | La sonde de capacité inclut 3·10⁻⁵ |
| ST/TT ≈ SS/TT < TS/TT sur C0–C3 | L'aplatissement LOD0 vient de l'étage 1 | Objectif séparé pour les incisions côte/plaine basse |

#### Ensuite, après la décision A/B

1. Calibration C0 à C7 (§3.6).
2. Test d'existence au premier ordre (C6).
3. Sonde de capacité (§4.6) si G0 laisse la question de la représentation ouverte.
4. Puis la paire G1, C contre H, avec les critères du §3.7.

Le choix entre G1 et la branche quantifieur dépend de G0b-2 et G0b-3, pas d'une préférence a priori.

---

## 7. Ce que je ne recommande pas

- Modifier la paire en cours, ou régler λ, μ ou κ pendant un run.
- Un passe-haut en h, ou pondéré par 2|q| de l'élève.
- Une MSE appariée seule, ou protégée par un simple déficit d'énergie (je corrige mon avis précédent).
- Une égalisation symétrique d'énergie par bande (`log(E_s/E_t)²`) : elle peut inventer de l'énergie.
- Une cible TT pour la HF.
- Conclure de la sonde INT8 qu'un QAT mixte échouerait, ou que son gain LOD0 sur la plaine-mer est réel.
- Juger le niveau de B en montagne avec le plancher actuel comme seuil de succès, ou trancher les vallées LOD3 sur 18 à 26 pixels.
- Écarter les mondes 202/404, coarse2 ou les cas défavorables de toute décision.
- Viser 50 à 100 Mo en réduisant les bits avant d'avoir validé un objectif HF.

---

## 8. Limites de cette revue

- Les colonnes dérivées du §2.1 combinent des médianes de R et de c prises séparément : ce sont des ordres de grandeur. G0a les recalcule par région.
- Je n'ai pas lu `arrays.npz`. L'énergie HF absolue du teacher en q, les altitudes des scènes et la part réelle de l'arrondi BF16 sont donc **inconnues**. Les calculs H-obj et H-BF16 sont indicatifs.
- La sonde INT8 et le témoin A72k reposent sur une scène et un bruit par cas.
- Les pentes C0–C3 et C4 de l'attribution sont calculées toutes fréquences confondues sur 32×32, et peuvent être diluées par du bruit dans le teacher.
- Je n'ai vu ni les résultats de B, ni ceux des mondes 202/404, ni les valeurs des scalaires au-delà des premières lignes de l'audit.
- Les seuils des §3.7 et §6 sont des propositions à pré-enregistrer, pas des prédictions. Je ne promets aucun succès et je n'avance aucune durée.