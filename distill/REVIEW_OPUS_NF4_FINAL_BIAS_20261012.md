# Revue technique : biais NF4, cohérence de la cible physique et suite depuis EMA70k

Lecture seule : je n'ai rien lancé, rien modifié et fait aucune action réseau.

Chaque affirmation porte un statut :
- **[Constat]** : lu dans les données ou les rapports.
- **[Démontré-code]** : découle du code ou d'un calcul exact, quelles que soient les données.
- **[Démontré-exp]** : établi par l'expérience appariée A/B.
- **[Hypothèse]** : compatible avec les données mais non prouvé ; je propose un test.

---

## 0. Ce que j'ai lu, et ce que je n'ai pas pu lire

- **Rapports** : `REPORT.md`, les deux revues `REVIEW_OPUS_NF4_*`, `BASE_NF4_ABLATION_42500`, `BASE_NF4_100K_STATUS`, `BASE_NF4_DECISION`, `BASE_NF4_AUXILIARY_PLAN`.
- **Code** :
  - auxiliaire : `native_field_auxiliary.py`, `native_auxiliary_training.py`, `native_auxiliary_train.py`, `native_auxiliary_calibrate.py` ;
  - entraînement : `native_weight_quant_train.py`, `native_weight_quant_resume.py`, `native_weight_quant.py` ;
  - chaîne et évaluation : `native_fusion.py`, `native_decoded.py`, `decoded_loss.py`, `native_weight_quant_evaluate.py`, `terrain_shape.py` ;
  - amont : `edm_unet.py`, `mp_layers.py`, `laplacian_encoder.py`.
- **Données** :
  - CSV EMA coarse4 de A et B à 50k ;
  - JSON complet de B à 50k (morphologie à toutes les échelles σ) ;
  - JSON de calibration auxiliaire à 40k (32 régions).
- **Non lu** :
  - `b/train.jsonl`, donc pas la distribution réelle de `auxiliary_scale` ;
  - les CSV par scène du 70k ;
  - la décomposition biais/reste des mondes 202 et 404 (ma recherche dans `eval/native-nf4-70k-historical-final-coarse4` n'a rien trouvé) ;
  - les valeurs des scalaires dans les checkpoints (fichiers binaires).

---

## 1. Verdict

1. **[Démontré-exp]** La dégradation de B vient du terme auxiliaire, et elle est **globale**, pas locale.
   - Les 14 écarts de biais B − A (7 scènes × 2 LOD) sont négatifs.
   - Les trois montagnes, que l'échantillonneur auxiliaire ne tire jamais (`auxiliary_record`, `native_auxiliary_training.py:77`, quatre catégories plates ou côtières seulement), descendent de 43 à 81 m.
   - B viole aussi ses propres garde-fous pré-enregistrés sur les vallées. Il aurait été rejeté même sans regarder le biais.
2. **[Démontré-code]** Le passe-haut apparié calculé sur `h = q|q|` peut réduire sa perte en **baissant le niveau des terres**.
   - Cette voie existe dès que l'élève a un excès d'énergie haute fréquence, ce qui est précisément le défaut visé.
   - Un deuxième mécanisme s'y ajoute, propre à toute MSE appariée sur une bande mal corrélée : l'amplitude optimale vaut c·σ_t, donc la perte lisse.
   - Enfin, la métrique de grain en `h` est elle-même confondue par le niveau.
   - Je n'ai pas isolé expérimentalement le terme passe-haut du terme côte. Mais le terme côte pousse dans le sens opposé à la dérive observée : il ne peut pas l'expliquer.
3. **[Hypothèse forte, testable sur CPU en quelques minutes]** Les **scalaires globaux** (`net.out_gain`, les `emb_gain` des blocs) servent de bouton d'amplitude globale. Trois éléments se combinent :
   - Adam normalise chaque coordonnée, donc un scalaire avance à ~lr par pas, quel que soit le plafond global ;
   - le forward arrondit tout en BF16 (`NativeWeightQuantBase.forward`, `native_weight_quant.py:122`) ;
   - un seul ulp BF16 de `out_gain` vaut, en ordre de grandeur, des dizaines de mètres en montagne.

   Cette hypothèse explique à la fois la fuite globale de B, la dérive du niveau de A d'un checkpoint à l'autre et la « chance » du 70k. Indice direct dans la calibration : pour chaque région, les cosinus côte/régulier et passe-haut/régulier sont presque identiques.
4. **Cible.** Pour le **grain de l'étage 2**, la cible T2(S1) « même entrée » est adaptée : l'attribution montre que l'étage 2 porte ce défaut. En revanche, elle ne fait pas référence pour le **niveau** ni pour le **signe du champ final**. L'échec de B ne vient pas du choix de cible mais de la forme de la perte. Je recommande de **séparer par bande** :
   - basse fréquence : cible TT complète sur bruits neufs ;
   - haute fréquence : cible même entrée.
5. **Niveau moyen et vallées.**
   - Pondérer les résidus en q par le jacobien **du teacher, détaché**. On garde l'échelle en mètres sans échappatoire de niveau.
   - Utiliser un terme de **basse fréquence seule** : son gradient en sortie est lisse, donc il ne vise pas directement les incisions.
   - Couvrir aussi C0–C3. Le niveau du LOD0 dépend de la moyenne du résidu décodé, que `laplacian_denoise` compte deux fois.
   - Calculer les métriques d'amplitude aussi « à niveau égalisé ».
6. **Reprise depuis EMA70k.** Elle est faisable et honnête si on la présente comme une **réinitialisation** : maîtres = EMA70k BF16, Adam neuf avec pré-passe de moments à lr=0, EMA repartant des mêmes poids. Un **bras témoin A′** est obligatoire, à cause de l'effet du redémarrage et du biais de sélection du 70k.
7. **Suite.**
   - Une phase 0 en forward et CPU seulement, qui tranche entre les hypothèses.
   - Un essai minimal R1 qui **ne vise que le niveau**, avec les scalaires exclus de l'auxiliaire.
   - Le grain ensuite (R2), avec un passe-haut en q protégé contre le déficit d'énergie.
   - Le passage à 50–100 Mo, et à la vitesse, n'a de sens qu'avec un objectif ainsi corrigé.

---

## 2. Constats chiffrés A/B à 50k (EMA, coarse4, une scène et un bruit par cas)

### 2.1 Biais (m)

| Scène | LOD3 A | LOD3 B | Δ B−A | LOD0 A | LOD0 B | Δ B−A |
|---|---:|---:|---:|---:|---:|---:|
| Désert 10294 | −4,20 | −6,01 | **−1,81** | −5,07 | −6,73 | **−1,67** |
| Tempérée 10296 | −2,43 | −6,75 | **−4,32** | −3,05 | −8,04 | **−4,99** |
| Côte humide 10594 | −3,31 | −5,62 | **−2,31** | −3,12 | −4,35 | **−1,23** |
| Plaine-mer 10296 | −0,99 | −1,95 | **−0,96** | −0,29 | −0,61 | **−0,32** |
| Montagne 620001 | **+27,60** | −21,13 | **−48,7** | **+12,70** | −38,65 | **−51,4** |
| Montagne 620002 | **+14,55** | −28,31 | **−42,9** | **+22,17** | −21,82 | **−44,0** |
| Montagne 620003 | **+18,74** | −49,58 | **−68,3** | **+32,29** | −48,33 | **−80,6** |

- **[Constat]** Le « biais négatif persistant » du rapport est vrai pour les **plaines** de A, **pas pour ses montagnes**, positives à 50k. Pour 620001, le biais vaut −31,5 m à l'EMA30k (revue plateau), +12,7 m pour A50k (sur la 4090) et −56,9 m à l'EMA100k (5090). Le niveau dérive donc de plusieurs dizaines de mètres entre checkpoints. Ce n'est pas un défaut fixe du NF4.
- **[Démontré-exp]** Les 14 écarts B − A sont négatifs. Leur ampleur en montagne dépasse de loin l'écart raw/EMA de A à 42,5k (environ 2 à 6 m). Deux réserves : une seule graine, et une divergence chaotique on-policy.

### 2.2 Morphologie au LOD0 (σ=8, sauf mention)

| Scène | Amplitude TPI A / B | Rappel des vallées A / B | Ratio HF A / B |
|---|---:|---:|---:|
| Désert | 0,950 / 0,921 | 0,818 / 0,812 | 0,892 / 0,862 |
| Tempérée | 1,015 / 0,992 | 0,939 / 0,930 | 0,954 / 0,945 |
| Côte humide | 0,959 / 0,905 | 0,725 / 0,609 | 0,905 / 0,816 |
| Plaine-mer | 0,696 / 0,539 | 0,297 / 0,066 | 0,705 / 0,508 |
| 620001 | 0,993 / 0,961 | 0,950 / 0,929 | 1,004 / 0,977 |
| 620002 | 0,986 / 0,935 | 0,953 / 0,912 | 0,987 / 0,946 |
| 620003 | 0,943 / 0,901 | 0,917 / 0,878 | 0,950 / 0,915 |

- **[Constat]** B aplatit le relief au LOD0 dans tous les cas. En montagne, l'amplitude TPI à σ=16 tombe à 0,948 / 0,928 / 0,904.
- **[Constat]** Confrontés aux garde-fous pré-enregistrés (revue plateau §6.8), les résultats de B les violent :
  - rappel des vallées ≥ A − 0,02 : violé sur les trois montagnes (−0,021, −0,041, −0,039) ;
  - |amplitude TPI − A| ≤ 0,05 : violé sur 620002 (−0,051) ;
  - HF LOD0 des plaines ≥ 0,9 × A : violé sur la plaine-mer (0,72×).
- **[Constat, à porter au crédit de B]** Sur les **plaines au LOD3**, B améliore réellement la forme :
  - corrélation TPI à σ=8 : 0,610 → 0,681 (désert), 0,592 → 0,634 (tempérée) ;
  - rappel des vallées LOD3 : 0,563 → 0,642 et 0,601 → 0,688.

  Dans son domaine, l'idée d'un terme de grain n'est donc pas morte. Ce qui fuit vers le reste, c'est son implémentation.
- **[Constat]** Déjà dans A, le LOD0 des plaines est **trop lisse** (HF 0,70 à 0,95), alors que le LOD3 est **trop granuleux**. Ce sont deux défauts opposés, portés par des canaux différents :
  - le grain LOD3 vient du C4 haute fréquence ;
  - le LOD0 dépend de C0–C3 via le decoder, car `laplacian_denoise(σ=5)` filtre le C4 haute fréquence.

  Un objectif « moins de haute fréquence » global aggrave donc mécaniquement le LOD0. C'est ce qu'on observe.

### 2.3 Calibration à 40k (`eval/native-nf4-aux-calibration40k.json`)

- **[Constat]** Le cosinus entre deux batchs **réguliers indépendants** vaut souvent 0,35 à 0,88 (par exemple 0,614 / 0,612 / 0,882 / 0,697 / 0,639). Le gradient régulier contient donc une **forte composante persistante**.
- **[Constat]** Pour chaque région où les deux termes sont actifs, cos(côte, régulier) et cos(passe-haut, régulier) sont presque égaux :

  | Région | cos(côte, régulier) | cos(passe-haut, régulier) |
  |---:|---:|---:|
  | 0 | 0,6259 | 0,6271 |
  | 1 | 0,5835 | 0,5833 |
  | 3 | 0,4116 | 0,4120 |
  | 4 | 0,9056 | 0,9054 |
  | 9 | 0,3910 | 0,3914 |
  | 17 | 0,6010 | 0,6013 |

  Or ces deux pertes ont des supports disjoints : pixels à ±20 m d'un côté, passe-haut des terres de l'autre. Leurs gradients sont donc **presque colinéaires**, dominés par une même direction dans l'espace des paramètres.
- **[Constat]** La revue gradients demandait des ratios **par tenseur** (`out_conv`, première convolution, embeddings, gains). Le JSON ne contient que des normes globales. Le plafond était donc aveugle à une éventuelle concentration.

---

## 3. Question 1 : le passe-haut de `h = q|q|` réduit-il son amplitude en baissant le niveau ?

### 3.1 Dérivée exacte [Démontré-code]

`low_field_terms` (`native_field_auxiliary.py:33-39`) calcule, sur les pixels de terre du teacher :

`L = moy_terre (HP(h_s) − HP(h_t))² / P_cat`

avec `HP = id − G_1` (linéaire), `h = q|q|` et `dh/dq = 2|q|`.

Décalons uniformément l'élève : `q_s → q_s + δ`. Comme HP est linéaire :

**dL/dδ = (2/(N·P)) · Σ_terre (HP(h_s) − HP(h_t)) · HP(2|q_s|)**

Il n'y a aucune approximation dans cette expression.

Pour l'interpréter, écrivons q = m + r (m basse fréquence, r passe-haut en q), sur la terre, avec |r| < m :
- `HP(h) ≈ 2m·r`, puisque HP(m²) ≈ 0 pour un m lisse à σ=1 px ;
- donc `HP(2|q_s|) ≈ 2 r_s` ;
- d'où **dL/dδ ≈ (8/(N·P)) · Σ (m_s r_s² − m_t r_t r_s)**.

Avec m_s ≈ m_t = M, ce terme vaut 8M(‖r_s‖² − ⟨r_s, r_t⟩)/(N·P). Par Cauchy-Schwarz, il est **strictement positif dès que ‖r_s‖ > ‖r_t‖** : l'excès de grain est exactement le défaut visé. La descente de gradient fait alors **baisser le niveau des terres**.

Optimum le long de cette seule direction : M_s/M_t = ⟨r_s, r_t⟩/‖r_s‖² = c·‖r_t‖/‖r_s‖. Avec un ratio HF de 2 et une corrélation c ≈ 0,5, cet optimum vaut environ 0,25. C'est énorme. Seule la perte principale s'y oppose, et les revues précédentes ont montré qu'elle voit à peine le niveau.

Le gain multiplicatif du latent C4 donne la même structure. Pour `C4 → (1+ε)C4`, on a `dq/dε = q + 31,4`, d'où `dL/dε = (2/(N·P))·Σ (HP h_s − HP h_t)·HP(2|q_s|(q_s+31,4))`. Réduire ce gain baisse **en même temps** m et r, le niveau et le détail : c'est la manière la moins coûteuse de réduire L.

### 3.2 Deuxième voie : l'atténuation de Wiener [Démontré-code]

Même en espace q, sans échappatoire de niveau, une MSE appariée sur une bande vérifie :

min_a ‖a·r_s − r_t‖² ⇒ a* = c·‖r_t‖/‖r_s‖, et l'énergie finale vaut **c²** fois celle du teacher.

Dans le JSON de B à 50k, les corrélations TPI au LOD3 à σ=2 valent 0,44 (désert), 0,34 (tempérée), 0,36 (plaine-mer) et 0,54 à 0,61 (montagnes). L'optimum du passe-haut apparié serait donc une bande à **12–37 % de l'énergie du teacher** : **c'est un objectif de lissage**, pas de « retrait du grain ». Le plafond l'a seulement freiné.

Ces corrélations sont mesurées sur la chaîne complète. Avec la cible même entrée de B, c est probablement plus élevé, mais non mesuré (test D3).

### 3.3 La métrique de grain est confondue par le niveau [Démontré-code]

`high_frequency_rms_ratio` (`native_weight_quant_evaluate.py:60-65`) et `compare_shape` (`terrain_shape.py:36-52`) travaillent en `h`. Or :

`‖HP(h_s)‖/‖HP(h_t)‖ ≈ (|m_s|/|m_t|)·‖r_s‖/‖r_t‖`

De plus, le seuil des vallées vaut `−0,75·scale_teacher` (`terrain_shape.py:43-44`) : le rappel des vallées est sensible à l'amplitude. **Baisser le niveau fait donc baisser mécaniquement le ratio HF, l'amplitude TPI et le rappel**, à détail inchangé.

- Plaine-mer (h de l'ordre de quelques mètres) : la baisse du ratio HF LOD3 de B (2,17 → 1,64) **n'est pas interprétable** sans le niveau. Une bonne partie peut venir de |m|.
- Désert et tempérée : si ces plaines sont à quelques centaines de mètres (je n'ai pas lu leur altitude moyenne), l'effet de niveau sur le ratio HF est inférieur à 1 %. Leur baisse d'environ 6 % est alors réelle en q.
- Montagnes : un biais de −50 m sur environ 2 000 m n'explique qu'environ 1,3 % de baisse de l'amplitude TPI, pour 3 à 5 % observés. Le reste est une vraie perte d'amplitude.

### 3.4 Pourquoi le terme côte n'explique pas la dérive [argument, non isolé]

- `root_coast` (`native_auxiliary_training.py:27-37`) applique une KL en q. Son gradient en un pixel vaut (σ(q_s/τ) − σ(q_t/τ))/τ : il pousse toujours vers le signe du teacher, sans échappatoire.
- Dans A, la plaine-mer est déjà trop basse : le terme côte pousse donc **vers le haut**.
- Or B finit plus bas, et son désaccord côtier double (11,7 → 20,8 %).
- Si ce terme passait par un gain global, le relever ferait monter les montagnes. Elles descendent.

**Réponse à la question 1 :**
- Oui, la voie existe : démontré sur la perte.
- Elle est cohérente avec toutes les données de B : 14 écarts négatifs sur 14, plus forts en montagne, côte opposée.
- Elle n'est pas isolée par une expérience : il n'y a pas de bras « passe-haut seul ».
- Elle s'ajoute à une atténuation de type Wiener, et la métrique de grain en h la masque en partie.

---

## 4. Le canal global : scalaires, Adam et BF16 [Hypothèse forte]

### 4.1 Les faits du code [Démontré-code]

- `EDMUnet2D.out_gain` est un scalaire appris (`edm_unet.py:103`). Il multiplie la sortie de `out_conv` pour **les cinq canaux et les deux étapes** (l.179).
- `UNetBlock.emb_gain` (`unet_block.py:72`) est un scalaire par bloc, qui module le conditionnement.
- `out_conv` et la première convolution ne sont pas quantifiés (`native_weight_quant.py:95-96`). Ils sont normalisés à chaque appel (`MPConv.forward`, `mp_layers.py:211-212`) : leur norme est figée, **seul `out_gain` porte l'amplitude de sortie**.
- `NativeWeightQuantBase.forward` passe tous les paramètres en BF16 avant le forward (`native_weight_quant.py:122`). Le scalaire n'agit donc qu'à la granularité BF16 : 2⁻⁸ ≈ 0,39 % dans [0,5 ; 1), 2⁻⁷ ≈ 0,78 % dans [1 ; 2).
- Adam (`betas .9/.99`) normalise chaque coordonnée. Un scalaire au gradient de signe stable avance d'environ **lr = 3e-6 par pas**, quel que soit le plafond global. `combine_gradients` (`native_auxiliary_training.py:108-117`) borne des **normes globales**, ce qui ne donne aucune garantie par coordonnée.
- `dL/d out_gain = Σ⟨∂L/∂v, v⟩/out_gain` agrège tous les pixels, canaux et fenêtres. Tout gradient de sortie corrélé à v s'y projette fortement. Cela suffit à rendre colinéaires des gradients de pertes très différentes (§2.3).

### 4.2 Ordre de grandeur

`window_outputs` (`native_fusion.py:22-28`) donne x0 = cos t·inputs + sin t·v. Un changement relatif ε de v donne δx0 = ε(x0 − cos t·inputs).
- Étage 1 : inputs = sin t·bruit, donc δx0 ≈ ε·x0.
- La basse fréquence traverse l'étage 2, d'où δC4 ≈ ε·k·C4, avec k d'ordre 1.
- En q : Δq = ε·k·(q + 31,4). En hauteur LOD3 : **Δh ≈ 2|q|·ε·k·(q+31,4)**.

Pour **un ulp** (ε = 0,0039, k = 1) :

| Terrain | q | Δh par ulp |
|---|---:|---:|
| Montagne vers 2 000 m | 44,7 | ≈ −27 m |
| Plaine vers 300 m | 17,3 | ≈ −6,6 m |
| Plaine côtière vers 2 m | 1,4 | ≈ −0,36 m |

Ce sont exactement les ordres de grandeur des oscillations de niveau observées. Le rapport montagne/plaine des écarts mesurés (10 à 40×) est cependant plus grand que ce que prédit ce modèle seul (environ 4 à 8×, selon des altitudes que je n'ai pas). **Un seul gain n'explique donc pas tout.** De même, A à 50k a des plaines négatives et des montagnes positives : il y a au moins une autre composante, par exemple affine ou propre à C0–C3.

### 4.3 Ce que cette hypothèse expliquerait

1. La fuite globale de B vers des scènes jamais échantillonnées.
2. La dérive du niveau de A entre checkpoints : l'objectif réclame un gain non représentable en BF16, d'où un cosinus régulier/régulier élevé et persistant, et des sauts d'ulp.
3. L'écart raw/EMA, puisque les maîtres EMA et raw franchissent les frontières d'arrondi à des moments différents.
4. Le choix du 70k, probablement favorisé par un bon niveau sur les sept scènes, avec un score dominé par l'offset (revue plateau §1.4).
5. Le biais négatif du NF4 brut au pas 0 (−245 m en montagne). Une hypothèse annexe : le bruit de quantification réduit la part cohérente des activations après normalisation MP, donc l'amplitude cohérente de sortie.

### 4.4 Test décisif (D0, CPU, quelques minutes)

Lire `net.out_gain` et tous les `emb_gain`, en FP32 maître et en valeur BF16 effective, pour :
- le teacher ;
- le NF4 brut au pas 0, puis 6k et chaque EMA de 15k à 100k ;
- les raw 20k, 50k et 100k ;
- A et B à 42,5k, 45k, 47,5k et 50k, en raw et en EMA.

Les comparer ensuite aux biais par scène des CSV existants.

- **Hypothèse confirmée** si `out_gain` de B à 50k est inférieur d'au moins un ulp BF16 à celui de A à 50k, et si les sauts de niveau du run principal coïncident avec des changements d'ulp.
- **Hypothèse rejetée** si ces valeurs BF16 n'ont jamais bougé. Dans ce cas, la direction commune est ailleurs (`out_conv`, première convolution, embeddings) : c'est le rôle du test D4.

---

## 5. Question 2 : cible teacher sur l'état élève (T2(S1)) ou TT complète ?

### 5.1 Ce que B a utilisé [Démontré-code]

`auxiliary_fields` (`native_auxiliary_training.py:40-63`) procède ainsi :
1. étage 1 élève sans gradient, sur un bruit neuf ;
2. construction de `inputs1` ;
3. teacher sur ces `inputs1` pour la cible ;
4. étage 2 élève avec gradient.

La cible est donc **T2(S1(z))**, une configuration de type « ÉT ». Dans l'attribution, ÉT ne diffère de TT que de 3,31 % contre 1,39 % pour le désaccord côtier, et de 0,058 contre 0,011 pour |log HF|.

### 5.2 Ce que l'attribution montre, et ce qu'elle ne montre pas

- **[Constat]** TÉ ≈ ÉÉ : 12,62 % contre 14,23 %, et 0,974 contre 1,009. Le grain et le désaccord côtier LOD3 viennent de l'**étage 2 élève**, même à partir d'un état teacher.
- **[Constat]** L'attribution ne mesure **ni le biais, ni la basse fréquence, ni le LOD0**. On ne sait pas quel étage fixe le niveau.
- **[Démontré-code]** T2(S1) est une cible **auto-référente pour la basse fréquence**. Si S1 porte une erreur de niveau, l'étage 2 du teacher la conserve en grande partie, car le spectre rouge du terrain domine le bruit blanc en basse fréquence au temps de l'étage 2. Cette cible ne peut donc pas servir de référence de niveau pour le champ déployé.

### 5.3 Ma réponse

- **Pour le niveau et le signe du résultat complet : oui**, il faut exécuter les **deux étapes teacher** sur les **mêmes** bruits neufs (z0, z1), avec les mêmes conditions, labels et variantes de coarse.
  - C'est la quantité déployée.
  - Le teacher y atteint une perte nulle ; c'est son optimum.
  - Le surcoût se limite à 25 fenêtres teacher de plus, sans gradient.
- **Pour le grain haute fréquence : garder la cible même entrée.** Face à TT, les différences de haute fréquence nées en S1 sont irréductibles pour S2. Une MSE appariée les transformerait en pression de lissage (§3.2). La cible même entrée réduit l'erreur irréductible à la seule capacité de S2.
- **La compensation en basse fréquence par S2 est réaliste.** À son temps t2, S2 voit nettement le niveau de S1. On peut donc demander à S2 de corriger un niveau légèrement faux.
- **Cette compensation crée une tension réelle** avec l'opérateur on-policy, qui exige S2 = T2 sur les états de S1. Le test D2 mesure quelle compensation est nécessaire : si le niveau ÉT ≈ niveau TT, il n'y a pas de tension.
- **L'échec de B ne vient pas de la cible.** Avec TT et la même perte en h, l'échappatoire de niveau, l'atténuation et la fuite par les scalaires auraient été les mêmes.

| Terme | Cible | Gradient |
|---|---|---|
| Niveau, basse fréquence de C4 et C0–C3 | TT, bruits neufs | S2. Aussi S1, avec checkpointing, si D2 montre que S1 porte ≥ 30 % du niveau |
| Grain haute fréquence de C4 | T2(S1), même entrée, bruit neuf | S2 |
| Signe côtier (KL en q) | TT (c'est un phénomène de niveau) | S2 |

À noter : TT recalculé contre la cible stockée donne déjà 0,304 m de MAE (attribution). Il faut donc calculer la cible TT par **le même chemin numérique** que l'élève (`forward_batches`, `blend`, `core_crop`), pas comparer à `region.target`.

---

## 6. Question 3 : protéger le niveau moyen sans lisser les incisions

### 6.1 Ni q nu, ni h : des résidus en q pondérés par le jacobien du teacher, détaché

- **h** : l'échelle est physique, mais le poids 2|q_s| dépend de l'élève. Cela crée l'échappatoire du §3.1.
- **q nu** : pas d'échappatoire, mais les mètres sont mal pondérés (une même erreur en q pèse 30 fois plus en mètres à 2 000 m qu'à 1 m).
- **Linéarisation au teacher** : Δh_lin = 2|m_t|·(q_s − q_t), avec |m_t| **détaché**.
  - La perte est quadratique en q_s, avec des poids fixes.
  - Elle a la même échelle en mètres au premier ordre.
  - **Elle n'offre aucun moyen de baisser la perte en changeant le niveau de l'élève** [Démontré-code].
  - Le poids s'annule au trait de côte : la côte garde donc sa KL en q (déjà correcte).

### 6.2 Un terme de basse fréquence seule

`L_LF = moy[(w_t·G_σ*(q_s − q_t))²]/τ_k²  +  ν·Σ_{c=0..3} moy[(G_σ*(C_c,s − C_c,t))²]/(var_c·τ_c²)`

- w_t = 2·max(|G_σ*q_t|, 1), détaché.
- σ = 8 pixels latents sur le core 64² (G_σ avec réflexion ; la moyenne est incluse).
- **Garde-fou structurel** [Démontré-code] : le gradient de sortie vaut G_σ*(W²·G_σ*Δq). C'est un champ lisse à environ σ√2 ≈ 11 pixels latents, soit environ 90 px LOD0. Il ne vise directement aucune vallée des échelles TPI σ = 2 à 16 au LOD0.
- **Limite honnête** : les paramètres sont partagés, donc un gradient lisse en sortie peut changer la haute fréquence par effet de bord. La protection reste **mesurée** (§9.3), pas garantie.
- **τ_k sans VAL** : RMS de G_σ*(h_FP32 − h_BF16) du teacher sur TRAIN, par catégorie (test D6). « 1 » signifie alors « aussi bon que l'écart de précision du teacher lui-même ». La cible devient le plancher, sans réglage.

### 6.3 Le LOD0 ne dépend pas que de C4 [Démontré-code]

`reconstructed_height` (`decoded_loss.py:76-80`) appelle `laplacian_denoise(high, low, σ=5)`. Celle-ci (`laplacian_encoder.py:134-137`) renvoie le résidu inchangé et un nouveau `low = blur(down(résidu + up(low)))`. Le décodage final vaut donc résidu + up(blur(down(résidu))) + … : **la basse fréquence du résidu decoder est comptée deux fois**, puis multipliée par 2|s| dans le carré signé.
- Les latents C0–C3 de l'élève déplacent ce résidu, donc le niveau du LOD0, indépendamment de C4.
- D'où le terme C0–C3 dans L_LF.
- Si le niveau LOD0 reste faux alors que le LOD3 est corrigé : terme basse fréquence **décodé** (variante R1b), avec `OriginalDecoder` (`native_decoded.py:116-142`).

### 6.4 Défaut de mesure à corriger [Démontré-code]

- Le LOD3 est évalué sur tout le core 64² (`core_crop`, `native_decoded.py:73-79`).
- Le LOD0 est recadré deux fois, `[64:-64]` (`decoded_loss.py:79`, puis `evaluate.py:51`), sur les **256 px centraux**, soit les 32² latents centraux.
- **Les biais LOD3 et LOD0 ne portent donc pas sur la même surface** : leur comparaison actuelle (par exemple 620001 : −21 contre −39 m pour B) n'est pas interprétable telle quelle. Il faut les rapporter sur la même emprise.

### 6.5 Métriques « à niveau égalisé »

Remplacer la basse fréquence de l'élève par celle du teacher, q′_s = q_s − G_σ*(q_s − q_t), puis recalculer h. On mesure alors HF, TPI et vallées sur h(q′_s), à côté des métriques brutes. Cela sépare le détail du niveau, ce qui manquait pour juger B.

---

## 7. Défauts persistants du 70k (modèle d'essai, pas accepté)

| Défaut | Statut | Cause |
|---|---|---|
| **Niveau** : la MAE est souvent ≈ \|biais\| ; il change de signe entre checkpoints ; montagnes au LOD0 à 24,45 m pour un plancher FP32/BF16 de 25,44 m | Constat | Objectif quasi aveugle au niveau (revues précédentes). Mécanisme précis : hypothèse §4. Le 70k est probablement favorisé par la sélection sur sept scènes (hypothèse) |
| **Grain LOD3** : pentes jusqu'à 1,63×, dernière bande 1,53 à 3,84× | Constat | Localisé dans l'étage 2 (attribution : démontré comme localisation, pas comme mécanisme). Mécanisme probable : bande pondérée environ 1e-6 dans la variance latente (hypothèse) |
| **LOD0 des plaines un peu lisse** (HF 0,70 à 0,95 dans A50k ; 70k non relu) | Constat (A) | Probablement l'amplitude de C0–C3 (hypothèse) |
| **Mondes 202 et 404 à environ 2× le plancher** (93,85 contre 41,92 m ; 66,93 contre 33,69 m) | Constat | **Inconnue.** Il faut une décomposition biais/reste (D1). Je ne l'ai pas trouvée |
| **Raccords en batch16** : maxima aggravés | Constat | Numérique, dépendant du batch (démontré : canonique batch1 exactement nul pour les deux modèles) |
| **Taille et vitesse** : 149,6 Mo, vue LOD3 plus lente de 11 % | Constat | NF4 poids seuls = convolution BF16 déquantifiée (démontré). Aucun gain de calcul possible dans ce format |

La **MAE n'est pas une cible physique** : le plancher FP32/BF16 est de 17 à 25 m en montagne, 21–22 m sur les lieux historiques. Pour décider, il faut le biais, la morphologie et la haute fréquence, chacun comparé à son propre plancher FP32/BF16. Le plancher du biais n'est pas publié : à mesurer.

---

## 8. Question 4 : reprise honnête depuis EMA70k sans l'Adam 70k

### 8.1 Ce qui existe

- L'EMA70k en BF16 (`native_weight_quant_resume.py:138`) et le paquet 70k.
- **Pas** de maîtres FP32 à 70k, **pas** d'Adam, **pas** de raw à 70k (raw seulement à 20k, 50k et 100k, l.159).
- `restore_training` refuse un snapshot (l.24-25) : la reprise exacte est impossible par construction.

### 8.2 Procédure proposée (« réinitialisation 70k », jamais « reprise »)

1. **Maîtres FP32 = valeurs BF16 de l'EMA70k.**
   - Vérifier en forward déterministe l'égalité bit à bit avec `load_packed` sur trois régions.
   - Comme le forward arrondit en BF16 puis en NF4, le réseau de départ est **exactement** le 70k déployé.
2. **Temps mort attendu.** Chaque maître est au centre de son intervalle BF16. Il faut environ un demi-ulp de dérive de signe stable pour changer le forward : quelques centaines de pas à 3e-6 pour un poids d'ordre 1, et davantage pour un code NF4. Les premiers snapshots seront presque identiques au 70k. Ce n'est pas un bug.
3. **Adam neuf** (betas 0,9/0,99, même eps), avec une **pré-passe de 500 pas à lr=0**.
   - Seuls m, v et le compteur sont mis à jour ; les paramètres ne bougent pas.
   - Cette pré-passe utilise des pas d'échantillonneur disjoints, par exemple `SeedSequence([seed, 1_000_000+i])`.
   - Chaque bras la fait **avec son propre objectif**, auxiliaire compris.
   - Ensuite, montée linéaire de 0 à 3e-6 sur 500 pas, puis LR constant.
   - On évite ainsi les premiers pas « en signe » d'Adam, où le pas vaut environ ±lr sur chaque coordonnée, scalaires compris.
4. **EMA** initialisée aux poids de départ, demi-vie 500.
5. **Pas de l'échantillonneur à partir de 70 000.** On rejoue les batchs du principal entre 70k et 80k, mais le compteur Adam repart de 0.
6. **Bras témoin A′ obligatoire.** Le redémarrage change la dynamique, et le 70k est un gagnant sélectionné : A′ risque de régresser vers la moyenne. On compare B′ à A′, jamais au 70k seul.

### 8.3 Améliorer le paquet de 150 Mo sans entraînement (test D5)

- **Principe.** Si D1 montre une erreur de niveau de faible dimension, on ajuste sur TRAIN, en forme close, une **correction affine par canal** (5 gains et 5 offsets sur les latents finaux), contre des cibles TT à bruits neufs. Pour C4, on pondère par le jacobien du teacher.
- **Variante minimale.** Le seul `out_gain` gardé en FP32 dans le forward (4 octets), si D0 le désigne.
- **Usage.** Tester sur **tous** les bancs, mondes 202 et 404 compris. On obtient soit une amélioration du paquet à taille constante, soit une borne haute de ce que le terme basse fréquence peut gagner.
- **Ce n'est pas un contournement** : une calibration ajustée uniquement sur TRAIN est un objet explicite, de quelques scalaires. Elle ne change pas le relief tant que le gain reste à ±1 % de 1.

---

## 9. Question 5 : prochain essai minimal, garde-fous et preuves

### 9.1 Phase 0 : forward et CPU uniquement, flags déterministes

| Test | Contenu | Ce qu'il tranche |
|---|---|---|
| **D0** | Scalaires `out_gain` et `emb_gain`, en FP32 et BF16, sur tous les checkpoints, contre les biais des CSV | L'hypothèse §4. Le gel des scalaires est-il nécessaire ? |
| **D1** | Pour EMA70k, A50k, B50k, EMA100k et raw100k, sur TRAIN et VAL (7 scènes de progression, 6 montagnes réservées, mondes 202 et 404) : régressions par pixel ΔC_k ~ C_k (gain) ou ~ constante (offset) ; Δh ~ 2\|q\|(q+31,4) ou ~ 2\|q\| ; biais LOD3/LOD0 **sur la même emprise** ; décomposition biais/reste de 202 et 404 | Nature du niveau : gain latent, offset C4 ou chemin decoder |
| **D2** | Attribution ÉÉ/TÉ/ÉT/TT, avec biais, MAE basse fréquence (σ=8) et biais LOD0, sur 24 captures TRAIN **plus 8 montagnes TRAIN** | Quel étage fixe le niveau, donc gradient S1 ou non |
| **D3** | Sur des champs TRAIN réels : la dérivée exacte §3.1 (L_HP en h selon δ, plate attendue en q) et la corrélation c sous cible même entrée | Démonstration numérique de l'échappatoire et ampleur du lissage de Wiener |
| **D4** | Gradients **par tenseur** (régulier, basse fréquence, côte en q, passe-haut en q) à EMA70k ; signaler tout tenseur au ratio préconditionné > 10 | La direction commune du §2.3 ; calibration de λ |
| **D5** | Calibration affine en forme close (§8.3) | Une amélioration possible sans entraînement |
| **D6** | Planchers FP32/BF16 du teacher sur TRAIN pour biais, basse fréquence, TPI, vallées et HF | τ_k et seuils des garde-fous |

### 9.2 Recette R1 : niveau seul

- **Bras** :
  - A′ : recette principale inchangée, depuis la réinitialisation du §8.2 ;
  - B′ : A′ plus L_LF.
  - Si D0 est positif, ajouter **A′f**, identique à A′ avec scalaires gelés aux valeurs du teacher, et faire de B′ un bras gelé lui aussi. A′f/A′ teste alors causalement l'hypothèse des scalaires, et B′/A′f teste le terme basse fréquence. Deux paires par GPU si on veut paralléliser (4090 et 5090), chaque comparaison restant appariée sur un seul GPU.
- **Commun** : même graine, pas 70 000 → 80 000 (10k par bras), blocs entrelacés de 2 500 pas, mêmes flags déterministes, raw et EMA à chaque bloc, état complet à chaque bloc, K=2 (auxiliaire sur les pas off-policy, comme B).
- **L_LF** :
  - forme du §6.2 ;
  - cible TT à bruits neufs (z0, z1) dérivés de `SeedSequence([seed, step, 0xA0])` par position de fenêtre, comme `refresh_noise`, sur des objets séparés du `QATBank` ;
  - gradient par S2, ou par S1 et S2 avec checkpointing selon D2.
- **Échantillonnage** : la distribution du principal, TRAIN seulement, avec contrôle `HOLDOUT_SEEDS`, plus un tiers stratifié sur les cinq strates critiques (les quatre précédentes et les montagnes humides TRAIN) × trois coarses.
- **Pas de passe-haut ni de côte dans R1.** On isole une seule hypothèse : « le niveau est la cause dominante des défauts de MAE et de côte ».
- **Garde-fous d'implémentation** :
  - gradient auxiliaire **mis à zéro sur tous les scalaires globaux** ;
  - plafond global à 0,5 conservé ;
  - journal par tenseur ;
  - smoke B′ λ=0 ≡ A′, au bit près, sur tout le chemin (pré-passe comprise) ;
  - aucun réglage en cours de run.
- **Calibration** : 32 régions TRAIN dont 8 montagnes, λ = 0,25/médiane(r̃), avec r̃ = max(ratio brut, ratio préconditionné).

### 9.3 Mesures et règle de décision (à pré-enregistrer avant le premier résultat)

- **Panel** :
  - 7 scènes de progression, 12 lieux critiques, 6 montagnes réservées, 7 lieux historiques, **mondes 202 et 404 compris** ;
  - coarse4 et coarse2 ;
  - bruit stocké plus 3 bruits ;
  - LOD3 et LOD0 sur la même emprise ;
  - métriques brutes et à niveau égalisé ;
  - raw et EMA (décision sur l'EMA) ;
  - évaluations à 77,5k et 80k (72,5k et 75k en descriptif).
- **Critère principal** : |biais| moyen par scène, sur les bruits et les deux LOD. B′ doit faire **au moins 40 % de mieux que A′** aux deux points, avec un signe d'écart cohérent sur au moins 70 % des couples scène × bruit.
- **Garde-fous des incisions** (relatifs à A′, et au plancher D6) :
  - rappel des vallées LOD0 à σ = 4/8/16 sur les montagnes : ≥ A′ − 0,01 ;
  - amplitude TPI LOD0 à σ = 8/16 : |B′ − A′| ≤ 0,02 ;
  - corrélation TPI ≥ A′ − 0,005 ;
  - pentes au LOD0 des montagnes dans [0,96 ; 1,03] ;
  - ratio HF LOD0 des plaines ≥ 0,95 × A′ ;
  - ratio HF LOD3 **à niveau égalisé** ≤ 1,05 × A′ ;
  - `channel0_mse` et opérateur de l'étage 1 ≤ 1,10 × A′ ;
  - raccords canoniques batch1 exactement nuls.
- **Ensuite** : planches teacher / EMA70k / A′ / B′ sur les mêmes scènes, mondes 202 et 404 compris, puis verdict visuel du propriétaire.
- **Confirmation finale** : un jeu de graines neuves, jamais regardé, avec les mêmes types de mondes difficiles, utilisé **une seule fois** pour le go/no-go. Les bancs actuels deviennent des bancs de suivi.

| Résultat | Lecture | Suite |
|---|---|---|
| Niveau corrigé, garde-fous tenus | Le niveau était le défaut dominant | Candidat à 150 Mo, puis R2 |
| B′ ≈ A′ sur le niveau | Le niveau n'est pas atteignable par les poids à ce LR et avec ce quantifieur | D5, scalaires en FP32, ou quantifieur (blocs de 256/128, échelles apprises) |
| LOD3 corrigé, LOD0 non | Chemin decoder et C0–C3 | R1b : basse fréquence décodée |
| A′f stabilise le niveau par rapport à A′ | Hypothèse des scalaires démontrée | Gel ou FP32 dans toutes les recettes suivantes |
| Garde-fous violés | λ trop fort ou fuite | Lire le journal par tenseur, réduire λ |

### 9.4 Recette R2 : le grain, après R1

- C4 seul, en q, avec poids du jacobien du teacher détaché, cible même entrée sur bruit neuf.
- Passe-haut **apparié** (bandes σ = 1 et 2), plus une **pénalité unilatérale de déficit** d'énergie par bande et par fenêtre, sur la terre : (1 − E_s/E_t)₊².
  - Le terme apparié retire l'excès de grain.
  - La pénalité de déficit empêche de descendre sous l'énergie du teacher, ce qui bloque l'optimum c² du §3.2.
  - Elle ne récompense jamais un excès d'énergie.
  - Risque : là où l'élève est déjà trop lisse, elle peut ajouter de l'énergie non corrélée. On le surveille par la corrélation TPI.
- Toutes les strates, montagnes comprises. Scalaires exclus. Mêmes garde-fous, plus l'amplitude de C0–C3 au LOD0.
- Variante quasi gratuite : pondérer par bande l'opérateur de l'étage 2 sur C4 dans les batchs principaux. À entrée identique, la haute fréquence de sortie vaut sin t2·HP(Δv_C4) : il n'y a pas de forward auxiliaire. Il faut pondérer par le poids de fusion, puisque la fusion moyenne les fenêtres 2×2.

### 9.5 Taille et vitesse, après R1 et R2

- NF4 poids seuls à 254 M paramètres : environ 150 Mo, et **aucun gain de calcul** possible.
- NF3/NF4 mixte par sensibilité de couche : environ 100 à 120 Mo, sans gain de vitesse non plus.
- **Le seul chemin vers 50–100 Mo avec gain de vitesse** : un élève plus étroit. Par exemple, passer de 192 à 128–144 canaux donne environ 0,45 à 0,56× les paramètres et les FLOPs, soit environ 55 à 85 Mo en NF4. Il faudrait l'initialiser par élagage guidé par le teacher.
- Un élève plus petit aura **plus** de dérive de niveau et de grain. L'objectif corrigé par R1 et R2 est le prérequis. Tous les acquis s'y transfèrent : basse fréquence sur TT, q linéarisé au teacher, scalaires protégés, métriques à niveau égalisé.

---

## 10. Ce que je ne recommande pas

- Prolonger la même recette, ou rechoisir un checkpoint avec un score dominé par l'offset.
- Un passe-haut en h, ou une MSE appariée seule sur une bande mal corrélée.
- Une cible d'énergie seule.
- Comparer un bras neuf au seul 70k.
- Régler λ en cours de run.
- Retirer du jugement les mondes 202 ou 404, ou tout autre cas défavorable.
- Interpréter une baisse du ratio HF en h sans le niveau.

---

## 11. Limites de cette revue

- Statistiques A/B : une scène et un bruit par cas, une seule graine. Les signes (14 sur 14) sont solides ; les amplitudes restent indicatives.
- L'hypothèse des scalaires repose sur le code, sur la colinéarité des cosinus et sur des ordres de grandeur. Je n'ai lu aucune valeur de `out_gain` (test D0).
- Altitudes moyennes des scènes inconnues : les ajustements « gain » ou « offset » ne sont pas résolus (test D1).
- Je n'ai lu ni `b/train.jsonl`, ni les CSV du 70k, ni les tables des bancs finaux par scène. Les causes des mondes 202 et 404 restent ouvertes.
- Les coûts et la mémoire de R1 sont estimés d'après le code et les mesures précédentes, pas mesurés ; le smoke les mesurera.