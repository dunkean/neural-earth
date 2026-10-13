# Addendum gradients NF4 : analyse et corrections du protocole

J'ai lu le nouveau rapport et le script de la sonde, `distill/native_field_gradient_probe.py`, pour savoir exactement ce qui a été mesuré. Je n'ai rien écrit ni lancé, et je n'ai utilisé aucun outil GPU. Le run principal à 100k n'est pas touché.

## 0. Ce qui change par rapport à ma revue

1. **Les échelles brutes sont incompatibles entre elles.** À poids 1, le gradient du terme plaine vaut 6 600 à 25 400 fois le gradient régulier, et celui de la côte 0,06 à 13 fois. Les poids utiles sont donc de l'ordre de 1e-5 pour la plaine et 1e-2 pour la côte. Les ratios varient d'une scène à l'autre d'un facteur 4 (plaine) à 200 (côte). Une médiane seule ne peut pas borner chaque cas : il faut un **plafond appliqué à chaque pas**.
2. **Le gradient régulier est petit (0,06 à 0,18), donc le clip à 1,0 ne joue jamais aujourd'hui.** Un auxiliaire mal dosé l'activerait, et B deviendrait une autre recette que A.
3. **Les cosinus ne sont pas interprétables sans une référence.** Ils vont de +0,15 à +0,66, sauf la côte en plaine tempérée à −0,62. Il manque le cosinus entre deux batchs réguliers indépendants : sans lui, ces valeurs peuvent refléter une simple direction commune à tous les gradients.
4. **La mémoire ne pousse plus vers l'étage 2 seul.** La trajectoire complète (25+9 fenêtres) atteint 9,2 Go au pic. Ce pic ne compte ni Adam ni l'EMA, mais il inclut trois jeux de gradients. En entraînement, j'estime environ 11 à 12 Go alloués.
5. **La distillation de trajectoire complète est bien posée** si les deux étages teacher reçoivent le même bruit neuf que l'élève. La compensation n'est pas invalide en soi ; je précise et corrige mes réserves au §3.
6. **Le primaire n'est pas un réplicat**, même informel au sens strict. Le seuil de bruit doit être mesuré sur la 4090.
7. **Nouveau risque d'implémentation :** `refresh_noise` modifie sur place les régions mises en cache par `QATBank` (§6).

---

## 1. Ce que la sonde mesure exactement

| Élément | Constat dans le code | Conséquence |
|---|---|---|
| Poids | `load_weight_quant(saved)` charge l'**EMA15k** comme maîtres FP32 (l.45) | Les gradients sont pris sur les poids EMA, pas sur les raw. Or l'entraînement calcule ses gradients sur les raw : il faut recalibrer sur le **raw 40k**. |
| Batch régulier | Un batch de 8 fenêtres par région, avec un générateur à part (`2026101259+index`, l.56). On-policy pour les index pairs (l.58) : désert et côte découpée on-policy, tempérée et plaine/mer off-policy | Un seul batch par cas. Le cosinus négatif de la tempérée vient d'un batch off-policy, donc l'on-policy ne l'explique pas. Avec n=4, on ne peut rien séparer. |
| Auxiliaire | `decoded_region_forward(..., checkpointed=True)` sur les entrées et le bruit **stockés**, contre `region.target` stocké (l.66-68) | Trajectoire complète appariée au bruit stocké. KL dans l'espace h, sans minimum par classe, poids 1. Ce n'est pas la configuration prévue. |
| Normes | Sur tous les paramètres, y compris les tenseurs sensibles non quantifiés | Le ratio global peut cacher des tenseurs où l'auxiliaire domine. |
| `peak_bytes` | `base_grads`, `coast_grads` (calculés avec `retain_graph`) et `flat_grads` sont **tous trois en mémoire** au pic (l.60-70), ≈ 3 Go en FP32. Ni Adam ni EMA | Voir le §5. |
| Flags | `cudnn.benchmark=True` (l.38) | Les tailles d'espace de travail pourront différer en mode déterministe. |

---

## 2. Lecture des chiffres

### 2.1 Ratios par cas et poids qui donneraient 25 % du gradient régulier

| Cas | ‖g_côte‖/‖g_rég‖ | ‖g_plaine‖/‖g_rég‖ | λ_côte pour 0,25 | λ_plaine pour 0,25 |
|---|---|---|---|---|
| Désert c4 (on-policy) | 0,061 | 6 588 | 4,1 *(terme quasi inactif, perte 2e-4)* | 3,8e-5 |
| Tempérée c2 (off-policy) | 4,14 | 7 344 | 0,060 | 3,4e-5 |
| Côte découpée c4 (on-policy) | 11,2 | — | 0,022 | — |
| Plaine/mer c2 (off-policy) | 13,0 | 25 407 | 0,019 | 9,8e-6 |

- **Un poids plaine de 0,01** donnerait 66×, 73× et 254× le gradient régulier. Votre estimation « environ 100× » est juste.
- **Avec un poids plaine fixé à la médiane** (3,4e-5), la plaine/mer recevrait 0,86 fois le gradient régulier, soit 3,5 fois la cible. Pour la côte, la médiane des 4 cas (0,041) donnerait 0,54 à la plaine/mer et 0,46 à la côte découpée. Si on exclut le désert, inactif, on obtient 0,29 et 0,25.
- **Ces chiffres sont seulement indicatifs.** Ils viennent de l'EMA15k, d'une KL dans l'espace h, de la cible stockée, et de 4 échantillons.

### 2.2 Pourquoi le terme plaine est si grand

C'est une question d'unités, pas d'importance du terme.
- La hauteur vaut h = q|q| avec q = 38,6·C4 − 31,4, donc dh/dC4 = 77,2·|q|. Pour une plaine vers 300 m (|q| ≈ 17), cela fait environ 1 300 m par unité de C4.
- Le terme se mesure en m², puis est divisé par 2,3 m², voire 0,055 m² pour la plaine/mer.
- La loss régulière, elle, est normalisée par la variance latente.

Le rapport gradient/perte le confirme : environ 10 à 20 pour la loss régulière, 170 à 250 pour le terme plaine.

Les pertes plaine (3,9 à 7,0 en unités normalisées) correspondent à une erreur passe-haut appariée de 4 à 7 fois la puissance moyenne du teacher. C'est cohérent avec un ratio HF LOD3 d'environ 2, plus la divergence de trajectoire.

### 2.3 Cosinus

- **Pour les lire, il faut deux références**, à mesurer aux mêmes poids :
  - le cosinus entre deux batchs réguliers indépendants : s'il vaut lui aussi 0,3 à 0,6, les cosinus auxiliaire/régulier ne font que refléter un gradient moyen commun ;
  - le cosinus entre deux régions auxiliaires de la même catégorie, pour savoir si la direction auxiliaire est stable.
- **Côte en plaine tempérée (−0,62, avec un gradient 4 fois le régulier).** C'est suspect pour une plaine. Deux causes probables : une classe de quelques pixels qui pèse autant qu'une grande (pas de minimum par classe), ou l'écart de trajectoire face à la cible stockée. Le passage à l'espace q avec 16 pixels minimum peut faire disparaître ce cas ou le transformer : il faut refaire la sonde.
- **Un cosinus élevé et persistant entre gradients indépendants serait une signature importante.** Il indiquerait une direction que l'objectif réclame mais que le forward NF4 ne parvient pas à suivre : plateau limité par le STE, le quantifieur ou le LR. Le KD sur 37 blocs, qui ne peut pas atteindre zéro en NF4, pourrait entretenir une telle poussée constante. C'est une hypothèse, testable au §7.

### 2.4 Clip

- La norme régulière mesurée vaut 0,06 à 0,18, très en dessous du clip à 1,0 (`resume.py:113`). Les `grad_norm` de `train.jsonl` à 30-40k permettent de vérifier sur les poids raw.
- Si la somme dépassait 1, le clip réduirait aussi la composante principale. Ce serait un facteur de confusion propre à B.
- Avec le plafond du §4, la norme totale reste au plus 1,5 fois la norme régulière, soit environ 0,3 au maximum : le clip reste inactif. Il faut quand même le journaliser.

---

## 3. La cible de trajectoire et la compensation

**Réponse directe : je ne considère pas la compensation comme invalide en soi.** Avec le même bruit neuf (noise0, noise1), la même condition, la même variante de coarse et les mêmes labels pour les deux étages teacher et élève, la sortie finale du teacher est une fonction déterministe des entrées. C'est exactement ce que l'échantillonneur à deux pas de l'élève doit reproduire.
- **La cible est cohérente :** le teacher l'atteint avec une perte de trajectoire nulle et une perte opérateur nulle. Les deux objectifs n'entrent en tension qu'à cause de la contrainte NF4.
- **Le bon critère est la sortie finale :** l'objet déployé est l'échantillonneur, pas l'opérateur. Compenser l'erreur NF4 en amont est donc légitime.

**Ce que je corrige dans ma revue.** J'ai écrit que la cible de trajectoire « apprend une autre fonction que le teacher » et « pousse l'étage 1 vers des états non-teacher ». C'est exact, mais ce n'est pas un défaut : un élève NF4 ne peut pas être le teacher. Ce qui reste, ce sont des points d'ingénierie et de mesure.

| Réserve | Statut |
|---|---|
| Bruit stocké figé, avec risque de corrections propres à ce bruit | **Levée**, à condition de recalculer les deux étages teacher sur le bruit neuf à chaque pas auxiliaire. `region.target` sert seulement à l'attribution et à l'évaluation au bruit stocké. |
| Écrasement de l'objectif principal | **Réel**, comme le montrent les mesures : c'est l'échelle (§4), pas la cible, qui pose problème. |
| Tension avec l'opérateur on-policy | **Réelle mais attendue.** Sur les pas on-policy (50 % des pas principaux), la loss opérateur demande v_s = v_t à l'état de l'élève, donc elle pénalise exactement la compensation que demande l'auxiliaire. L'équilibre dépend de λ et du plafond : on doit s'attendre à une compensation partielle. Il faut mesurer les cosinus séparément pour les batchs on-policy et off-policy. |
| Dérive de l'étage 1 (pré-compensation) | Permise par l'objectif. L'ancrage par l'opérateur et le KD est une hypothèse de protection, pas une garantie. **Nouveau garde-fou** : erreur opérateur de l'étage 1 et `channel0_mse`. |
| Variance | Une région par pas, 34 fenêtres et un bruit neuf : variance élevée. Le panel TRAIN la mesure. |
| Seul C4 est supervisé | La compensation n'agit que sur ce que voit le LOD3. L'effet au LOD0 reste à mesurer, non présumé. |

**L'attribution reste utile, mais pour un autre motif.** Ce n'est plus un argument mémoire : elle sert à choisir entre coût et variance.
- Définition : part_étage1 = 1 − E(TÉ)/E(ÉÉ), sur le désaccord LOD3 près de la côte et sur |log HF| au LOD3. ÉÉ = élève sur les deux étages ; TÉ = étage 1 teacher, étage 2 élève.
- Si part_étage1 ≥ 0,2 sur l'un ou l'autre critère : **B = trajectoire complète avec bruit neuf. C'est désormais le défaut.**
- Sinon : B = étage 2 seul, cible même entrée. Elle est équivalente dans ce régime, moins chère (environ 2,5 fois moins par pas auxiliaire) et moins variable.

---

## 4. Calibration corrigée

**Point de départ.** Le **raw** du `latest.pt` complet à 40k. Pas l'EMA15k : les gradients d'entraînement sont pris sur les poids raw, et l'état Adam est nécessaire.

**Échantillon.**
- 32 régions TRAIN : 8 par catégorie, variantes de coarse équilibrées, chacune avec un bruit neuf dans la configuration finale (espace q, minimum 16 pixels, cible retenue).
- Chaque région est appariée à **deux** batchs principaux, un on-policy et un off-policy, tirés avec le vrai `SeedSequence([seed, step])` à des indices de step disjoints de la fenêtre d'entraînement (par exemple S+50 000 et suivants).

**Mesures par région.**
- ‖g_rég‖, ‖g_côte‖ et ‖g_plaine‖, chacun par un backward séparé.
- Leurs cosinus.
- Les normes **préconditionnées par Adam**, avec P = 1/(√v̂ + ε) pris dans le checkpoint : ‖P⊙g‖. Adam décide du pas par paramètre ; un auxiliaire concentré sur des paramètres où le gradient principal est faible a donc plus d'effet que ne le dit le ratio brut.
- Les ratios par tenseur, à signaler s'ils dépassent 10, surtout pour `out_conv`, la première convolution, les embeddings et les gains.
- Les pertes et le nombre de pixels par classe.

**Références.** 16 paires de cosinus régulier/régulier, et des paires auxiliaire/auxiliaire au sein de chaque catégorie.

**Poids fixes.** λ_k = 0,25 / médiane_i(r̃_k,i), avec r̃ = max(ratio brut, ratio préconditionné). La médiane porte uniquement sur les régions où le terme est actif.

**Plafond appliqué à chaque pas**, qui est la seule vraie borne par cas :
- `g = g_rég + s · g_aux`, où `g_aux = ∇(λ_c·L_côte + λ_f·L_plaine)` est calculé en un seul backward ;
- `s = min(1, 0,5·N̄_rég/‖g_aux‖, 0,5·N̄ᴾ_rég/‖P⊙g_aux‖)` ;
- N̄ est une moyenne glissante (demi-vie 50 pas) des normes régulières, brutes et préconditionnées, initialisée à la médiane de calibration. On évite ainsi qu'un seul batch régulier faible plafonne trop fort.

**Pourquoi cette combinaison.**
- Un quantile seul, par exemple le 90e percentile sur 32 cas, repose sur 3 échantillons et ne borne pas les cas jamais vus.
- Une normalisation pure (s = 0,25·N̄/‖g_aux‖ à chaque pas) amplifierait les auxiliaires presque nuls, comme la côte en désert, et donc leur bruit.
- La médiane fixe la force typique sans amplifier les cas faibles, et le plafond borne les cas forts, comme la plaine/mer.

**Ce qu'on journalise** à chaque pas auxiliaire : s, les quatre normes, cos(g_rég, g_aux), l'activation du clip, et le taux de pas où le plafond s'applique. **Pas de réglage en cours de run.** Si le plafond s'applique sur plus de 50 % des pas, on le signale : on teste alors un « auxiliaire normalisé à 0,5 », ce qui reste un objectif défini.

**Mise en œuvre.**
1. Backward régulier dans `.grad`.
2. Calcul des normes, puis copie dans un tampon (+1 Go FP32).
3. Backward auxiliaire.
4. Combinaison avec le facteur s, puis le clip habituel et le pas d'optimisation.

---

## 5. Mémoire et coût

**Mémoire.** Le pic mesuré est de 9,19 Go. Ajustement pour l'entraînement : on retire deux des trois jeux de gradients (−2,03 Go), on ajoute Adam m,v (+2,03), l'EMA FP32 (+1,01) et le tampon (+1,01). On obtient **environ 11,2 Go alloués**, plus les forwards teacher du bruit neuf, transitoires et sans gradient. Avec la mémoire réservée, la fragmentation et le contexte CUDA, il faut compter environ 13 à 15 Go, pour environ 19,5 Go disponibles. Le smoke doit mesurer `max_memory_reserved` avec l'état complet et les flags déterministes, avec au moins 3 Go de marge.

**Coût par pas** (estimation par comptage des passes, non mesurée) :

| Pas | Équivalents forward |
|---|---|
| Principal off-policy | ≈ 32 |
| Principal on-policy | ≈ 68 |
| Auxiliaire, trajectoire complète (34 fenêtres élève avec gradient et recalcul, 34 fenêtres teacher) | ≈ 170 |
| Auxiliaire, étage 2 seul | ≈ 61 à 70 |

Rapport entre B en trajectoire complète et A :
- auxiliaire un pas sur deux (K=2) : environ 2,7 ;
- auxiliaire un pas sur quatre (K=4) : environ 1,85.

**Règle :** choisir K **à partir du débit mesuré au smoke et avant tout résultat**. K=2 si les deux bras de 10 000 pas tiennent dans la fenêtre disponible, sinon K=4. On ne remonte jamais ρ pour compenser un K plus grand.

---

## 6. Autres corrections au protocole

- **Réplicat.** Je retire « le primaire comme réplicat gratuit » : GPU différent, `benchmark=True`, arrondis différents. Il ne reste qu'une référence descriptive grossière.
  - Sur la 4090 en mode déterministe, relancer A donnerait un résultat identique au bit près, donc aucune information.
  - Seuil de bruit minimal : la variation de A et de B entre S+5k, S+7,5k et S+10k, plus l'évaluation VAL sur plusieurs bruits.
  - Réplicat honnête facultatif : **A′ = A avec une autre graine de sampler**. Règle fixée d'avance : si l'effet se situe entre 1 et 3 fois l'écart-type temporel, on lance A′ avant de conclure.
- **Mutation des régions en cache (risque de confusion grave).** `DecodedRegion.refresh_noise` écrase `stages[*]['inputs'/'target']`, `noise`, `teacher_initial`, `target` et `decoder_noise` (`native_decoded.py:81-102`). Or `QATBank` met ces objets en cache et y lit les entrées des batchs principaux (`native_fp8_training.py:18-48`). Si l'auxiliaire rafraîchit une région partagée, **les batchs principaux de B changent**.
  - Il faut une instance séparée, ou une fonction qui ne modifie rien.
  - Le test « B avec λ=0 identique à A » doit exécuter tout le chemin auxiliaire, rafraîchissement compris, pour détecter ce problème.
- **Bruit neuf.** Le construire par position de fenêtre, comme `refresh_noise` : `gaussian_noise_patch(s+5819, y, x, …)` pour les 25 fenêtres d'initialisation et `s+5820` pour les 9 de raffinement, avec `s` dérivé de `SeedSequence([seed, step, 0xA0])`. Teacher et élève reçoivent exactement les mêmes tenseurs.
- **Nouveaux garde-fous** (aucun ne déclenche d'arrêt anticipé) :
  - `channel0_mse` d'évaluation ;
  - erreur opérateur de l'étage 1 sur le panel TRAIN : B ≤ A × 1,10 ;
  - cosinus moyen glissant cos(g_rég, g_aux), pour l'interprétation.
- **Cache, disque et archives.** J'approuve votre plan, avec quatre précautions :
  - un cache LRU borné sur des chargements déterministes ne doit pas changer le tirage ;
  - ne rien archiver dans le répertoire du primaire pendant que son watch tourne : il ne garde la liste des steps traités qu'en mémoire et réévaluerait tout s'il redémarrait ;
  - vérifier le SHA de la copie avant toute suppression ;
  - garder la copie 40k et son SHA hors des archives.
- **Code.** Les 8 fichiers dont le SHA est vérifié par le contrat restent inchangés. Tout le nouveau code va dans de nouveaux fichiers, comme le prototype actuel.

---

## 7. Diagnostics devenus possibles

1. **Rapport signal/bruit d'Adam au 40k** (CPU uniquement). Pour chaque tenseur, calculer ‖m‖/‖√v‖ à partir de `exp_avg` et `exp_avg_sq`.
   - Pour un bruit pur, avec β1=0,9, la valeur attendue est √(0,1/1,9) ≈ 0,23. Pour un gradient constant, elle vaut 1.
   - Une valeur nettement supérieure à 0,23 sur de nombreux tenseurs, avec une référence régulier/régulier élevée et peu de changements de codes NF4, indiquerait **une direction persistante non réalisée par les codes** : plateau limité par le quantifieur ou le LR. L'auxiliaire risquerait alors de buter sur le même mur, et l'ablation pertinente suivante serait le LR ou le quantifieur.
   - Une valeur proche de 0,23, avec une référence proche de 0, indiquerait un objectif convergé dominé par le bruit : l'auxiliaire a de la marge.
2. **Référence régulier/régulier et cosinus on-policy/off-policy**, mesurés pendant la calibration.
3. **Attribution par étage** (ÉÉ, TÉ, ÉT) sur les mêmes 32 régions, au bruit stocké contre `region.target`, en forward seulement : on mutualise les passes.

**Ordre de la phase 0 :**
1. Contrôle BF16.
2. Déterminisme de l'évaluation.
3. Rapport signal/bruit d'Adam et changements de codes (CPU).
4. Une seule session GPU de sonde et calibration avec la configuration finale : raw 40k, espace q, 16 pixels minimum, bruit neuf, les deux cibles, les références, l'attribution.
5. Choix de la cible selon la règle du §3.
6. Smoke avec λ=0 sur tout le chemin, puis mesure de la mémoire et du débit, puis choix de K.

---

## 8. Ce qui ne change pas

- Deux bras appariés sur la même 4090.
- Même source : raw, EMA, Adam et RNG restaurés exactement.
- Même graine de sampler, LR hérité constant de 3e-6, EMA de demi-vie 500, clip à 1,0.
- Blocs entrelacés de 2 500 pas, 10 000 pas par bras.
- Évaluation EMA et raw sur le panel VAL multi-bruits ; mondes 101, 303 et 404 jamais utilisés.
- Critères E1/E2 (visés), E3 (transfert au LOD0, non présumé) et E4, garde-fous montagne et HF inchangés.
- Planches A, B et teacher, puis votre verdict visuel.
- Rappel : ce proxy ne corrige pas l'offset du désert et de la plaine tempérée.

---

## 9. Limites

- **n=4** : un batch et une région par catégorie. Aucune statistique fiable ; les ratios et cosinus ci-dessus sont indicatifs.
- Toutes les mesures portent sur l'**EMA15k, l'espace h, sans minimum par classe et avec la cible stockée**. Les quatre éléments changent dans la configuration finale.
- Les estimations de mémoire et de coût viennent du code et du pic mesuré, pas d'un smoke avec l'état complet.
- La signature « direction persistante » est une hypothèse, que les diagnostics du §7 peuvent confirmer ou réfuter.
- Je n'ai lu ni `train.jsonl`, ni la sortie JSON brute de la sonde au-delà du rapport, ni l'état Adam du 40k.