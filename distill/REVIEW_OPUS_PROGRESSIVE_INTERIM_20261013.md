# Revue du palier 192→160, état à 14k (rapport intermédiaire 6k)

## Verdict

- **Laisser le run 20k aller à son terme sans rien changer.** Je n'ai trouvé aucune erreur dans la projection progressive, les normalisations MP, les skips ou la QKV. La compaction est vérifiée : noyaux exacts, MSE relative 1,1e‑5.
- **La récupération en 160 ne doit pas être une simple prolongation de la recette actuelle.** La loss voit très peu les trois défauts qui comptent visuellement :
  - le grain du LOD3 (la hauteur latente, lue directement par le serveur) ;
  - la forme des côtes ;
  - le niveau des montagnes.

  Je recommande une **modification ciblée** de la loss, puis une récupération en 160 avec des points de contrôle chiffrés. Le passage vers 128 viendra plus tard, après correction de la sélection des canaux.
- **Rien n'indique une limite de capacité.** Les données plaident plutôt contre.
- **Il n'existe pas encore de candidat rapide** : le réseau est encore physiquement en 192. Le compact 20k pourra être montré en aperçu live, avec ses défauts documentés. Il n'est pas intégrable en l'état, à cause des côtes et du grain LOD3.

---

## 1. État mesuré à 14k (coarse4, EMA)

Toutes les valeurs de cette section sont des **faits** lus dans les CSV/JSON. L'étalon est la différence teacher FP32/BF16 sur les mêmes scènes et les mêmes bruits.

### Relief et vallées humides (LOD0)

| Montagne | Amplitude TPI σ2 | Corrélation TPI | Vallées retenues | Biais |
|---|---:|---:|---:|---:|
| 620001 | 0,79 | 0,872 | 0,62 | +11 m |
| 620002 | 0,96 | 0,868 | 0,74 | +143 m |
| 620003 | 0,71 | 0,868 | 0,49 | +191 m |
| Étalon | ≈1,00 | 0,998–0,999 | ≈0,97 | +11 / +17 / +27 m |
| Projeté sans entraînement (retrait 0,216) | 0,13 / 0,22 / 0,25 | 0,65–0,76 | — | — |

- **Organisation conservée.** La corrélation est stable à toutes les échelles (σ2–16) et le cosinus des pentes vaut 0,87–0,93. Les vallées restent visibles sur la planche.
- **Amplitude et vallées dégradées** sur 2 montagnes sur 3 : relief aplati de 21 à 29 %, la moitié des pixels de vallée de 620003 déplacés.
- **L'amplitude progresse pendant que le retrait augmente.** EMA 6k : 0,76 / 0,75 / 0,61. EMA 14k : 0,79 / 0,96 / 0,71. C'est un signal favorable sur la capacité.

### Grain des plaines

- **LOD3, très dégradé.**
  - Ratio HF : désert 4,9×, tempéré 6,2× (étalon 0,89 / 0,95 ; B75 environ 1,5× sur son propre panneau).
  - Corrélation TPI −0,03 / +0,03 (étalon 0,86 / 0,79), cosinus des pentes ≈0. Le grain n'est pas du détail amplifié, c'est du bruit **décorrélé** du teacher.
  - Montagnes LOD3 : 5,2 à 6,2×.
  - Trajectoire en plaine : 2k 1,8–2,0× ; raw 6k 2,9–3,3× ; EMA 6k 4,3–4,9× ; raw 12k 4,7–5,9×.
- **LOD0, pas de grain mais un micro-relief amorti.** Amplitude désert 0,74, tempéré 0,58 ; corrélation 0,91 / 0,78.
- **Le LOD3 est bien visible dans le produit.** Le serveur lit directement le canal 4 latent pour les vues macro (`backend/terrain_server.py:662-668`).

### Côtes

- **Côte humide, coarse4.**
  - Désaccord mer/terre LOD0 : 4,0 % (étalon 0,09 % ; 2k 0,32 %).
  - Amplitude du relief côtier 0,28 (étalon 1,02 ; 2k 0,87), vallées retenues 0,06 (étalon 0,98).
  - Planche : le relief du rivage est aplati.
- **Côte humide, coarse2.** Désaccord LOD0 11,9 % (étalon 0,98 %). Planche : le cap et son détail au large disparaissent.
- **Plaine basse / mer.**
  - Désaccord LOD0 24 % (étalon 3,9 % ; 2k 12 %), avec une MAE de seulement 1,1 m.
  - Planche 14k : le réseau de chenaux et d'îlots du teacher est remplacé par une lagune unique simplifiée. À 2k, il était intact.

### Biais d'altitude (LOD0)

- Plaines : désert −11,7 m, tempéré −4,3 m, côte humide +13,9 m, plaine basse +0,6 m.
- Montagnes : voir le tableau ci-dessus (jusqu'à +191 m).
- La « baisse » du biais moyen entre 12k et 14k vient de 620001. 620003 passe de +153 à +191 m. Le signe varie d'un checkpoint à l'autre (620001 : −21 → +59 → +11).

### Continuité entre tuiles

**Non mesurée** sur ce palier. Les raccords sont prévus au moment de la compaction 20k.

### Vitesse (banc RTX 4090, BF16)

Source : `native-progressive160-harness-graph-4090.json`. Architecture seule, poids non entraînés.

| Configuration | Original | Compact 160 | Gain |
|---|---:|---:|---:|
| Graphe CUDA, batch 16 | 2,023 ms/fenêtre | 1,513 ms/fenêtre | ×1,33–1,34 |
| Graphe CUDA, batch 1 | — | — | ×1,15–1,19 |
| Eager, batch 16 | — | — | ×1,41–1,44 |

- Paramètres : 253,7 M → 178,1 M (−30 %).
- L'application utilise déjà des graphes CUDA par taille de batch, de 1 à 16 (`backend/terrain_cuda_graphs.py:185-213`). Le gain réaliste du base seul se situe donc entre ×1,15 et ×1,34.
- **Estimation** : les poids entraînés ne changent pas le coût d'un calcul dense. La mesure à 20k sera une confirmation, pas une découverte.
- Le gain sur toute la chaîne (decoder, coarse, CPU) n'est pas mesuré.

---

## 2. Bruit tolérable ou perte de structure

**Tolérable, compatible avec votre priorité vitesse :**
- léger amortissement du relief LOD0 en montagne (amplitude ≥0,8), avec une organisation conservée ;
- décalages de niveau de ±10 m en plaine loin de la mer ;
- grain LOD3 de l'ordre de celui de B75 (environ 1,5–2×).

**Perte réelle :**
- réorganisation des côtes basses (chenaux et îlots perdus) ;
- relief côtier écrasé (vallées 0,06) ;
- déplacement du trait de côte en coarse2 (12×–13× l'étalon) ;
- grain LOD3 à 5–6×, décorrélé, visible en vue macro ;
- montagnes surélevées de 140–190 m. Ce dernier point est peut-être peu visible seul, mais il trahit une direction de niveau mal contrainte.

---

## 3. Audit du code

### 3.1 Vérifié correct (faits)

- **Interpolation progressive** (`native_progressive_width.py:43-76`).
  - Interpolation des noyaux effectifs entre la version complète et la version compacte réinjectée. Le gradient remonte vers les coordonnées retenues.
  - La normalisation MP globale (`normalize`, dim=None) et le fan-in recalculés sur le sous-tenseur correspondent exactement au réseau étroit.
  - Gain de sortie ancré.
- **Normalisations** (`:79-87`) : normalisation par pixel de l'encodeur et RMS de l'embedding sur les canaux retenus. C'est cohérent avec `UNetBlock.forward`, qui normalise bien `c` (`unet_block.py:131`).
- **Skips** (`:117-142`) : décalages `skip_indices + x.shape[1]` identiques à `project_state`.
  - `mp_concat` est appliqué avec les effectifs compacts.
  - Les deux groupes perdent 1/6 des canaux, donc les facteurs de concaténation ne changent pas.
- **QKV** (`native_compression.py:91-97`).
  - La disposition (tête, canal, q/k/v) donne bien `out*3+t`.
  - Les têtes sont retenues entières : normalisation par tête et échelle 1/√64 exactes.
  - **Fait** : l'attention n'existe que dans `dec.64x64_in0`. `attn_resolutions=[8,16]` ne correspond jamais aux résolutions 512…64 calculées avec `image_size=512`. La liste des noyaux de `proof.json` le confirme.
- **Contrôles de `proof.json`** : identité exacte à retrait 0, noyaux compacts exacts y compris `attn_qkv` et `attn_proj`, forward compact avec MSE relative 1,1e‑5 et écart maximal 0,027.
- **Clipping** : jamais actif. La norme des gradients reste entre 0,03 et 0,6 pour un seuil de 1.

### 3.2 Écarts concrets dans le code

**A. Sélection par blocs de 64 hors attention** (`native_compression.py:46-52`, et couplage aux têtes en `:95-97`).
- **Fait.** Dès que l'ancien et le nouveau nombre de canaux sont multiples de 64, la sélection se fait par blocs contigus, même sans attention.
  - Niveau 384→320 (aucune attention) : suppression du bloc 128–191 (`ARCHITECTURE.json`). Ce bloc contient les canaux 158 (score 2,72), 170 (2,02) et 151 (1,82). Les 64 canaux de plus faible score (≈0,8–1,0) sont dispersés ailleurs (`channel-calibration.json`, clé « 384 »).
  - Niveau 768→640 : la sélection du flux résiduel de tout le goulot est confondue avec celle des têtes d'attention. Elle est classée avec l'énergie résiduelle, qui ne mesure pas l'importance d'une tête. Les canaux 0–127 sont supprimés.
- **Conséquence** : point de départ moins bon. Il n'y a **pas** d'erreur de forward.
- **Pour 128, l'écart est bloquant** : 320→256 et 640→512 seraient encore sélectionnés par blocs.
- **Correction** : sélection canal par canal du flux résiduel, et sélection séparée des têtes pour l'entrée de `attn_proj` et la sortie de `attn_qkv`. Les deux comptes restent égaux, donc c'est compatible avec l'architecture.
- **Contrôle** : erreur du réseau projeté sans entraînement à retrait 1 avec les deux sélections. Quelques minutes sur la 4090.

**B. Journalisation repliée** (`native_progressive_train.py:109, 127, 132, 154`).
- **Fait.** Les lignes journalisées tous les 100 pas correspondent toujours à un pas impair, donc toujours hors politique. L'auxiliaire y est toujours de type `wet-high-mountain`.
- **Conséquence** : `train.jsonl` ne montre jamais la loss sur l'état courant de l'élève, ni l'auxiliaire en plaine ou sur la côte.
- **Correction** : moyennes sur la fenêtre de 100 pas, ventilées par catégorie.

**C. Point mineur.** Les instantanés bruts sont en BF16. Pour initialiser la récupération depuis les poids bruts, partir des maîtres FP32 de `latest.pt['model']`. L'outil de compaction lit `saved['ema']`, ce qui fonctionne aussi pour les fichiers `raw/` puisqu'ils y stockent les poids bruts.

### 3.3 Déséquilibres de loss

Aucun ne justifie un arrêt. Tous justifient une correction avant la récupération.

1. **La loss principale est quasi aveugle au mètre.**
   - **Fait** : opérateur, gradient et x0 sont normalisés par la variance latente (`native_weight_quant_train.py:26-40` ; variance du canal 4 = 1,06).
   - **Estimation** : pour une plaine vers 300 m, 1 unité latente ≈ 2|q|·38,6 ≈ 1 340 m. Un grain de 10 m vaut donc environ 1e‑5 sur l'opérateur moyen, soit environ 0,02 % de sa valeur actuelle (≈0,05). En montagne à 3 000 m, un grain de 100 m vaut environ 0,2 %.
2. **L'auxiliaire bout-à-bout est filtré basse fréquence et sans terme de côte.**
   - **Fait** : `level_terms` floute à σ=8 sur 64 px (`native_level_objective.py:16-23`). `root_coast` et `auxiliary_terms` existent (`native_auxiliary_training.py:27-37, 66-71`) mais ne sont pas branchés dans ce trainer.
   - **Fait** : l'auxiliaire tourne 1 pas sur 4. Il n'atteint le plafond de 25 % (version préconditionnée Adam) que sur 15 des lignes montagne journalisées, la plupart après 9k.
   - **Fait** : sa perte de niveau en montagne plafonne à 30–170× l'étalon² depuis environ 6–8k.
3. **Le feature KD a dérivé.**
   - **Fait** : calibré à 25 % de l'opérateur (ratio médian 0,70 à retrait 0,15 → poids 0,1749).
   - **Fait** : aujourd'hui, 0,175×0,318 = 0,056 contre un opérateur de 0,054. Le terme pèse donc autant que l'opérateur, soit environ 4× la part prévue.
   - **Hypothèse** : la correspondance fixe des coordonnées freine la redistribution de l'information des canaux supprimés.

### 3.4 Supervision des deux passes

- **Passe 1** : supervisée fenêtre par fenêtre. Son entrée (bruit) est identique en politique et hors politique.
- **Passe 2** : la moitié en DAgger. L'état est la passe 1 de l'élève, fusionnée sur 9 fenêtres, et la cible est le teacher appliqué à cet état (`native_fp8_training.py:35-46`). L'implémentation est correcte.
- **Hypothèse** : à faible bruit, le teacher conserve le grain que la passe 1 de l'élève lui transmet. La cible DAgger ne corrige donc pas ce grain.
- **Conséquence** : la sortie évaluée (TT, toutes fréquences, en mètres) n'est optimisée que par l'auxiliaire basse fréquence.

### 3.5 Retard de l'EMA

- **Fait.** Âge moyen de l'EMA ≈721 pas. Avec la dérivée de la courbe en S, l'écart de retrait vaut ≈0,045 à 6k, 0,052 à 12k et 0,045 à 14k. Il tombe à ≈0,01 à 19k et ≈0 à 20k.
- **Fait.** L'écart raw/EMA porte surtout sur le niveau des montagnes : raw 12k −22 / +104 / +77 m contre EMA 12k +59 / +164 / +153 m. Les plaines et les côtes sont proches.
- **Le retard est plausible mais ne suffit pas.** Pour 620003, le signe du biais EMA ne suit pas la direction de la projection (−150 m sans entraînement).

---

## 4. Explication des défauts

Ce sont des hypothèses. Chacune est accompagnée d'un test.

- **H1. Recette plutôt que capacité.** À 2k, avec seulement 2,8 % de retrait, on voyait déjà : grain LOD3 de 1,8–2,0×, corrélation LOD3 de 0,54–0,68, amplitude LOD0 tempérée de 0,78. Les losses étaient alors de l'ordre de 1e‑4, et Adam fait des pas de taille ~lr quelle que soit la force du signal. **Hypothèse** : les poids dérivent dans les directions que la loss ne voit pas.
  - *Test* : contrôle à retrait 0 (teacher contre lui-même) pendant 2k pas avec la même recette. Environ 1 h sur la 5090. Si le grain apparaît, c'est la recette qui le produit.
- **H2. Origine du grain.** Passe 1 ou passe 2 ?
  - *Test* : chaînes hybrides élève passe 1 + teacher passe 2, et l'inverse, sur les 7 scènes. Sans entraînement.
- **H3. Côtes.** Pas de terme de côte, flou σ8, jacobien de hauteur ≈2 près de 0 m, et environ 15 % de cas rares dans l'échantillonnage. Cohérent avec la perte des chenaux.
- **H4. Niveau des montagnes.** Direction peu contrainte : auxiliaire faible, signe instable, moyenne EMA par-dessus.
- **H5. Risque de quadrillage.** Si le grain est décorrélé entre fenêtres, la fusion linéaire module son amplitude de 1 à 0,5 avec une période de 32 px. Cela peut dessiner une grille en vue macro.
  - *Test* : pic spectral à 1/32 px⁻¹ sur un champ plus large.

---

## 5. Réponses aux questions du rapport 6k

1. **Le calendrier 20k + récupération séparée reste adapté.**
   - L'opérateur (≈0,03–0,07) et les features (≈0,25–0,33) plafonnent depuis environ 7k alors que le retrait passe de 0,3 à 0,8.
   - Les métriques EMA plafonnent entre 10k et 14k et l'amplitude des montagnes progresse.
   - Faits qui pousseraient à corriger plutôt qu'à conclure à une limite de capacité : le contrôle à retrait 0, les chaînes hybrides, le réseau projeté sans entraînement à retrait 1, puis la pente des métriques en mètres sur les 5–10k premiers pas d'une récupération corrigée.
2. **Le retard EMA est une explication partielle.** Je recommande :
   - à 20k, compacter **raw et EMA** et les évaluer tous les deux ;
   - en récupération, redémarrer l'EMA et suivre deux demi-vies (500 et 2 000) ;
   - choisir entre elles sur le panneau d'évaluation.
3. **Pas d'erreur bloquante dans la projection.** Les corrections à faire sont dans §3.2 (A, B) et §3.3 (1–3). Aucune ne justifie de modifier le run en cours.
4. **Recette et critères** : voir §6.

---

## 6. Recommandation

### Maintenant
Ne rien changer au run. Il reste environ 5,5k pas, soit ≈2,7 h à ≈1,76 s/pas.

### À 20k
- Évaluer le réseau projeté sans entraînement à retrait 1, puis raw et EMA compactés.
- Évaluer **B75 sur le même panneau** : il manque une référence produit mesurée avec le même protocole et le même decoder.
- Utiliser 3 bruits par scène.
- Mesurer les raccords prévus, plus le désaccord des fenêtres dans leurs zones de recouvrement et le pic spectral à 32 px.
- Lancer le benchmark aux batchs 1, 4 et 16.
- Faire une **ventilation des temps de la chaîne réelle** (base / decoder / coarse / CPU) sur le teacher, pour estimer le gain réellement atteignable.

### Récupération en 160 avec modifications ciblées
Budget de 20–30k pas, avec points de contrôle à 3k, 6k et 10k.

- **Initialisation** : le meilleur des deux compacts. Taux d'apprentissage : montée sur 500 pas vers 2e‑5, puis décroissance cosinus vers 2e‑6. B75 a été entraîné avec 3e‑5 sans problème.
- **Terme en mètres sur le canal 4**, sur toutes les fréquences, dans l'espace x0 :
  - pondéré par le jacobien 2|q_ref|·38,6 ;
  - appliqué aux deux passes ;
  - visant environ 25–35 % du gradient.

  C'est une MSE par rapport à la cible, pas une loss d'énergie : elle ne récompense pas un grain inventé. L'étalon montre que le LOD3 du teacher est reproductible (corrélation 0,79–0,98), donc cette cible n'est pas chaotique.
- **Auxiliaire bout-à-bout tous les 2 pas** :
  - garder la composante basse fréquence ;
  - ajouter une composante pleine résolution LOD3 en mètres et `root_coast` sur les côtes et plaines basses ;
  - plafond à 0,5.
- **Feature KD** : poids ramené à environ 0,045 (part de 25 %), ou décroissance vers 0 sur 5k pas.
- **Échantillonnage** : environ 20 % de côtes et plaines basses, coarse2 inclus. Garder 25 % de montagnes humides.
- **Journalisation corrigée** (§3.2 B), plus la part de gradient de chaque terme.

### Critères proposés pour un candidat utile
Ces seuils sont mes propositions ; votre jugement visuel prime.

| Zone | Critère |
|---|---|
| Montagnes, LOD0 | corrélation TPI ≥0,93 ; amplitude 0,85–1,15 ; vallées retenues ≥0,75 ; biais ≤±60 m |
| Plaines, LOD3 | ratio HF ≤2 (B75 ≈1,5) ; corrélation TPI ≥0,5 |
| Côte humide, LOD0 | désaccord ≤1 % (≤3 % en coarse2) ; amplitude ≥0,7 |
| Plaine basse, LOD0 | désaccord ≤10 % et réseau de chenaux visiblement conservé |
| Raccords | désaccord en recouvrement ≤1,5× celui du teacher ; aucune grille visible |

**Palliatif à tester en live, sans entraînement** : un léger flou (σ≈1 px) sur le canal 4 latent, uniquement pour les vues macro. Il masquerait le grain décorrélé. Il ne répare pas les côtes.

### Compromis vitesse et suite du programme
Les gains combinés sont des **estimations**.

| Option | Gain sur le base | Risque principal |
|---|---|---|
| 160 récupéré | ×1,15–1,34 ; poids −30 % (NF4 ≈105 Mo) | faible |
| 160 + 2 blocs | ≈×1,5–1,7 | qualité à mesurer |
| 160 + une passe distillée vers la sortie TT | ≈×2,3–2,7 en régime établi ; davantage sur une zone froide isolée | détail fin et raccords |
| 128 (3 blocs, 2 passes) | FLOPs ×2,25 idéal | batch 1 probablement autour de ×1,3–1,4 seulement |

- **Une passe distillée.** Dans la géométrie d'évaluation, un cœur 64² demande 25+9 forwards (`native_decoded.py:37-41`). Les raccords sont le risque, car on perd la seconde fusion. Il faut un contexte élargi et un entraînement à travers la fusion.
- **Pourquoi pas 128 en priorité.** À batch 1, régime fréquent pendant la navigation, le passage à 160 ne donne que ×1,15–1,19 : réduire la largeur s'y convertit mal en vitesse.
- **Ordre recommandé** : 160 récupéré, puis **passe unique distillée** et/ou 2 blocs, puis 128 avec la sélection des canaux corrigée et la recette corrigée dès le départ.
- **Garder 160 à deux passes comme solution de repli.** Le gain ×2 attendu de la largeur seule n'est pas acquis à petit batch.
- **Amdahl** : la ventilation de la chaîne indiquera quand le decoder devient le goulot.

---

## 7. Preuves utilisées

- **Rapports et programme**
  - `ckpt/native-progressive160-v1/{INTERIM_006000, ADDITIONAL_EVIDENCE, MILESTONE_002000, 004000, 010000, 012000, 014000}.md`
  - `self/neural-earth/distill/BASE_PROGRESSIVE_SPEED_20261013.md`
  - `distill/BASE_COMPACT_RELEASE.md`
- **Configuration et entraînement**
  - `ckpt/.../ARCHITECTURE.json`
  - `ckpt/.../preparation-v2/{proof.json, channel-calibration.json}`
  - `ckpt/.../anneal/{contract.json, train.jsonl, status.json}`
- **Code**
  - `distill/native_progressive_{width, prepare, train, evaluate, compact, features}.py`
  - `distill/native_compression.py`, `native_weight_quant_train.py`, `native_level_objective.py`, `native_auxiliary_training.py`, `native_fp8_training.py`, `native_decoded.py`, `native_fusion.py`, `terrain_shape.py`
  - `terrain-diffusion/terrain_diffusion/models/{mp_layers, unet_block, edm_unet}.py`
  - `backend/terrain_server.py:662-688`, `backend/terrain_cuda_graphs.py`
- **Résultats** (dans `_base/`)
  - `base-progressive160{,raw,untrained}-*-coarse{4,2}.{csv,json}` pour 0, 2k, 6k, 12k (raw) et 14k
  - planches PNG lues : 14k coarse4, 14k coarse2, raw 12k coarse4, 2k coarse4
- **Étalon et vitesse**
  - `eval/native-progressive160-precision-floor-v2-4090/{metrics.csv, report.json}`
  - `eval/native-decoded-probe/channel-statistics.json`
  - `bench/native-progressive160-harness-graph-4090.json`

La première tentative de revue a échoué sur une limite d'usage (HTTP 429). La nouvelle tentative programmée, `review/opus-progressive160-interim6000-retry/`, est cette session. Il n'existait donc pas d'avis antérieur à lire.