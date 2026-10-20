# Modèles fournis dans Neural Earth

Le menu **NN** fait partie du serveur habituel (`python launch_terrain.py`).
La carte continue démarre avec **coarse8/25k, base original, décodeur64/200k**.
Les trois modèles originaux restent sélectionnables séparément. Coarse2/200k
et décodeur16/10k sont les seules autres variantes de la carte continue.
Les anciens choix coarse4/coarse1 migrent vers coarse8/coarse2 ; les anciens
bases expérimentaux migrent vers l'original. Les réglages valides et le reset
sont conservés. Les choix s'appliquent à tous les onglets du même serveur.

Le lien **Tuile40×40km** ouvre `/tile40/`. Ce mode utilise coarse2/200k et
le **base résiduel48/8k brut**, avec décodeur16/10k par défaut ou décodeur64/200k
pour la qualité. Le résiduel est entraîné pour une tuile finie ; il n'est pas
installé dans le pipeline de carte continue. Seuls le sketch et le terrain
final sont affichés. **Lieu du calcul** propose **Backend Python (serveur)**
par défaut pour comparer rapidement la qualité : CUDA automatique ou CPU,
avec possibilité de forcer l'un des deux. Les modèles restent en mémoire
entre les générations ; changer de décodeur remplace la configuration résidente.
Le choix **Navigateur** utilise un worker sur CPU ou WebGPU, avec repli CPU
en mode automatique. Les deux modes reçoivent le même croquis, climat et seed.

| Poids d'inférence distribués | Sélection |
|---|---|
| `distill/weights/coarse8.pt` | EMA25k, solveur8étapes |
| `distill/weights/coarse2.pt` | EMA200k, solveur2étapes |
| `distill/weights/base-residual8k.pt` | Poids bruts8k, altitude résiduelle |
| `distill/weights/decoder.pt` | Décodeur64, EMA200k |
| `distill/weights/decoder16-10k.pt` | Décodeur16, EMA10k contre l'original |

Le manifest contient étapes, configurations et SHA256. Les paquets ne
contiennent ni optimiseur ni deuxième copie EMA. Aucun autre poids distillé
n'est distribué ; les archives de recherche restent hors Git et ne peuvent
pas ajouter de choix au catalogue. Le constructeur de distribution est
`python -m distill.bundle_selected_models --write`, depuis les checkpoints
externes. L'ancien exporteur coarse1/2 ne peut plus écrire dans Git.

Les trois originaux viennent de `xandergos/terrain-diffusion-30m`, révision
`9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Le chargement normal les télécharge
automatiquement ; on peut les installer ou vérifier séparément :

```bash
python tools/download_models.py
python tools/download_models.py --check
```

Le mode Python utilise directement les cinq poids fournis et la configuration
originale épinglée, sans export ONNX. La première sélection de **Navigateur**
prépare les ONNX localement, avec contrôle
numérique, sans banque d'entraînement ni checkpoint externe. Les fichiers
vont dans `TERRAIN_RUNTIME_ROOT/browser-models/` et sont partagés entre les
deux choix de décodeur. Les paquets complets pèsent42,63Mo avec décodeur16,
55,10Mo avec décodeur64. Une installation peut préparer les exports en avance :

```bash
python -m distill.tile40_prepare
# Serveur autonome du mode40km, sans serveur CUDA de carte continue :
python -m distill.tile40_demo --port 8782
```

Dans le menuNN, **Appliquer les NN** recharge les modèles et conserve la vue.
**Reset après Orogen** invalide seulement les caches neuronaux, en conservant
Orogen, le seed, la caméra et les réglages de génération. Les sélections et
les empreintes participent à l'identité des caches. Le serveur de revue
supervisé reste compatible (`python -m distill.live_server --port 8765`) ;
il utilise le même catalogue et les mêmes routes que le serveur normal.

La qualité du résiduel reste expérimentale. Les temps natifs des réseaux ne
sont pas des temps navigateur et l'intégration n'ajoute aucune nouvelle
acceptation de qualité ou de vitesse.

## Réentraînement résiduel et climats, sous 100 Mo

`distill.tile40_climate_run` prépare la même recette que le meilleur résiduel
48 canaux / brut 8k : flow conditionnel, bruit spatialement corrélé et altitude
résiduelle autour du coarse2 disponible à l'inférence. Le nouveau modèle part
de poids neufs. En gardant cette architecture et des poids FP32, la largeur
maximale multiple de 16 est **112**, soit **24 772 725 paramètres** et environ
**99,12 Mo** par export d'inférence ; la largeur 128 dépasse déjà 129 Mo en
tenseurs seuls. Le plafond strict est 100 000 000 octets, contrôlé sur chaque
export brut/EMA réel. Les états d'entraînement avec Adam et EMA sont plus gros.

La banque reste identique : 960 TRAIN, 30 VAL, 30 TEST réservés. L'échantillonnage
consacre 70 % aux montagnes, conserve collines et plaines et associe deux des
cinq climats au même croquis, crop, bruit initial et temps du flow. L'objectif
de flow reste inchangé. Le contrôleur accepte un budget explicite avec évaluation des 30
tuiles VAL complètes, brut et EMA, tous les 2k pas, avec le même décodeur et
les mêmes huit appels de flow que la référence 8k. Les diagnostics par climat
incluent spectre, direction, pente et variation spectrale par rapport au climat
tempéré pour le même croquis. La sélection exige un progrès des montagnes et
limite les régressions par relief/climat ; elle ne vaut pas acceptation visuelle.

Préparer le plan sans entraîner, puis lancer dans une session ayant accès CUDA
et au stockage externe persistant :

```bash
.venv/bin/python -m distill.tile40_climate_run
# Job durable, sur le GPU choisi ici par --gpu :
.venv/bin/python -m distill.jobs start residual-climate112 --gpu 1 -- \
  .venv/bin/python -m distill.tile40_climate_run --run --steps 40000 \
  --batch-size 8 --evaluation-gpu 0
.venv/bin/python -m distill.jobs status residual-climate112
```

Les sorties vont dans `~/data/distill/tile40-20261020/residual-climate112-v1/` :
contrat, checkpoints reprenables, exports `train/inference/`, comparaisons PNG
des terrains VAL et sélection JSON. Relancer la même commande reprend le même
contrat après inspection d'un arrêt ; une évaluation incomplète doit être
inspectée avant reprise. L'entraîneur conserve sa réserve de 20 Gio de disque.
Sans CUDA, le contrôleur échoue explicitement avant de créer les checkpoints
du nouvel entraînement. Le candidat doit être évalué avant toute distribution.

Avec `--evaluation-gpu 0`, la 5090 entraîne en continu en BF16, avec maîtres
FP32 et EMA, pendant que la 4090 évalue les vrais exports d'inférence. Chaque
batch contient quatre paires de climats. `status.json` suit les deux processus,
`train/status.json` suit la loss et la mémoire CUDA, et `status-watch.json` suit
les évaluations. Le contrôleur arrête les deux processus si l'un échoue. La
reprise choisit le checkpoint complet le plus avancé, même après un arrêt sans
mise à jour de `latest.pt`. Les serveurs de démonstration occupant les GPU sont
arrêtés avant ces calculs ; leurs configurations sont conservées.

Le suivi des scores tourne séparément sur CPU, sans modifier l'entraînement :

```bash
.venv/bin/python -m distill.jobs start residual-climate112-scorewatch --gpu cpu -- \
  .venv/bin/python -m distill.tile40_climate_report \
  --root ~/data/distill/tile40-20261020/residual-climate112-v1 --every 2
```

Un bilan est généré après chaque paire de checkpoints dont les 30 VAL bruts
et les 30 VAL EMA sont terminés : 4k, 8k, 12k, etc. Les fichiers vont dans
`score-reports/point-XXXXXX.{json,txt}` et `score-reports/latest.{json,txt}`.
Le bilan compare score global, montagnes et climats à la référence48/8k ;
un gain positif indique une amélioration, et une distance plus basse est
meilleure. Il indique séparément si les critères de non-régression sont remplis.
Le suiveur émet aussi un dernier bilan en cas d'arrêt entre deux jalons.

Les évaluations réutilisent les bandes FP64 et les statistiques fixes des
références dans `~/data/distill/cache/tile40-reference-metrics-v1/`. Les clés
incluent le contenu du terrain, les versions NumPy/SciPy et le code d'évaluation ;
les bandes stockées sont contrôlées par SHA256 et les entrées abîmées sont
recalculées. Les tests exigent l'égalité exacte de toutes les mesures avec et
sans cache. `--no-reference-metric-cache` permet de refaire le calcul complet.

Le run du 20 octobre a terminé 40 000 pas et ses 40 évaluations VAL
(brut/EMA tous les 2 000 pas). Aucun candidat n'a passé les critères. Le meilleur
score montagnes du 112, à 4k brut, était 0,1299 contre 0,1004 pour le meilleur
**distillé existant48/brut8k**. Ce dernier est le comparateur des bilans ;
le modèle original fournit les terrains cibles. Le pilote de reprise à faible
learning rate n'a pas passé les critères non plus. Résultats complets dans
`residual-climate112-v1/final-result.json` et `final-scores.csv`, sous la racine
externe du 20 octobre. Les modèles distribués restent ceux du catalogue.

L'utilisateur a précisé que le temps d'inférence n'est pas une contrainte :
l'objectif est la qualité de l'original, notamment les montagnes selon le
climat, sous 100 Mo. Le plafond du run112 concernait la base seule ; coarse et
décodeur sont supplémentaires. Les poids existants base48/coarse2/decoder16
pèsent respectivement 18 284 418, 11 216 897 et 860 920 octets, soit
30 362 235 octets ensemble. Un export112 n'établit donc pas un pipeline complet
sous100Mo. Le diagnostic séparé `sampling-quality-unbounded-v1/` compare les
mêmes 30VAL à 4 et 32 intervalles de midpoint, sans réentraînement ; son
`request.json` conserve cette clarification et `result.json` les mesures.

À 32 intervalles, le 112/brut6k atteint 0,1494 au global et 0,1007 sur les
montagnes. Le comparateur48/brut8k à quatre intervalles reste à 0,1654/0,1004.
Les critères par climat ne sont toujours pas satisfaits. Les checkpoints4k
et8k, ainsi que le comparateur48, ont aussi été mesurés à32intervalles.

Un paquet expérimental séparé, `quality-priority-bundle-v1.tar`, contient
le 112/brut6k stocké en FP16, coarse2, decoder16, leurs configurations et un
chargeur Python pour le code installé. Il reste sous100Mo pour l'ensemble des
poids et métadonnées, et non seulement pour la base. Le chargeur reconstruit
la précision FP32 pour le calcul ; aucun objectif de vitesse n'est imposé.
Les 30VAL de cet export donnent 0,1491/0,1000. La petite variation par rapport
aux poids FP32 ne prouve pas un progrès de fidélité. Le chargement a été vérifié
sans téléchargement ni lecture de poids originaux, et une tuile régénérée
par ce chargeur est exactement identique à la tuile évaluée. La qualité de
l'original et la réponse climatique restent non acceptées ; le paquet n'est
pas installé dans le catalogue. Les preuves et les comparaisons sont dans
`quality-priority-bundle-verification.json` et `quality-priority-comparison.csv`,
sous la même racine externe du20octobre. Les deux serveurs de démonstration
ont été relancés et leurs pages répondent HTTP200.

## Base originale sous100Mo,conditionnement climatique conservé

L'utilisateur a ensuite précisé que le plafond concerne **la base seule**, avec
priorité à la qualité de l'original et aux climats, sans plafond de latence.
L'essai `native3-climate-v1` conserve donc l'U-Net MP original complet
(253688389 paramètres), son attention, ses58 conditions et ses deux passes.
Le décodeur original est conservé ; sa taille est hors budget de la base.

`tile40_native3` stocke les convolutions en3bits, les projections d'embedding
en4bits et les projections d'entrée du climat/temps ainsi que les extrémités
sensibles enBF16. Les échelles par bloc256 sont apprises et stockées enFP16.
La normalisation MP originale reste appliquée aux poids bruts reconstruits.
L'export initial complet,manifeste inclus,pèse99215524octets. Le chargeur
vérifie les empreintes et reconstruit les poids BF16 pour le calcul ; ce
format ne prétend pas accélérer l'inférence ni limiter la mémoire de travail.

`tile40_native3_prepare` capture de vrais appels64px des deux passes de
l'original sur12géométries TRAIN de montagnes,4de collines et4de plaines,
chacune sous les cinq climats. Trois géométries VAL distinctes donnent15cas
opérateur ; TEST reste réservé. Les cinq climats partagent le même bruit
initial et le même ordre de fenêtres,vérifiés avant calcul des statistiques.
La première tentative de capture a échoué avant de produire des données à
cause d'un compteur de version PyTorch en `inference_mode` ; la banquev2
utilise `no_grad` et reste séparée de cet essai échoué.

Le QAT part des poids originaux BF16 avec maîtres FP32. Chaque batch contient
deux paires de climats sur le même relief et la même fenêtre,une par passe,
avec70% de montagnes. La loss conserve les positions des détails et ajoute
l'écart de réponse entre climats ; tous ses dénominateurs proviennent de TRAIN.
AdamW ajuste les poids et les échelles avec des taux distincts. Un smokeGPU
a vérifié trois pas réels et l'égalité exacte entre le forward QAT figé et
le modèle rechargé depuis les octets compressés avant le lancement durable.

```bash
.venv/bin/python -m distill.jobs status native3-climate-captures-v2 --compact
.venv/bin/python -m distill.jobs status native3-climate-train --compact
.venv/bin/python -m distill.jobs status native3-climate-eval --compact
```

Les données et résultats restent sous `~/data/distill/tile40-20261020/` :
`native3-climate-captures-v2/`, `native3-original-control-v1/`,
`native3-climate-initial-v1/`, `native3-climate-v1/`. Le budget de l'essai
est6000pas avec checkpoints reprenables et exports bruts/EMA tous les1000pas.
Le premier GPU évalue les30VAL entièrement décodées pendant que la5090
entraîne. `native-selection.json` conserve les scores,les corrélations
spatiales et les différences de terrains entre climats,pas seulement leurs
puissances spectrales. `score-reports/latest.{json,txt}` est renouvelé après
chaque paire de checkpoints évalués. Le suiveur compare aussi au contrôle de
répétition original ; l'ancien distillé48/brut8k reste clairement identifié.
Les critères numériques de proximité à l'original sont distincts de
l'acceptation visuelle. Aucun modèle du catalogue n'est remplacé automatiquement.

Le run est terminé : 6000pas, six checkpoints bruts/EMA et 30VAL physiques
par version. Le meilleur checkpoint est le **brut5000**, sélectionné sur le
critère déclaré combinant morphologie des montagnes et réponse climatique.
La base complète pèse **99215527octets**, manifeste inclus. Les scores VAL
global/montagnes sont0,05573/0,03499, contre0,16536/0,10037 pour l'ancien
distillé48/brut8k. Le contrôle original répété donne0,01031/0,00536 : ces
distances diagnostiques ne constituent pas un pourcentage de qualité.

Trois vérifications supplémentaires n'ont pas amélioré le meilleur résultat :
résidus de rang8 sur les projections de conditionnement,2000pas;
13convolutions finales passées en4bits sans apprentissage;réajustement de cette
précision pendant800pas. Leurs bases approchent99,96Mo mais restent moins bonnes
selon le critère VAL. Les historiques complets restent dans
`native3-climate-residual-v1/`, `native3-output4-v1/` et
`native3-output4-recovery-v1/`. Les maîtres FP32 et états d'optimiseur restent
séparés des paquets d'inférence;aucun surcoût caché n'est requis par le chargeur.

Le brut5000 a été figé avec empreintes **avant** l'ouverture de30TEST réservés,
sur de nouvelles géométries. Preuve : `native3-final-candidate-freeze.json`.
Les résultats TEST donnent0,04742au global et0,03287sur les montagnes.
Les corrélations des détails de montagne par climat vont de0,9642à0,9762.
Les réponses spatiales aux changements de climat restent imparfaites :
erreur relativeRMSE0,328à0,460 et corrélation0,891à0,945. Le seuil numérique
de proximité à l'original reste **non atteint**, en VAL comme en TEST.
Ces deux géométries de montagnes par climat ne prouvent pas une qualité générale.
Aucun entraînement ni nouveau choix de checkpoint n'utilise les résultats TEST.

Les dix comparaisons TEST de montagnes ont été examinées : crêtes et réseaux
de ravines conservés, différences visibles entre froid,tropical,aride,tempéré
et océanique,avec quelques détails fins atténués ou modifiés. Cela ne valide pas
une équivalence complète à l'original. Les mesures et observations textuelles
sont réunies dans `native3-delivery-v1/comparison.html`, avec tableaux et CSV.

Le paquet livré `native3-delivery-v1/native3-base99m-v1.tar` pèse99235840octets
sans compression d'archive. Il contient les poids,le manifeste,un chargeur
indépendant des poids parents,la provenance et la licence amont. Le dossier
complet représente99221916octets. Le chargeur requiert l'architecture
`terrain_diffusion` et les dépendances installées;il reconstruit les poids BF16.
Les58conditions et les deux passes restent celles de l'original. Le décodeur
original et coarse2 sont supplémentaires;le plafond concerne le stockage
de la base,ni sa mémoire de travail ni l'ensemble du pipeline.

`native3-final-result.json` conserve la conclusion non acceptée et la revue
visuelle;`native3-delivery-v1/inventory.json` authentifie tous les fichiers.
Les démonstrations8765et8782 ont été restaurées,HTTP200,avec leur sélection
existante. Le nouveau modèle reste disponible comme paquet expérimental.

Validation finale :19tests ciblés réussis,`git diff --check` réussi.
Le chargeur livré a reconstruit tous les tenseurs et produit un forward64px
exactement identique au modèle évalué,surRTX5090,avec lecture des poids parents
et téléchargement explicitement interdits. Preuve dans
`native3-delivery-v1/loader-verification.json`.
