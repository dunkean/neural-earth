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
