# Démo navigateur : croquis → tuile de40km

La démo calcule les1334×1334altitudes dans un worker du navigateur. Le serveur
local ne fait que servir le code et les graphesONNX. Le croquis contient16×16
altitudes en mètres et couvre exactement40×40km. Il n'est pas normalisé par tuile.

Depuis le dépôt :

```bash
source distill/env.sh
python -m distill.tile40_demo --models /home/delete/data/distill/tile40-20261014/browser-bundle-encoder20k-decoder8-original2k-v1
```

Ouvrir `http://127.0.0.1:8780/`. Choisir une plaine, des collines, un massif,
une crête, une vallée ou un plateau, puis modifier les altitudes au pinceau.
Les champs ligne/colonne permettent aussi l'édition au clavier. Sélectionner
un climat et une graine, puis générer. Le calcul automatique essaieWebGPU
et utiliseWASM surCPU si nécessaire. Le modeCPU peut être choisi explicitement.
Les en-têtesCOOP/COEP du serveur permettent les threadsWASM ; un hébergement
ultérieur devra les conserver pour obtenir les mêmes conditionsCPU.

Seul le relief décodé final est affiché, avec un ombrage et une exagération1,5
identiques entre terrains. L'image n'est pas contrastée automatiquement selon
sa propre amplitude. Les altitudes et statistiques accompagnent l'image.
Les sorties téléchargeables sont lePNG, les hauteursFloat32 little-endian
en ordre ligne/colonne (`terrain40.f32`) et les paramètres/provenance enJSON.
LeJSON fournit les coordonnées et le pas exact40000/1334m ; conserver les
deux fichiers pour réutiliser leDEM. Un nouveau calcul conserve les inputs
capturés au clic pour que l'export corresponde exactement à son résultat.

Ce bundle est un prototype : sa qualité n'est pas acceptée. Il a passé une
mesure complèteCPUWASM8threads dansEdgeWindows sur un casVAL montagne aride,
avec5,46–5,69s à chaud et32,64Mo de graphes au total. Ces temps n'incluent
pas le téléchargement initial et ne couvrent pas tous les terrains/ordinateurs.
La démo affiche séparément le temps d'inférence et le temps incluant le
chargement. Les six formes de croquis servent aussi à éprouver les entrées
hors du petit jeu pilote ; elles ne constituent pas six qualités déjà validées.

La capture réelle et les exports d'un croquis de crête modifié sous climat
froid sont vérifiés par `webgpu/tile40_demo_probe.cjs` viaPlaywright.
La route du serveur n'expose ni latents teacher ni hauteur de référence.
Les identités des poids et du runtime sont vérifiées avant le démarrage.
Les fichiers de modèles et de captures restent horsGit.
