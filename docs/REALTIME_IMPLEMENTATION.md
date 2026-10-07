# Optimisations temps réel — état de l’implémentation

7 octobre 2026. Machine vérifiée : RTX 3090, PyTorch 2.11 + CUDA 12.8, modèle 30 m en BF16. Les 20 étapes du solveur coarse, les deux étapes de fusion et les cinq sources de conditionnement du modèle sont conservées. Le profil `natural` conserve l'entrée originale ; le profil `earth` fournit une nouvelle organisation continentale et climatique.

## Navigation et cache

- La caméra s’affiche via `requestAnimationFrame`, indépendamment de la génération. Les besoins de caméra sont publiés au maximum toutes les 50 ms.
- Les abonnements de caméra portent une session et un epoch. Le serveur partage les calculs identiques entre clients, retire les travaux abandonnés en attente et donne la priorité à la couverture puis aux détails visibles. Le préchargement prédit le déplacement et reste limité.
- Les tuiles parentes restent affichées pendant l’affinement. Le cache navigateur et les textures WebGPU ont chacun un budget de 96 Mio. Les très grandes fenêtres adaptent explicitement le LOD à leur budget.
- Un cache physique FP32 avec halo de 24 pixels sert au rendu WebGPU et au PNG. L’écriture sur disque et le rendu PNG se font hors du verrou d’inférence. Le profil numérique entre dans l’identité du cache.
- Le cache disque du profil actif a un quota LRU groupé de 16 Gio, configurable par `TERRAIN_DISK_CACHE_GIB`. Les fichiers utilisés par un calcul ou une réponse restent protégés, y compris pendant l’envoi PNG sous Windows. Le quota peut être dépassé temporairement tant que tous les groupes candidats sont protégés. Les exports initiaux, poids et anciens profils sont conservés.
- Les vues macro du profil `earth` échantillonnent directement le conditionnement continental, sans calcul neuronal planétaire. Le profil `natural` conserve l'échantillonnage des champs appris par petits blocs : son calcul à grande échelle reste coûteux.

## Calcul et rendu GPU

Les poids normalisés des couches MP sont conservés une fois en évaluation, avec invalidation par version des paramètres. Les embeddings coarse constants sont réutilisés et les lectures scalaires synchrones du solveur sont supprimées. Les fenêtres intermédiaires, leur fusion, le décodage Laplacien et le calcul du climat restent sur CUDA ; seule la tuile finale nécessaire à l’affichage est transférée.

Dans le navigateur, WebGPU calcule deux filtres gaussiens, le hillshade et la palette une fois à l’arrivée d’une tuile. Les textures restent résidentes pour le pan et le zoom, sans readback par image. Le bouton **Rendu GPU** permet de comparer avec PNG/Canvas ; une perte du GPU déclenche ce repli.

Le profil mondial `earth`, les palettes climatiques et la correction de l'historique des LOD sont décrits dans [WORLD_GENERATION.md](WORLD_GENERATION.md). Le rendu utilise les cinq champs climatiques, y compris les deux variabilités, pour sa classification heuristique. Les résolutions sources réelles sont indiquées : coarse 7 680 m, latents 240 m et décodeur 30 m. Interpoler une tuile coarse à 480 m ne crée pas de détails neuronaux à 480 m.

Ce chemin utilise encore l’agent Python/CUDA pour les NN. L’inférence NN entièrement WebGPU et les moteurs natifs C++/Rust décrits dans le design ne sont pas encore livrés.

Les CUDA Graphs sont activés pour le coarse et les petits batches base (jusqu’à 4). Le décodeur et les gros batches base gardent leur chemin classique. Les constantes des couches MP et de l’attention sont conservées pour éviter des transferts CPU pendant la capture. Chaque bucket possède ses entrées statiques et rend une copie de sa sortie ; un replay suivant ne peut pas écraser le résultat précédent. Les buckets sont bornés et le profil de cache inclut les options de graphs. Une comparaison exacte à la première capture sur le GPU courant bloque un replay divergent et déclenche un repli visible dans les diagnostics.

## Mesures de la première tranche

| Cas | Référence | Optimisé | Portée |
| --- | ---: | ---: | --- |
| Forward coarse | — | ×1,59 | Médiane de 8 essais, poids MP en cache ; sorties réseau identiques |
| Forward base | — | ×1,62 | Même protocole |
| Forward decoder | — | ×1,18 | Même protocole |
| Crop froid 512² + cinq sorties climat | 3,440 s | 2,529 s | Poids résidents ; un crop par profil, hors rendu/disque |
| Même crop déjà en cache | 27,5 ms | 9,0 ms | Assemblage CPU remplacé par CUDA |
| Pic mémoire allouée PyTorch | 1,08 Gio | 1,68 Gio | Ce harness ; exclut bureau, navigateur et mémoire réservée |

L’optimisation avec fenêtres CPU reproduit exactement la référence. L’assemblage CUDA change légèrement l’arrondi FP32 : écart maximal observé de 0,00269 m sur le relief et de 0,000244 sur les sorties climat. Les cinq canaux sont contrôlés séparément. Les seeds 42 et 0 se reproduisent exactement après changement et retour de seed ; le cas seed 0 corrige aussi un tirage aléatoire upstream.

Les batches coarse 2/4 et decoder 2/4 ont été **rejetés** après comparaison numérique et ne peuvent pas être activés dans le profil de production. Les batches latents et les limites mémoire sont configurés selon la mémoire disponible ; aucun gain sur les 4090/5090 n’est annoncé sans mesure sur ces machines.

`TERRAIN_CUDA_DEVICE=auto` sélectionne une carte CUDA visible par une heuristique matérielle déclarée ; `0` ou `1` impose l’ordinal visible. Cette sélection est réappliquée dans les threads du serveur. Le scénario 4090/5090 et le passage à un nouveau thread sont vérifiés par mocks ; seule la 3090 a été mesurée physiquement. Le navigateur choisit son adaptateur WebGPU indépendamment. Cette version utilise un seul worker CUDA, sans répartition simultanée sur les deux cartes. `TERRAIN_CUDA_GRAPHS=0` permet de désactiver les graphs au prochain démarrage.

Sur l’API HTTP locale, la tuile native froide de référence est passée de 2,734 s à 2,082 s. La tuile voisine a pris 0,405 s de calcul mais 1,016 s au total, dont 0,538 s en file ; la référence avait pris 0,845 s au total. Ces observations de session montrent pourquoi il faut distinguer calcul, attente et rendu ; elles ne constituent pas une distribution de latences ni une garantie de temps réel.

## Deuxième tranche : CUDA Graphs

Cette comparaison repart du profil CUDA résident optimisé, avec et sans graphs dans le même harness. Elle ne doit pas être multipliée mécaniquement par le gain de la première tranche : les séries sont distinctes.

| Crop froid + cinq sorties climat | CUDA optimisé classique | CUDA Graphs | Gain |
| --- | ---: | ---: | ---: |
| Seed 42, 512² | 3,364 s | 1,957 s | ×1,72 ; première capture incluse |
| Seed 43, 256² | 1,786 s | 1,258 s | ×1,42 |
| Seed 0, 256², coordonnées négatives | 1,931 s | 1,203 s | ×1,60 |
| Seed 42, autre région 256² | 2,372 s | 1,461 s | ×1,62 |

Les quatre crops ont une élévation et cinq sorties climat **exactement identiques** entre ces deux profils. Le pic mémoire PyTorch du harness graphs est de 1,79 Gio. Le forward coarse isolé passe d’environ 12,5 à 1,85 ms ; le gain sur le décodeur était trop faible pour justifier sa capture en production. Les captures et replis par réseau sont exposés par `/api/status`.

Le premier crop classique remplit aussi le cache des poids MP, alors que le premier crop avec graphs bénéficie de ce cache déjà chaud. Le ×1,72 est donc une observation de premier crop, avec cette asymétrie de chauffage, plutôt qu’un gain isolé des graphs. Les trois comparaisons suivantes, avec poids chauds des deux côtés et contexte neuronal absent, montrent un gain supplémentaire de ×1,42 à ×1,62. Elles restent des mesures ponctuelles, sans p95.

## Vérifications et reproduction

`test_terrain_navigation.py` vérifie le partage de calculs, les priorités, les annulations, les epochs, l’expiration des sessions et le cache physique. `verify_navigation_mock.cjs` exerce une grande fenêtre, les changements de seed, les rafales de caméra et les changements de mode sans lancer de NN.

`verify_realtime.cjs` teste le vrai serveur dans Chrome, le déplacement pendant la génération et la conversion GPU ↔ PNG. `verify_terrain_renderer.cjs` contrôle les filtres et l’ombrage sur quatre champs synthétiques, avec une tolérance d’un octet couleur, ainsi que l’éviction et la perte du GPU. Les captures bureau/mobile ont été inspectées.

Pour refaire les mesures NN, arrêter le serveur puis lancer `benchmark_inference.py` ou `verify_inference_runtime.py`. Les variantes expérimentales ne sont pas activées par ces harness dans le serveur. Les rapports détaillés, tableaux et références numériques sont dans `E:\TerrainDiffusionRuntime\inference-realtime`, `inference-decoder`, `inference-graph-pipeline.json`, `inference-runtime-verification.json` et `realtime-qa`. Le benchmark HTTP se lance avec `benchmark_navigation.py` et une seed absente du cache.

Ces changements améliorent le chemin actuel. Les téléportations froides, le téléchargement des poids et la création de grandes surfaces restent des coûts distincts d’une navigation à 60 Hz.
