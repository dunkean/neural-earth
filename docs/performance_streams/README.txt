Mesures et intégration multistream — 9 octobre 2026

Intégration actuelle : le panneau Rendu / Calcul neuronal expose 1/2/4/8/16
streams coarse, quatre par défaut (TERRAIN_COARSE_STREAMS au démarrage).
GET/POST /api/inference/streams permet de lire/appliquer le réglage par moteur.
Les poids sont partagés, les graphes et états du solveur sont privés. La fusion
et la persistance restent ordonnées sur le thread hôte. Le nombre de streams
ne modifie ni les batches neuronaux ni l'identité physique du cache. Les
contrats persistés continuent de déclarer batch_size=1. Les groupes de fond
respectent les budgets exprimés en fenêtres et rendent la priorité entre
groupes. En cas de refus de capture, repli explicite au chemin séquentiel,
visible dans les diagnostics et dans le panneau. Les captures se réchauffent
au premier groupe ; leurs temps ne font pas partie des scores ci-dessous.

Base et décodeur : mesures sur RTX 3090, PyTorch 2.11.0+cu128, seed 0.
Les entrées sont récoltées pendant une vraie lecture de WorldPipeline.residual
sur 768x768 pixels. Les tailles rencontrées sont base16 et base1 (fins de
groupes), puis décodeur1. Les scores isolés portent sur huit appels préparés
base16 (128 fenêtres) et huit appels décodeur1 (huit fenêtres). Aucun batch
n'est redécoupé. Les fins de groupes base1 sont récoltées et mentionnées dans
le reçu, mais ne font pas partie du score base16. Les sorties de tous les
appels mesurés sont contrôlées octet par octet, y compris en mode mixte.

Médianes murales de sept répétitions, ordre 1/2/4 puis 4/2/1 alterné,
captures chaudes et pools co-résidents :
Streams | Base batch16     | Gain base | Décodeur batch1 | Gain décodeur
1       | 0,6985 s         | 1,000x    | 0,4612 s        | 1,000x
2       | 0,6551 s         | 1,066x    | 0,4292 s        | 1,075x
4       | 0,6456 s         | 1,082x    | 0,4239 s        | 1,088x

Test mixte : huit solveurs coarse1, huit appels base16, huit décodeurs1,
avec quatre streams privés par famille. Toutes les entrées sont déjà prêtes :
cela ne représente pas l'ordonnancement complet des dépendances en navigation.
Séquentiel entre familles : 1,1978 s ; familles simultanées : 1,2063 s,
soit un gain 0,993x (0,7 % plus lent). L'ordre séquentiel/simultané est alterné
sur sept répétitions. Les durées GPU médianes par famille passent de
0,129/0,650/0,419 s à 0,376/0,870/0,828 s. L'exécution concurrente est possible,
mais ne conserve pas les gains isolés et augmente la latence de chaque famille.
Le premier essai donnait +1,6 % avec coarse4/base2/décodeur2 : ce faible gain
n'est pas reproduit par la série alternée avec coarse4/base4/décodeur4.
Ne pas attribuer la différence à la seule chauffe : les comptes diffèrent.

Les scripts gardent les poids du checkpoint immuables, les streams et buffers
privés et un contrôle de formes préchauffées ; aucun repli eager silencieux
n'est admis dans les scores. La préparation CPU, les dépendances, la fusion,
la persistance et HTTP sont exclues. Les processus GPU externes et l'affichage
ne sont pas maîtrisés ; la carte atteint 81 °C. Ces résultats locaux n'établissent
ni un gain sur d'autres GPU, ni un gain du moteur serveur complet.
L'intégration coarse de production a également passé la comparaison des
fenêtres pondérées à 4 et 16 streams contre le chemin scalar1 réel.

Reçus : models-mixed-alternated.json (série retenue), models-mixed-initial.json
(premier essai). Les hashes correspondent aux sources au moment des essais ;
les ajustements ultérieurs des diagnostics ne réécrivent pas ces reçus.
Reproduction :
& .venv/Scripts/python.exe benchmark_model_streams.py E:/TerrainDiffusionRuntime/coarse-streams/models-mixed-new.json
& .venv/Scripts/python.exe -m unittest test_terrain_coarse_streams test_terrain_coarse_graph test_terrain_stream_settings test_terrain_server_world test_terrain_windows test_terrain_coarse_async test_terrain_coarse_priority test_terrain_background -v
node verify_terrain_streams.cjs

Historique du prototype coarse isolé :

Le prototype tourne hors du serveur, en PyTorch CUDA BF16 et batch 1.
Chaque slot possède son solveur à 20 étapes, son instance de CUDA Graph,
ses buffers et son stream. Les poids sont partagés en lecture seule.
Le thread hôte soumet les travaux, joint les événements, puis consomme les
résultats dans l'ordre d'entrée. Aucune mutation concurrente du store.

Essai final : RTX 3090, PyTorch 2.11.0+cu128, seed 42, 32 fenêtres,
5 répétitions par variante, ordre des variantes alterné. Le conditionnement
naturel est utilisé avec des gains SNR régionaux explicites ; cinq politiques
différentes ont été rencontrées. Ce n'est pas un test du bootstrap terrestre.

Streams | Solveur, médiane | Fenêtres pondérées, médiane | Débit fenêtres/s
1       | 0,826 s          | 0,882 s                    | 36,3
2       | 0,595 s          | 0,640 s                    | 50,0
4       | 0,518 s          | 0,557 s                    | 57,4
8       | 0,510 s          | 0,537 s                    | 59,6
16      | 0,509 s          | 0,529 s                    | 60,5

Le solveur gagne environ 1,60x dès quatre streams et plafonne ensuite.
Avec dénormalisation/pondération, quatre streams gagnent 1,58x ; seize,
1,67x. Le petit supplément au-delà de quatre streams ne permet pas de
conclure à un gain garanti en navigation sous charge réelle.

La durée GPU médiane d'un groupe de fenêtres pondérées est respectivement
28, 40, 70, 134 et 265 ms. Si l'annulation/priorité du serveur restait à la
frontière du groupe, les gros groupes retarderaient davantage le premier plan.
Les allocations supplémentaires des captures sont d'environ 8,3 Mio par
slot, soit 33 Mio pour quatre et 132 Mio pour seize (compteurs PyTorch,
hors poids et mémoire CUDA réservée ; les pools sont co-résidents à l'essai).

Toutes les variantes passent les comparaisons octet par octet au solveur
actuel : chaque fenêtre, chaque répétition mesurée et le rectangle fusionné
par le vrai InfiniteTensor. L'ordre de fusion est identique ; l'écart
d'altitude du rectangle coarse fusionné est nul. Une autre série de
32 fenêtres naturelles, seed 0, a passé les mêmes contrôles à 1/2/4 streams.
Six tests automatisés passent, couvrant notamment les états privés,
conditions/embeddings dynamiques, résultats ordonnés et non écrasés,
modification des poids, rejet de groupe invalide et refus explicite de capture.

La trace Nsight montre jusqu'à quatre streams exécutant des kernels en même
temps, contre un au contrôle. L'analyse porte sur 28 096 kernels solveur par
variante. La trace modifie les temps : elle prouve le recouvrement, pas le gain.
Le rapport complet est conservé sur E:/TerrainDiffusionRuntime/coarse-streams/
replay-trace-20261009.nsys-rep ; overlap.json en conserve l'analyse et le SHA
de l'export SQLite. L'algorithme d'analyse a aussi passé trois cas synthétiques.

Limites : les mesures excluent préparation CPU/bruit, persistance, HTTP,
rendu et démarrage/capture. L'activité graphique et celle d'autres processus
GPU ne sont pas maîtrisées. Les snapshots GPU sont dans les reçus. Ces essais
ne certifient ni un gain serveur complet, ni d'autres cartes/pilotes, ni
l'indépendance du base envers l'ordre de navigation.

Décision : prototype concluant ; commencer une intégration par quatre streams
coarse, en gardant l'ordre de fusion et les points de contrôle d'intérêt.
À cette étape du prototype, le serveur n'était pas modifié. Les modèles base et
decoder, leurs batches et les identités des caches de production sont inchangés.
Huit/seize streams sont utiles à comparer pour le débit de préparation de fond.
Il n'existe pas de limite matérielle déduite à seize : c'est une borne de ce test.

Reproduction depuis la racine du projet (PowerShell) :
& .venv/Scripts/python.exe -m unittest test_terrain_coarse_streams test_terrain_coarse_graph -v
& .venv/Scripts/python.exe benchmark_coarse_streams.py --regional-snr --windows 32 --repeats 5 --streams 1 2 4 8 16 --output E:/TerrainDiffusionRuntime/coarse-streams/new-report.json

Reçus : regional-32-final.json, natural-seed0-32.json, overlap.json.
Chaque reçu de benchmark contient les hashes des sources exécutées.
Les hashes des anciennes séries diffèrent : l'instrumentation de profilage,
les comptes de streams autorisés et les durées de groupes ont été ajoutés
entre les séries. Les reçus n'ont pas été réécrits pour masquer ces différences.
