# Bootstrap terrestre : reprise du 7 octobre 2026

La demande actuelle retire toutes les initialisations Earth/Macro A1–A4 et interdit
la réutilisation de city-generator. Le LOD validé est conservé. Trois agents Sol high
ont étudié les projets et la littérature puis implémenté le heightmap natif,
le conditionnement physique et la validation indépendante. Après l'interruption
du serveur d'agents, trois agents Sol high ont repris les fichiers présents.

## Quatre blocs

1. **Heightmap natif** : cœur `world-builder-rs` pinned, plaques et champs fractals,
   reconstruction continue des faces du cube, calibration ETOPO par aire sphérique,
   quatre morphologies, sélection bornée de candidats déterministes.
2. **Conditionnement** : même heightmap à chaque requête, cinq canaux physiques,
   WorldClim par latitude, lapse une fois, encodage altitude signée/racine une fois.
3. **Intégration et nettoyage** : anciens profils refusés, interface et API nouvelles,
   seed initiale aléatoire, identités complètes et caches isolés, LOD maximum 11.
4. **Qualité et runtime** : plusieurs seeds et morphologies, actual NN coarse/latent/DEM,
   revue Astra high + Opus 5.5 CLI, corrections, navigation navigateur réelle.

## Projets réellement récupérés

- `../world-builder-rs` : fetch HTTPS puis fast-forward, déjà au commit
  `009efe18c457757da12ffc8a6215806efb87f012` ; source du générateur retenu.
- `research_projects/map_denoise` : clone authentifié du dépôt demandé,
  commit `5d3182246cddeb93870b9ec5fec9481738c22e15` ; contrat du conditionnement
  et de la hiérarchie étudié, aucune prétention qu'il résout seul la carte initiale.
- `../planet_heightmap_generation` : Orogen, pull fast-forward au commit
  `cc2662b4edd52231c4f65d8765f3ef12cd82d9b7` ; génération headless exécutée
  et comparée, sans intégrer son code dans l'application.
- Le dossier local `world_builder` contient des modifications utilisateur ; il a
  été étudié en lecture seule. City-generator n'a pas été étudié pour cette reprise.

Les mesures et liens primaires figurent dans les trois documents d'étude.

## Contrat géographique et limites

Le raster terrestre couvre ±20 000 km en x et ±10 000 km en y, nord pour y négatif.
Le heightmap source est périodique en longitude ; le chemin NN existant ne garantit
pas de raccord périodique de ses bruits ou de son relief. La caméra conserve un monde
fini. Les bassins sous le datum sont affichés comme mer selon `height<0` ; le masque
océan physique upstream est conservé uniquement comme diagnostic.

La calibration monotone garde les signes **aux pixels source**, mais elle peut
déplacer les zéros après interpolation bilinéaire. Le NN est comparé au sampler
calibré effectivement injecté, jamais à un masque géographique appliqué après coup.
La taille d'un crop est indiquée dans chaque artefact ; une côte absente d'un crop
ne prouve pas sa disparition sur la planète.

La résolution source est 39 062,5 m par pixel dans les deux axes de la carte ;
7 680 m désigne l'espacement du conditionnement NN. Ces valeurs sont séparées dans
les métadonnées et dans les en-têtes de provenance des tuiles.

## Reviews et corrections

La première review Astra a identifié deux P2 corrigés : résolution source annoncée
à tort à 7,68 km et vérification d'artefacts utilisant une identité native mémorisée.
`verify_implementation_identity` contrôle maintenant les octets actuels sans reconstruire
le binaire ; une génération utilise une copie privée vérifiée de l'exécutable.
Les deux passes sont dans `reviews/BOOTSTRAP-RESTART-astra-r1.md` et `-r2.md`.

La validation NN et la review Opus sont consignées dans les rapports associés.
L'acceptation visuelle et les limites ne se déduisent pas des seuls tests CPU.

La seconde passe a trouvé un P1 introduit par le découplage de la référence :
le dossier disque restait identique après une modification native. Les tuiles
physiques et PNG vérifient désormais l'identité complète du monde, et chaque
mode de vue mondiale possède son reçu d'identité. Deux tests reproduisent une
révision native et exigent une nouvelle génération, même sous le même chemin.
Les erreurs Git/Cargo, les erreurs d'initialisation et celles des endpoints de
génération deviennent du JSON 503. Un snapshot natif tardif refuse un fichier
Python différent du module chargé au démarrage.

Les remarques Opus qui restent des limites de portée sont évaluées ainsi :

- Le produit est une planète déroulée en projection équirectangulaire. Les
  statistiques planes et sphériques sont publiées côte à côte. Un archipel testé
  atteint 16,46 % des terres dans sa plus grande composante sur la carte plane,
  contre un critère de 15 % sur la sphère ; cette différence n'est pas masquée.
- Le climat WorldClim extrapolé aux pôles est hors distribution du checkpoint.
  Les sorties NN polaires contiennent réellement des précipitations négatives.
  Elles sont mesurées et ne sont pas acceptées comme climat physique final ;
  cette reprise livre le heightmap et sa navigation, avant le raccord au moteur.
- Le module de conditionnement commun appartient encore à l'identité Natural,
  car il participe à son aperçu et à son dispatch. Une modification de ce fichier
  peut donc invalider son cache ; ses mathématiques et ses sorties sont protégées
  par tests. Les dépendances Rust et le source heightmap sont exclus de Natural.
- Les NPZ de bootstrap occupent environ 3,8 Mio par monde dans cette série. Leur
  cache persistant n'a pas encore de quota global. Les tuiles physiques ont un
  quota global de 16 Gio ; les fenêtres coarse ont un quota de 2 Gio **par monde**,
  sans éviction globale entre seeds ou versions. Une préparation NN mondiale
  complète représentait environ 707 Mo dans le benchmark antérieur. Elle devient
  donc une action explicite, via le bouton ou `prepare=1`, pour éviter qu'une
  succession de mondes aléatoires lance des préparations complètes non demandées.

## Mesures CPU et vérifications

Les **98 tests Python** passent après les contrôles d'admission finaux décrits ci-dessous,
ainsi que les tests de LOD, promotions, fraîcheur,
navigation simulée et l'ordre de raffinement du navigateur. Les **deux tests Rust**
vérifient la reconstruction continue. Les sorties au format de production
256²×6 → 1024×512 sont identiques avec Rayon 1 et 4, seed u64 maximale.

Sur une nouvelle seed aléatoire (`12550389142729498754`), la création complète
mesurée du heightmap prend **1,18 à 2,34 s**, dont **0,59 à 1,67 s** dans le
générateur Rust. Le cas continents a nécessité trois candidats. La relecture
disque vérifiée prend **37 à 45 ms** ; une recherche résidente prend environ
3 microsecondes. Ces mesures CPU n'incluent ni le NN, ni le transport, ni l'affichage,
ni une compilation Rust initiale. La charge desktop n'est pas contrôlée.

Reçu : `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/cpu-bootstrap.json`.
Les 32 cartes de qualité CPU, les sites NN et les limites mesurées figurent dans
`TERRESTRIAL_BOOTSTRAP_QA.md`. Les rapports r1/r2 conservent leur état au moment
de la review ; le présent document distingue leurs demandes des corrections finales.

## Décision sur le conditionnement et dernières observations

Le bruit d'altitude est fixé à **0,05**, les quatre bruits climatiques à **0,5**.
Le choix privilégie le maintien des côtes observé sur les inférences réelles.
La comparaison sur quatre empreintes identiques ne prouve pas un gain de relief
universel : les sites initialement nommés `mountain-median` ont été sélectionnés
par altitude et représentent des **altitudes élevées médianes**, parfois des
plateaux. Sur ces trois sites, le ratio de pente p95 de 0,05 à 0,5 est 0,632,
0,657 et 1,267. Sur la plaine Earthlike, il est **0,407**, soit une réduction
de **59,3 %** ; le ratio d'amplitude p95−p05 est 0,531 et celui du détail après
retrait de la composante à environ 1 km est 0,485. L'étude supplémentaire
ci-dessous utilise des sites sélectionnés par relief local, au lieu de déduire
leur caractère accidenté de leur altitude.

L'identité complète a été mesurée indépendamment sur CPU : sérialiser et
vérifier un manifeste terrestre de 83 128 octets prend **1,25–1,30 ms** par
appel. Deux à quatre vérifications peuvent intervenir sur une requête déjà en
cache. Ce coût est connu ; aucune modification supplémentaire de cette sécurité
n'est introduite dans cette reprise. Reçu :
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-final-cpu/world-identity-timings.json`.

La référence Natural seed 42, tuile LOD 2 −14/10, a été réexécutée sur CUDA et
comparée à la référence sauvegardée avant cette reprise : **391 444 octets
identiques**, altitude et climat, erreur absolue maximale zéro. Reçu :
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/natural-server-replay-final.json`.

Le premier test Chrome réel a révélé un défaut que les inférences directes ne
traversaient pas : `CoarsePreparation.install` conservait l'admission des anciens
profils macro et refusait les noms terrestres. Le reçu de cet échec est conservé
dans `browser-before-coarse-fix.json`. Le contrôle reconnaît désormais explicitement
les quatre nouveaux profils tout en refusant un manifeste d'un autre monde.
Deux nouveaux tests traversent la vraie construction serveur, le bind du pipeline,
la construction du manifeste et l'installation coarse, avec modèles et source
native simulés. Les contrôles de seed, SNR, précision et identité restent actifs.
Astra r4 vérifie le correctif, ses tests et le reçu navigateur réel.

## Navigation réelle après correction

Chrome headless, renderer WebGPU, serveur RTX 3090, vue 900×650, Earthlike seed 42,
centre −14 003 910 / 6 097 680 m. La préparation mondiale automatique et le
préchargement prédictif sont désactivés pour isoler le parcours mesuré.

| Mesure | Tuiles froides, processus neuf | Revisite, même processus | Après redémarrage serveur |
|---|---:|---:|---:|
| Ouverture jusqu'à aperçu et renderer prêts | 3,859 s | 1,126 s | 1,194 s |
| Première soumission LOD 4 après zoom | 4,570 s | 68 ms | 57 ms |
| Première soumission LOD 3 | 6,800 s | 196 ms | 178 ms |
| Première soumission LOD 2 | 10,518 s | 255 ms | 235 ms |
| Première soumission LOD 1 | 12,072 s | 373 ms | 365 ms |
| Première soumission LOD 0 | 12,247 s | 488 ms | 473 ms |
| Couverture native complète, 10/10 tuiles | **12,443 s** | **570 ms** | **569 ms** |
| Cache physique des requêtes de zoom | 22 misses | 22 hits | 22 hits |
| Nouveaux forwards coarse / base / decoder | 80 / 36 / 40 | 0 / 0 / 0 | 0 / 0 / 0 |

L'ordre des réponses est 4 → 3 → 2 → 1 → 0, LOD 11 reste la limite de recul,
et le DEM final est à 30 m/pixel. L'ouverture et le zoom ont des origines de temps
distinctes. Le premier zoom comprend le chargement des modèles ; le heightmap
de cette seed existe déjà sur disque. Ces temps mesurent des soumissions de
dessin échantillonnées et la couverture finale, pas la présentation GPU, ni un
débit froid soutenu sur toutes les régions. Une autre charge desktop peut exister.
Les manifestes JSON bruts, headers cache, captures et SHA256 sont conservés dans
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/browser-final-cold.json`,
`browser-final-warm.json` et `browser-final-after-restart.json`. Les trois parcours
passent sans erreur JavaScript ni réponse HTTP en échec. Le dernier commence
avec un serveur sans monde resident ni forward exécuté ; les paquets physiques
proviennent du disque. Les hashes de sources et ceux reconstruits depuis le JSON
brut du manifeste serveur correspondent aux fichiers livrés.

## Clôture de la validation NN

Après le correctif d'admission, les 21 crops natifs, huit sondages de 32 points
et le sondage Earthlike de 512 points ont été réexécutés sur CUDA. Leurs manifestes
actuels et NPZ ont été attestés sur CPU ; les anciens reçus conservent leur
identité historique. Dossiers finaux : `final-hardened` et
`final-hardened-earthlike-global512`, sous
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/`.

Huit autres sites ont été choisis avant inférence : un maximum de relief local
par style et un site de hauteur p99. Les 24 variantes 0,05/0,1/0,5 gardent 100 %
de terre. À 0,05, les quatre sites accidentés ont **599–2 085 m** de relief
p95−p05 et conservent **71,6–102,9 %** de la pente p95 mesurée à 0,5.
Le détail après retrait d'une composante à 1 km est cependant réduit de
**58,5 % sur le site Archipelago**. Le réglage ne supprime donc pas ces reliefs
forts, mais ne garantit pas la conservation de tout leur détail.
Les huit lignes de résultats, les règles de sélection et les limites climatiques
figurent dans [TERRESTRIAL_BOOTSTRAP_QA.md](TERRESTRIAL_BOOTSTRAP_QA.md).

Reçu supplémentaire : `final-hardened-relief-stress/relief-comparison.json`.
Les 24 exports sont attestés avec les sources et le checkpoint actuels. Les
harnesses ont été figés et copiés avant les inférences. Le script exact des
16 anciens exports comparatifs a également été récupéré octet pour octet,
SHA256 `873747617734dff6a54cf6fd9dbc3dea3f1052aec6cd402499ad19f92b0903a9` ;
la remarque Opus sur sa provenance est donc résolue sans réattribuer de nouveaux
hashes aux anciens résultats.

## Évaluation de la review Opus r4

Le P1 de documentation a été fermé : les résultats de stress sont publiés,
et 0,05 reste un choix explicite de maintien des côtes avec les pertes de détail
mesurées ci-dessus. Les sorties ne sont pas décrites comme un gain universel.

Les corrections finales ne changent aucun calcul neuronal :

- `_dtype=None` est reconnu comme FP32 par l'admission ; les tests couvrent
  cette branche, une précision incorrecte et un vrai manifeste d'un autre style.
- Un monde terrestre doit annoncer son profil et fournir un heightmap dont
  le SHA256 correspond à celui du manifeste avant toute persistance.
- Une erreur interne d'installation coarse devient `RuntimeError` dans le
  serveur et donne du JSON 503 ; un paramètre HTTP invalide reste un 400.
  Les deux branches sont testées par le vrai endpoint height.
- Le test de référence Natural passe par le vrai `server.get_world`, puis
  une reconstruction après vidage de la mémoire, avec fenêtres coarse
  persistées. Les deux sorties restent identiques aux 391 444 octets du baseline.
  La seconde passe relit quatre fenêtres sans nouveau forward coarse ; base et
  decoder recomputent respectivement 16 et 25 forwards. Le reçu contient le
  manifeste admis du serveur avec sa précision `bf16`.

Astra r5 a vérifié les cinq tests de construction/admission et n'a trouvé aucun
nouveau problème P1/P2 prouvé. Les premiers reçus `browser-cold` / `browser-warm`
précèdent uniquement ces derniers contrôles et restent historiques. Les trois
nouveaux reçus `browser-final-*` ci-dessus traversent les contrôles définitifs.
Les 45 exports natifs et neuf sondages mondiaux ont été réexécutés et attestés
contre ces mêmes sources. Leurs 75 payloads NPZ numériques sont identiques
octet pour octet aux arrays des reçus pré-hardening ; les manifestes sont nouveaux,
donc aucune identité d'archive NPZ complète n'est revendiquée.

Les identités des exports de recherche NN et celles du serveur ont des contextes
d'exécution distincts, notamment leurs profils de runtime et la notation de
précision. Elles ne sont pas annoncées égales. Les entrées géographiques,
checkpoints et sources sont attestés dans chaque contexte ; les mesures de
navigation utilisent le manifeste brut du serveur réellement interrogé.

Un diagnostic CPU a également examiné les falaises anguleuses visibles sur un
site Gondwana : le parent physique et le coarse appris sont lisses, les formes
apparaissent dans le latent puis le DEM. Les lignes de fenêtres et l'expansion
nearest n'expliquent pas de discontinuité démontrée. Aucune correction numérique
étroite n'est justifiée par ce seul diagnostic ; cette qualité visuelle locale
reste une limite, sans attribution certaine au modèle ou à la précision.
Reçus : `post-admission-relief-stress/gondwana-rectilinear-diagnosis.json` et
`gondwana-rectilinear-stage-diagnosis.png`.

## Évaluation de la review Opus r5

Opus 5.5 CLI accepte avec limites et confirme la fermeture de toutes les demandes
de r4, sans P0/P1. Le nouveau P2 sur le stockage est retenu : la préparation de
toute la planète est activée uniquement par le bouton ou `prepare=1`. Le NN
des régions visitées reste calculé à la demande, et leurs caches restent actifs.
Le stockage coarse/NPZ entre mondes n'a pas de quota global ; la politique évite
la préparation complète implicite, sans prétendre remplacer une éviction globale.

Deux remarques latentes sont conservées comme limites de développement : une
configuration interne invalide avant l'installation coarse peut encore arriver
en 400, et certains helpers Python importés tardivement peuvent différer du
snapshot si leurs fichiers sont édités après démarrage. Le profil livré est
valide, les tests et mesures utilisent des sources figées, et le serveur a été
arrêté avant chaque modification de ces sources. Elles ne justifient pas de
modifier encore le chemin neuronal validé dans cette livraison. Un changement
de code requiert un redémarrage pour former une nouvelle identité cohérente.

Astra r6 clôture les reçus GPU, la relecture Naturel et les trois parcours Chrome
de la version Python définitive. Les falaises anguleuses apprises, le climat
polaire, le débit froid soutenu et la présentation GPU restent hors certification.

## Clôture de la politique de préparation

Chrome 154, avec API simulée, confirme zéro POST de préparation au démarrage
normal et après une nouvelle seed aléatoire, un POST correct après clic sur le
bouton, et le déclenchement automatique avec `prepare=1`. Le reçu et le harness
figé sont `manual-world-preparation-policy.json` et
`manual-world-preparation-policy-harness.cjs` dans le dossier runtime. Les tests
LOD, raffinement, fraîcheur et navigation grand viewport passent également.

Le seul changement du produit après les trois parcours `browser-final-*` est
la condition HTML `get('prepare')!=='0'` devenue `get('prepare')==='1'`.
L'ancien HTML est conservé exactement dans
`index-before-manual-world-preparation.html`, SHA256
`bb02e9a4abd162f9ca615e662cc9777123994ea2771edc919ff47b14914be8b9`.
Ces trois parcours désactivaient déjà la préparation avec `prepare=0` ; leurs
sources Python et leur comportement mesuré restent ceux livrés. Leur reçu HTML
précède explicitement ce changement, sans lui attribuer une nouvelle identité.

Un quatrième parcours Chrome avec le serveur réel et le HTML final
(`afffd309a0e9e317b801fcc327608fb389f48c2ba37cbd91e88656b76175583a`)
passe : aperçu en 1 144 ms, couverture finale à 30 m/pixel en **594 ms**, réponses
LOD 4 → 3 → 2 → 1 → 0, 22 hits physiques, zéro forward neuronal et aucun monde
résident avant ou après. Reçu : `browser-ui-final-cache.json` dans le même dossier.
Cette mesure porte sur la relecture de la région déjà persistée, pas sur une
région nouvelle. Le P2A de la review Opus r5 est ainsi corrigé et vérifié.
Astra high r7 a vérifié indépendamment le diff exact, les SHA256, le reçu de
politique et le parcours avec serveur réel, y compris le manifeste brut et les
sources courantes. Aucune correction supplémentaire demandée ; review :
[BOOTSTRAP-RESTART-astra-r7.md](reviews/BOOTSTRAP-RESTART-astra-r7.md).
