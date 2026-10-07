# Runtime et initialisation continentale — 7 octobre 2026

Travail demandé après la clôture des cinq blocs : accélérer réellement LOD 3,
puis LOD 2 ; exclure LOD 12 du dézoom ; montrer les améliorations intermédiaires
lors d'un zoom fort. Un agent Sol xhigh traite l'inférence ; un agent Sol high
traite séparément l'initialisation continentale au LOD 11. Reviews Astra high et
Opus 5.5 CLI après les blocs significatifs, corrections avant livraison.

## Navigation progressive

Le client plafonne le dézoom à 61 440 m/pixel et les requêtes au LOD 11,
y compris fit, boutons et molette. Une petite fenêtre peut donc ne pas contenir
tout le monde : le fit respecte cette limite.

Pour une zone sans détail déjà disponible, les besoins passent par
**LOD 4 → 3 → 2 → 1 → 0**, jusqu'au niveau demandé. LOD 4 est le premier
affichage du coarse appris ; LOD 3 utilise le latent, LOD 2–0 le decoder.
Les niveaux plus éloignés partagent la même source coarse et ne sont pas
imposés avant ce premier affichage utile. Chaque étape couvre les besoins
visibles avant d'engager la suivante ; la cible finale reste celle de la caméra.
Les parents restent affichés pendant le calcul. Une revisite déjà couverte
saute ses ancêtres ; le préchargement prédictif attend la fin de l'affinement.
La tuile intermédiaire doit avoir participé à un `drawFrame` soumis avant
l'ouverture de l'étape suivante. Un rAF suspendu retient donc le calcul plus
fin. Cette barrière ne mesure pas la présentation physique de l'écran.
L'image native initiale, lorsqu'elle existe, est protégée contre les patches
de secours moins précis ; les nouvelles tuiles natives peuvent la remplacer.
Lors d'un pan, une bande nouvelle peut demander une étape plus grossière.
Les calculs natifs et intermédiaires déjà admis restent abonnés s'ils couvrent
encore la vue ; seule l'admission de nouveaux calculs plus fins attend la
barrière. Le test retient les réponses natives puis déplace la caméra de 16 km
et vérifie les abonnements conservés. Ces calculs peuvent occuper les quatre
transferts et retarder l'amélioration du bord entrant ; le coût en parcours
froid continu reste à mesurer. Une revalidation intermédiaire conservée reçoit
actuellement deux pénalités de priorité de 1000, sans accumulation ultérieure.

Vérifications exécutées : `node test_terrain_lod.cjs`,
`node test_terrain_freshness.cjs`, `node test_terrain_refinement.cjs`,
`node verify_progressive_zoom.cjs`,
`node verify_navigation_mock.cjs`. Le test Chrome à transport simulé bloque
volontairement LOD 3 et vérifie LOD 4 présenté avant son lancement, aucune
requête plus fine pendant le blocage, l'ordre 4/3/2/1/0, aucune requête LOD 12,
un refresh chaud sans demandes superflues, l'annulation lors du dézoom et
l'image initiale sans recouvrement moins précis. Le test VM de couverture
partielle vérifie qu'un parent plus fin déjà présent évite ses ancêtres.
Ce test ne mesure pas le débit NN ni la présentation matérielle.
Une tuile qui échoue durablement retient l'étape entière pendant ses reprises.
Le mock large écran vérifie le budget adaptatif ; une torture d'éviction avec
génération réelle concurrente n'a pas été exécutée.
Après Opus r3, les tests vérifient aussi l'annulation des vols hors vue,
l'absence de nouvelles requêtes fines pendant une étape retenue, puis la
complétion des requêtes conservées sans abort ni redémarrage. Le fallback
autour de l'image initiale doit être non vide, pour éviter un succès vacu.

## Inférence — mesures avant optimisation

Serveur arrêté, une tâche NN à la fois ; charge graphique du bureau non contrôlée.
Poids résidents, stores NN neufs,
pas de cache physique ni de mip final. Cinq répétitions par LOD, sortie physique
304² avec halo 24. Le premier usage/capture est séparé de ces répétitions.

| LOD | Médiane | Maximum, cinq échantillons | Travail |
|---|---:|---:|---|
| 3 | 2,978 s | 3,164 s | 169 fenêtres latent initial + 121 finales ; 19 forwards base ; quatre coarse |
| 2 | 4,593 s | 4,741 s | 202–244 fenêtres base ; 25 decoder |

Une relecture du même store NN vivant mesure 12–23 ms, séparément.
Ces chiffres ne sont pas des temps de complétion de viewport.

Le batching de préparation et les caches scalaires retirent environ 0,7 s de
préparation au LOD 3 : médiane finale 2,117 s. Le LOD 2 apparié donne
4,508 → 3,829 s, mais un run final défavorable à 5,356 s interdit de certifier
une latence stable. Les 92 arrays comparés sont identiques octet par octet.
Les paramètres, fenêtres, batches et checkpoints sont conservés. Les fichiers
runtime modifiés et l'admission A4 changent l'identité des caches ; les anciennes
préparations restent conservées dans leur namespace, mais ne sont pas réutilisées.
Mesures, CPU supplémentaires et limites : [optimisation NN](RUNTIME_LOD_OPTIMIZATION.md).

## Parcours réel du navigateur

Le [run naturel](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-natural-progressive.json>)
sur une seed nouvelle, poids déjà résidents, reçoit 25/25 réponses natives
en miss physique : premier LOD 4 dessiné échantillonné à 1,113 s, LOD 3 à
3,090 s, LOD 2 à 5,392 s, puis couverture native confirmée à 7,571 s.
Viewport Chrome 900×650, préchargement et préparation mondiale désactivés.
Le temps est une observation de session ; GPU et présentation non chronométrés.
Le [run A4 v1](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-continental-progressive.json>)
est historique : LOD 11 conditionnant visible, 26/26 misses sur le passage
natif, complétion 11,346 s. Premier aperçu mondial 11,548 s. Le cache persistant des statistiques
retire environ 0,35 s de recalcul, sans expliquer seul cette première latence. La première tentative de harness a
échoué uniquement sur l'attribution d'une réponse mondiale tardive au scénario
natif ; son reçu est conservé, et le harness corrigé attribue à l'émission
de la requête. Aucun chiffre historique n'est réattribué aux futures sources.

Le [run A4 v2](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-continental-v2.json>)
rejoue le centre côtier original, après redémarrage du serveur : aperçu initial
9,377 s ; premier LOD 4 à 5,182 s, modèles chargés à ce premier accès ;
couverture native en 12,394 s, 25/25 misses physiques. Il ne se compare pas
directement au parcours naturel avec poids déjà résidents. La
[revisite v2](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-continental-v2-warm.json>)
dans un navigateur neuf réutilise 25/25 réponses physiques : vue native en
0,556 s, aperçu prêt en 1,097 s. Le cartouche de profil A4 a été corrigé entre
les captures ; aucun calcul terrain n'a changé. Ces reçus précèdent le
durcissement schema 2 du cache de statistiques et gardent leurs manifests.

Sur les **sources finales schema 2**, le
[parcours seed 42](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-continental-final.json>)
termine en 13,541 s, 25 misses, modèles chargés au premier accès. Aperçu prêt
en 12,500 s. La suite CPU a été lancée parallèlement au début de ce parcours :
ce reçu valide les étapes et le rendu, sans isoler leur coût. Les tuiles de
preview initiales attendent 9,70–10,65 s en file pour 0,28–0,34 s de compute.
Le timeline serveur n'a pas été enregistré pour attribuer cette attente.
La seconde mise en vue LOD 11 à 0,886 s est postérieure à l'aperçu initial,
et ne remplace pas ses 12,500 s à froid. Le
[parcours seed 0, modèles résidents](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-continental-final-resident.json>)
est indépendant de ces tests CPU : premier LOD 4 à 1,502 s, LOD 3 à 3,618 s,
LOD 2 à 7,188 s, couverture native en 11,326 s ; 28 misses physiques. Aperçu
initial 10,653 s. Le viewport et les footprints diffèrent du benchmark unitaire.

La [revisite finale avec provenance brute](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-continental-final-provenance.json>)
reçoit 25 hits et termine en 0,543 s dans un navigateur neuf. Son entrée
`worldResponses.rawUTF8` conserve la réponse `/api/world` et son SHA : le
manifest canonique Python se reconstruit exactement. Le champ `worldManifest`
des anciens reçus est la représentation JSON du client, susceptible de
normaliser `1.0` en `1`, et ne constitue pas ce texte canonique brut. Aucun
reçu précédent n'a été réécrit pour lui ajouter une provenance rétroactive.

L'[ouverture par défaut](<E:/TerrainDiffusionRuntime/runtime-lod-optimization/default-profile-final.json>)
sans paramètre `profile` sélectionne A4, affiche son nom correct, puis permet
de rouvrir `natural` ; aucune requête LOD 12. Le serveur final reste disponible
sur `http://127.0.0.1:8765`. Recharger les onglets déjà ouverts pour le nouveau
client. La préparation complète coarse NN est distincte de cet aperçu.

## Initialisation

Le profil naturel original reste une référence protégée et fragmentée.
Les variantes macro A1–A3 n'ont pas de raccord Rust actif. Le profil A4 v2
produit de grandes masses séparées, côtes, relief intérieur, chaînes et bassins
attachés aux continents. Il réutilise la distance capsule Rust exacte et
adapte sa structure de côte sous forme portable documentée ; ce n'est pas une
liaison à un binaire Rust. Deux défauts v1 relevés par Opus ont été corrigés :
bande côtière sans pente et sommets écrêtés. Les deux côtes originales ont
37,80 % et 20,02 % d'eau dans leurs sorties NN v2 ; les crêtes restent détaillées.
L'aperçu LOD 11 indique explicitement les entrées de conditionnement ; le probe
de 512 points appris ne constitue pas une planète NN complète. Les limites
de distribution des altitudes, de côtes souvent raides et de plaines basses
rares, de climat et de géologie
restent documentées dans [initialisation A4](CONTINENTAL_BOOTSTRAP.md).

Le corpus final comprend huit sites natifs, dont une côte médiane et la paire
côtière de plus faible saut axial par seed. L'inventaire est échantillonné et
ces minima sont presque tangents à la côte ; ils ne sont pas les minima de
norme de gradient planétaires. La côte faible polaire seed 0 perd une grande
part de sa terre au NN : 7,04 % de terre au lieu de 45,84 % dans le lattice
interpolé. Ce cas reste publié et interdit une revendication universelle de
fidélité du littoral. A4 est le défaut **expérimental d'initialisation** ;
natural reste disponible. Les quatre sites initiaux sont inchangés octet par
octet sur les 28 arrays comparés malgré la nouvelle identité des sources.

Le cache exact des statistiques évite le tri/décodage après redémarrage, sans
changer les valeurs. Schema 2 inclut l'ordre des rasters et la version GDAL ;
neuf tests CPU passent, dont cache inaccessible et changement d'implémentation.
Les edits de sources invalident l'identité mondiale, y compris natural, et
imposent une nouvelle préparation. Les anciens résultats sont conservés.

[Décisions et corrections après reviews](reviews/RUNTIME-corrections.md).

Validation finale : **85 tests Python**, quatre tests JS/Chrome de couverture
et progression, mock grand écran 5760×3240, trois parcours réels et revisite
de provenance, puis ouverture A4 et retour natural. Les cinq tests du harness
passent aussi sous Python `-O`. Reviews et limites restent archivées, avec
Astra high et Opus 5.5 CLI ; les nouveaux chemins numériques restent ceux du
checkpoint, aucune procédure ne modifie le DEM appris.
Deux tests CPU supplémentaires protègent les coordonnées des diagnostics
côtiers et leur mapping raster. Les inventaires A4 et ETOPO n'étant pas
appariés, leur ratio observé ne certifie pas une distribution côtière terrestre.
