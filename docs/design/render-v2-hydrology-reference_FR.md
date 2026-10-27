# Complément hydrologique réel de Render v2

[English translation](render-v2-hydrology-reference.md).

Ce complément reste distinct des 60 situations sans hydrologie et de leurs 900 images. Il comprend six centres sélectionnés dans le réseau source du même monde : rivière humide, rivière tempérée, rivière aride intermittente, lac continental froid à été tempéré, lac polaire et lac aride fermé. Chaque centre est prévu aux cinq emprises 153,6 km, 40 km, 10 km, 2,56 km et 160 m, en juillet V2/Render3D et janvier V2. Le DEM reste respectivement latent 240 m au LOD3 puis décodeur 30 m au LOD0 ; le dernier zoom agrandit ce DEM et éprouve le matériau, sans créer de géométrie submétrique.

La sélection est conservée dans `output/render-v2-hydrology-reference/manifest.json`. Les niveaux, surfaces et chemins du réseau dérivent de l'atlas à 19,53 km : ce sont des repères, pas une validation des rives du DEM neuronal. Les hydrogrammes mensuels sont modélisés, non observés. Le cas intermittent actuellement sélectionné est BSh, avec juillet alimenté et janvier sec dans son hydrogramme ; il ne doit pas être présenté comme une savane Aw.

## Affichage des lacs et limites

`DISPLAY_LAKES` reste désactivé pour les couches historiques. Render v2 dispose d'une grille d'affichage source séparée, demandée par `v2_lakes=1` et `v2_lake_min_area_m2`. Elle sélectionne les lacs par surface et conserve uniquement les échantillons dont le terrain brut est inférieur au niveau du lac. Elle n'active pas les petits bassins régionaux Base et n'effectue aucun carving : le DEM retourné doit être strictement identique. Les cellules émergées restent sèches et le masque physique fait autorité sur le vectoriel, y compris pour une vue contenant seulement un lac.

Ce choix borné ne corrige pas l'écart éventuel entre les niveaux source et le relief neuronal. Il faut examiner visuellement les rives, îles et transitions aux cinq zooms. À partir du LOD4 natif coarse, le chemin actuel conserve le vectoriel historique ; il ne faut pas annoncer de gel saisonnier V2 des lacs dans ce chemin.

## Protocole

Le sélecteur refuse de construire un réseau absent. Les éventuelles constructions sont déclenchées séparément et une seule fois via l'API du serveur existant. Le réseau de l'identité courante `bd36737b…` a été construit en 9,6 secondes ; les compteurs d'inférence sont restés inchangés. Les résultats antérieurs de `f5cddc89…` constituent uniquement des preuves historiques, sans migration de reçus ni présomption de pixels neuronaux identiques.

```powershell
.venv/Scripts/python.exe tools/verification/select_render_v2_hydrology_reference.py --world-info output/render-v2-reference/world-info-bd36737b.json
$env:REVIEW_CASES='lake-temperate'
node tools/verification/review_render_v2_hydrology_reference.cjs --freeze
node tools/verification/review_render_v2_hydrology_reference.cjs --run
```

Ces captures utilisent le serveur existant et peuvent compléter les caches neuronaux de cette nouvelle identité. Le runner vérifie le gel des sources et du protocole, les dimensions PNG, la caméra exacte, la source réelle du DEM et l'identité de chaque clé. Il exige également `hydrology.status=ready`, `fetching=false` et les en-têtes `X-Terrain-Hydro-State: ready` des buffers effectivement envoyés au renderer ; `visible.pending=0` seul ne suffit pas. Les vues V2 exigent quatre canaux physiques, le mois effectif et une coupe maximale nulle. Le gros plan du lac Dfc doit contenir de vrais échantillons lacustres ; une vue lac seule doit masquer la peinture vectorielle. Les captures composent le viewport réel afin de conserver le vectoriel historique dans le témoin Render3D.

Après gel du cœur et autorisation de la campagne complémentaire, supprimer le filtre du premier essai pour produire les 90 vues des six centres :

```powershell
Remove-Item Env:REVIEW_CASES -ErrorAction SilentlyContinue
node tools/verification/review_render_v2_hydrology_reference.cjs --plan
node tools/verification/review_render_v2_hydrology_reference.cjs --freeze
node tools/verification/review_render_v2_hydrology_reference.cjs --run
```

Le `--plan` doit annoncer six sites et 90 vues. Cette procédure est préparée ; la série complète attend le gel du cœur. La préparation coarse globale doit avoir terminé avant d'ouvrir une nouvelle page du viewer, car son ouverture peut interrompre ce travail de fond.

Le succès de ces assertions vérifie le protocole, pas la qualité visuelle. La lisibilité des rives et du relief, l'apparence de l'eau et de la glace ainsi que la saison doivent être revues séparément sur les images.

## Première preuve de la nouvelle identité

La passe working Dfc `291ff73314ce-efa1c53c-b2e60d57` a produit les 15 vues et satisfait le protocole, sans constituer la série finale figée. Les images et son rapport sont dans `output/render-v2-hydrology-reference/captures-reference/captures/` sous cet identifiant. Le carré de 160 m contient 20 échantillons lacustres du DEM à 30 m ; le raster V2 est prêt pour les deux mois, sans coupe et sans peinture vectorielle superposée. Eau en juillet et surface gelée pâle en janvier sont visibles. Le grossissement expose une rive diffuse à l'échelle du DEM, ainsi que des couronnes sombres et un relief de canopée faible sous neige : ces limites restent à revoir visuellement.

`real-lake-raster-bd36737b.json` vérifie indépendamment la vraie tuile native de cette identité et son reçu : 26 395 pixels lacustres sur 92 416, autorité partout, DEM et sortie historique strictement inchangés. Le point ciblé est à 278,44 m sous un niveau de 281,88 m ; le centre source voisin à 284,17 m reste sec. Les pixels de cette tuile diffèrent de la version historique ; aucune égalité neuronale entre versions n'est revendiquée.

Le témoin Render3D utilise bien la capture composite `#viewport`, mais son overlay possède ici zéro pixel peint et zéro lac dessiné alors que le réseau contient un lac. Le culling historique utilise seulement le premier anneau du lac source, hors vue pour ce centre ; ce chemin source est par ailleurs en contour seul au LOD3 ou plus fin. L'absence de lac dans cette image correspond donc à l'état actuel du produit historique, pas à une capture qui aurait oublié son canvas. Elle ne peut pas servir de référence d'un lac rempli réaliste. Les layers historiques n'ont pas été modifiés dans cette adaptation V2.


## Rivières régionales et bord écran

La revue réelle a révélé deux problèmes distincts : le raster rapide pouvait conserver le cours source alors que Render3D affichait déjà son remplacement régional ; sa fraction de couverture était intégrée sur 30 m puis interpolée bilinéairement. Ce double filtrage étirait la berge sur 20–30 m au gros plan. Le contrôleur hydrologique expose désormais au renderer exactement les mêmes segments sélectionnés et projetés que le témoin vectoriel, avec leur largeur physique multipliée par le facteur mensuel existant. Aucune largeur cartographique minimale n'est introduite dans l'eau V2.

Un passage GPU rasterise les quads de ces segments dans un masque écran. La rasterisation matérielle limite le travail aux rubans concernés ; aucun fragment ne parcourt le réseau entier. La couverture suit une tranche de boîte et conserve les cours plus fins qu'un pixel. Un test de profondeur choisit la couverture maximale aux unions, sans trou ni accumulation de transparence. Le niveau signé est transporté en deux composantes demi-précision (partie haute et résidu), puis reconstruit en mètres. Le format RGBA16float et la profondeur 24plus représentent environ 12 octets par pixel actif, plus le buffer de segments ; un masque vide prêt occupe seulement 1×1 pixel. La géométrie projetée et son Float32Array sont mémorisés tant que réseau, caméra et mois ne changent pas.

Ce contrat remplace seulement l'apparence rapide des rivières V2 en carte aux LOD 0–3. Il ne change ni le DEM, ni le routage, ni les quatre canaux physiques, ni le serveur ou l'identité des tuiles. Lacs, mer et autorité sèche sous le niveau marin restent prioritaires. L'ancienne exclusion de population et le supplément de berge sur le ruban source sont écartés dans cette voie ; les racines résolues échantillonnent le nouveau masque régional. En attente de réseau ou de compilation du masque, lorsqu'une surface GPU V2 fine est disponible, le vectoriel Render3D sert de repli identifié, sans ancien ruban raster en doublon. Un acquittement `surfaceKey` distingue cette surface du masque réellement prêt (`frameKey`). Sans surface GPU consommatrice — GPU décoché, PNG CPU ou shader V2 initialement indisponible — la politique précédente de raster et d'overlay reste inchangée ; aucune nouvelle surimpression régionale n'est ajoutée au PNG. La nouvelle précision régionale n'est donc pas revendiquée pour les exports CPU. Le carving détaillé, le globe et le coarse natif conservent leurs voies antérieures.

Le snapshot expose `hydrology.geometryCurrent`, `queryRevision` et `currentTerrainRevision`, ainsi que `renderer.riverDisplay` : état prêt/repli, clé caméra/réseau/mois, révision, nombre de segments réellement chargés, format, dimensions, mémoire et erreur. La capture attend une géométrie à jour après les dernières arrivées de terrain, puis le masque soumis et prêt. Son readback distingue les pixels réellement mouillés du masque écran des anciens échantillons de classe 3 du payload physique. Les deux preuves ne sont pas interchangeables.

## Preuve ciblée des trois rivières

La passe réelle `15f91fbd07b9-d1820dd7-cec2659e` comprend 36 images : trois rivières et un lac témoin, 40 km/2,56 km/160 m, juillet V2/Render3D et janvier V2. Elle est conservée dans `output/render-v2-river-display/reference/`. Les quatre cas passent le protocole et la revue bornée du correctif : mêmes coudes que Render3D, bords nets, confluence sans trou, oued sec en janvier et lac inchangé. Elle ne remplace pas la campagne finale de 990 images ni une validation universelle du matériau d'eau.

Les centres de rivières sont désormais pris sur un vrai segment régional, puis conservés à tous les zooms : humide [-4985167.1720028585,1063774.1584098355], tempérée [4915129.518241145,-5336995.5385306], aride intermittente [2978506.019704413,2705087.4193684384]. Les écarts aux centres source sont respectivement 5112,49 m, 5672,22 m et 13,37 m. IDs de fragment et de parent, segment, débits mensuels, largeur, statut prêt et SHA du réseau sont conservés dans le manifeste et `regional-focus-selection.json`. Le manifeste précédent 024388ad est archivé octet à octet, sans relabel des captures antérieures. Le paysage tempéré mesuré est mixte DFB/DFC avec une marge CFB ; il ne constitue pas une preuve de biome CFB pur.

Au centre humide, le readback à 160 m de juillet contient 256058 pixels mouillés, contre 468306 en janvier ; l'oued de janvier contient 0 pixel mouillé dans un masque 1×1 prêt. Le lac témoin conserve 20 échantillons lacustres et un masque rivière vide aux deux mois. Les tests autonomes vérifient également unions, sous-pixel 0,02 → couverture 0,019989, niveaux signés, absence de trouée de l'ancien ruban, déplacement, redimensionnement, repli, reset et dispose.

Le matériau d'eau reste simple ; le réseau régional garde ses propres limites de résolution et de raccordement au terrain neuronal. Un bord net ne prouve pas que le lit est physiquement incisé, ni que le niveau ou les débits sont observés. Les lacs polaires restent notamment à examiner pour les empreintes anguleuses de propriété source et les petites îles issues du DEM.


## Passe hydrologique 89516 archivée, version rejetée

La version de production 89516 a ensuite été rejetée : la vraie plaine EF humide `polar-plain-candidate-01` du corpus principal expose trop de sol gris en été alors que le mois chaud reste sous 0 °C. Le calcul V2 manque de rétention annuelle/permanente de la neige. La passe hydrologique reste une preuve technique archivée ; elle doit être recapturée après correction et aucun verdict n'est transféré au futur run.

Le run `89516e9f2c58-e4b29729-cec2659e` conserve 90 images réelles, six situations et cinq étendues, avec les trois vues V2 juillet / Render3D juillet / V2 janvier. Les six protocoles passent ; l'assemblage `--sheets --require-complete` confirme les 90 fichiers et leurs mesures natives. Le verdict humain, lié à ce run dans `reviews.json`, est distinct : quatre cas acceptés pour les contrats ciblés, deux rejetés comme preuves suffisantes de qualité lacustre.

Les trois rivières conservent le tracé régional apparié et les berges nettes ; l'oued devient sec en janvier. Le lac DFC montre sa rive à 160 m, de l'eau en juillet et de la glace en janvier. Les petits affluents sont moins visibles au loin que le trait cartographique Render3D : la couverture physique sous-pixel est intégrée sans largeur minimale artificielle. Le matériau d'eau reste homogène, et les coudes du réseau source restent parfois anguleux.

Le lac polaire reste un contre-exemple : limites droites et îlots rayés ou carrés au croisement du plan lacustre et du DEM ; Render3D affiche lui-même un très large connecteur anguleux dans cette région. Le gel saisonnier correct ne valide pas cette géométrie. Le lac aride est visible aux étendues 153,6 et 40 km, mais son centre source est sec sur le terrain natif ; les vues 10 km, 2,56 km et 160 m ne sont donc pas des témoins de rive. L'autorité sèche est conservée et le manifeste figé n'est pas déplacé pour cacher cet échec de cadrage. Ces deux cas restent dans la galerie avec leur rejet explicite, sans prétendre à une validation photoréaliste universelle.

## Cadrage aride préparé pour la prochaine passe

Le manifeste historique `e4b29729ec6c9b732faac75c4da55e6c4263829ae1eea215cf7d2464d76f2d3d` est désormais archivé octet à octet dans `captures-reference/manifests/`. Le manifeste actif prépare une nouvelle campagne de **six sites / 90 images** : `lake-arid-closed-wet-shore`, au centre `[-6764655,3340995]`, remplace le seul cadrage sec du même lac source 194. Le complément séparé de 15 images n'est plus requis. Le rapport, les captures et les verdicts du run 89516 restent attachés à l'ancien manifeste et ne sont pas transférés.

La sélection lit uniquement les tuiles et reçus déjà présents : profondeur native de 0,356 m, propriété lacustre 194, 14 échantillons sous le niveau et 11 émergés dans une fenêtre de 150 m ; 208338 échantillons mouillés sont recensés dans la fenêtre initiale de 40 km. La preuve est `output/render-v2-hydrology-reference/arid-lake-wet-focus.json`. Les statistiques source ont été recalculées en CPU au nouveau centre par `select_render_v2_reference.source_stats` : dominance BWK de 100 %, altitude moyenne source 1518,82 m. Le centre, l'enregistrement et les statistiques originaux sont conservés dans `original_source_record` ; ces statistiques à 19,53 km ne prouvent pas la rive à 30 m.

Ce remplacement n'est pas une acceptation : la future revue indépendante doit constater eau et rive dans la vue de 160 m. Aucune nouvelle capture n'a été effectuée ; le protocole `cec2659e` reste inchangé, en attente du nouveau gel de production et du GO explicite du coordinateur.
