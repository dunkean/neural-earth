# Corpus réel de référence Render v2

[English translation](render-v2-reference-protocol.md).

Le corpus prévu comprend **60 centres distincts × cinq zooms × trois vues = 900 PNG**. Les images proviennent exclusivement du viewer sur les tuiles réellement générées du monde `7985403534069300490`. Les statistiques de l’atlas Orogen servent à choisir les centres ; elles ne constituent aucune preuve du relief natif ou de l’aspect final.

Le manifeste et tous les résultats résident dans `output/render-v2-reference/`, séparément des premières captures `output/render-v2-40km/`. Les centres communs aux listes initiales sont dédupliqués à un millimètre et conservent leurs identifiants d’origine dans `aliases`. Le corpus réunit la matrice initiale, les autres situations réelles, les positions exactes de l’utilisateur, les candidats de remplacement et trois compléments côtiers/arides. Deux centres supplémentaires visent explicitement une plage déjà visible dans la capture réelle et un point de mer dans la banquise ; leurs échantillons natifs et leur provenance sont consignés.

| Étendue carrée | LOD forcé | Source exigée | Empreinte image, 1024² |
| --- | --- | --- | --- |
| 153,6 km | 3 | latent réel, 240 m | 150 m/pixel |
| 40 km | 0 | decoder réel, 30 m | 39,0625 m/pixel |
| 10 km | 0 | decoder réel, 30 m | 9,765625 m/pixel |
| 2,56 km | 0 | decoder réel, 30 m | 2,5 m/pixel |
| 160 m | 0 | même decoder, 30 m | 0,15625 m/pixel |

Le dernier zoom vérifie le matériau et la composition des couronnes agrandis sur le DEM à 30 m. Il **ne démontre pas un relief submétrique** et ne déclenche aucun raffinement de LOD négatif. Les vues sont Render v2 juillet, Render3D de référence et Render v2 janvier. Juillet est l’hiver austral et janvier l’été austral au sud de l’équateur. Le comparateur Render3D conserve son propre contrat historique ; il n’acquiert pas le calendrier v2 par son simple libellé.

Les étendues désignent des carrés du chart géographique utilisé par le DEM. Ce ne sont pas des carrés géodésiques de dimensions physiques égales à toutes les latitudes. L’atlas source a une maille de 19,53 km dans ce chart ; les 33 × 33 échantillons bilinéaires de présélection ne sont pas 1089 observations indépendantes.

## Familles et reliefs mesurés

Les six familles sont explicitement groupées : AF = Af/Am ; CFB = Cfb/Cfc ; BSK = BSk/BSh ; ARID = BWh/BWk ; DFC = Dfc/Dsc ; POLAR = ET/EF. Le nom de famille ne signifie pas que chaque situation appartient à une classe Köppen pure. Les fractions de toutes les classes sont conservées.

| Relief demandé | Critère de mesure sur le vrai carré de 40 km |
| --- | --- |
| Plaine | Écart-type des hauteurs < 45 m, P90−P10 < 130 m, pente P95 < 0,15 |
| Plateau | Au moins 19 % de hautes surfaces planes (hauteur ≥ P60 et pente < 0,06), plus de 8 % de pentes > 0,2 et P90−P10 > 300 m ; confirmation visuelle du plateau nécessaire |
| Montagnes | Écart-type > 160 m, P90−P10 > 450 m et pente P95 > 0,30 |
| Ondulé | Écart-type ≥ 40 m et P90−P10 ≥ 120 m, après exclusion des catégories précédentes |

Ces seuils sont des critères de sélection, pas une taxonomie géomorphologique universelle. Ils sont évalués dans l’ordre plaine, plateau, montagnes, ondulé et ne sont pas modifiés pour transformer un échec en succès. Un bassin boisé entouré d’un rebord peut satisfaire certains chiffres d’un plateau tout en échouant à la revue visuelle. Les situations côtières et sans terre sont consignées séparément.

## Préparation et exécution

```powershell
.venv/Scripts/python.exe tools/verification/select_render_v2_reference.py
node tools/verification/review_render_v2_reference.cjs --plan
```

Ces commandes ne lancent ni navigateur de terrain, ni requête serveur, ni inférence. Le premier outil lit uniquement l’atlas, les listes et les reçus locaux. Le mode `--measure fichier.json` mesure les tuiles natives déjà présentes, sans demander leur génération. Les océans sans aucun pixel de terre produisent des champs terrestres `null` et des fractions de biome vides.

Après le gel explicite du code de production et du manifeste :

Le gel reste soumis à un préflight visuel des six familles aux niveaux latent
et décodeur, complété par une vraie plaine EF, la plage et le pack marin.
Le pack marin ne remplace pas le contrôle de neige terrestre permanente.
Une réussite technique
(sources exactes, fichiers et compteurs) ne vaut pas acceptation du rendu.
Un défaut majeur absent de Render3D arrête la campagne : les captures partielles
restent dans leur version d'origine, sans être renommées ni utilisées pour
valider le correctif suivant. Le préflight `db083bf3841a…` a ainsi été arrêté
après un réseau blanc vermiculé sur DFC au LOD3 ; les 60 images principales et
30 hydrologiques décrivent cet état rejeté, pas une galerie finale terminée.
Le préflight suivant `790e34695a49…` conserve 105 images principales et
75 hydrologiques. Son contrôle rapproché a révélé des couronnes translucides
sur la plage ; la comparaison hydrologique a aussi montré un désaccord entre
tracé physique rapide et tracé régional affiché. Ces images restent des preuves
de défauts à corriger, et ne deviennent pas une galerie finale par leur seul
succès de protocole. Toute nouvelle campagne utilisera son propre gel.

La passe `89516e9f2c58…` a été arrêtée après 495 images principales complètes
et 90 hydrologiques. Une vraie plaine EF humide, dont le mois chaud reste sous
zéro, devenait gris beige en juillet alors que Render3D conservait son manteau
blanc. V2 n'utilisait que la neige instantanée et omettait la composante annuelle
permanente. Ce défaut est bloquant ; les preuves et verdicts restent dans ce
run rejeté. Aucune image ni appréciation n'est transférée à la version corrigée.

```powershell
node tools/verification/review_render_v2_reference.cjs --freeze
node tools/verification/review_render_v2_reference.cjs --run
```

Le mode `--run` utilise le serveur existant, charge la recette partagée et laisse le viewer demander les tuiles nécessaires. Il peut donc compléter les caches des nouveaux centres. Il n’installe ni ne lance un autre modèle ou serveur. `REVIEW_CASES=id1,id2` limite une invocation et permet une reprise progressive. Une image complète n’est réutilisée que si son fichier et son SHA256 correspondent exactement à la même version du code et du manifeste. Chaque version possède son sous-répertoire de captures ; les anciennes preuves restent intactes.

La campagne peut répartir les centres restants entre deux invocations au plus,
dans des répertoires de travail distincts. Chacune garde le manifeste complet
et le gel de production identiques ; seul `REVIEW_CASES` sélectionne son lot.
La réunion copie les PNG sans transformation, vérifie leur SHA256 et conserve
les reçus originaux dans `worker-reports/`. Elle ne fabrique ni terrain ni
images et ne transfère aucune appréciation visuelle entre versions. Le runner
du répertoire commun doit être arrêté avant cette réunion pour éviter deux
écrivains concurrents. Le complément hydrologique compte aussi dans la limite
de deux navigateurs de capture simultanés.

## Préchauffage des caches avant la campagne finale

```powershell
node tools/verification/warm_render_v2_reference.cjs --plan
$env:REVIEW_CASES='af-rolling,cfb-rolling'
node tools/verification/warm_render_v2_reference.cjs --run
```

Sans `REVIEW_CASES`, le driver parcourt les 60 centres. Il utilise le viewer existant pour une seule vue de travail de 40 km, au LOD0 forcé et à 30 m de source réelle par centre. L'hydrologie, le préchargement périphérique et le raffinement sous le LOD0 sont désactivés ; la mer reste activée pour calculer les centres océaniques. Il vérifie l'identité du monde, son profil de cache et les étapes source avant le démarrage. Une reprise vérifie également que les tableaux et reçus natifs existent toujours sur disque pour cette identité.

Le dossier `output/render-v2-reference/warm/` est distinct de la campagne finale. Ces PNG de travail servent à préparer les caches, sans comparer les dates et sans être comptés parmi les 900 images. Un préchauffage peut calculer de nouvelles tuiles neuronales ; il faut coordonner son lancement avec les autres travaux GPU. Le mode `--plan` ne lance aucune requête ni aucun navigateur. Le rendu peut encore évoluer pendant ce travail : seule la campagne après gel peut démontrer une qualité cohérente d'une même version.

## Contrats et lecture des résultats

La campagne courante utilise l'identité `bd36737badf0cedf99200ba94bf7dc761070aca686a98784373247c86efe87dd`, profil de cache `96d1b3a5fb584090`, dont les métadonnées sont conservées dans `output/render-v2-reference/world-info-bd36737b.json`. L'ancienne identité `f5cddc89751589636d1052bfa46529f6f0a400b1fd1b176a35dbb14ce3590a1a` et ses reçus restent historiques. Le manifeste précédent est archivé ; aucun reçu physique n'est renommé ou migré. Les identifiants source, relief, érosion et climat sont inchangés, mais cela ne démontre pas l'identité des pixels neuronaux. Tous les centres doivent donc être remesurés dans les nouveaux caches natifs. L'option `--world-info chemin.json` du sélecteur fixe explicitement cette version de travail et propage son identité au runner.

- Les empreintes de tous les JavaScript du viewer et des modules de rendu CPU sont figées. Une modification du code ou du manifeste arrête la passe ; les versions ne sont pas mélangées.
- Le protocole de capture et de mesure possède également son empreinte figée ; un changement des outils pendant la passe est refusé. Les dimensions PNG sont vérifiées dans l’en-tête du fichier, en plus des dimensions du canvas.
- L’identité du monde et les étapes de génération sont contrôlées par un GET de métadonnées. Les clés de cache de chaque image doivent désigner cette identité.
- Chaque capture exige `pending=0`, un renderer prêt, le LOD/source/résolution réels attendus, la caméra exacte, un canvas 1024², le mode et la date attendus. Un aperçu de conditionnement ou un ancien DEM fin ne satisfait pas le contrat du latent à 240 m.
- `Render sea` est activé pour que les cas purement océaniques soient réellement calculés au LOD demandé. Prefetch, grille, HUD et minimap sont masqués/désactivés pour cette session. L’hydrologie est désactivée ; seules la mer et la côte du DEM sont incluses dans ce corpus.
- Les statistiques natives portent sur le carré de 40 km réellement décodé. Les reçus doivent porter `decoder`, 30 m et l’identité attendue ; une couverture partielle du carré est refusée. Les biomes sont reconstruits à partir du climat continu de la source sur les hauteurs du DEM, tous les 120 m. Ce classement est un outil de mesure, pas une observation de terrain.
- Les seuils morphologiques, fractions de biomes et mélanges sont conservés séparément de l’acceptation du protocole. Un plateau demandé mais mesuré montagne reste un échec de sélection. La qualité visuelle ne se déduit ni du pourcentage de vert ni de la réussite technique.

## Galerie

```powershell
node tools/verification/audit_render_v2_reference.cjs
```

L’audit produit `coverage.json` et `coverage.md` : une ligne par combinaison des quatre reliefs et des six familles. Une case n’est mesurée que si un vrai DEM confirme le relief demandé et la fraction de famille. Les seuils de 80 % strict et de 70 % représentatif restent distincts ; une transition naturelle ne devient pas artificiellement un biome pur. Un candidat source ne compte jamais comme mesure. Les appréciations `reliefAccepted` et `status: accepted` dans `reviews.json` sont nécessaires pour marquer une case comme visuellement acceptée. Aucune réaffectation automatique d’une montagne au plateau demandé ne remplit la matrice.

Deux indicateurs sont conservés. `requestedTargetMatches` mesure la réussite des cibles de présélection. `actualNativeCoverage` mesure les reliefs et familles réellement présents dans le DEM, indépendamment de la cible initiale. Ainsi, un centre demandé comme plateau mais mesuré montagne à 99 % CFB reste un échec de présélection plateau et peut servir de référence montagne CFB, sans changer son nom. Son identité, ses fractions natives, sa cible échouée et sa provenance sont conservées. Son acceptation visuelle exige `actualReliefAccepted: true` et `status: accepted` dans la revue liée au `runId` courant ; les mesures seules ne valident pas l'apparence du rendu. Un rapport d'une ancienne identité de monde est ignoré pour les comptes courants.

Le préchauffage des 60 centres dans l'identité `bd36737b…` a été entièrement mesuré : 18 cibles strictes sur 24, 20 avec transitions représentatives ; la couverture des reliefs réellement mesurés atteint 20 cases strictes sur 24. Le portfolio de 60 situations variées n'est donc pas une matrice complète des quatre reliefs × six familles. Quatre lacunes restent ouvertes : plateaux AF, CFB, DFC et POLAR. Parmi les 21 candidats, trois lacunes ont été comblées au niveau natif : plaine AF, plateau BSK et plaine POLAR. Les seuils n'ont pas été modifiés. Ces constats de sélection sont séparés de la campagne finale de 900 images et de sa validation visuelle.

Une recherche supplémentaire, bornée et sans requête de génération, a mesuré
820 autres carrés de 40 km entièrement couverts par les caches natifs actuels :
147 AF, 151 CFB, 206 DFC et 316 POLAR. Les 1 148 décalages sans couverture
complète ont été rejetés. Aucun ne satisfait simultanément les trois critères
du plateau ; le meilleur AF atteint 17,34 % de hautes surfaces planes, sous
le seuil de 19 %. Les quatre lacunes restent donc ouvertes. Le code, les
mesures exactes à 30 m et la provenance sont conservés dans
`output/render-v2-reference/cached-plateau-search/`. Il ne s'agit pas d'une
preuve d'absence de plateaux dans tout le monde, seulement d'une recherche
explicite dans le terrain déjà disponible.

Les revues sont liées à la version par `runId` ; une ancienne appréciation n’est pas réutilisée sur une nouvelle version du rendu. Exemple de structure :

```json
{"runId": "identifiant-copie-du-report", "sites": {"af-rolling": {"status": "accepted", "reliefAccepted": true, "notes": "Comparaison aux cinq zooms et aux deux dates."}}}
```

Les nombres de cases acceptées strictes et de transitions représentatives acceptées sont conservés séparément dans `coverage.json`. Une transition visuellement valable à 74 % de sa famille peut donc rester utile sans être annoncée comme une case pure à 80 %.

Pour les deux centres ciblés, le runner mesure également le carré natif de 160 m : présence de mer pour `user-pack-ocean`, rivage mixte pour `warm-coast-beach-detail`. Cette vérification géométrique ne démontre ni la qualité de la banquise ni la conservation visuelle de la plage ; ces points restent à examiner sur les PNG.

```powershell
node tools/verification/assemble_render_v2_reference.cjs
node tools/verification/assemble_render_v2_reference.cjs --sheets --require-complete
```

L’index et les pages par centre affichent les cinq zooms et donnent accès aux PNG à leur résolution complète. Les planches regroupent au plus quatre centres et cinq zooms, pour chaque vue. L’assemblage lit uniquement les captures PNG réelles ; il ne fabrique aucun DEM ni texture synthétique. Sans `--require-complete`, le statut reste explicitement provisoire et les cellules manquantes sont affichées telles quelles. Les appréciations humaines peuvent être consignées par identifiant dans `reviews.json` ; leur état reste distinct des 900 captures techniquement vérifiées.

Chaque cellule capturée propose aussi « Ouvrir dans Neural Earth » : monde
partagé, centre, emprise, taille, LOD forcé, layer, saison et mois hydrologique
sont repris du reçu. Ce lien ouvre le viewer local courant ; le PNG et son hash
restent la référence de la version figée si le logiciel évolue ensuite.

## Entrée commune terre et hydrologie

L'assembleur accepte `REVIEW_REFERENCE_BASE` pour l'ensemble principal ou le complément hydrologique. Les nombres de sites, zooms, styles et images attendues proviennent de chaque manifeste : 60/900 pour le principal et 6/90 pour l'hydrologie.

```powershell
$env:REVIEW_REFERENCE_BASE='output/render-v2-reference'
node tools/verification/assemble_render_v2_reference.cjs
$env:REVIEW_REFERENCE_BASE='output/render-v2-hydrology-reference/captures-reference'
node tools/verification/assemble_render_v2_reference.cjs
```

Ces commandes préparent les pages HTML sans navigateur ni requête au viewer. Le point d'entrée commun est `output/render-v2-reference/index.html` : 66 situations, cinq zooms et 990 images prévues. Les liens entre galeries sont relatifs et utilisables en ouvrant ce fichier local. Un ensemble non encore assemblé dispose d'une page d'attente explicite. Les index donnent le `runId`, l'identité du monde et le hash de production de chaque ensemble ; la campagne commune ne devient complète que lorsque les deux ensembles sont complets et partagent la même production et le même monde. `gallery-summary.json` conserve ces informations sans fusionner les reçus ou les versions des captures.

Les pages hydrologiques résument les preuves par image : mois effectif, nombres d'échantillons lac/rivière, coupe maximale et overlay réellement peint. Le témoin vectoriel historique est identifié comme tel lorsqu'il ne possède pas de raster physique. Les mesures natives sur 40 km restent requises, y compris pour les six sites hydrologiques. L'option `--require-complete` s'applique à l'ensemble sélectionné, refuse tout manque de fichiers ou de mesures, et vérifie l'identité du manifeste figé. Les SHA des PNG présents sont vérifiés lorsqu'ils figurent dans le reçu. Un assemblage complet de deux versions de production différentes est refusé ; aucun index ne le présente comme une référence cohérente de 990 images.

Le protocole du masque régional attend aussi `geometryCurrent` : la révision
du réseau doit correspondre aux dernières arrivées de terrain, après le délai
de rafraîchissement. Pour V2 fast, il exige `renderer.riverDisplay.ready`, le
mode `regional-mask`, la même `frameKey` que l'acquittement de l'overlay et le
mois demandé. Un repli vectoriel reste utile au produit mais ne valide pas
ce chemin GPU. Le runner lit réellement la texture RGBA16float soumise et
enregistre ses pixels mouillés, solides et sa couverture intégrée. Un masque
vide valide mesure 1 × 1 avec zéro pixel mouillé. Les anciens échantillons
physiques de classe rivière restent consignés comme sources ; ils ne servent
plus de preuve du tracé effectivement rendu.

L'option `--sheets` ajoute les planches PNG par un navigateur CPU sur un serveur local d'assemblage. Elle n'ouvre jamais le viewer de terrain sur 8765. Les sources de l'assembleur sont distinctes du protocole de capture : une amélioration de navigation ou de mise en page n'invalide pas les images déjà capturées.
