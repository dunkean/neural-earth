# Render v2 : implémentation et vérification

2026-10-23. Spécification : [render-v2.fr.md](render-v2.fr.md).

## Résultat et architecture

Un nouveau matériau `render-v2` et quatre diagnostics (`render-v2-substrate`,
`render-v2-moisture`, `render-v2-vegetation`, `render-v2-snow`) sont intégrés dans
Neural Earth et le Terrain Bench de Magna Urbis. Le menu présente un seul layer
**Render v2** ; les quatre diagnostics sont un sélecteur subordonné dans ses
réglages, à côté de la saison. Les anciens liens de diagnostic restent valides.
Les styles historiques et les
choix de moteur par défaut restent disponibles. Aucun travail agricole, poids,
DEM, routage ou pin de sous-module n'est modifié.

Le module NE `web/terrain_render_v2.js` définit l'évaluateur WGSL. Le miroir NumPy
est `backend/terrain_render_v2.py`. `tools/assets/sync_render_v2.cjs` génère les
équations GLSL versionnées dans le dépôt voisin ; aucune dépendance entre dépôts
n'est nécessaire à l'exécution. La fixture JSON commune permet de vérifier les
trois évaluateurs sur des entrées identiques, au lieu de supposer leur parité.

Le v2 réutilise le BRDF, les normales analytiques et les candidats de couronnes
de Render3D ; les sols, le couvert et les plages historiques servent de référence.
Son microhabitat, ses couleurs proches, son calendrier, sa composition d'eau et
ses diagnostics enrichissent cette référence. Il ne se contente pas de renommer Render3D.

La présence de ces fonctions ne vaut pas acceptation visuelle. L'[audit sur les
vraies tuiles](render-v2-audit-40km.fr.md) conserve les régressions de référence
et leurs comparaisons. Render et Render3D restent le seuil de qualité : les
plages, bancs sédimentaires et ombres du relief doivent rester lisibles.
Une amélioration de palette ou de détail ne compense pas leur disparition :
la comparaison porte sur les mêmes coordonnées, tuiles et emprises, et un
élément réussi du rendu historique devient un critère de non-régression.

### État courant : corrections et validation sur les tuiles réelles

- Les six couches de contrôles V2 sont transportées en `rgba16float`, notamment les températures
  saisonnières ; Render3D conserve son transport historique `rgba8unorm`. Cela
  réduit la quantification thermique sans changer les tuiles physiques.
- Le transport de pluie reprend les gradients thermiques saisonniers réels de
  la tuile. Une constante de lapse rate ne doit pas remplacer ces données.
- Le calendrier V2 alimente aussi le matériau de sol hérité. Changer de date
  recalcule ses textures de couleur nécessaires, en réutilisant les textures
  physiques et le terrain ; aucun nouveau décodage n'est demandé.
- La plage physique exclut le couvert végétal. Les flats sédimentaires utilisent
  un masque distinct : ils peuvent conserver une végétation estuarienne.
  L'exclusion des arbres résolus est décidée à leur racine, avec un masque
  de référence fixe à 30 m. Multiplier le reveal par la fraction de plage du
  pixel rendait les couronnes translucides dans le sable ; ce défaut a été
  retiré. Les six vraies captures côtières à 2,56 km et 160 m sont conservées
  dans `output/render-v2-floor/beach-root-final/`, avec deux témoins Render3D
  pixel-identiques. Les survivants gardent leur opacité, la moyenne lointaine
  conserve son approximation de couverture surfacique.
- Le détail du sol traverse les mélanges de sol ouvert et de litière. Les
  normales minérales ne deviennent pas des normales de couronnes.
- Le guide fermé conserve la couleur de canopée avant les anciennes saisons et
  la neige. Le gradient macro filtré dispose de son propre transport : sa
  normalisation d'éclairage ne doit pas supprimer les pentes fines du DEM.
- La modulation minérale est continue et de faible amplitude. L'ancien clamp
  saturait la forme avant de l'afficher et produisait deux aplats de couleur.
  La neige conserve son relief grâce à une compression des hautes valeurs après
  éclairage, plutôt qu'un écrêtage de chacun des canaux.
- Le couvert moyen utilise la population qui engendre réellement les couronnes.
  Sol ouvert, litière et canopée sont éclairés séparément puis mélangés en lumière
  linéaire ; les détails résolus rejoignent cette moyenne selon leur empreinte.
  Le bake coarse conserve cette réponse non écrêtée en half-float. Il utilise
  toutefois une espérance régionale du peuplement, et non une intégration exacte
  des candidats fins.
- Les perturbations forestières de 100–400 m éclaircissent désormais le couvert
  au lieu de vider des cellules entières : `min(.55, 2*c*sigmoid(score/1.15))`.
  Le taux `c` reste lié au contrôle et au milieu ; en forêt humide il est réduit.
  Les ouvertures physiques et les trouées entre candidats restent distinctes.
  L'ablation sur les vraies tuiles CFB retire les petites taches ocres sans
  assombrissement global, avec des contrôles Render3D pixel-identiques.
- La variation de densité d'un peuplement agit sur sa couverture optique
  (`1 - (1-cover)^density`), au lieu de multiplier directement le pourcentage
  de couvert puis de le saturer. Cette dernière opération creusait de grands
  réseaux clairs dans les forêts froides au LOD3. La formule est un contrôle
  d'apparence saturant ; elle ne constitue pas une mesure de LAI.
- La neige est composée au-dessus de la litière, dans les chemins fins et
  coarse. Le contrôle GPU sous neige complète compare 2 098 pixels de sol
  forestier au sol enneigé ouvert : différence d'albédo nulle. La litière ne
  doit plus réintroduire du brun par-dessus cette neige.
- Le manteau neigeux conserve une composante annuelle : le maximum de la
  neige saisonnière et d'un plancher alimenté par le mois le plus chaud,
  l'humidité et la rétention par pente. Ce plancher utilise l'intervalle
  thermique permanent du matériau historique, sans masque de biome ajouté
  ni bruit de peuplement. Il corrige une vraie plaine EF devenue gris beige
  en été dans la passe rejetée `89516…`. La preuve `2b6b939d8e0e…` contient
  27 images réelles aux trois échelles : les trois vues V2 de juillet EF
  retrouvent un manteau clair ; les 18 témoins DFC/arides restent pixel-identiques.
  Render3D et janvier EF sont aussi inchangés. Reçu et comparaison dans
  `output/render-v2-snow-annual-reference/`. Cela reste une approximation
  climatique sans bilan d'accumulation ou de fonte interannuelle.

Ces corrections demandent encore une comparaison visuelle complète après
stabilisation du shader, notamment pour l'exposition des forêts tropicales.

Les tests courants vérifient 77 cas Python de rendu (27 historiques, 8 Torch,
42 V2), la compilation GPU des
chemins fins et coarse, le transport des températures/pluies, le changement de
saison sans nouvel upload physique, les plages et l'isolation sol/couronne.
Ils ne remplacent pas l'acceptation des vraies images. Les chiffres de captures
et performances plus anciens ci-dessous décrivent leurs versions respectives.

Le transport de canopée corrige aussi une confusion de canaux : l'humidité de
berge occupait le canal historique de ton forestier. Elle ne doit donc plus
assombrir une couronne opaque. Le contrôle GPU d'isolation donne une différence
nulle entre les deux humidités de berge pour ces pixels. Le champ écologique
de diagnostic utilise désormais le même couvert effectif que la couleur.

La campagne préparée contient 60 positions terrestres/côtières et six situations
hydrologiques, chacune à cinq emprises et dans trois vues (V2 juillet,
Render3D juillet, V2 janvier). Les images finales ne sont pas encore acceptées :
les répertoires de travail ne doivent pas être présentés comme les 990 captures
finales. Le manifeste distingue les reliefs demandés des reliefs réellement
mesurés ; quatre combinaisons plateau/biome ne sont pas couvertes strictement.

Le rechargement du serveur nécessaire au raster de lacs a changé l'identité
physique de `f5cddc…` en `bd3673…` : le manifeste de provenance inclut aussi
`index.html` et le serveur. Les sources hauteur, climat et érosion restent
identiques. Les anciens caches et preuves ne sont pas renommés ; la nouvelle
campagne recalcule ses vraies tuiles et conserve ses propres reçus.

## Évaluateur de milieu

Entrées : position canonique en mètres, graine, altitude/gradient physiques,
position topographique locale, températures et pluies des deux saisons de
référence, sable régional, enveloppe riveraine et date annuelle. Sorties :
température locale, humidité potentielle/actuelle, exposition, province
minérale, sable/argile/organique, affleurement, tapis herbacé, potentiel ligneux,
hauteur, part résineuse, feuillaison, neige, potentiel de gel et ton de peuplement.
Le champ séparé de provenance glaciaire est décrit ci-dessous.

Les provinces de 180/690 km mélangent noir/gris, brun, beige/calcaire et rouge
ferrugineux, sans la neutralisation grise du matériau historique. Une translation
de domaine dépendant de la graine évite que toutes les petites cartes locales
autour de l'origine aient la même province. Les mosaïques de 2,4 km et 670 m
modulent modestement le microhabitat avec les fonds, crêtes et versants. Ce sont
des proxys d'apparence, pas une nouvelle météo ou une carte géologique mesurée.

Sable, argile et matière organique sont bornés et leur somme ne dépasse pas un.
Roche et herbacées se partagent le sol ; la canopée forme un étage distinct. La
neige est une rétention continue dépendant du froid, de l'apport et de la pente,
sans seuil de bruit qui disperse des taches blanches isolées. Le sable reste une
famille de matériaux : aucune dune géométrique ou incision fictive n'est ajoutée.

Les bandes sont filtrées selon l'empreinte du pixel, sans renormalisation des
octaves restantes. Les bruits de microhabitat disparaissent dans les vues
lointaines ; ils ne sont pas échantillonnés sans filtrage à 20 km/pixel. Les
normales fines conservent les coordonnées entières/fractions de Render3D. Le
tapis herbacé a un gain de normale faible et une variation de sward bornée à
2,5 %, sans transformer de petites couronnes en brins d'herbe.

## Calendrier, peuplements et diagnostics

Le slider visible **Saison** a son propre état `v2season`, restauré depuis l'URL
et les recettes. Le slider historique est masqué pendant l'affichage v2. Le
calendrier pilote couleur et opacité des feuilles, ombres, tapis herbacé, humidité,
neige et glace. Les candidats, textures de contrôle et graines ne sont pas
reconstruits au déplacement du slider. Décembre/janvier se raccordent exactement.
Dans Magna, le déplacement de la saison met à jour les uniforms carte/perspective
sans reconstruire les textures projetées. NE utilise ses références saisonnières
spatiales, donc les hémisphères suivent
leur propre calendrier. Le mois hydrologique des requêtes v2 et la clé du cache
d'eau changent aux frontières mensuelles ; cela ne relance pas l'inférence.
Avec l'hydrologie active, ces frontières peuvent demander de nouveaux buffers
d'eau via le réseau local et réuploader une tuile combinée ; l'absence d'upload
physique vérifiée par le probe de calendrier concerne le matériau sans ce
changement mensuel. Il n'y a pas de simulation hydrologique continue par frame.

Les quatre diagnostics appellent le même évaluateur que la couleur finale. Ils
montrent le substrat, l'humidité locale, les étages de végétation et neige/glace.
La couverture de glace de l'eau appelle le même compositeur `v2Ice` /
`ice_material` que le matériau final, y compris plaques et icebergs. Elle n'est
pas une simple palette du potentiel de gel. Ils ne remplacent pas les anciens
diagnostics climatiques physiques, qui gardent leur signification régionale.

Le champ stable v2 réemploie les candidats de couronnes déterministes et les
écotones de Render3D. La galerie riveraine NE est une enveloppe étroite (65 m
maximum dans cette adaptation raster) liée aux échantillons de rivière faisant
autorité. Magna conserve son enveloppe géométrique existante, ses occupations
appliquées et ses exclusions de bâtiments. Une racine sur l'eau est rejetée ;
l'eau est composée en dernier. Le retrait saisonnier des feuilles réduit aussi
la couverture et les ombres, sans tirer de nouvelles positions d'arbres.

## Eau, glace et provenance des icebergs

Les RGBA physiques transportent niveau, fraction mouillée, classe et autorité.
Un échantillon sec faisant autorité reste terrestre même sous le niveau marin.
Le v2 compose une eau dépendant du substrat, de l'humidité, de la profondeur et
de la fraction du chenal. Le bord peu profond et le centre profond diffèrent.
En mer, après l'absorption des faibles profondeurs, une teinte bathymétrique
conserve les nuances du plateau continental jusqu'aux abysses, entre 150 et
3 000 m. Cela préserve le repère de profondeur de Render3D dans les vues larges ;
il ne s'agit pas de rendre le fond abyssal optiquement visible. Les lacs et
chenaux ne reçoivent pas cette teinte marine.
Les glacements lacustre, fluvial et marin ont des seuils distincts ; les cours
en mouvement gèlent plus prudemment. Le froid pilote une concentration légèrement
renforcée dans les faibles profondeurs côtières. La profondeur est un proxy
littoral disponible, pas une nouvelle distance géométrique à la côte.

La banquise compacte conserve de grandes plaques jointives d'environ 1,2 km,
faiblement déformées (65 m), séparées par de minces chenaux et leurs crêtes. Les
isolignes de bruit fermées initialement employées comme petites fissures ont été
retirées : elles produisaient des contours artificiels visibles à 5 m/pixel.
La marge moins concentrée sélectionne des plaques opaques stables : parents de
2,6 km, sous-fragments de 470 m dans certains parents, déformations XYZ
indépendantes de 215 m. La correction marginale élimine les plaques entières
semi-transparentes et les réseaux cyan sous les fragments absents : sélection
binaire, liseré limité à la glace présente. Le passage compact/marginal est
continu entre concentrations 0,68 et 0,90.

Le champ cellulaire compare les deux distances les plus proches parmi 27 points
d'une grille 3D déplacée déterministement. La marge, les icebergs et la glace
intérieure gardent leurs contours `F2-F1`. Le pack marin compact emploie la
distance à la bisectrice des deux sites : `(d2²-d1²)/(2|site2-site1|)`.
Ce changement évite que la largeur d'une fracture grossisse en poche près des
sommets. Les chenaux de quelques mètres sont intégrés à l'empreinte du pixel,
et le réseau secondaire de glace refondue est atténué. Le relief de compression
reste étroit ; aucun relief de ces plaques ne déplace le DEM.
Le générateur adapte la base cellulaire de
[Worley (1996)](https://doi.org/10.1145/237170.237267), sans son semis Poisson
général. Les coordonnées sont sphériques dans NE et métriques régionales dans
Magna ; aucune graine locale de tuile n'est introduite. Les largeurs des bords
augmentent avec l'empreinte du pixel ; entre 140 et 450 m/pixel, couverture,
clarté et liserés convergent vers une moyenne analytique calibrée. Ce n'est pas
une intégrale exacte de l'aire de chaque plaque. Au-delà, les cellules et les
fortes variations provinciales ne sont plus échantillonnées. Aucun de ces effets
ne déplace le DEM ni les niveaux d'eau.

NE réutilise les débits mensuels lorsqu'ils existent. Une série de repli uniforme
ne prouve pas un oued sec. Le mode hydrologique rapide fournit au v2 un raster
d'autorité sans carving : l'élévation physique reste l'originale. Les overlays
bleus historiques ne repeignent pas les emprises déjà composées par le v2.

Dans la voie rapide, `v2_lakes` ajoute un raster des grands lacs de source
(seuil par défaut 1 km²) au compositeur V2. Seuls les points du DEM sous le niveau
du lac sont mouillés ; les îles restent sèches et le cut/fill est strictement
nul. Cette option n'active pas les petites dépressions locales historiques.
Au gros plan, dans le support lacustre existant, le bord est l'intersection
du plan d'eau avec le DEM affiché, antialiasée selon son gradient et le pixel.
Il ne s'agit plus de dilater la fraction bilinéaire 30 m sur toute la rive.
Le passage conserve progressivement cette fraction filtrée entre 15 et 60 m/px ;
la mer et les chenaux ne sont pas modifiés par ce traitement réservé aux lacs.
Les chemins coarse natifs de LOD 4 et plus utilisent encore les vecteurs : leur
gel lacustre n'est pas couvert par ce nouveau raster. Une tuile 30 m réellement
recalculée sous `bd3673…` vérifie 26 395 échantillons lacustres, une autorité
valide et aucune modification du DEM ou de la sortie legacy.

Le canal de transport 50, ajouté aux 50 canaux historiques, est une proximité
canonique de côtes glaciaires potentielles. Il est calculé et mis en cache depuis
l'atlas source, avec terre côtière, été perpétuellement froid et apport minimal.
Il est continu à l'antiméridien et dans le chart polaire ; il ne dépend pas des
bords de la tuile affichée. Voir la [méthode de provenance](../methods/render-v2-glacier-provenance.md).
Les icebergs utilisent des fragments cellulaires érodés et lobés, de tailles
variées jusqu'à plusieurs centaines de mètres, plutôt que des disques. La
provenance règle la densité de sélection, pas la transparence de leur surface.
Un plateau blanc, des versants plus sombres, un liseré immergé cyan et une ombre
orientée donnent du volume satellite. L'ombre est déduite de la même cellule,
sans seconde recherche cellulaire. La profondeur atténue la provenance mais
laisse possibles les fjords profonds. Le froid marin seul ne crée aucun iceberg.
Entre 115 et 290 m/pixel, les blocs convergent vers leur aire moyenne approchée.
Ce sont des proxys localisés, sans vêlage, dérive, courants ou thermodynamique
simulés. Les références visuelles sont les
[stades NSIDC](https://nsidc.org/learn/parts-cryosphere/sea-ice/science-sea-ice)
et l'[atlas aérien NASA IceBridge](https://svs.gsfc.nasa.gov/11143).

Magna utilise au plus huit groupes de cellules sources côtières et froides du
champ conservé, préparés une fois côté CPU ; ses fragments ne recherchent pas
une côte dans des centaines d'échantillons de terrain. Son climat est un preset
annuel : l'adaptation nord été/hiver reprend l'amplitude continentale
`6 + 0.5 max(0, 10 - T_annuel)`. Elle ne reconstitue pas un climat planétaire.

## Lointain, vue proche et sorties

NE utilise le même milieu à grande échelle pour les aperçus CPU/PNG, blocs
coarse, quads WebGPU et globe. Le lointain ne dessine pas d'arbres ou de pierres
individuels non résolus. Il n'y a pas trois couches artificiellement indépendantes
Orogen/Coarse/Base : leurs données et leur empreinte déterminent les fréquences
visibles. Les détails géométriques du terrain restent ceux des stades existants.

Dans Magna, la carte et la perspective utilisent le même GLSL, les mêmes champs
forestiers et les masques conservés. Le raster/export utilise le même matériau
et son intégration à la résolution demandée ; il ne revendique pas une égalité
pixel par pixel de couronnes entre une exportation et une caméra d'une autre
résolution. Les mers v2 restent dans le raster complet : ni extension des couleurs
terrestres ni clip terrestre ne les effacent. La note anglaise
`city_generator/docs/methods/render-v2.md` est embarquée dans le viewer de méthodes
et accessible via le bouton d'aide de la saison/style, y compris hors ligne.

## Vérifications reproductibles

Les artefacts générés restent ignorés par Git.

| Vérification | Commande / résultat |
| --- | --- |
| CPU environnement, calendrier, fractions, eau et diagnostics | `python -m unittest discover -s tests/python -p 'test_terrain_render_v2*.py'` : 22 réussites, dont 6 contrats polaires (provenance, opacité, pack compact, stabilité, intégration LOD et renforcement côtier) |
| Eau polaire CPU/WGSL réelle | `node tools/verification/verify_render_v2_polar.cjs` : 96 positions/empreintes × 7 champs, erreur absolue maximale 0,000002 ; atlas mer seule et zoom sur un vrai iceberg sous `output/render-v2-polar/` |
| WGSL réel sur fixture commune | `node tools/verification/verify_render_v2_parity.cjs` : 9 cas × 18 champs, erreur absolue maximale 0,000003 |
| GLSL réel et raster Magna | Depuis `../city_generator/web`, `node scripts/render_v2_ui_check.mjs http://127.0.0.1:5173 --raster-only` : mêmes 9 cas × 18 champs, erreur maximale 0,000003 ; export côtier identique au matériau raster (erreur maximale 0/255), sans clip terrestre ; captures et rapport sous `web/out/render-v2/` |
| Menu réellement visible NE | `node tests/browser/test_terrain_render_v2_menu.cjs` : une seule entrée v2, saison, diagnostics, sélection et restauration des anciens liens ; aucune erreur JavaScript |
| Pixels WebGPU v2 | `node --test tests/browser/test_terrain_render_v2.cjs` : six milieux, répétition exacte, boucle annuelle exacte, champ stable entre dates, eau exclue du diagnostic végétal, gel/dégel lacustre et captures proches ; la côte froide utilise 100 m/échantillon pour couvrir une zone assez grande pour les rares icebergs |
| Régressions NE | 56 tests render, 16 routes, 4 provenance glaciaire, 4 cache hydrologique, 9 world et 46 navigation réussis ; navigation exécutée avec `TERRAIN_DISTILL_LIVE=0` pour neutraliser une hypothèse antérieure teacher/student du test de préchauffage |
| Neige, métrique et héritage écologique | `node tools/verification/verify_render_v2_snow.cjs` : 29 contrôles réussis, 64 vues, neige plane, éclairage, isotropie aux latitudes 0/67,8/85, maintien du vert aux DEM 30/240 m et aucun candidat ligneux en climat EF riverain |
| Régressions GPU historiques | Render3D et hydrologie WebGPU compilés et exécutés sans erreur |
| Magna | Depuis `web/` : `npm run typecheck`, `npm run build`, `npm run build:benches` ; 29 tests ciblés via `npx vitest run tests/terrain3dStitch.test.ts tests/terrain3dPicking.test.ts tests/render.forest-clearings.test.ts tests/render.ground-appearance.test.ts` |

Les captures synthétiques NE sous `output/render-v2/` couvrent aride minéral,
tempéré, forêt humide, côte sableuse, montagne enneigée et côte froide, été/hiver
et vue proche. Une capture `bank` est centrée volontairement sur un chenal
vertical synthétique ; `close` vise le peuplement terrestre, pour distinguer une
vraie berge d'une rupture de peuplement. Les captures réelles Orogen stabilisées
(63 patches, aucune tuile visible en attente) et globe sont sous
`output/render-v2-review/`. Aucun échec console/HTTP n'a été observé dans cette
vérification de l'application réelle. Les captures ne constituent pas une
comparaison métrique avec une image Google Earth.

L'atlas polaire contrôle océan chaud, pack marginal et compact, mer glaciaire,
bergs dans le pack, lac et rivière à 0,5 / 5 / 50 / 500 m/pixel. Le zoom détaillé
est centré sur un fragment présent, pas sur une ancre arbitraire qui pourrait
montrer uniquement de l'eau. Les PNG d'itérations conservent les comparatifs :
plaques compactes jointives, correction des halos marginaux, puis retrait des
isolignes artificielles à l'intérieur du pack compact.

La première validation NE portait sur Orogen, les blocs source et le globe,
avec des grilles synthétiques pour les détails. La revue utilisateur du 23 octobre
a ensuite exercé le décodeur réel sur le monde partagé
`7985403534069300490`, sans changer sa recette : 77 patches à 30 m pour
le site côtier de latitude de chart 67,83° et 84 patches à 30 m pour le site
continental de latitude −1,36°. Les comparaisons gardent caméra, saison et
éclairage identiques et utilisent aussi Render et Render3D. Elles ont révélé
des régressions absentes des fixtures : plage perdue, stries de microhabitat,
couvert trop sensible à la pente du décodeur et saturation sur certaines crêtes.
Les captures initiales sont `user-after-v2.png` / `user-after-render.png`
et `green-before-{v2,render,render3d}.png` sous `output/render-v2-review/`.
Le cas lointain `arid-after-{v2,render,render3d}.png` utilise les sources Coarse
avec une caméra à 3152,08 m/pixel. Ces constats exigent une comparaison visuelle
après correction ; la réussite des probes synthétiques ne suffit pas à déclarer
la qualité satellite terminée.

La carte et la perspective Magna ont également été vérifiées dans l'application
réelle sur ANGLE Vulkan 1.4.341 / NVIDIA RTX 3090. Les captures été/hiver
`output/render-v2-review/cg-map-hardware.png`,
`cg-perspective-hardware.png` et `cg-perspective-winter-hardware.png` montrent
le terrain rendu après réception des textures, sans erreur console. Un premier
placeholder opaque pouvait apparaître brièvement bleu ; son alpha est désormais
nul à l'entrée du style v2 avant réception de l'overlay réel. Les autres styles
conservent leur initialisation.

L'export côtier contrôlé `city_generator/web/out/render-v2/coast-export.png`
présente terre et mer dans le même raster, sans clip terrestre. Cinq positions
de la côte vers le large correspondent exactement au shader raster (écart
maximal 0/255 ; mer RGB [6,31,54]). Le rapport `raster-report.json` conserve les
neuf comparaisons GLSL et la différence été/hiver moyenne de 22,247/255.

## Corrections issues de la revue utilisateur

La première passe de correction conservait les couleurs ouvertes, herbacées,
affleurements, couvert ligneux et plages de Render3D à moyenne et grande distance.
La substitution de palette entre 6 et 1,5 m/pixel a depuis été retirée : la
moyenne ouverte héritée est conservée à tous les footprints, enrichie par une
variation minérale bornée avant les sédiments, plages et neige. Une pente fine du
décodeur ne déclenche plus la substitution générale du vert par la roche v2.
La canopée reste saisonnière. Le renfort riverain respecte le froid, la neige
pérenne et la hauteur thermique, sans créer de couronnes dans les milieux EF.

Les provinces minérales restent fixes ; leur contraste affiché s'efface entre
600 et 2400 m/pixel. Les rouges ferrugineux sont désaturés. Le modelé régional
reprend le hillshade canonique, avec les réglages d'exagération, d'ambiance et de
contraste de l'utilisateur ; le BRDF ajoute seulement les réponses locales.
La neige froide peut couvrir entièrement un terrain plat ; elle atténue le grain
minéral au lieu de laisser systématiquement 18 % de substrat visible.

Les bruits locaux terrestres, y compris ceux du sol hérité, utilisent un domaine
cylindrique dans le chart du DEM. Le témoin polaire à 85° et 41,9 m/pixel passe
d'un rapport de variation horizontale/verticale de 0,0765 à 0,973 pour la neige
partielle ; le sol nu est proche de 0,99. Les champs marins sont évalués séparément
en coordonnées sphériques. Les réglages fins de banquise gardent les plaques
jointives et déplacent leurs bords ensemble avec un champ vectoriel, plutôt que
de faire varier le seul seuil des chenaux. Le modelé mat reste faible ; il n'y a
ni dynamique de fracture ni validation photoréaliste de ces plaques.

Les captures après correction `green-after-v2.png`, `coast-after-v2.png` et leurs
comparatifs Render3D montrent les deux sites réels au décodeur, à caméra et
éclairage identiques. Les captures `green-lod3-after-*` conservent aussi la même
caméra avec le latent 240 m forcé. Le site de glace est vérifié au LOD −1 réel
(15 m, 84 patches), à 24,43 m/pixel. Les noms d'itérations et rapports sont sous
`output/render-v2-review/` ; les captures intermédiaires ne sont pas une promesse
de conformité visuelle de toutes les cartes.

Dans Magna, les trois voies carte/perspective/raster évaluent une seule fois le
matériau historique et transmettent son sol avant neige à v2. La plage est
conservée, et la neige du calendrier v2 est composée ensuite. L'appel partagé
évite de dupliquer les grandes fonctions forestières dans le shader.

## Coûts mesurés et limites

Sur la révision `15f91fbd07b9…`, après les corrections de peuplement,
neige/litière et plage, le benchmark RTX 3090 sans autre travail GPU mesure
sur sa fixture 1920 × 1080 : bake **0,900 ms/tuile GPU**, upload avec attente
**2,050 ms/tuile**, draw à 16 / 4 / 1 / 0,25 / 0,02 m/px :
**1,289 / 4,627 / 1,782 / 5,837 / 6,910 ms GPU**. Saison puis draw :
**2,019 ms GPU**. Initialisation : 5,181 s ; compilation paresseuse partagée :
8,119 s. Rapport : `output/render-v2-floor/final-benchmark-15f91fbd.json`.
Cette fixture n'active pas un réseau hydrologique : elle ne mesure pas la
régénération du masque régional lors d'un déplacement de caméra. Un masque
non vide occupe environ 12 octets par pixel plus ses segments, soit 12,58 Mo
à 1024² ; le masque vide valide mesure 1 × 1. Ces chiffres ne sont pas une
cadence de navigation incluant génération, réseau et composition navigateur.

Sur les sources figées `db083bf3841a…`, le même benchmark a été réexécuté
sans autre capture GPU ni calcul neural actif. Sur RTX 3090, fixture
1920×1080 : bake V2 **0,902 ms/tuile GPU**, upload avec attente de fin
**1,944 ms/tuile** ; draw à 16 / 4 / 1 / 0,25 / 0,02 m/px :
**1,114 / 4,006 / 1,610 / 5,121 / 6,692 ms GPU**. Changement de saison et
draw : **1,839 ms GPU**. Initialisation : 6,776 s ; compilation paresseuse
partagée : 7,797 s. Rapport : `output/render-v2-floor/final-benchmark.json`.
Les coûts ne sont pas monotones avec le zoom : moyenne statistique et couronnes
résolues empruntent des branches différentes. Les timestamps GPU et médianes
murales sont des mesures séparées ; ce probe ne mesure ni l'inférence, ni le
réseau, ni une cadence complète de navigation. Le probe polaire courant mesure
0,691 ms GPU pour 2 359 296 pixels ; dispatch et lecture CPU prennent 40,3 ms.

Avant la révision polaire, sur RTX 3090/Ampere et une fixture synthétique
1920×1080, le benchmark existant lancé
avec `node tools/benchmarks/benchmark_render_layer.cjs --render-v2` mesure des
timestamps GPU séparés. Bake : 0,480 ms/tuile ; upload et fence : 0,922 ms/tuile.
Draw à 16 / 4 / 1 / 0,25 / 0,02 m/pixel :
0,882 / 2,631 / 3,274 / 5,943 / 7,324 ms. Un changement de saison suivi d'un draw
mesure 2,894 ms sur cette fixture, sans nouveau bake du champ stable. Compilation
paresseuse partagée : environ 4,9 s à froid, initialisation environ 3,2 s. Ce ne
sont ni une garantie de 60 fps, ni un temps complet de génération, transfert
réseau, composition navigateur et présentation. Le probe polaire final mesure
0,637 ms de timestamp GPU pour 2 359 296 pixels d'atlas synthétique sur la même
carte ; les 43,8 ms de dispatch et lecture de
32 Mo incluent le transfert CPU. Cette mesure isolée n'est pas une cadence de
l'application. Le GPU court-circuite l'eau sur les pixels terrestres ; le CPU
saute la composition d'eau sur les surfaces entièrement sèches. La
recherche cellulaire est aussi sautée quand le pack ou la provenance sont
absents et au-delà de leurs empreintes d'intégration. Les timings des petits probes
de compilation GLSL sous SwiftShader ne sont pas des mesures GPU utilisateur.

Limites explicites :

- Le filtre de profondeur minimale des lacs agit sur l'overlay historique,
  mais n'est pas transmis au raster de lacs V2. Sa valeur ne filtre donc pas
  encore ces surfaces ; le filtre d'aire dispose bien de son paramètre dédié.
  La revue du raccord rivière a confirmé ce défaut distinct, sans le corriger
  par une exclusion approximative de tous les lacs. Les captures de référence
  utilisent le seuil par défaut de zéro.
- Le modèle reste une apparence naturelle synthétique, sans météo historique,
  accumulation neigeuse, transport sédimentaire ou géologie mesurée nouveaux.
- Les climats, débits et tracés de source gardent leur précision disponible. Le
  raffinement n'invente pas un débit saisonnier absent. Magna ne possède pas de
  séries mensuelles de débit : aucune rivière n'y est arbitrairement asséchée
  par sa seule aridité locale.
- La couverture lointaine est une moyenne d'apparence calibrée, pas une
  conservation exacte de l'aire de chaque champ cellulaire. La concentration
  suit le climat disponible et la profondeur côtière ; aucune chaleur océanique
  ni mémoire interannuelle de glace n'est disponible.
- Les normales et calottes d'icebergs sont des effets de matériau satellite,
  pas des silhouettes géométriques 3D ni une simulation de dérive.
- Le microhabitat terrestre et les bruits locaux hérités suivent la métrique
  du chart qui porte le DEM, avec un domaine cylindrique périodique en longitude.
  Les provinces larges et les matériaux marins gardent la sphère canonique.
  Sur le globe, relief et texture subissent la même projection ; cela ne rend
  pas le DEM cartographique intrinsèquement métrique sur toute la sphère.
  Le passage entre charts normal et polaire peut modifier le détail terrestre
  local : son identité entre charts n'est pas garantie.
- Les gradients/TPI des champs stables restent ceux des tuiles physiques et de
  leurs halos ; un nouveau DEM/LOD peut encore modifier légèrement l'éligibilité
  des arbres d'un bord. Les graines et positions candidates sont stables, mais
  une invariance absolue des couronnes entre tous les raffinements n'est pas
  démontrée. Aucune parité parfaite entre des DEM de résolutions différentes
  n'est revendiquée.
- En perspective Magna, une texture d'eau intérieure sans classe séparée est
  traitée prudemment comme eau en mouvement pour le gel. Sa profondeur intérieure
  inférée depuis la couverture est un proxy d'apparence, pas de la bathymétrie.
  L'aperçu CPU représente
  la couverture moyenne, pas la géométrie de chaque couronne rapprochée.
- Le compilateur ANGLE/D3D matériel de cette machine bloque aussi dans un probe
  du constructeur TerrainGl issu de HEAD (aucune génération de terrain), avec
  watchdog à 60 s ; le constructeur courant atteint le même watchdog. Le probe
  HEAD garde seulement les petites additions de uniforms/atténuation du module
  forestier partagé : il ne constitue pas une mesure d'une baseline entièrement
  isolée. ANGLE Vulkan sur NVIDIA RTX 3090 a toutefois compilé l'application
  réelle et rendu carte et perspective été/hiver sans erreur. La limite porte
  sur le backend D3D de cette machine ; aucune cadence universelle ni performance
  matériel de perspective n'est déduite des contrôles SwiftShader.

Les références d'apparence sont [PBRT Noise](https://pbr-book.org/3ed-2018/Texture/Noise),
[Filament](https://google.github.io/filament/main/filament.html) et
[NSIDC Sea Ice](https://nsidc.org/learn/parts-cryosphere/sea-ice). Les adaptations
utilisent cohérence spatiale, filtrage et séparation matériau/éclairage ; elles
n'implémentent pas les modèles physiques complets de ces références.
