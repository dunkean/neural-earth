# Render v2 — paysage satellite procédural multiscale

Date : 2026-10-22. Spécification d'implémentation pour Neural Earth et Magna Urbis.

## Objectif et périmètre autorisé

Créer un nouveau layer **Render v2** (`render-v2`, avec compatibilité explicite si une API exige `Render_v2`) qui représente un paysage satellite naturel, du premier aperçu Orogen aux vues rapprochées. Livrer son implémentation dans Neural Earth et dans le projet voisin `city_generator`, côté Terrain Bench/Rust bridge et vue en perspective existante. Conserver les layers et réglages actuels comme alternatives ; ne pas changer le moteur ni le style par défaut.

Précisions utilisateur : des vues/traitements séparés **Orogen**, **Coarse** et **Base** sont autorisés, avec raffinement fin à partir des décodeurs. Sa dernière consigne laisse toutefois le choix technique à l'implémentation (« fais comme au mieux ») : cette séparation est une possibilité, pas une obligation de créer trois nouvelles vues. Choisir les transitions offrant le meilleur résultat avec les données disponibles ; ne pas forcer le même shader ni la même définition entre étapes et ne pas confondre détail procédural et détail physique appris.

L'agriculture est exclue : aucune génération de champs, cultures, parcellaire ou calendrier agricole. Les emprises bâties/eaux déjà disponibles restent des contraintes d'exclusion de végétation. Ne pas modifier les poids neuronaux, le DEM physique, le routage hydrologique, les révisions amont ou le sous-module Magna Urbis pour réaliser un effet d'apparence.

Le résultat recherché est la variété spatiale d'une image satellite : provinces minérales, versants, vallées, dépôts, mosaïques de végétation, gradients d'humidité, écotones et eaux contextualisées. Le relief fin doit rester sobre et lisible. Un bruit supplémentaire ne constitue pas, à lui seul, une amélioration.

## Problèmes à résoudre explicitement

1. Petites herbes peu convaincantes, végétation sur l'eau, limites coupées par le filtrage et le flou.
2. Champs de climat/sols trop grossiers qui bavent au zoom ; détails graphiques non accompagnés de propriétés locales cohérentes.
3. Neige, sable et autres couvertures qui deviennent des taches contrastées de type dalmatien à certaines échelles.
4. Palettes trop uniformes dans un même biome ; roches arides toutes semblables au lieu de provinces noires, brunes, beiges et rouges ; mêmes limites pour sables et prairies.
5. Peuplement forestier, clairières, densité et transitions de frontière insuffisamment naturels.
6. Absence de banquise structurée, glace lacustre et icebergs plausiblement localisés.
7. Eau trop uniforme et trop présente dans les cours saisonniers/secs ; manque de variation transversale du chenal.
8. Premier niveau Orogen peu défini : enrichir aussi le lointain, les blocs natifs et l'aperçu initial, sans attendre les tuiles fines.

## Contrat commun de surface

Définir un évaluateur déterministe des propriétés du milieu, à position mondiale et date données, utilisé par le rendu et ses diagnostics. La grille source fournit les conditions régionales ; elle ne doit pas être l'unique source de détail après interpolation.

Entrées en unités explicites : position en mètres, graine, altitude et gradient physiques, relief régional stable, climat des saisons de référence et gradients thermiques, substrat régional, eau faisant autorité et éventuels segments de rivière, date annuelle normalisée. Dans Magna Urbis, un preset annuel et la continentalité existante constituent un adaptateur déclaré, pas un climat planétaire reconstruit.

Propriétés : température saisonnière locale, humidité potentielle et actuelle, exposition, roche mère/minéralité, fractions sable/argile/matière organique, sol apparent, gravier/éboulis, herbacées, couvert ligneux, hauteur et type fonctionnel, neige/glace, classe et fraction d'eau. Les couvertures concurrentes d'un même étage sont bornées et normalisées ; substrat, couverture et canopée sont des étages distincts.

Le raffinement est un modèle d'apparence et de microhabitat, non une nouvelle simulation météorologique ni une mesure géologique. Les couches climatiques physiques historiques conservent leur sens. Fournir au minimum des diagnostics Render v2 pour le substrat, l'humidité locale, la végétation et la neige/glace, dérivés du même évaluateur que les pixels finaux. Pas de diagnostic décoratif calculé séparément.

## Cohérence spatiale et multiscale

Séparer identité du milieu et filtrage d'affichage : une position conserve son matériau et ses candidats végétaux lors d'un changement de tuile, de caméra ou de LOD. Le niveau de détail règle les fréquences visibles et leur intégration, pas une nouvelle graine ou un nouvel écosystème.

Employer des coordonnées canoniques mondiales, avec séparation cellule entière/fraction locale pour conserver la précision. Sur la planète, les champs de province doivent être continus sur la sphère ; les détails locaux doivent éviter les ruptures à l'antiméridien et documenter leur métrique vers les pôles. Ne pas utiliser une origine de tuile comme graine.

Le climat régional est raffiné par des perturbations bornées et corrélées au relief : versants exposés/abrités, fonds humides, crêtes drainées et variabilité régionale. Ne pas réinventer des ombres pluviométriques fines par du bruit blanc. Préserver les moyennes régionales et limiter les amplitudes.

Les champs stables régionaux sont préparés et mis en cache ; les motifs résolus sont évalués par shader. Filtrer selon l'empreinte réelle du pixel : fréquences non résolues vers couleur/couverture moyenne et rugosité ; fréquences résolues vers texture. Pas de renormalisation qui amplifie les dernières octaves et fait apparaître des taches lors du zoom.

Échelles indicatives, à adapter aux unités du monde : provinces de dizaines à centaines de km ; versants et mosaïques de 100 m à plusieurs km ; peuplements/écotones de quelques mètres à centaines de mètres ; éléments de sol de quelques cm à quelques m. Les amplitudes physiques restent modestes à proximité.

## Vue d'ensemble Orogen et transition vers le proche

Le nouveau layer doit être présent dans les aperçus serveur/PNG, blocs natifs/coarse, carte et globe. Des traitements distincts pour Orogen, Coarse et Base sont souhaitables lorsque leurs données diffèrent ; leurs petites structures n'ont pas à reproduire celles des décodeurs. Enrichir les premières images par une variation de province et de versant, des réseaux alluviaux lorsque disponibles, des gradients de végétation et des structures minérales corrélées. Partager les grandes identités de milieu et le calendrier ; réserver le raffinement réel des petits détails aux sorties des décodeurs.

Ne pas dessiner des cailloux, arbres individuels ou faux ravins non résolus dans le lointain. Si une variation de normale sous-maille est ajoutée, la déclarer comme apparence procédurale, filtrée et sans déplacement du DEM. L'amélioration du premier niveau est un livrable à vérifier, pas une conséquence supposée du shader proche.

## Roche, sols, sable et herbacées

Roche : familles minérales continues et spatialement cohérentes, dont gris/noir, brun, beige/calcaire et rouge ferrugineux. Conserver cette diversité en évitant la neutralisation globale actuelle. Corréler affleurements, fractures, strates et débris au même matériau parental. Orienter motifs selon relief et province ; éviter un sinus horizontal universel.

Sols : mélanges continus, texture et humidité, contraste modéré. Dépôts de bas de pente et alluvions seulement avec des facteurs de relief/eau disponibles. Déclarer les proxys lorsqu'il n'existe pas de transport sédimentaire.

Sable : variété minérale et traces/rides orientées, sobres ; pas de grandes îles de couleur à haute fréquence. Grandes dunes uniquement si leur forme est compatible avec les données ou explicitement procédurale. Transition avec gravier et sol selon milieu.

Herbacées : tapis fin et touffes de petite hauteur uniquement lorsqu'elles sont résolues ; variation de vert, paille, sécheresse et sol visible selon humidité, température, substrat et saison. Éviter des couronnes de buisson rétrécies utilisées comme brins d'herbe. Les détails de normale doivent rester doux.

## Végétation et écotones

Construire une mosaïque de communautés fonctionnelles : feuillus, conifères, tropicaux, forêt sèche, savane/buissons, prairie/lande, zones humides lorsque les entrées le justifient. Varier hauteur, densité, teinte et regroupement à plusieurs échelles corrélées.

Conserver des candidats déterministes et une couverture moyenne stable. Définir les frontières par aptitude écologique et écotones irréguliers, avec changements de stature et de densité, plutôt qu'un découpage binaire ou un flou étalé. Ne pas ouvrir artificiellement une forêt tropicale humide par un bruit de clairières générique.

Les couronnes gardent leurs attributs à leur centre fixe. Les herbes, troncs/candidats et ombres de sol doivent respecter les masques d'eau et d'occupation ; exclure les centres immergés et composer la couverture d'eau en dernier. Un éventuel surplomb de couronne ne peut devenir un arbre enraciné dans l'eau. Les berges utilisent la vraie géométrie/largeur hydrologique, pas la largeur cartographique minimale.

Raccorder les galeries riveraines existantes de Magna Urbis et adapter leur enveloppe au réseau de Neural Earth. Contrôler largeur, hauteur, densité, transition et sol humide selon contexte ; pas de halo uniforme autour de toutes les eaux.

## Saison commune

Ajouter un slider visible **Saison** pour Render v2, avec mois/période lisible, cycle annuel continu et sauvegarde dans URL/recette/session. Un seul état de saison pilote tous les composants v2 et, lorsqu'il est pertinent, le mois hydrologique affiché ; les modes historiques conservent leurs propres réglages.

Le changement ne relance ni génération du relief ni inférence. Séparer champs stables et état saisonnier pour éviter une reconstruction générale à chaque cran. Garantir le raccord décembre/janvier et les saisons inversées des hémisphères.

Effets : verdissement/sénescence des herbes, feuillaison et couleurs des feuillus, persistance des conifères, saison sèche tropicale si le climat l'indique, humidité du sol, neige, glace et eau saisonnière. Les arbres ne se redistribuent pas au déplacement du slider. Une réduction du couvert foliaire doit affecter l'opacité/structure et les ombres, pas seulement la teinte.

Neige : couverture cohérente avec température, apport et rétention, roche exposée sur forte pente, variations de profondeur/apparence douces. Préférer des nappes liées au relief et des lisières fines aux taches isotropes contrastées. Une réponse saisonnière déterministe avec retard approché est acceptable, mais ne pas prétendre simuler une accumulation historique.

## Eau et cryosphère

Respecter l'autorité hydrologique et sa fraction surfacique. Conserver les dépressions sèches sous le niveau marin comme terre. Eau lacustre, rivière et océan partagent une famille de matériaux, contextualisée par profondeur, sédiment/substrat, milieu et couverture de glace.

Dans un chenal résolu, varier transversalement faible profondeur, lit visible, eau plus profonde et bord humide. Les bancs découverts demeurent terrestres. Employer la profondeur et la couverture disponibles ; une approximation visuelle ne modifie pas la bathymétrie.

Un oued peut être sec et montrer son lit : utiliser le débit mensuel/régime lorsqu'il existe. Une série absente ou un débit uniforme de repli ne prouve pas une saison sèche. Le v2 ne doit pas recouvrir ensuite ce chenal sec d'un ancien overlay bleu. Préserver les rivières permanentes alimentées depuis l'amont, même dans un désert local.

Glace lacustre : masque cohérent avec saison/température et type d'eau, transition de rive, variations douces, fissures résolues seulement. Glace fluviale plus prudente avec écoulement. Banquise : couverture saisonnière, plaques/floes et chenaux au niveau résolu, moyenne blanche/bleutée à distance. Ne pas appliquer les seuils d'eau douce à l'océan sans distinction.

Icebergs : éléments stables sur l'eau près de sources glaciaires côtières plausibles, densité bornée et taille physique, silhouettes/ombres de matériau adaptées à la vue satellite. Pas de semis d'icebergs dans tous les océans froids, ni assimilation des fragments de banquise à des icebergs. Sans vêlage ni courant simulés, présenter cette distribution comme un proxy procédural localisé ; ne pas revendiquer une dynamique physique. Leur affichage doit exister dans l'implémentation, pas seulement dans le document.

## Intégration technique

Neural Earth : module v2 distinct, branchement explicite aux sélecteurs, aperçu, modes CPU/PNG et WebGPU, blocs coarse, globe, liens et partage. Réutiliser transport et données existants ; ajouter seulement les canaux nécessaires. Les diagnostics v2 doivent aussi fonctionner en zoom rapproché. Préserver Render/Render3D/Brush et leurs sorties.

Magna Urbis : adaptateur vers le même contrat, shader WebGL2 et intégration du Terrain Bench/Rust bridge ; réutilisation en perspective et export/raster avec comportement déclaré. Ne pas introduire de moteur concurrent ni dépendance réseau runtime vers l'autre dépôt. Conserver les emprises et l'eau appliquées, même si les overlays correspondants sont masqués. Ajouter la note de méthode anglaise et son lien d'aide comme demandé par AGENTS.md.

Chercher à partager des constantes/fixtures/contrats reproductibles lorsque possible. Les différences WGSL/GLSL/Python doivent être vérifiées sur un jeu d'entrées commun ; ne pas copier sans contrôle trois modèles qui divergent. Pas de promesse de parité exacte si l'adaptateur climatique ne fournit pas les mêmes données.

## Validation et critères de livraison

Les vérifications doivent exercer le vrai shader compilé, pas uniquement rechercher des chaînes dans son source. Réutiliser les infrastructures existantes ; pas de grand framework de tests.

- Nouveau layer sélectionnable dans les deux applications ; slider et partage restaurent date et rendu ; anciens modes accessibles et defaults conservés.
- Captures à plusieurs échelles d'une région aride minérale, prairie/forêt tempérée, forêt humide, côte sableuse, montagne enneigée et lac/mer froids ; échantillons Orogen globaux et vues proches obligatoires.
- Série de zooms aux échelles critiques : pas de taches contrastées soudaines neige/sable, pas de découpe floue de peuplement et pas de perte de couleur moyenne injustifiée.
- Eau : rivière/lac/côte, dépression sèche, cours permanent désertique et lit saisonnier sec ; végétation exclue de l'eau, eau finale non réintroduite par overlay historique.
- Hiver/été : herbe et feuillus différents, neige/glace réactives ; banquise et icebergs visibles dans les contextes appropriés et absents des contextes chauds.
- Continuité de graine, tuile, LOD et décembre/janvier ; pas de dérive par déplacement de caméra ; vérification antimeridien/pôles ou limites explicites si non résolu.
- Diagnostics : leurs valeurs de couvert, humidité, substrat et neige/glace correspondent aux décisions du rendu au même point et à la même date.
- Exécuter tests ciblés, compilation des shaders, typecheck/build Magna et contrôles de console navigateur. Mesurer séparément préparation, transfert, bake et frame ; ne pas déduire le temps réel d'un simple coût de shader isolé.
- Documenter commandes, captures, résultats, mesures et limites restantes dans un rapport. Toute exigence non réalisée doit être nommée ; ne pas présenter un changement de palette ou un alias de Render3D comme livraison complète.

## Références et nature des adaptations

Les références guident des adaptations de rendu ; elles n'impliquent pas l'implémentation de leur modèle physique complet.

- [PBRT, Noise](https://pbr-book.org/3ed-2018/Texture/Noise) : cohérence déterministe et contrôle fréquentiel des textures procédurales. Adaptation au filtrage selon empreinte et aux motifs de matériau.
- [Filament, Physically Based Rendering](https://google.github.io/filament/main/filament.html) : séparation matériau/éclairage, normales et rugosité. Réutiliser les principes du shader existant sans imposer un moteur.
- [NSIDC, Sea Ice](https://nsidc.org/learn/parts-cryosphere/sea-ice) : banquise comme eau de mer gelée, extension saisonnière ; distinguer banquise et glace continentale. Le modèle v2 reste une apparence saisonnière, sans océan/thermodynamique couplés.
- [NSIDC, Science of Sea Ice](https://nsidc.org/learn/parts-cryosphere/sea-ice/science-sea-ice) : morphologie et différence de congélation entre eau douce et eau salée ; ne pas traiter les lacs et l'océan comme un seul milieu.
- [NOAA, What is an iceberg?](https://oceanservice.noaa.gov/facts/iceberg.html) : origine glaciaire des icebergs. La présence de banquise seule ne constitue pas une source de vêlage.
- Code existant : `web/terrain_renderer.js`, `web/terrain_render3d.js`, `backend/terrain_render.py`, `backend/terrain_pedology.py`, `backend/terrain_hydrology_render.py` ; dans Magna Urbis : `rust/bridge/terrainGround.ts`, `terrainForest.ts`, `terrainForestRiparian.ts`, `terrainForestWater.ts`, `web/src/ui/terrain3d.ts`.

## État de réalisation

Ce document spécifie la cible autorisée. Au moment de sa création, Render v2 n'est pas encore implémenté. Le rapport d'implémentation doit distinguer fonctionnalités effectivement livrées, validation effectuée et limites. La cible visuelle « Google Earth » désigne une direction esthétique, pas une équivalence mesurée à des images satellite réelles.
