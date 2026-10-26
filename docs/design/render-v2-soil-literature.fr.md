# Sols Render v2 : références et application au shader

23 octobre 2026. Recherche demandée par l'utilisateur pendant l'audit des vraies tuiles. Les références orientent les corrections ; leurs algorithmes complets ne sont pas présentés comme déjà implémentés.

## Décision de conception

Le sol doit rester un matériau détaillé sous les herbes, buissons et couronnes. La végétation en masque la partie réellement occultée ; ses trouées laissent voir terre, litière, graviers ou mousse. À distance, ces éléments doivent conserver une couleur moyenne et une rugosité cohérentes. Un voile de couleur verte sur un sol lisse ne suffit pas.

Pour Render v2, je retiens un matériau de sol continu, composé avant la végétation, avec des bandes spatiales explicitement filtrées. Les structures géologiques et écologiques fournissent la variété à grande échelle ; les détails de matière apportent le grain résolu. Les deux ne doivent pas se remplacer au zoom.

## Références primaires consultées

| Référence | Apport documenté | Application proposée ici |
|---|---|---|
| Cook et DeRose, **Wavelet Noise**, SIGGRAPH 2005 ([article Pixar](https://graphics.pixar.com/people/derose/publications/WaveletNoise/paper.pdf)) | Analyse de la perte de détail et de l'aliasing des bruits usuels, notamment quand un bruit 3D texture une surface 2D ; construction de bandes mieux délimitées. | Vérifier le contenu fréquentiel des bandes et leur projection sur le terrain. Le symptôme de détail qui apparaît tardivement au zoom est directement pertinent pour nos captures. |
| Lagae, Lefebvre, Drettakis et Dutré, **Procedural Noise using Sparse Gabor Convolution**, SIGGRAPH 2009 ([publication et code des auteurs](https://graphics.cs.kuleuven.be/publications/LLDD09PNSGC/)) | Contrôle de fréquence, largeur de bande et orientation ; bruit de surface et filtrage anisotrope. | Candidat pour des détails orientés : rides de sable, strates ou fibres. Le choix du spectre doit empêcher les grandes taches parasites d'un bruit destiné au grain. |
| Heitz et Neyret, **High-Performance By-Example Noise**, HPG 2018 ([article des auteurs](https://eheitzresearch.wordpress.com/722-2/)) | Synthèse de textures stochastiques à partir d'un petit exemple, avec mélange préservant leur histogramme. | Option pour sable, humus, mousse ou granite lorsque le bruit analytique ne reproduit pas leur distribution visuelle. Préserver les couleurs et le contraste des matériaux lors des mélanges. |
| Deliot et Heitz, **Procedural Stochastic Textures by Tiling and Blending**, GPU Zen 2, 2019 ([chapitre et démonstration](https://eheitzresearch.wordpress.com/738-2/), [prototype Unity Labs](https://github.com/UnityLabs/procedural-stochastic-texturing)) | Variante pratique et préfiltrage de tables pour limiter la dérive des couleurs avec les mipmaps. | Référence d'implémentation pour un échantillonnage procédural sans répétition évidente. Le prototype Unity est une référence historique, pas une dépendance à installer dans notre renderer WGSL. |
| Olano et Baker, **LEAN Mapping**, I3D 2010 ([article](https://userpages.cs.umbc.edu/olano/papers/lean/)) | Filtrage des distributions de normales pour conserver la réponse spéculaire des bosses devenues sous-pixel. | Les détails non résolus doivent contribuer à la rugosité au lieu de disparaître sans compensation ou de scintiller. Adapter ce principe à notre BRDF nécessite une validation propre ; la formule actuelle n'est pas automatiquement une implémentation de LEAN. |
| Barré-Brisebois et Hill, **Blending in Detail**, 2012 ([article technique](https://blog.selfshadow.com/publications/blending-in-detail/)) | Comparaison des mélanges de normales et présentation du Reoriented Normal Mapping. | Si des normales de détail sont introduites, les orienter par rapport au relief affiché sans aplatir ses pentes. La somme actuelle de dérivées reste une option à comparer, pas un bug prouvé par cet article. |
| Pharr, Jakob et Humphreys, **Physically Based Rendering**, 4e édition, chapitre textures ([échantillonnage](https://www.pbr-book.org/4ed/Textures_and_Materials/Texture_Sampling_and_Antialiasing), [filtrage d'images](https://www.pbr-book.org/4ed/Textures_and_Materials/Image_Texture)) | Calcul de l'empreinte du pixel et élimination des fréquences non représentables ; filtrage anisotrope des textures. | Déduire le filtrage des dérivées de la surface projetée, en tenant compte de l'orientation du motif. Le LOD demandé ne suffit pas à définir quelles bandes sont visibles. |
| Jensen, Legakis et Dorsey, **Rendering of Wet Materials**, 1999 ([publication des auteurs](https://graphics.cs.yale.edu/publications/rendering-wet-materials)) | L'eau en surface et dans le matériau modifie son apparence par plusieurs mécanismes, au-delà d'une simple teinte sombre. | Distinguer humidité du sol et eau libre ; faire varier les paramètres du matériau selon son type. Une approximation temps réel doit être explicitement calibrée sur nos scènes. |
| Khronos, **PBR Lighting Implementation** ([documentation Vulkan](https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/Lighting_Materials/04_lighting_implementation.html)) ; Epic, **Color Grading and the Filmic Tonemapper** ([documentation](https://dev.epicgames.com/documentation/unreal-engine/color-grading-and-the-filmic-tonemapper-in-unreal-engine)) | L'éclairage peut dépasser la plage de l'affichage ; la compression des hautes lumières doit préserver leur forme avant l'écrêtage final. | Supprimer le plafond prématuré dans la voie BRDF V2 et appliquer une épaule continue après la calibration de l'éclairage. Notre courbe simple dans l'espace d'affichage hérité ne constitue ni ACES, ni un pipeline radiométrique HDR complet. |

Ces rapprochements avec Render v2 sont des propositions d'ingénierie. Les articles démontrent leurs méthodes dans leurs propres conditions, pas la qualité de notre intégration.

## Point d'intégration dans le code existant

Le défaut initial n'était pas seulement l'absence d'une nouvelle fonction de bruit. Dans `web/terrain_render_v2.js`, `v2Fragment` ajoutait `form` et `swardTone` à `m.albedo`, puis appelait `applyForest` avec un `guide.open` dépourvu de ce détail. La recoloration du sol et de la litière pouvait donc repartir d'une couleur lisse. Le chemin sans couronne remélangeait lui aussi le sol ouvert.

La correction transmet maintenant une modulation de sol explicite à la composition de sol ouvert et de litière, dans NE et Magna. Les normales de couronnes reprennent le relief macro, sans hériter du bruit minéral du sol. Le contrôle GPU isolé vérifie que le détail change le sol nu et les trouées, tout en laissant inchangée une couronne opaque ; les captures de vraies tuiles restent nécessaires pour juger l'aspect obtenu. La variance de pente non résolue contribue déjà à la rugosité du sol : c'est une approximation du shader, pas une revendication d'implémentation complète de LEAN.

La recapture réelle a ensuite révélé un autre défaut : borner directement `detail.x * 10` à ±0,08 saturait le relief dès 8 mm, alors que la première bande a une amplitude de 17 cm. Une matière continue devenait presque bicolore. La correction borne d'abord la forme normalisée à ±1, puis applique une amplitude d'albédo faible selon le matériau. Elle garde le champ spatial et son filtrage ; elle n'ajoute pas un autre bruit pour masquer le précédent.

Les provinces minérales enrichissent désormais la moyenne héritée par un rapport borné à une province neutre : au plus ±14 % sur roche entièrement exposée, moins de ±5 % sur sol minéral sans herbe, avec atténuation par matière organique et neige. Sédiments et plage sont composés ensuite ; la canopée reste indépendante. Cette variation ne remplace plus la palette du même sol entre 1,5 et 6 m/pixel.

L'intégration devrait transmettre un matériau complet de sol — albédo linéaire, normale, rugosité et occlusion — jusqu'à la composition des couronnes. Les zones sans hit de végétation doivent conserver ce matériau. Les ombres, la litière et la mousse doivent le modifier par des masques explicites, sans remplacer tout le sol par une couleur constante. Les détails de roche ne doivent pas être réutilisés comme silhouettes de petites herbes.

## Construction proposée

1. **Support physique stable.** Ancrer les champs aux coordonnées du terrain réellement affiché, indépendamment des origines de tuiles et du zoom. Partager ce contrat entre Neural Earth et Magna Urbis. Traiter projection, relief et texture ensemble pour éviter une nouvelle déformation isolée aux pôles.
2. **Mélange de matériaux.** Séparer roche, sol minéral, sable, sol organique/litière, mousse et neige. Déduire leurs poids de la pédologie, de l'humidité, du couvert, des pentes et des dépôts existants. La normalisation doit conserver une moyenne cohérente entre les vues éloignées et rapprochées.
3. **Détails à échelles connues.** Employer des bandes étroites pour les détails analytiques, ou un échantillonnage stochastique d'exemplaires de matière. Orienter les motifs seulement lorsqu'une direction physique pertinente est disponible. Mesurer le coût GPU avant de retenir un Gabor complet sur chaque pixel.
4. **Filtrage explicite.** Conserver une bande tant qu'elle est résolue, puis intégrer sa moyenne et sa variance. À 0,625 m/px, des grains centimétriques ne peuvent pas être représentés un par un : ils contribuent à l'apparence agrégée. Les gros détails résolus peuvent rester visibles dans les trouées végétales.
5. **Végétation au-dessus du sol.** Composer herbes, buissons et canopée à partir de leur vraie couverture. Conserver le matériau et ses normales dans les parties visibles du sol ; limiter la contribution du grain du sol aux surfaces qu'il représente, sans la projeter sur les couronnes.
6. **Saisons continues.** Garder les mêmes positions et graines aléatoires au fil de l'année. Faire évoluer humidité, litière, sénescence et neige par des paramètres continus. Les motifs doivent changer d'état sans se déplacer ni se régénérer arbitrairement.

La synthèse par exemple reste compatible avec un rendu procédural temps réel, mais demande de petits exemplaires et leur prétraitement. Une voie entièrement analytique reste possible. Le choix doit se faire sur de vraies tuiles et un budget GPU mesuré ; aucune de ces techniques ne justifie de remplacer le DEM généré par un terrain artificiel pour l'audit.

## Validation attendue

Reprendre les mêmes sites générés : forêt tropicale dense avec trouées, forêt tempérée et litière, steppe herbacée, désert sableux, roche aride incisée et neige partielle. Comparer à cadrage identique 40 km/10 km/2,56 km/640 m, puis dérouler le zoom. Vérifier ensemble visibilité du sol sous couvert discontinu, stabilité des couleurs moyennes, absence de camouflage au zoom intermédiaire, relief conservé et coût de rendu. Les anciennes grilles synthétiques ne font pas partie de cette validation visuelle.
# Trouées forestières : distinguer le champ continu et sa classification

La revue des vrais rendus a identifié une autre origine du camouflage : le
champ de population hérité transforme un FBM de 100–400 m en ouvertures presque
binaires. Le bon mélange radiométrique de ses surfaces rend ce défaut visible ;
il ne corrige pas le générateur des trouées. Un flou global ou un changement de
couleur ne constitue donc pas une correction de leur distribution.

[Asner et al. (2013)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0060875)
analysent la distribution continue des hauteurs de canopée avec plusieurs seuils
verticaux, et distinguent les contextes de plaine alluviale et de terre ferme.
[ForestGapR (Silva et al., 2019)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13211)
sépare détection, taille, agrégation spatiale et évolution des trouées.
Les [observations UAV/terrain d'Amazonie centrale (2023)](https://bg.copernicus.org/articles/20/3651/2023/bg-20-3651-2023.html)
montrent également que la distribution mesurée dépend du seuil minimal retenu.
Ces travaux ne prescrivent pas notre shader : ils motivent la distinction entre
densité continue, trouées structurelles et leur apparence filtrée.

Le profil progressif V2 en cours d'essai conserve le champ spatial, le climat et
les exclusions physiques ; il élargit sa transition de couverture. C'est une
correction d'apparence bornée, sans simulation de mortalité, succession ou
tempêtes, et sans prétendre reproduire les distributions empiriques de ces
études. Les vieux profils Render3D restent des témoins séparés.

## Saturation des peuplements denses

Le préflight suivant a révélé une autre source de grandes trouées : multiplier
directement le pourcentage de couvert par le bruit de peuplement. Une variation
symétrique est écrêtée du côté plein, mais creuse fortement le côté négatif.
L'ablation à couvert constant retire le motif DFC ; supprimer le bog ne le retire
pas. Les premières hypothèses d'éclairage ou de neige seule ne sont pas confirmées.

Le candidat en cours d'évaluation transporte la variation dans une densité
effective `tau=-log(1-cover)`, puis reconstruit `cover=1-exp(-tau)`.
La relation exponentielle aux fractions de trouées figure notamment dans
[Chianucci et al. (2014), analyse de photographie de couvert](https://iforest.sisef.org/contents/?id=ifor0939-007).
Cette publication distingue couverture des couronnes, couverture du feuillage et
porosité : nous ne les assimilons pas et ne prétendons pas estimer le LAI.
Le transfert proposé est une politique d'apparence inspirée de la saturation
d'une densité, sans calibration forestière empirique ni simulation de succession.

## Eau côtière : profondeur et clarté distinctes

Le préflight réel révèle une autre perte de variété : une extinction uniforme
de 0,15 par mètre efface presque entièrement le fond à 30 m, même en mer claire.
Le gradient bathymétrique de 150 à 3 000 m ne restaure pas les faibles fonds.
Le [produit MODIS Kd490 de la NASA](https://modis.gsfc.nasa.gov/data/dataprod/kd_490.php)
et l'[explication NOAA CoastWatch](https://eastcoast.coastwatch.noaa.gov/cw_k490.php)
distinguent justement profondeur, atténuation et clarté de la colonne d'eau.
Une atténuation exponentielle est donc un mécanisme approprié ; leur produit
spectral Kd490 n'est pas le coefficient directement mesuré de notre shader RGB.

Le candidat marine à 0,035 par mètre, avec les eaux intérieures inchangées à
0,15, a été comparé sur les vrais carrés côtiers de 40 et 10 km. Il conserve
une bande graduée de faibles fonds sans recolorer la terre ni remplacer toute
la mer par du turquoise. Ce choix reste une politique d'apparence selon la
classe d'eau, sans données de chlorophylle ou de matières en suspension ni
calibration optique locale. Preuves et override exact :
`output/render-v2-floor/marine-optics-causal/clear-water/`.
