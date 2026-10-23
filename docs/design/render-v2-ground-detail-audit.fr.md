# Sol détaillé sous végétation : audit et points d'intégration

Audit initial de code du 23 octobre 2026, rapproché exclusivement des captures de vrais DEM, puis correction de composition décrite en fin de document. Les défauts analysés ci-dessous correspondent à l'état **avant correction**. La recherche bibliographique et le choix final du spectre/filtre sont traités séparément : cette note décrit les contraintes concrètes du code à respecter pour les intégrer.

## Deux causes différentes du sol lisse

Le filtrage élimine effectivement toutes les bandes de `groundFieldWithClasts` dès que le footprint atteint 1,176 m/px. Ensuite, même lorsqu'une bande est résolue, la reconstruction du sol dans `applyForest` perd une partie importante de son albédo détaillé. Ajouter du bruit ou augmenter son gain ne corrige pas cette seconde cause.

Les sources principales sont `web/terrain_render3d.js` (`BANDS`, `groundVisibility`, `groundFieldWithClasts`, `applyForest`) et `web/terrain_render_v2.js` (`v2Open`, `v2Fragment`, substitutions de `drawShader`). L'assemblage final des shaders compte : les substitutions de v2 doivent être incluses dans la lecture.

## Bandes réellement visibles

Le filtre porte sur `wavelength × aspect × filterWidth`, et non sur la longueur d'onde nominale seule. La visibilité commence à baisser à 0,18 fois cette largeur et devient nulle à 0,42 fois cette largeur. C'est une coupure conservatrice ; la desserrer sans autre filtrage ferait réapparaître des détails insuffisamment échantillonnés.

| Bande | Forme actuelle | Longueur nominale | Largeur filtrée | Début de décroissance | Extinction |
| --- | --- | --- | --- | --- | --- |
| 0 | Crêtes arrondies, anisotropes | 8 m | 2,8 m | 0,504 m/px | 1,176 m/px |
| 1 | Rainures étroites | 2,2 m | 0,264 m | 0,04752 m/px | 0,11088 m/px |
| 2 | Clastes elliptiques, grille irrégulière | 0,32 m | 0,136 m | 0,02448 m/px | 0,05712 m/px |
| 3 | Grain fin | 0,055 m | 0,0396 m | 0,007128 m/px | 0,016632 m/px |

À 2,5 m/px (vraie fenêtre de 2,56 km), **les quatre bandes sont nulles**. À 0,625 m/px (vraie fenêtre de 640 m), seule la bande de crêtes de 8 m subsiste, à 95,63 % de son amplitude. Les clastes ne sont donc pas de petits points visibles dans ces deux zooms. À 0,1 m/px, la bande 8 m reste entière, celle de 2,2 m ne garde que 3,85 % et les deux autres sont encore nulles.

`groundSlopeVariance` transfère une approximation scalaire de la variance non résolue vers la rugosité GGX. Cette rugosité ne crée pas de texture spatiale visible. En v2, la référence BRDF utilise la même rugosité : quand aucune bande n'est résolue et qu'il n'y a pas de couronne, le ratio détail/référence ne peut pas inventer du détail absent.

## Ce que l'herbe change actuellement

`v2Open` mélange une couleur de sol et une couleur d'herbe ; il ne produit ni structure de brins ni matériau de litière. Le même champ géométrique de crêtes/rainures/clastes est ensuite employé pour tous les couverts.

Dans `v2Fragment`, hors neige, le gain des micro-normales vaut :

- roche nue : jusqu'à 0,52 ;
- sol sans herbe ni roche : 0,09 ;
- herbe pleine sans roche : 0,02 ;
- neige pleine : 0,008.

L'herbe atténue donc fortement le relief du champ rocheux, sans lui substituer son propre spectre. La rugosité de base reste 0,88 pour herbe, terre et sable si la fraction de roche est nulle ; l'humidité et la matière organique ne définissent pas ici un matériau BRDF différent.

Le terme `swardTone` est le **même** `detail.x`, multiplié par un masque d'herbe et `groundVisibility(.32, fp)`. Il s'éteint à 0,1344 m/px : il n'apporte rien aux captures à 2,5 ou 0,625 m/px. Il ne constitue pas un champ d'herbe indépendant. Le facteur `form` conserve au plus ±8 % d'albédo hors neige, mais rencontre ensuite le problème de composition ci-dessous.

## Où le détail est remplacé dans la forêt

Le chemin v2 est le suivant :

1. `open` contient le sol de référence sans `form/swardTone` ; `distantColor` y mélange la couleur de canopée.
2. `m.albedo` reçoit `distantColor × (1 + form + swardTone)` ; `m.normal` reçoit le gradient détaillé du sol.
3. `applyForest` reçoit les guides `open`, `closed` et `distantColor` **non détaillés**.
4. La substitution v2 de `drawShader` remplace `groundRecolor(guide.open)` par `guide.open`. Le sol reconstruit ne reçoit donc pas le facteur détaillé calculé à l'étape 2.
5. Même dans la branche **sans arbre** (`st.cover`, couvert voisin et mip tous presque nuls), `applyForest` remixe l'albédo vers ce guide lisse. À reveal complet, le facteur `form/swardTone` est entièrement perdu.
6. Les autres branches reconstruisent `litter`, puis `nearAlbedo`, depuis ce même sol lisse et des couleurs de litière. Les normales `nGround` subsistent dans les trouées ; l'albédo détaillé est remplacé.

Ce dernier point explique pourquoi il ne suffit pas de regarder le calcul de `m.albedo` pour conclure que le détail reste visible sous végétation. Le relief opaque d'une couronne doit naturellement cacher le sol : le problème concerne les trouées, les sols entre couronnes et le sous-bois effectivement visible, pas le sol derrière une couronne opaque.

Le sibling Cg possède déjà une autre stratégie dans `rust/bridge/terrainForest.ts:649` : `k = groundToDisplay(m.albedo) / renderGuideColor`, puis `openGround = renderGuideOpen × k`. Cela transmet le détail à son guide ouvert. NE v2 n'utilise pas ce transfert. Un facteur explicite serait plus sûr qu'une division par une couleur sombre, et évite de transporter par erreur une teinte de canopée sur une roche.

## Intégration proposée, en deux étapes bornées

### 1. Conserver le matériau du sol lors de la composition

Séparer explicitement la référence macro, le matériau du sol visible et le matériau de couronne :

```text
macroReference = couleur macro sans détail
floor = sol détaillé : albédo, normale, rugosité, occlusion
floor = appliquer la litière / herbe basse en conservant son détail filtré
visible = composer floor et crown selon leur couverture réellement résolue
final = éclairage macro × réponse de détail / réponse de référence
```

Dans le code actuel, le premier changement peut réutiliser le champ déjà calculé : transmettre explicitement le facteur d'albédo du sol à `applyForest`, l'appliquer aux branches sans arbre et aux albédo de sol/litière, **après** le choix de la couleur de litière. Le facteur ne doit pas modifier la couleur des couronnes. Ne pas reconstruire l'albédo à partir du seul guide lisse dans les retours précoces.

Les normales et rugosités doivent suivre la même séparation : conserver la normale du sol dans les trouées ; employer la normale macro et les pentes propres de la couronne sur sa surface. Aujourd'hui `nCrown` récupère `macroSlope` depuis `nGround`, qui inclut déjà les micro-pentes du sol : ce transport est à séparer lorsque le matériau du sous-bois sera enrichi.

Cette étape ajoute essentiellement des paramètres et des opérations de mélange, pas une seconde évaluation de bruit. Elle doit préserver le gain neige faible, les domaines marins, le relief DEM et les masques thermiques des couronnes.

### 2. Donner au sol, à l'herbe basse et à la litière leurs propres bandes

Introduire une description de détail par matériau plutôt qu'un seul champ rocheux dont on réduit l'amplitude. Le résultat d'une évaluation doit au minimum fournir hauteur/gradient, modulation d'albédo de moyenne contrôlée et variance des pentes non résolues. La fraction rocheuse conserve ses crêtes et clastes ; terre, sable, herbe basse et litière choisissent d'autres amplitudes, anisotropies et bandes physiques. Les coefficients doivent être évalués sur de vrais sites, pas fixés par une nouvelle série de terrains artificiels.

Pour qu'une variation soit visible à 0,625 m/px avec le filtre actuel, sa largeur doit dépasser 1,49 m ; à 2,5 m/px elle doit dépasser 5,95 m. À ces pas, il faut représenter de faibles variations de surface/agrégats résolues, pas dessiner des gravillons ou des brins de plusieurs mètres. Les détails plus fins continuent d'alimenter la rugosité filtrée. Il manque donc aussi des bandes de surface entre les grandes mosaïques écologiques de centaines de mètres et les bandes de sol coupées sous 1,176 m/px.

L'intégration d'un filtre orienté ou d'un bruit à spectre maîtrisé doit se faire dans l'évaluateur commun, en utilisant le footprint transformé dans le domaine du détail, et en conservant la variance non résolue. Remplacer seulement le générateur tout en gardant les reconstructions lisses d'`applyForest` laisserait le problème principal intact. Éviter une renormalisation arbitraire du contraste des bandes restantes à chaque LOD.

Conserver les coordonnées entières/fractionnaires du sol (`r3.cells`, `r3.fracs`) pour les nouvelles échelles ; ne pas passer directement une position monde f32 de plusieurs millions de mètres à un motif centimétrique. Les poids de matériau et les masques doivent être filtrés avant la composition ; des transitions rapides exigent une prise en compte cohérente de leurs dérivées ou une composition des réponses, plutôt qu'un faux gradient de hauteur.

## Portée et vérification sur les preuves existantes

Les captures BWH `arid-plain/rolling` montrent le sol presque uniforme en fenêtre 640 m, et les BSK le passage de masses floues aux couronnes. La canopée AF/CFB réelle perd du contraste local en v2 ; ces constats sont consignés dans [l'audit réel](render-v2-audit-real-open-ground.fr.md), avec méthodes et limites. Ils sont compatibles avec ce chemin de composition, sans constituer une ablation de chaque terme du shader.

Pour l'étape 1, reprendre les **mêmes positions et emprises réelles**, en particulier une zone BWH sans couronne, une trouée AF, une zone BSK entre arbres et un contrôle EF. Vérifier que la modulation du sol survit au retour sans arbre et dans les trouées, que les couronnes restent inchangées, et que le détail disparaît proprement lorsqu'il devient subpixel. Vérifier aussi un léger déplacement de caméra pour détecter le scintillement ; une seule image nette ne suffit pas.

Le port GLSL Cg doit suivre la même composition, sans cumuler son ratio existant et un nouveau facteur explicite. Le générateur `tools/assets/sync_render_v2.cjs` partage actuellement l'environnement, pas toute cette composition. Le backend NumPy produit l'albédo environnemental/macro et l'éclairage de relief, mais ne reproduit pas les bandes micro-BRDF/couronnes du GPU : la parité environnementale seule ne validera pas ce changement de détail.

## Première correction de composition réalisée

Dans `web/terrain_render_v2.js`, la voie fine v2 transmet maintenant le facteur d'albédo du sol aux branches sans arbre, litière moyenne et litière proche. Les couronnes reçoivent leur propre albédo et leur normale macro/couronne, sans modulation du sol. La rugosité de référence sans variance non résolue suit les mêmes mélanges de couverture que le matériau final : la variance du sol conserve sa réponse BRDF dans les trouées et ne remonte pas sur une couronne opaque.

La référence d'exposition est calculée après composition, à partir du matériau final et d'une pente macro **filtrée**, distincte de la pente physique du DEM. La correction finale est `shaded × (baked.a / flatHillshade) × Y(flatShade) / Y(macroReferenceShade)`. `flatHillshade` est calculé avec les mêmes force, ambient, contraste et direction solaire que le bake. Elle retire la double exposition de base et le surplus d'ombrage de pente, sans normaliser le niveau d'éclairage plat vers l'albédo nu. Les ombres directionnelles du DEM restent appliquées.

La sixième couche v2 transporte `aux5.rgb=canopy leaf-on` avant phénologie/neige et `aux5.a=macroGradient.x` ; `aux0.b=macroGradient.y`. Le gradient macro du bake vaut six fois la pente physique filtrée. La référence BRDF l'emploie divisé par trois, pour correspondre au gain ×2 de pente physique conservé dans le matériau. Ainsi, la référence retire le grand relief déjà présent dans le hillshade, mais laisse la contribution du petit relief DEM ; celui-ci n'est pas remplacé par un bruit de sol. Les couronnes utilisent la normale physique ×2 sans micro-pentes du sol. Les cinq couches byte de Render3D restent distinctes des six couches half-float de v2.

La couleur leaf-on héritée est multipliée par le ratio saisonnier `v2Canopy(snowFree) / v2Canopy(leafOn)`, puis reçoit la neige v2. Il n'y a pas de gain de palette global. Le contrat de couvert a ensuite été remplacé par la population effective aux2 filtrée : voir les trois surfaces ci-dessous. `applyForest` reste responsable de révéler les couronnes.

Le masque plage hérité, dont le stockage est multiplié par 0,8, exclut le couvert forestier et herbacé même lorsque la neige masque le sable. Le nouveau transport `aux0.g=r3Flats` conserve les barres/flats de sédiment au-delà de la plage stricte. Ce sédiment intervient après la canopée **moyenne** dans le guide lointain, comme dans Render ; il ne recolore pas les couronnes résolues. Les flats n'interdisent pas les arbres des estuaires. La moyenne du sol reste `macroColor` à tous les footprints, enrichie par `v2GroundTint` avant les overlays ; cette correction minérale faible n'est pas une deuxième palette proche. Le remplacement de palette entre 1,5 et 6 m/px a été supprimé.

`tests/browser/test_terrain_render_v2_floor.cjs` compile le shader de production et ablate uniquement sa réponse détaillée du sol sur de petites entrées unitaires de matériau. Ces entrées **ne constituent pas des preuves visuelles de terrain**. Résultat GPU Chrome/ANGLE avec six couches aux16float, après correction de la forme saturée : différence moyenne avec/sans détail de 0,617/255 sur le sol nu, 0,414/255 dans 1284 pixels de trouées/litière, **exactement zéro dans 14875 pixels de couronnes opaques**, et couverture forestière nulle sur la plage testée. Le test borne aussi le contraste du sol nu pour prévenir son retour en masque binaire. Erreur de couleur du sédiment attendu sur les flats : 0,358/255 ; les 14875 pixels de couronnes restent présents dans le contrôle d'estuaire. L'écart du terrain plat éclairé avec sa réponse BRDF directe, avec la même courbe d'affichage des deux côtés, n'est que de 0,093/255, maximum un niveau de quantification : la double exposition du terrain plat est éliminée. Rapport : `output/render-v2-floor/report.json`.

L'isolation de l'exposition sur ces couronnes donne maintenant Y55,941 pour l'albédo et Y57,152 avec la correction finale ; Y46,712 serait obtenu avec le seul albédo multiplié par le hillshade brut. La normalisation initiale vers l'albédo nu donnait Y38,042 sur l'ancienne palette. Ces étapes ont été isolées séparément ; le changement n'est pas un gain global ajouté à la palette.

La faible réflectance visible d'un couvert n'est cependant pas, seule, une preuve de mauvais albédo : [Wu et al., 2018](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.14939) mesurent une réflectance visible moyenne des feuilles d'Amazonie de 0,05 pour des feuilles matures et 0,04 pour des feuilles âgées. Ces mesures spectrales ne sont pas directement des couleurs RGB de couronne ; elles justifient de séparer le diagnostic d'albédo, l'éclairage et la structure avant d'éclaircir une palette à l'aveugle.

Cette première correction ne remplace pas le spectre des quatre bandes et ne prétend pas achever le matériau réaliste. La comparaison avant/après sur les vrais sites et la vérification du mouvement de caméra restent nécessaires avant validation visuelle finale.

La passe réelle intermédiaire `output/render-v2-floor/final-candidate-real/AUDIT.fr.md` a ensuite révélé deux défauts que ce test unitaire ne pouvait pas accepter visuellement : des aplats arides trop contrastés et la neige écrêtée après correction d'exposition. La forme du sol est désormais bornée dans [-1,1] **avant** multiplication par son amplitude0,04sol→0,06roche→0,006neige ; l'ancien `clamp(detail.x*10,-.08,.08)` saturait la bande8m dès8mm de hauteur. Le sward suit la même séparation de forme et amplitude. La voie V2 conserve les hautes valeurs BRDF jusqu'à une compression finale `v2DisplayShoulder` sur le pic RGB, avant l'eau ; la voie Render3D garde son clamp existant. Les références d'exposition restent non compressées. Cette courbe est une adaptation d'affichage à l'architecture hybride, pas une implémentation revendiquée d'ACES ni une simulation radiométrique complète.

Une ablation réelle isolée du mélange de la moyenne AF est consignée dans `output/render-v2-floor/mean-display-candidate/AUDIT.fr.md`. Le matériau distant décode maintenant la moyenne displayRGB héritée, plutôt que de remélanger les deux albédo linéaires ; la modulation reste appliquée à la seule fraction ouverte. Le test GPU vérifie cette égalité de moyenne sur une couverture mixte0,761, avec une erreur exactement nulle. Sur le vrai carré AF640m vu aux quatre zooms, cette correction ne baisse toutefois Y lointain que de1,15 % et ne change pas le proche. L'écart moyen lors de la résolution des couronnes demeure : il faut isoler la distribution des normales/ombres et l'occlusion avant de conclure à une erreur de palette.

## Réponse des couronnes non résolues

`TerrainRenderV2.canopyMeanShader` exporte une LUT de19altitudes solaires pour les coefficients diffuse et spéculaire des deux formes analytiques déjà employées par `forestCrowns`. La quadrature est uniforme en aire du disque projeté ; les pentes broadleaf1,1r et needle1,6r/sqrt(r²+0,015), l'AO intrinsèque1−0,4r² et la rugosité0,9 viennent du shader existant. `v2CanopyMeanRatio(rgb,needle,light,light2,strength)` rapporte la réponse intégrée à celle du même matériau plat. Ce ratio s'applique uniquement au guide fermé moyen, avant les sédiments/plages et en s'éteignant sous la neige. La vraie couleur des couronnes résolues reste distincte. Aucun calcul de quadrature ne se déroule par fragment : la LUT contient76coefficients, avec interpolation selon l'altitude des deux lumières.

`tools/verification/verify_render_v2_canopy_mean.cjs` vérifie60combinaisons contre la vraie BRDF compilée au GPU : erreur maximale0,000266 sur le ratio. La LUT conserve une moyenne de réponse **linéaire**, pas une moyenne exacte des codesRGB. Le biais maximal à cette dernière atteint0,099 en lumière rasante ; à45° il vaut environ0,02 pour les feuillus et0,03–0,05 pour les conifères. Le recouvrement, les ombres entre couronnes, les clumps et les sols sous couvert ne sont pas préintégrés. Cette portée limitée doit rester visible dans les affirmations de qualité.

La passe réelle historique `output/render-v2-floor/canopy-mean-af/AUDIT.fr.md` montrait une diminution de6,6 % de Y lointain surAF, sans changement du proche totalement résolu : amélioration partielle, résidu de29,5 % entre les deux vues du même carré640m. Un essai distinct de couvert aux2 moyenné au mip4 a été **retiré** : amélioration insuffisante et contrat coarse/fine non résolu. À cette étape, le guide conservait ecology.r. Les archives f5 ne valident pas la nouvelle identité bd367 apparue après rechargement du serveur : la nouvelle campagne conserve sa propre provenance.

## Population et moyenne de trois réponses lumineuses

L'ablation réelle bd367 `output/render-v2-floor/bd-tone-fixed/tone-comparison.json` a isolé une erreur de contrat : aux2.a contient l'humidité de berge, tandis que `ForestStand.tone` attendait un résidu centré sur0,5. Le passage de bank0 au tone canonique corrige le facteur indu de0,83 sans éclaircir globalement la palette. Sur le même carré central640m, le proche AF passe deY53,241 à59,097 ; les captures Render3D restent identiques. Le test GPU change désormais la berge de0 à1 et exige une différence exactement nulle dans les14875 pixels de couronnes opaques.

La population aux2 filtrée au footprint physique remplace ensuite le guide écologique comme autorité du couvert fin et de son diagnostic. Pour une couverture effective `c` et la densité interne exacte `d=forestDensity(c,kappa)`, la fraction de peuplement est `share=c/d`. Les trois aires valent `(1-share, share-c*leaf, c*leaf)`, avec exclusion de plage physique. Chaque surface est éclairée séparément : sol ouvert, litière issue de `forestFloor` avec AO0,72, couronne moyenne issue de `forestCrownColor` puis de la LUT intrinsèque, rugosité0,9. Les **réponses lumineuses linéaires** sont additionnées ; on ne mélange pas leurs AO ou leurs normales en un matériau artificiel. Flats et plage interviennent ensuite, avant la compression d'affichage. Le reveal original mélange cette moyenne avec la réponse proche ; les photons totalement résolus restent inchangés.

L'ablation `output/render-v2-floor/bd-three-state/` donne AF Y62,401→59,097 entre les vues40km et640m du même carré, soit5,59 %, et CFB88,328→84,839, soit4,11 %. L'activation directe du reveal proche avait donné des moyennes prometteuses mais des taches binaires inacceptables ; elle est rejetée. Le profil progressif des clairières conserve les mêmes bruits/positions et élargit le seuil en `sigmoid(4*(score-1.07*threshold))`. La passe réelle `bd-progressive-three-state/` garde des écarts similaires (AF5,61 %, CFB4,06 %) avec des transitions moins dures. Une mosaïque cartographique reste visible : ces mesures ne constituent pas une validation photoréaliste. La compensation1,07 est une approximation de variance ; sa quadrature gaussienne de référence est conservée dans `output/render-v2-floor/clearing-profile/gaussian-reference.json`, distincte des preuves visuelles.

## Moyenne coarse native

`terrain_render3d.js` exporte les fonctions pures exactes de climat, densité, litière et couleur de couronne, ainsi que `r3ForestExpected`. Ce dernier évalue la partie déterministe du même champ : climat à l'altitude, pente, vallée/crête, bog, roche, neige pérenne, hauteur et densité ; les résidus des lattices sont centrés à zéro. La fraction de clairière est intégrée avec le biais de relief au lieu d'évaluer un bruit en un point. Ce proxy n'est **pas** la moyenne exacte du champ fin : non-linéarités climatiques, covariance du bruit et perturbations rares ne sont pas intégrées. Il ne garantit pas une identité coarse/fine et ne crée aucune nouvelle donnée physique.

Le bake coarse V2 éclaire les mêmes trois surfaces sur un plan, conserve leur somme linéaire puis l'encode en RGB display HDR **RGBA16float**. Le fragment applique seulement le hillshade directionnel normalisé par sa valeur sur terrain plat, puis la shoulder une fois avant l'eau. La réponse dépend de la lumière : la clé inclut les vecteurs du LOD réellement affiché, la saison et les réglages forestiers. Render3D conserve son stockage RGBA8 et son shader historique. Aucun champ statistique ou texture supplémentaire n'est ajouté ; le tampon couleur V2 coûte huit octets par texel contre quatre pour Render3D, soit819200 contre409600 octets pour un bloc160² doublé.

`test_terrain_render_v2_climate.cjs` vérifie au GPU le calendrier, la reconstruction des températures/pluies, la lumière propre au LOD, l'invalidation forêt, les deux formats et leur comptage mémoire. Le passage V2→Render3D→V2 restitue exactement le total mémoire, et les octets Render3D lus avant/après sont identiques. Ce contrôle technique ne remplace pas les captures réelles LOD4/6 nécessaires à l'acceptation du proxy coarse.

## Correction causale suivante : éclaircissement des peuplements

Les étapes progressives ci-dessus restent des archives : le taux humide réduit n'avait pas supprimé les derniers blobs CFB. L'audit réel `output/render-v2-floor/cfb-causal/AUDIT.fr.md` isole leur origine. Zéro clairières les retire, zéro affleurements les conserve ; les pixels de population quasi nulle possèdent des herbacées proches de1 et une fraction rocheuse proche de0. Le problème est donc la distribution des perturbations, pas une absence de sol ou de relief.

Le profil partagé est désormais `thinning=min(.55,2*c*sigmoid(score/1.15))`, oùc est le taux naturel climatique ; le couvert reçoit `cover*=1-thinning`. La signature historique de `v2ClearingFraction(score,threshold)` décodec depuis son quantile. Un événement100–400m ne vide plus sa cellule ; les petites trouées viennent des candidats d'arbres, les ouvertures physiques restent indépendantes. Le proxy coarse retient la moyenne `min(.55,c)` au lieu de l'ancien CDF de seuil. Aucune conservation de la précédente moyenne de grandes ouvertures n'est imposée au-delà du plafond.

Sur les vrais candidats AF/CFB10km et640m, le défaut ciblé CFB10km est retiré tout en conservant le relief. La moyenne Y de toute la plaque CFB reste83,7598→83,7273, et le contraste chromatique20–160m baisse de26 % : ce n'est pas un gain global. Les contrôles Render3D sont pixel-identiques. Cette correction ne vaut pas acceptation photoréaliste globale ; les couronnes et les grands gradients restent ceux du moteur existant, à revoir dans la galerie réelle finale.

## Préflight DFC rejeté, puis densité optique et neige sur la litière

Le préflight réel db083 révèle encore un réseau blanc envahissant en DFC au LOD3. Le chemin vérifié est latent240m → uploadTile304² → neuf tuiles aux16float → r3Pipeline/v2Fragment ; le native-coarse n'est utilisé qu'aux LOD≥4. Les trois premiers probes de helpers coarse étaient donc non informatifs et sont explicitement écartés. L'audit causal complet est `output/render-v2-floor/dfc-snow-causal/AUDIT.fr.md`.

Les ablations utiles isolent la population fine : couverture constante retire le réseau, alors que neige0, éclairage0, bog0 et éligibilité neutralisée ne le retirent pas. La modulation directe du couvert par le résidu `stand` est remplacée par `v2StandCoverage`, qui applique les mêmes résidus à la densité optique effective. Le bruit, ses positions, hauteur/ton et exclusions physiques sont préservés ; aucun flou de DEM ni gain de palette n'intervient. Cette densité n'est pas un LAI mesuré. Les images DFC153,6/40km montrent le relief redevenu lisible ; AF/CFB et leurs proches160m restent cohérents. Les couronnes restent celles du moteur existant et encore stylisées.

Un défaut séparé reconstruisait de la litière brune après la neige. Le mélange V2 rejoint le contrat CG `litter=mix(litter,ground,v2GroundSnow)`, appliqué aux voies statistique/proche et au helper pure native-coarse. Le gate GPU neige pleine vérifie une différence d'albédo exactement nulle sur2098 pixels de sous-bois/trouées. Les deux vrais PNG Render3D post-port,153,6km et40km DFC, sont pixel-identiques au préflight de référence. Sources finales de cette étape :790e34695a49… ;25 tests Python et les deux régressions GPU floor/climate passent. Cela ne transforme pas le préflight rejeté en validation finale : une nouvelle version de la campagne doit conserver ses propres reçus et revues.

## Couronnes opaques au bord des plages

Le préflight 790e34695a49 est également une archive partielle : le vrai cadre `warm-coast-beach-detail`, 160 m, révèle des couronnes translucides sur le sable. La cause est identifiée dans `drawShader` : `reveal *= 1-v2BeachCoverage` modulait le mélange de toute la réponse proche par le masque de plage au pixel, y compris l'intérieur des couronnes. Le test antérieur « plage pleine sans arbres » ne couvrait pas cette erreur sur les fractions intermédiaires.

La plage devient une éligibilité de racine, stable pour chaque candidat. Le masque physique `B=clamp(aux4.a/.8,0,1)` est lu au centre de l'arbre, au mip 0 du bake : une valeur aléatoire dédiée et déterministe rejette le candidat si elle est inférieure à B. Cette décision ne dépend ni de la position du fragment dans la couronne, ni du footprint de la caméra. La disponibilité de surface `1-B` reste appliquée à la moyenne non résolue et aux masques de peuplement/ombre non résolus ; elle ne multiplie plus le reveal des arbres explicites. Ce mélange d'aires est une approximation locale de la transition côtière, pas une intégration exacte des intersections de couronnes. Les flats d'estuaire ne sont pas des exclusions de racine.

Le gate GPU d'une plage partielle B=0,5 conserve 10 932 pixels de couronnes pleinement opaques, avec une fuite du détail de sol exactement nulle. La plage pleine n'a aucun arbre ; les flats seuls conservent les 14 875 pixels opaques du contrôle forestier. La vraie paire plage 160 m / 2,56 km est capturée en juillet V2, Render3D et janvier V2 : six PNG, protocole passé, couronnes fantômes retirées et sable conservé. Les deux contrôles Render3D sont strictement pixel-identiques au préflight 790. Reçus : `output/render-v2-floor/beach-root-final/captures/eea7a9b2aa83-f870d039-a911f822/report.json` et `legacy-pixel-equality.json`. Cette correction ciblée est acceptée ; la campagne globale reste à refaire après le travail hydrologique séparé.

La même étape porte le candidat marin validé sur deux vrais cadres côtiers : atténuation artistique 0,035 pour la mer, 0,15 inchangé pour les eaux intérieures. Les fonds peu profonds restent lisibles à 20–30 m, sans prétendre calibrer un coefficient Kd mesuré. La palette bathymétrique 150–3000 m reste distincte de cette atténuation. Le miroir NumPy et le GLSL partagé sont synchronisés ; 26 tests Python V2 passent, dont la reconstruction exacte de l'ancienne absorption intérieure. Les références optiques primaires et les limites de ce modèle restent dans le document de littérature du sol.
