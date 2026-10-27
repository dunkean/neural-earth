# Audit Render v2 sur les tuiles générées

[English](render-v2-audit-40km.md)

Date : 23 octobre 2026. État : référence avant corrections, 24 sites réels capturés ; nouvelle validation en préparation.

Le rendu n'est pas validé visuellement. Les comparaisons de référence montrent une perte de luminosité au zoom, des plages recouvertes et une neige intermédiaire trop morcelée. Les constats et chiffres ci-dessous décrivent cette version de référence ; ils ne sont pas des mesures après correction. Les shaders ont depuis été modifiés et demandent une nouvelle inspection appariée.

## Seuil d'acceptation

Render et Render3D constituent le seuil de qualité des fonctionnalités déjà présentes. Render v2 doit conserver plages physiques, bancs sédimentaires, ombres du relief, lecture des pentes et détails visibles au sol, puis améliorer matériaux, saisons et transitions. La présence d'un nouvel effet ne compense pas la disparition d'une plage ou l'effacement du relief. Chaque comparaison utilise le même terrain, la même caméra et les mêmes réglages d'éclairage.

La galerie finale de 50 à 100 situations à cinq zooms sera produite après les corrections jugées satisfaisantes. Les captures intermédiaires servent au travail de correction et ne constituent pas cette livraison finale.

## Support de validation

Seules les captures du monde généré dans l'application servent à cet audit. Le premier essai de terrains analytiques a été écarté après la correction de l'utilisateur : il ne permettait pas de juger le paysage demandé. Ses images et ses chiffres ne comptent pas dans la couverture ci-dessous.

La matrice réelle croise six familles climatiques — forêt tropicale, tempérée, steppe, désert, taïga, toundra/polaire — et quatre morphologies — plaine, collines, plateau/escarpement, montagnes. Les codes réels et les altitudes sont mesurés sur les tuiles décodées ; un nom de fichier ne suffit pas à valider un biome ou un relief.

Chaque site est cadré sur 40 × 40 km, puis sur les mêmes coordonnées à trois niveaux rapprochés. Les captures de 1024 × 1024 pixels donnent respectivement :

| Étendue | Échelle d'affichage | Données de terrain |
|---|---:|---|
| 40 km | 39,0625 m/px | Décodeur LOD0, échantillons physiques de 30 m |
| 10 km | 9,765625 m/px | Mêmes tuiles LOD0 |
| 2,56 km | 2,5 m/px | Mêmes tuiles ; détails procéduraux visibles |
| 640 m | 0,625 m/px | Mêmes tuiles ; texture fine procédurale |

Le zoom agrandit ici les vraies données décodées et révèle les détails du shader. Il ne produit pas un DEM physique de 62,5 cm. Ce contrôle est complété par les vrais niveaux intermédiaires de la chaîne de génération.

Pour chaque zoom : Render v2 en juillet, Render3D en juillet, Render v2 en janvier. Même caméra, même terrain, même éclairage, sans HUD ni minimap. Juillet et janvier désignent des dates ; les saisons sont inversées entre hémisphères. Le contrôle des deux dates ne constitue pas une validation de l'animation annuelle complète.

Provenance : seed `7985403534069300490`, profil `orogen--g46261a320a644f43189ea849`, monde `f5cddc89751589636d1052bfa46529f6f0a400b1fd1b176a35dbb14ce3590a1a`. Le partage de génération est `gHY4ZFzncjegep8JTpf`. Les rapports conservent URL, caméra, résolution et stade réellement affichés, identité du monde, erreurs navigateur et empreintes des sources du rendu.

## Défauts déjà confirmés sur le monde généré

### Assombrissement au zoom

La forêt tropicale réelle perd fortement sa luminosité dans V2. Sur le même cadrage rapproché de 1,28 km à 2,5 m/px, le proxy de texture forestière donne une luminance d'affichage moyenne de 31,56/255, contre 57,89 dans Render3D ; 83,86 % des pixels du proxy sont sous 40/255, contre 12,18 %. La forêt tempérée donne 74,41 contre 90,68. Ces mesures portent sur des PNG réels appariés, hors HUD ; le proxy choisi sur Render3D n'est pas un masque sémantique et ne sépare pas albédo et ombre.

La matrice réelle montre aussi un assombrissement du sable au rapprochement. Il faut traiter la cohérence de l'exposition et de la couleur entre le matériau lointain et le matériau rapproché avant d'ajouter davantage de textures. Render3D sert de comparaison, pas de vérité satellite.

Preuves : [mesures réelles de canopée](../../output/render-v2-40km/real-detail/audit-real-canopy.json), [mesures des quatre zooms](../../output/render-v2-40km/real-matrix/readonly-image-metrics.json).

### Plages toujours recouvertes

Sur les vraies plaques `tropical-rainforest-low` et `warm-coast`, Render3D montre des bandes sableuses que V2 réduit à un liseré ou recouvre de vert. Le relief et la côte sont identiques. Le code mélange le sédiment de plage au sol ouvert, puis remélange la canopée sans retirer sa couverture sur cette plage : c'est une explication cohérente avec l'observation, à confirmer par une correction isolée et une nouvelle capture.

Il n'y a pas de métrique fiable de largeur de plage tirée d'un simple seuil RGB à cette échelle : eau peu profonde, sable et crêtes claires peuvent se confondre. Le constat repose sur les images appariées.

Preuves : `output/render-v2-40km/real/{tropical-rainforest-low,warm-coast}-{v2-july,render3d-july}.png` ; `web/terrain_render_v2.js`, fonction `v2Fragment`.

### Effet dalmatien aux niveaux intermédiaires

Sur les vraies vues `alpine-tundra`, `taiga-low` et la côte partagée `user-compact-pack`, V2 produit des nappes de neige morcelées et des plaques gris-vert/blanc qui concurrencent la lecture du relief. Le défaut est particulièrement visible aux niveaux 3 et 4. Il ne faut pas le confondre avec les ombres normales du DEM ni le réduire au seul grain de roche.

La matrice décodée révèle un second effet voisin, même sans changer de tuiles : au cadrage de 10 km, la taïga `dfc-rolling` et le site tempéré `cfb-rolling` montrent de grosses plaques de couvert/clairières. Au rapprochement, elles se résolvent en peuplements et trous forestiers. Le défaut de lecture existe donc aussi pendant la révélation du couvert ; supprimer seulement le bruit de grain du sol ne le résoudrait pas. La steppe réelle présente également ce comportement, tandis que les déserts inspectés montrent surtout une dérive de teinte et d'exposition.

Le décompte de pixels noirs ou rouges est insuffisant : un rendu peut ne contenir aucun pixel noir absolu tout en étant trop sombre, et un bruit peut former des taches très gênantes avec des couleurs pourtant naturelles.

Les vues intermédiaires sont des sorties réelles de l'application, avec les stades suivants :

| LOD demandé | Stade réellement affiché | Résolution physique source | Étendue du contrôle large |
|---|---|---:|---:|
| 3 | latent, qualité 3 | 240 m | 153,6 km à 300 m/px |
| 4 | coarse, qualité 4 | 7 680 m | 307,2 km à 600 m/px |
| 6 | coarse, qualité 4 réutilisée | 7 680 m | 1 228,8 km à 2 400 m/px |

Ces vues larges ne sont pas présentées comme des plaques de 40 km. Un second contrôle aux positions transmises par l'utilisateur compare les niveaux à cadrage de 40 km constant pour distinguer changement de données et changement d'échelle.

La voie coarse évalue le climat directement ; elle n'utilise pas le même tampon auxiliaire RGBA8 que la voie détaillée. Attribuer tous les gros amas blancs des niveaux 4/6 à une quantification thermique 8 bits serait donc faux. La rétention neigeuse sur un DEM de 7,68 km agrandi, ses pentes et le climat interpolé sont à examiner sur cette voie. Les détails de cette analyse figurent dans le complément sur les sols ouverts.

Sur les PNG réels, l'écart-type de la bande de luminance `Gauss(2 px) − Gauss(8 px)` atteint 10,30 dans V2 contre 5,43 dans Render3D sur l'alpin au niveau 3 ; 4,29 contre 2,79 sur la côte partagée. Au niveau 4, cette côte donne 2,71 contre 0,61. Le désert chaud, utilisé comme contrôle, donne 2,00 contre 1,93 au niveau 3 et 0,45 contre 0,46 au niveau 6. Cela appuie un défaut de couverture/neige dans les cas observés ; cette métrique d'image ne distingue pas à elle seule ombre, bruit et matière. [Mesures des vraies vues intermédiaires](../../output/render-v2-40km/audit-real-mid-metrics.json).

Preuves : [planches réelles intermédiaires](../../output/render-v2-40km/mid-sheets/), rapports `mid-lod3`, `mid-lod4`, `mid-lod6` et `transitions-lod*`.

### Banquise encore artificielle

Sur la côte exacte partagée, les chenaux forment encore un réseau très uniforme de petites cellules. À distance intermédiaire, l'ensemble ressemble à un motif graphique ; au niveau coarse les chenaux disparaissent. Les deux échelles ne donnent pas encore une matière crédible et continue. Les grandes surfaces marines presque uniformes renforcent l'aspect plastique. La capture hivernale de 40 km confirme le défaut ; la présence d'un layer de glace n'est pas une validation de sa qualité.

Preuves : `output/render-v2-40km/real/user-compact-pack-v2-july.png` et [comparaison de ses niveaux intermédiaires](../../output/render-v2-40km/mid-sheets/user-compact-pack.png).

## Couverture et analyse croisée

Les 24 sites ont été capturés aux quatre zooms et dans les trois styles/dates, soit 288 images réelles. Après réattribution documentée des deux reliefs arides, 15 croisements satisfont les seuils stricts de biome et de morphologie ; deux autres montagnes représentent des transitions naturelles mixtes avec au moins 70 % de la famille climatique demandée. Les sept catégories restantes demandent des remplacements. Ce bilan de classification ne vaut pas acceptation du rendu. La sélection sur l'atlas source à 19,53 km sert seulement à repérer des candidats ; elle ne valide pas leurs reliefs ou biomes après décodage.

Galerie : [vraies tuiles aux quatre zooms](../../output/render-v2-40km/real-matrix/index.html). Rapport machine : [captures, provenance et mesures DEM](../../output/render-v2-40km/real-matrix/report.json). Les cases dont les mesures contredisent la catégorie demandée restent marquées comme telles et ne sont pas comptées comme des croisements validés.

## Ordre de correction proposé

| Priorité | Changement à vérifier | Critère sur les mêmes vraies tuiles |
|---|---|---|
| 1 | Harmoniser l'exposition et la couleur entre matériau lointain, sol rapproché et couronnes | Pas d'assombrissement massif en passant de 10 km à 2,56 km ; relief encore lisible sous forêt et sur neige |
| 1 | Revoir la révélation des champs de couvert et de neige | Pas de plaques de camouflage qui apparaissent à un zoom intermédiaire avant les détails ; moyenne de couverture et de couleur conservée |
| 1 | Adapter la neige coarse à la résolution réellement disponible | Pas de gros amas issus de pentes/TPI de plusieurs kilomètres agrandis ; continuité à la substitution coarse → latent → decoder |
| 1 | Retirer le couvert végétal sur les plages retenues par le rendu | Les bandes sableuses des deux côtes réelles restent visibles, y compris sous les couronnes rapprochées |
| 2 | Retravailler la structure de banquise et sa transition d'échelle | Taille et contraste des chenaux crédibles, matière moins uniforme, passage au coarse cohérent |
| 2 | Vérifier les palettes minérales et les sols ouverts après ces corrections | Pas de saut d'ocre vers gris-brun uniquement provoqué par le zoom ; grain fin sobre et relief préservé |

Chaque correction doit être recapturée sur la même sélection réelle, puis inspectée en défilement. Ajouter du bruit supplémentaire avant de corriger ces transitions risquerait de renforcer les défauts observés.

## Limites de cet audit

Cet audit porte sur Neural Earth/WebGPU. Il ne revalide pas à lui seul l'intégration de `city_generator`. Les captures de base utilisent LOD0 forcé et une résolution physique de 30 m. Les contrôles intermédiaires montrent les vraies voies latent/coarse, mais des images statiques ne suffisent pas à juger tout le défilement automatique, le scintillement ou chaque transition temporelle. L'hydrologie est désactivée dans le partage de référence : la matrice terrestre ne valide donc ni tous les lacs/rivières saisonniers, ni tous les lits secs. Les champs agricoles restent exclus.
