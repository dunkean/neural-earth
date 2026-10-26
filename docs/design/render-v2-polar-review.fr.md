# Revue polaire réelle avant gel de Render v2

## Reproducteur et preuves

Monde `bd36737badf0cedf99200ba94bf7dc761070aca686a98784373247c86efe87dd`, centre océanique réel `[14657326.440143395, 7263946.615556731]`, hydrologie désactivée. Le centre voisin initial de l'utilisateur est terrestre ; le déplacement de 12 km sert à examiner effectivement la banquise. Les captures utilisent les vrais caches décodeur à 30 m, LOD0 forcé, et une caméra identique entre V2 et Render3D.

- Avant correction : `output/render-v2-polar-current/captures/0e195ba26a64-d11cd03a-a911f822/`, six vues à 40 km et 10 km.
- Après correction : `output/render-v2-polar-current/captures/db083bf3841a-6c77484e-a911f822/`, neuf vues à 40 km, 10 km et 5 m/pixel, pour V2 juillet, Render3D juillet et V2 janvier.
- La dernière emprise mesure 5,12 km sur 1024 pixels. Elle agrandit le DEM30 m et examine le matériau ; ce n'est pas une nouvelle géométrie à 5 m.
- Le rapport de chaque passe conserve les sources, résolutions, caméra, mois, clés de cache et hash de production. Les neuf vues finales satisfont le protocole ; cela ne constitue pas une validation photoréaliste de tous les milieux.

## Défaut observé et correction bornée

La première passe montrait un pack très pâle et lisse, des filaments sombres sinueux et un second réseau gris autour des plaques. La base cellulaire utilisait `F2-F1` comme épaisseur de fracture. Cette différence n'est pas une distance uniforme au plan séparant les deux sites, d'où des ouvertures larges et des variations d'épaisseur dans certaines configurations.

Pour le pack compact marin seulement, le même parcours des 27 voisins conserve désormais les deux vecteurs des sites les plus proches. Avec leurs distances `d1`, `d2` et leurs positions `A`, `B`, le champ devient :

```text
distance_bisectrice = (d2² - d1²) / (2 × |B - A|)
```

C'est la distance au plan équidistant de ces deux sites, issue directement de la géométrie euclidienne. Elle conserve les lieux `d1=d2`, les sites, les déformations du domaine, les concentrations et la sélection des plaques. Elle ne prétend pas calculer la distance à toute la frontière de Voronoï après déformation. Aucun second parcours cellulaire n'est ajouté.

La largeur nominale des chenaux compacts est ramenée à quelques mètres et intégrée sur l'empreinte du pixel. Les fractures sous-pixel contribuent par leur fraction au lieu de devenir des fils opaques. Le gris de glace refondue est atténué ; son ancien contraste dessinait un second réseau trop visible. Un modelé étroit de compression et un grain filtré plus fin remplacent une partie de cette variation, sans rajouter de grandes marbrures d'albédo. Lacs et rivières gardent leur ancien champ cellulaire et leur apparence ; le pack marginal et les icebergs conservent également leur chemin.

Les cellules sont une adaptation procédurale des fonctions de texture cellulaire de [Steven Worley, *A Cellular Texture Basis Function*, SIGGRAPH1996](https://itn-web.it.liu.se/~stegu76/TNM084-2019/worley-originalpaper.pdf), pas une simulation de dynamique de glace. Le principe de banquise jointe, fractures, chenaux et glace refondue s'appuie sur les descriptions du [NSIDC](https://nsidc.org/learn/parts-cryosphere/sea-ice/science-sea-ice). Le relief de compression est un indicateur de matériau ; aucune géométrie de crête ni photométrie spectrale n'est revendiquée.

## Résultat et limites

Sur les vraies vues de 40 km et 10 km, le second réseau est atténué et les fractures ressemblent moins à un fil de fer. Les grandes plaques restent jointives. La surface demeure claire et mate ; certaines sinuosités du champ cellulaire restent visibles au détail. Le relief terrestre est conservé, mais les taches de neige partielle observables sur la terre sont distinctes de cette correction marine.

La vue continentale `output/render-v2-floor/final-overview-lod6/` est à environ 3152 m/pixel avec une source coarse à 7680 m : elle ne résout pas les plaques kilométriques. Son front bleu–blanc très étalé correspond au champ moyen de gel à cette échelle et ne démontre pas, à lui seul, un matériau plastique. L'eau de Render3D et son gel uniforme ne constituent pas non plus une référence de plaques physiques.

Vérifications réalisées : 37 tests CPU Render v2 passent, dont huit polaires. Sur 4096 positions, les quatre sorties du matériau sont strictement identiques avant/après pour le pack marginal, les icebergs, les lacs et les rivières. La comparaison réelle ne masque pas les limites du proxy procédural. Les gates GPU de parité, les coûts et la campagne complète sont effectués séparément par la passe globale après gel.
