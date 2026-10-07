> Rapport historique : les profils Earth et Macro A1-A4 sont retirés depuis la reprise du 7 octobre 2026. Voir [le nouveau bootstrap](TERRAIN_BOOTSTRAP_RESTART.md).

# Monde terrestre : géographie, climat et niveau de détail

Cette tranche répond aux cinq observations sur la navigation mondiale. Le modèle installé reste Terrain Diffusion 30 m : aucun nouvel entraînement ni remplacement des poids n'est effectué.

## Pourquoi le monde ressemblait à un archipel

Le conditionnement procédural original emploie une fréquence de bruit de 0,05 sur une grille de 7 680 m, soit une longueur caractéristique de l'ordre de 154 km. Étendre ce bruit sur une carte de 40 000 km répète de petites régions sans créer une organisation continentale. Le choix du conditionnement explique donc une grande partie du résultat, indépendamment de la capacité du réseau à produire du relief local.

Le profil `earth` introduit une géographie continentale à plusieurs échelles, des intérieurs à faible pente, des chaînes et des dépressions pouvant accueillir des bassins et des mers intérieures. Ces champs alimentent le réseau avant la génération. Aucune silhouette continentale n'est appliquée après coup à son relief. Le réseau peut modifier les côtes et les bassins de son entrée.

Le monde plat couvre **40 000 × 20 000 km**, avec le nord en haut et une latitude de +90° à −90°. Cette convention climatique ne constitue pas une projection sphérique ni une gestion des raccords au méridien. Le profil `natural` conserve le conditionnement procédural original pour comparaison.

## D'où vient le climat

Le modèle possède déjà des canaux climatiques. L'application précédente les utilisait à l'inférence, mais leur arrangement procédural n'était pas lié à la latitude et l'affichage montrait seulement une palette d'altitude.

Le nouveau conditionnement utilise les distributions des cinq sources originales : **ETOPO**, **WorldClim BIO1**, **BIO4**, **BIO12** et **BIO15**. Leur contenu entre dans l'identité du cache. Les distributions climatiques sont calculées par bandes de latitude sur les rasters complets, y compris les pôles ; la longitude réelle n'est pas copiée. Les graines produisent ainsi d'autres géographies avec des climats polaires, tropicaux, tempérés et des régions sèches possibles.

Les champs sont l'élévation en mètres, la température annuelle en °C, la variabilité thermique BIO4 en °C × 100, les précipitations annuelles en mm et leur coefficient de variation BIO15 en %. Le gradient thermique avec l'altitude suit la formule du pipeline existant. Le transport vers le navigateur encode une température de référence, les trois autres BIO et le gradient : le shader reconstruit la température à l'altitude de chaque pixel.

Le profil terrestre réduit le bruit du conditionnement de `0.5` à `0.1` sur les cinq canaux, en gardant les poids, 20 étapes coarse, deux étapes de fusion et BF16. Dans la formule upstream, ce paramètre plus petit renforce le signal fourni. Les essais `0.5`, `0.25` et `0.1` sur quatre régions ont favorisé `0.1` pour conserver l'entrée continentale et polaire. Le profil naturel reste à `0.5`.

Les entrées polaires sont éloignées de la moyenne thermique du checkpoint et ont révélé des pluies négatives. Dans `earth`, BIO4, BIO12 et BIO15 sont donc projetés vers zéro au minimum après la sortie coarse, avant fusion et conditionnement des latents. Ce correctif ne touche pas l'altitude, la température ou la statistique de relief p5 à cette étape ; les latents reçoivent bien le climat corrigé. `/api/status` expose les nombres de cellules corrigées. Il ne remplace pas un entraînement explicitement validé sur les pôles.

Les vues **biomes**, **température** et **précipitations** rendent ces données lisibles. Les biomes sont une classification visuelle de climat et de relief ; ils ne simulent pas la végétation, les glaciers ou l'hydrologie. La couleur d'une région polaire ne prouve pas que sa géométrie a été érodée par un glacier.

## Ce que cela change pour les fjords et les plaines

Il faut distinguer trois facteurs : la structure fournie au réseau, les régions visitées et ce que les poids ont appris. Le rectangle initial ne représentait pas la diversité d'un monde entier. Le nouveau profil fournit des candidats pour de grandes plaines et des bassins intérieurs, au lieu de compter sur une succession de petits sommets procéduraux.

Le modèle n'expose pas de commande « fjord », de bassin hydrographique imposé, ni de simulation glaciaire. Un climat froid ne garantit donc pas un fjord. Pour contrôler ces formes explicitement, une prochaine tranche devrait ajouter des contraintes géomorphologiques et vérifier leur conservation par le réseau, ou introduire un entraînement adapté. Ajouter une érosion GPU après génération reste envisageable, mais exige la cohérence des frontières, des échelles et du cache ; ce n'est pas une propriété acquise de cette tranche.

L'[appendice F du papier](https://arxiv.org/html/2512.08309v4) exclut explicitement les tuiles au-delà de ±60° et décrit des rotations/flips des crops pour apprendre du terrain sans direction privilégiée. La grille de téléchargement du dépôt local contient aussi les bornes −60°/+60°. Le checkpoint 30 m n'est pas exactement celui du dataset 90 m présenté dans le papier, et son manifeste d'entraînement complet n'est pas installé : l'exclusion publiée est donc un indice solide de limite polaire, sans prouver à elle seule chaque source de ce checkpoint. Le climat fourni ici par les rasters complets ne constitue pas un réentraînement polaire.

## Carte globale et raffinement

La vue mondiale est un **aperçu du conditionnement** utilisé par le réseau, et non une planète entièrement générée à 30 m. Elle est calculée directement à une résolution d'affichage bornée. Le client indique cette provenance. Quand la caméra se rapproche, les tuiles passent aux sorties apprises coarse, latentes puis au décodeur 30 m.

À un zoom donné, le rendu sélectionne le LOD cible et utilise seulement des parents recadrés pour combler les trous pendant la génération. Les anciennes tuiles plus fines restent éventuellement en cache, mais elles ne se superposent plus au niveau courant après un dézoom. Le filtrage des hauteurs et l'échantillonnage du rendu limitent aussi l'aliasing aux transitions d'échelle.

Un parent provisoire peut encore être moins détaillé qu'une tuile cible arrivée : ce compromis permet de naviguer pendant le calcul. Il disparaît à couverture complète. L'aperçu mondial plus dense apporte une couverture immédiate sans lancer un calcul neuronal sur toute la surface terrestre.

Les tuiles `earth` de LOD 7 et supérieur utilisent l'aperçu ; les LOD 4 à 6 utilisent le coarse appris à 7 680 m, le LOD 3 les latents à 240 m et les LOD 0 à 2 le décodeur à 30 m avec réduction. La grille affichée et la résolution source sont indiquées séparément. Le grand intervalle coarse → latents reste une limite pour ajouter du **détail neuronal réel** aux vues intermédiaires sans un coût élevé ; une interpolation plus dense ne le résout pas.

## Vérifications sur la 3090

L'audit `verify_earth_world.py --gpu`, serveur arrêté, a produit un aperçu 2048 × 1024 en **1,81 s CPU**, après **0,42 s** de préparation des distributions. Pour la graine 42, la fraction de terres est **27,77 %** et le plus grand composant terrestre de l'aperçu couvre **63,1 millions de km²**. Ces mesures décrivent le conditionnement, pas les côtes finales du réseau.

Quatre petits crops froids ont exécuté les trois réseaux sur CUDA et produit cinq sorties climatiques finies. Leurs médianes finales sont :

| Région testée | Altitude | Température | Précipitations annuelles |
| --- | ---: | ---: | ---: |
| Terre équatoriale | 162 m | 26,19 °C | 3 628 mm |
| Océan équatorial | −4 130 m | 26,62 °C | 3 275 mm |
| Terre polaire | 125 m | −37,04 °C | 3,02 mm |
| Plaine subtropicale sèche | 181 m | 26,35 °C | 142,63 mm |

Ces crops vérifient la transmission des champs aux réseaux et leur plausibilité locale ; ils ne prouvent pas l'hydrologie ni la diversité géomorphologique de toute la planète. Les tests CPU vérifient aussi plusieurs graines, les deux pôles, les plaines sèches, la reproductibilité des crops et l'isolation du profil original. Le rapport détaillé est `E:\TerrainDiffusionRuntime\earth-world-verification.json`.

Le parcours `verify_earth_viewer.cjs` vérifie le vrai serveur dans Chrome/WebGPU : monde global, quatre palettes, zoom sur la plaine sèche auditée via la minimap, décodeur 30 m, retour au LOD mondial avec les anciennes tuiles fines encore en cache, basculement PNG/GPU et affichage mobile. Il passe sans erreur JavaScript et avec les budgets mémoire respectés. Le premier parcours mondial a pris 5,39 s jusqu'à couverture complète des tuiles ; le parcours suivant avec cache disque a pris 1,42 s. Ces deux mesures locales incluent l'initialisation de page et ne sont pas un benchmark de débit ni un p95. Rapports et captures : `E:\TerrainDiffusionRuntime\earth-qa`.

## Références

Le [README officiel de Terrain Diffusion](https://github.com/xandergos/terrain-diffusion#quick-start) décrit le conditionnement grossier procédural ou dessiné, les entrées climatiques et l'affinement. Le [modèle 30 m installé](https://huggingface.co/xandergos/terrain-diffusion-30m) reste distinct du modèle 90 m proposé pour des mondes plus vastes. Le [papier](https://arxiv.org/abs/2512.08309) décrit la génération continue ; il ne promet pas une commande explicite pour chaque forme géomorphologique.
