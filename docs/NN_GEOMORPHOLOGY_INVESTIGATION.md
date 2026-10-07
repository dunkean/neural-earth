# Enquête : continents custom et perte de géomorphologie

Enquête du 7 octobre 2026, après le retour visuel de l'utilisateur. Lecture des
sources installées, du checkpoint épinglé, du papier et du dépôt officiel ;
réexamen CPU des exports NN précédents. Deux agents indépendants ont relu le
contrat et l'interprétation. Aucun changement du produit ni nouvelle inférence
GPU pendant cette enquête. Les exports cités restent les résultats historiques
des expériences déclarées dans leurs reçus.

## Conclusion et niveau de preuve

La meilleure première voie est de modifier **le conditionnement Natural** pour
piloter ses seules grandes structures continentales, en conservant ses
variations régionales, sa construction climatique et les réglages du checkpoint.
La faisabilité est plausible et cohérente avec l'architecture ; la qualité
Natural avec les quatre morphologies demandées n'est pas encore démontrée.

L'acceptation précédente établissait les contrats numériques, les côtes et le
runtime. Elle ne démontrait pas une géomorphologie comparable à Natural. Des
pentes élevées, une bonne fraction de terre ou des NPZ finis ne démontrent ni
l'organisation des vallées ni la qualité des crêtes.

## Ce que le modèle reçoit et apprend

Le checkpoint installé est `xandergos/terrain-diffusion-30m`, révision
`9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Son `config.json` indique une
résolution native de 30 m et `cond_snr=[0.5,0.5,0.5,0.5,0.5]`.

Natural est un générateur procédural d'entrée, pas un autre réseau ni une carte
terrestre mémorisée : cinq bruits Perlin transformés par quantiles puis corrections
climatiques (`synthetic_map.py:182–270`). Le canal d'altitude emploie quatre
octaves, fréquence de base 0,05 sur la grille coarse, lacunarité 2 et gain 0,5.
Le climat utilise notamment un résidu de régression pour BIO4 et une correction
de BIO15 en fonction de la pluie. Les cinq champs ne sont donc pas équivalents à
cinq tirages indépendants de distributions physiques marginales.

Dans `training/datasets/coarse_dataset.py`, les altitudes ETOPO sont transformées
par racine signée, agrégées spatialement en moyenne et moyenne−p5, puis associées
aux quatre variables climatiques. L'entraînement crée son conditionnement avec
les cinq canaux de la cible réelle auxquels il ajoute du bruit gaussien
(`:418–424`). Il ne fabrique pas explicitement une entrée continentale à 39 km
par floutage ou décimation de cette cible. Le checkpoint peut néanmoins accepter
des esquisses ; cette différence ne constitue pas une preuve d'impossibilité.

Le coarse prédit six champs, dont moyenne d'altitude et relief moyen−p5. Le
pipeline transforme le second en p5 avant le latent. Le latent reçoit une
empreinte 4×4 de moyenne/p5, un masque et le climat ; le decoder développe le
relief fin. Un conditionnement de relief incorrect peut donc influencer toutes
les étapes, même si la moyenne en mètres est correcte.

Le [papier v4, §§5.3–5.4](https://arxiv.org/html/2512.08309v4#S5.S4) décrit
explicitement des esquisses ou cartes procédurales guidant le coarse, puis le
latent et le decoder. Son coarse a un champ réceptif volontairement limité.
Les résultats publiés concernent la hiérarchie 90 m, pas une certification de
nos entrées custom du checkpoint 30 m.

## Écarts vérifiés dans notre branche custom

| Élément | Natural installé | Quatre bootstraps |
|---|---|---|
| Altitude d'entrée | Perlin multioctave évalué sur la grille NN | Raster physique 1024×512 interpolé |
| Pas de source d'altitude | Grille coarse à 7,68 km | 39,0625 km |
| Climat | Factory synthétique et corrections originales | Quantiles WorldClim par latitude, bruits indépendants plus lents |
| Bruit d'altitude | 0,5 | 0,05 |
| Bruits des quatre climats | 0,5 | 0,5 |

Le raster custom couvre 40 000×20 000 km. L'interpoler sur le lattice 7,68 km ne
reconstitue pas ses fréquences absentes. Natural contient au contraire des
octaves de paramètres 0,05/0,1/0,2/0,4. La relation fréquence/longueur d'onde n'est
pas celle d'une sinusoïde pure, mais l'écart de bande spatiale est réel. Cela ne
signifie pas que le NN est incapable d'inventer du détail à partir d'une carte
lisse ; cela modifie les conditions dans lesquelles il doit le faire.

Notre configuration native fixe également `erosion_steps=0`, `detail_m=0` et
plusieurs options de détail côtier à zéro (`native/terrain_bootstrap/src/main.rs:84`).
Le heightmap injecté n'est donc pas le résultat détaillé que produit le projet
world-builder avec ses autres configurations. Cela ne prouve pas que réactiver
l'érosion suffirait à faire fonctionner le NN ; la chaîne a été conçue pour lui
confier le relief fin, mais son signal de conditionnement reste à adapter.

Le nom `cond_snr` est trompeur. Dans `world_pipeline.py:1023–1030,1073–1075`
et notre implémentation conservée, avec `s=cond_snr` :

`condition = normalized_input / sqrt(1+s²) + noise * s / sqrt(1+s²)`

`s=0,05` ajoute environ 0,04994 de bruit normalisé, contre 0,44721 pour 0,5.
Le rapport bruit/signal diminue exactement dix fois. C'est une contrainte plus
forte sur **toutes** les fréquences de l'altitude, pas un contrôle séparé des
continents et du relief. L'ancienne sélection 0,05 privilégiait la tenue des côtes.
L'hypothèse est qu'elle ancre aussi trop fortement le signal régional custom ;
elle ne suffit pas à expliquer à elle seule toutes les formes observées.

Les unités et la racine signée sont conformes dans l'adapter. En revanche,
distributions marginales, unités et determinisme ne garantissent pas la
distribution jointe ni les statistiques spatiales vues à l'entraînement.

## Preuves visuelles et limites des anciennes comparaisons

Les planches dans
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/matched-relief/`
comparent des sorties NN complètes à palette et lumière fixes, même seed et
empreinte. Sur `archipelago-s0-mountain-median-relief-board.png`, les quatre
variantes sont entièrement terrestres. Les variantes custom 0,05/0,1/0,5 montrent
des reliefs en terrasses anguleuses tandis que Natural montre des formes de
drainage plus organisées. Revenir simplement à 0,5 ne corrige donc pas tout ce
cas visuel. Ce site se trouve toutefois vers 64° sud : les climats et les hauteurs
des variantes diffèrent. Il ne constitue pas une isolation causale du climat,
du spectre d'altitude ou du bruit.

La planche Earthlike « plain-median » montre aussi un signal custom plus lisse,
mais Natural n'y contient que 52,8 % de terre. Une comparaison de toute
l'empreinte y confond terre et mer. Les sites nommés historiquement
« mountain-median » étaient sélectionnés par hauteur, pas par qualité de massif.
Les nouveaux stress tests mesurent du relief fort, sans certifier son organisation.

## Affichage : ne pas confondre les étages

`terrain_server.py:388–456` montre que :

- LOD≥7 peut servir `conditioning-preview` : aucun NN n'a calculé ce relief.
- LOD4–6 sert le coarse appris à 7,68 km ; l'affichage plus fin est une interpolation.
- LOD3 sert le low-frequency latent à 240 m, avant le DEM final décodé.
- LOD0–2 sert le decoder, sauf relecture d'un mip de DEM déjà finalisé.

Ces règles sont communes aux profils. L'affichage intermédiaire peut masquer une
partie de la géomorphologie finale ; cela n'explique pas à lui seul la différence
entre Natural et custom. Natural peut en outre afficher un ancien `terrain.png`
dans sa région initiale, mais seulement pour la seed correspondante, en relief
et au LOD0 (`index.html:34,54`). Cela n'explique pas un avantage Natural au LOD3/4.

## Voie proposée et expérience décisive

Construire le prototype depuis le factory Natural brut. Piloter une composante
continentale à grande longueur d'onde, préserver ses variations régionales et
la calibration du signal envoyé au NN, puis appliquer les corrections Natural
avec la nouvelle altitude. Commencer avec les cinq bruits originaux à 0,5.
Les transitions côtières doivent rester continues. Une addition constante ou
un masque dur ne garantit pas les bonnes distributions ni l'absence d'îles
parasites ; le compositeur doit être évalué en espace encodé et physique.

World-builder peut fournir l'organisation macro, sans imposer chaque altitude
interpolée comme une vérité presque exacte. Les constantes de séparation des
bandes, les amplitudes et la tolérance de déplacement des côtes restent à mesurer.

L'[implémentation officielle, modification du monde](https://github.com/xandergos/terrain-diffusion#modifying-world-generation-advanced)
propose précisément la modification de `synthetic_map.py` ou l'entraînement
du petit modèle coarse. Son import de cartes conseille 7,7 km/pixel pour le
checkpoint 30 m ; il expose aussi la force de raffinement des cinq champs.

L'expérience minimale doit isoler six configurations : Natural original à
0,5 ; Natural original à 0,05 ; altitude custom et factory climatique Natural
à 0,5 ; altitude et climat custom à 0,5 ; custom actuel à 0,05 ; Natural à
contrôle continental à 0,5. Lorsqu'on remplace l'altitude avant `finalize`, le
gradient thermique est recalculé : conserver la méthode Natural ne signifie
pas conserver la température finale octet pour octet.

Utiliser plusieurs seeds, côtes et intérieurs non polaires, tenir constants les
poids, bruits NN, empreintes et lumières. Comparer les entrées, le coarse
(moyenne **et p5**), le latent, les DEM et leurs vrais étages LOD. Évaluer
l'organisation des vallées/crêtes, les spectres et autocorrélations en plus
des côtes, fractions terrestres, pentes et amplitudes. Un FID calculé sur quelques
crops ne serait pas une validation sérieuse.

Si cette adaptation ne permet pas de réunir continents et qualité Natural,
la prochaine voie est un fine-tuning du **coarse** avec des conditionnements
explicitement floutés/décimés à plusieurs échelles et des cibles réelles. Garder
d'abord latent et decoder inchangés, et contrôler la compatibilité de leurs
entrées. Avec la même architecture et le même solveur, ce fine-tuning n'impose
pas à lui seul davantage de coût d'inférence ; son coût d'entraînement n'a pas
été évalué ici. Aucun besoin de réentraîner tout le pipeline n'est établi.
