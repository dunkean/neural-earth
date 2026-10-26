# Ressources naturelles multi-échelles — potentiel, stock, flux

Date : 26 octobre 2026. Branche locale `resources-layers` de Neural Earth, rebasée sur `origin/main` `051dd1f`, qui contient le merge Render v2 `d6cbb78`. Destinataires : Neural Earth, qui produit, Chronos, qui consomme et porte le temps, et Magna Urbis, qui consomme localement et construit les cités.

## 0. Statut et lecture

Ce document distingue trois statuts :

- **Implémenté** : code présent dans `backend/terrain_resources.py`, `backend/terrain_resource_layers.py`, `web/terrain_resources.js` et les tests de `tests/python/test_terrain_resources.py`.
- **Mesuré** : valeur obtenue sur le monde Orogen en cache (seed `7985403534069300490`, style `continents`), avec la commande indiquée.
- **Proposé** : conception non encore codée, ou calibration provisoire.

Toute valeur numérique sans référence vérifiée est marquée « provisoire ». Les sources sont listées au §17. Seules les valeurs relues sur la source y figurent ; les autres sont signalées comme non vérifiées.

Les ressources sont un **a priori procédural**. Elles ne constituent ni une carte géologique, ni un inventaire agronomique, ni une prévision. Leur intérêt est la cohérence : avec la tectonique d'Orogen, avec le climat et entre les échelles.

Ce document reprend la spécification des 35 couches (`chronos/NeuralEarth-Resources-Starter.md`) et le catalogue (`chronos/NeuralEarth-Resources.md`). Il en donne une réalisation et un contrat multi-échelle ; il ne les recopie pas.

## 1. Décisions

1. **Une seule chaîne de ressources, dans Neural Earth, alignée sur ses LOD.** L'autorité est monde, décidée depuis l'atlas Orogen. Chaque LOD plus fin ne fait que répartir ce que son parent a engagé. C'est l'architecture de `docs/design/multiscale-hydrology.md` : autorité globale, raffinement régional, propriété des frontières.
2. **Un seul adressage** : la grille des LOD d'affichage (`30 × 2^LOD` m, origine au centre de la carte). **Un seul hasard adressé** : `key(seed, couche, LOD, cellule)`, identique à la fonction de Chronos.
3. **L'état naturel initial est immuable**. Chronos porte le **delta historique** : extraction cumulée, défrichement, coupe. Toute échelle lit « naturel + delta » à une date donnée.
4. **Le rendu n'est pas l'état physique.** Les champs de Render v2 sont consommés comme proxys documentés. Les réglages d'affichage, comme le curseur de forêt, ne modifient jamais une ressource.
5. **Magna Urbis n'est pas un niveau à part.** Son compilateur de sol de tuile 40 km / 30 m est un raffinement Neural Earth, hébergé provisoirement. Voir la note `docs/RESOURCE_TILE_REFINEMENT.md` du dépôt Magna Urbis.

## 2. Ontologie et unités

| Nature | Définition | Agrégation entre échelles | Exemples (identifiants de champs) |
|---|---|---|---|
| **Potentiel** (aptitude, prospectivité) | Possibilité d'un usage ou d'une présence, en indice de 0 à 1 ou en intensité par unité de surface | Moyenne pondérée par la surface pertinente, jamais une somme | `cereal_suitability`, `prospect_<famille>`, `stone_ashlar`, `transhumance` |
| **Stock** | Quantité présente à la date de référence, non renouvelable ou très lente | Somme exacte d'entiers | `stock_<métal>` (t ; kg pour l'or et l'argent), `wood_stock_m3`, `peat_stock_m3`, gisements |
| **Flux** | Renouvellement ou production par an ou par mois | Somme exacte d'entiers par période | `cereal_net_t_year`, `wood_flux_m3_year`, `fish_*_month_t`, `sea_salt_month_t`, `bog_iron_t_year`, `water_yield_m3` |
| **Saison et accès** | Mois où un flux est disponible ou un travail possible | Pas de somme : calendrier par cellule | `harvest_month`, `felling_months`, `sledge_months`, `v2_snow_month` |

Le comportement dans le temps dépend de la classe de la ressource. C'est Chronos qui fait évoluer ces quantités (§10).

| Classe | Règle | Délai typique | Ressources |
|---|---|---|---|
| Épuisable | `R(t+1) = R(t) − E(t)`, sans régénération | aucun | minerais, sel gemme, charbon, pierre rare |
| Régénérable avec délai | croissance logistique `G = r·S·(1 − S/K)` | taillis : 7–35 ans selon l'essence ; futaie : plusieurs décennies | bois, gibier, poisson, fer des marais |
| Renouvelable annuel | flux annuel sans stock, limité par la surface engagée | une saison | céréales, fourrage, sel marin, eau |
| Quasi fossile | régénération physique mais négligeable à l'échelle historique | tourbe : environ 0,4 mm/an, soit environ 12 cm en 300 ans | tourbe |

Le délai du fer des marais repose sur Singewald 1911, cité par l'USGS OFR 2003-346 : des dépôts épuisés « à nouveau exploitables après quelques années ». C'est une observation anecdotique du Maryland, pas une vitesse mesurée. Le taux de la tourbe est un calcul fait à partir de Loisel et al. 2014 (23 g C/m²/an, densité 0,12 g/cm³, 47 % de carbone) ; il n'est pas publié tel quel.

**Quanta d'export (implémenté, `QUANTUM`).** Chaque quantité extensive est un entier dans une unité fixe :

| Champ | Quantum |
|---|---|
| surface arable | 1 m² |
| céréales nettes | 1 kg/an |
| stock de bois | 1 m³ |
| flux de bois | 0,01 m³/an |
| eau | 1 000 m³/an |
| tourbe (surface, volume) | 1 m² et 1 m³ |
| fer des marais | 1 kg/an |
| métaux | 1 t (or et argent : 1 kg) |

Les entiers rendent la conservation vérifiable : `somme(enfants) == parent`, à l'unité près, en Python comme en Rust.

## 3. Architecture multi-échelle

### 3.1 Adressage (implémenté)

Une cellule `(LOD, i, j)` couvre `[i, i+1) × [j, j+1) × 30·2^LOD` mètres de carte. L'origine est au centre de la carte et y croît vers le sud, exactement comme les tuiles d'affichage (`terrain_server.py` : `xs = tx·256·30·2^LOD + …`).

- **Identifiant 64 bits** : 5 bits pour `LOD+16`, 30 bits pour `i + 2^29`, 29 bits pour `j + 2^28`. Il est valable du LOD −6 (47 cm) au LOD 15.
- **Parent** : `(i >> k, j >> k)`, c'est-à-dire une division entière arrondie vers le bas, aussi pour les indices négatifs.
- **Aire exacte sur la sphère** pour une cellule équirectangulaire : `Δx · R · (sin φ_nord − sin φ_sud)`. Les cellules du bord (antiméridien, pôles) sont rognées aux limites du monde ; leur aire est celle de la partie rognée.
- La somme des aires vaut `4πR²` à 10⁻¹² près, testé aux LOD 10, 12 et 14. Les aires des enfants somment exactement celle du parent.
- **Limite (implémenté) :** seul le diamètre par défaut est pris en charge (bornes ±20 000 × ±10 000 km). Un monde plan ou redimensionné est refusé avec une erreur explicite.

### 3.2 Hasard adressé (implémenté)

- `key(seed, parts)` est la fonction de `chronos-core/src/keys.rs` : FNV-1a sur les parties, séparateur `0xff`, puis splitmix64. Le test `test_keys_match_chronos_rust` compare cinq valeurs produites par la compilation de `keys.rs` avec rustc 1.99.
- **Par cellule** : `splitmix(key(seed, [domaine…, "lod<L>"]) XOR cell_id)`, puis `splitmix(… XOR k)` pour le k-ième tirage.
- Aucun état aléatoire n'est partagé et aucun `default_rng` n'est utilisé. Le résultat ne dépend ni de l'ordre de calcul, ni de la fenêtre demandée.
- Le bruit de valeur des provinces lithologiques est aussi adressé par cellule, au LOD 13 (245,76 km), puis interpolé. Un même point donne donc la même valeur, quelle que soit la grille qui l'échantillonne.

### 3.3 À quel LOD chaque fait est décidé

| LOD (pas) | Faits décidés | Origine de l'information | Calcul | Statut |
|---|---|---|---|---|
| Atlas Orogen (≈ 19,5 km) | Distances aux frontières, contextes tectoniques, lithologie à la résolution de l'atlas | `crust`, `boundaries`, `convergence`, `hotspot`, `backArc`, `uplift`, climat sur deux saisons | à la compilation monde | implémenté |
| **10 (30,72 km)** | **Autorité** : fractions de contexte et de roche, prospectivités, climat mensuel, bilan hydrique, NPP, céréales, pâture, bois, gibier, pêche, sel, tourbe, matériaux ; **nombre, tonnage et teneur des gisements** ; budgets entiers par cellule | atlas échantillonné, bilan hydrique partagé, évaluateur Render v2 | une fois par monde, environ 65 s, puis cache disque (§15) | implémenté |
| 11 et plus (≥ 61 km), régions Chronos | Sommes exactes des cellules du LOD 10 ; une région est une union de cellules du LOD 10 | agrégation | à la demande | implémenté (`aggregate`, `region_summary`) |
| 9 à 4 (15,36 km – 480 m) | Budgets entiers : sommes des enfants au LOD 3. **Champs affichés : descente avec bruit adressé et conservation exacte des moyennes du parent (§16)** | agrégation ; relief neuronal du LOD | à la demande, par tuile | implémenté |
| **3 (240 m)** | **Position des gisements** ; budgets aréaux répartis exactement depuis le LOD 10 | intensités interpolées du LOD 10, aire exacte des enfants, facteur canonique optionnel (relief latent à 240 m, proposé) | à la demande, par cellule parente | implémenté (`descend_budget`, `_place_in_cell`) |
| 2 à 0 (120–30 m) | Champs affichés descendus jusqu'au décodeur à 30 m, modulés par le DEM neuronal et les rivières sources (§16) ; emprises de gisements (filons, amas, lentilles, placers, halo, chapeau de fer) ; mélanges continus de roches à strates et intrusions | DEM neuronal, hydrologie source | à la demande, par tuile | implémenté (viewer) |
| 0 (30 m), tuile 40 km | Répartition exacte de chaque bloc de 240 m entre ses 64 cellules ; carrières, glaisières et tourbières locales ; géologie fine de `surface_context.py` conditionnée | champs de tuile (`surface_context.py`) | à la demande par tuile | proposé (note Magna Urbis) |
| −1 à −6 (15 m – 47 cm) | Champs affichés : bruit adressé seulement, conservé sur le LOD 0 ; aucun relief amplifié n'est lu. Apparence : textures, teintes d'oxyde, croûtes de sel, tourbe visible | même chaîne ; aucun nouveau stock | à la demande ; rendu | implémenté (champs), proposé (textures, §12) |

La règle de l'hydrologie s'applique ici aussi : **pas de hiérarchie supplémentaire sur une grille interpolée qui n'apporte pas d'information**. Les LOD 9 à 4 sont des agrégats, pas des niveaux de décision.

Le LOD 3 est le premier niveau qui a sa propre information neuronale : la reconstruction latente à 240 m. C'est là que les gisements prennent position, et que le relief latent pourra pondérer les budgets (proposé).

### 3.4 Règles multi-échelles

1. **Conservation.** Toute quantité extensive est un entier engagé au LOD 10. Les enfants le répartissent par `apportion` : plus forts restes, poids entiers ≤ 2^20, égalités départagées par identifiant de cellule croissant. Les parents et les régions le somment. Les tests vérifient l'égalité exacte.
2. **Budgets hérités.** Un enfant ne crée ni ne détruit de quantité. Si tous ses poids sont nuls, le total est réparti uniformément, sans perte. Un gisement appartient à sa cellule du LOD 10 (`owner_cell`) : sa position au LOD 3 est une descendante de cette cellule (testé).
3. **Indépendance à l'ordre.** Le résultat d'une descente est une fonction pure du parent, de son voisinage immédiat (interpolation) et de la seed. Des descentes demandées dans un ordre différent donnent des octets identiques (testé).
4. **Le raffinement ne contredit jamais le parent.** Un affleurement fin peut révéler ou cacher un gisement, mais ne change ni son tonnage ni sa famille. La géologie fine est contrainte à reproduire les fractions du parent (note Magna Urbis).
5. **Intensives et extensives.** Les rendements, aptitudes et teneurs ne se somment pas. Un rendement régional se recalcule à partir des surfaces engagées et des flux.
6. **Propriété des frontières.** Un gisement, un banc de pêche ou une forêt qui traverse une limite de région garde un seul propriétaire, sa cellule du LOD 10. Les régions Chronos sont des unions de cellules : il n'y a donc pas de double compte.
7. **Fenêtres et halos.** Une tuile qui ne couvre qu'une partie d'un parent lit l'allocation du parent entier au LOD 3, puis ne raffine au LOD 0 que les blocs complets de 240 m qu'elle couvre. La tuile 40 km est étendue à 1 336 = 167 × 8 cellules.

## 4. Entrées Orogen et contexte tectonique (implémenté)

Les entrées viennent de l'atlas persistant de 2 048 × 1 024 (`terrain_orogen.py`) :

- `crust` : 1 = plaque océanique ;
- `boundaries` : 1 convergente, 2 divergente, 3 transformante ;
- `convergence` : contrainte propagée ;
- `hotspot`, `backArc` (négatif dans un bassin arrière-arc), `tecActivity`, `uplift` ;
- `temperature_*` (`T = 90·t − 45` °C) et `precip_*` (`P = 1000·p` mm par semestre).

Les zones de frontière sont classées dans un voisinage 5 × 5 de `crust` :

- **subduction** : frontière convergente mêlant croûte océanique et continentale ;
- **arc intra-océanique** : convergente entièrement océanique ;
- **collision** : convergente entièrement continentale ;
- **rift continental** : divergente entièrement continentale.

Les distances sont calculées par transformée de distance, avec la longitude bouclée.

| Contexte | Poids (avant normalisation) | Lecture géologique |
|---|---|---|
| Arc | `3·exp(−(d_sub/300 km)²)` côté continental, `exp(−(d_io/200)²)` sur plaque océanique | arc magmatique, porphyres, épithermal |
| Collision | `3·exp(−(d_col/250)²)` | chaîne métamorphique, granites de collision |
| Rift | `3·exp(−(d_rift/200)²)` | grès rouges, basaltes, évaporites |
| Point chaud | `3·clip(hotspot/0,25)` | basaltes de plateau |
| Arrière-arc | `1,5·clip(−backArc/0,06)²` | bassins volcano-sédimentaires |
| Ophiolite | `2,5·(0,7·plaque océanique émergée + 0,3·exp(−(d_col/80)²))` | croûte obductée, suture |
| Craton | `3·(continental)·smooth(350, 900, d_conv)·(1 − smooth(0,3, 0,9, contrainte))·(1 − smooth(0, 0,08, uplift))` | boucliers anciens |
| Marge passive | `1,5·exp(−(d_mer/250)²)·(1 − max(arc, collision))` | plates-formes carbonatées littorales |
| Plate-forme | 0,5 sur croûte continentale | couverture sédimentaire intracontinentale |
| Océanique | plaque océanique immergée + 0,02 | fond océanique |

**Mesuré** sur le monde de référence, en part de la surface émergée :

| Contexte | Part |
|---|---|
| Collision | 24,7 % |
| Craton | 24,2 % |
| Plate-forme | 20,6 % |
| Arc | 11,2 % |
| Rift | 8,5 % |
| Arrière-arc | 4,4 % |
| Ophiolite | 3,5 % |
| Marge passive | 2,0 % |

Sur ce monde, `tecActivity` dépasse 0,64 sur 95 % des terres continentales : il ne peut pas désigner les cratons. Le critère retenu est donc la distance aux zones convergentes et la faible contrainte.

**Limite :** la métrique est plane dans l'atlas, ce qui surestime les distances est-ouest vers les pôles. Les tampons sont donc plus larges aux hautes latitudes. La correction par distance de corde, comme `terrain_glacier.py`, est proposée.

## 5. Géologie, provinces métallogéniques et gisements

### 5.1 Lithologie (implémenté)

Huit classes : granite, calcaire, grès, argilo-marneux, basalte, métamorphique, ultrabasique, évaporites. Les cinq premières sont celles de `surface_context.py` de Magna Urbis. Le métamorphique et l'ultrabasique alimentent O1 (§12).

Les fractions sont le produit des fractions de contexte par une table contexte × roche (`LITHOLOGY`, proposée, non calibrée). Deux modulations s'y ajoutent :

- **Évaporites** : `+0,10 · bassin · basse terre · max(aridité actuelle, 0,6 · paléo-aridité)`. La paléo-aridité est un champ adressé au LOD 13. Les évaporites enregistrent un climat passé : l'aridité actuelle n'est qu'un indice parmi d'autres.
- **Variation de province** : `× exp(0,35 · bruit adressé)` par classe, au LOD 13.

**Mesuré**, en part des terres : granite 21 %, grès 18 %, métamorphique 18 %, argilo-marneux 15 %, basalte 13 %, calcaire 13 %, ultrabasique 1 %, évaporites 0,7 %.

### 5.2 Familles de gisements (implémenté, calibration en partie vérifiée)

Les associations entre contexte, roche hôte et type de gîte suivent les modèles descriptifs de l'USGS (Cox & Singer 1986, Bulletin 1693). Les numéros ci-dessous ont été vérifiés. Le fer oolithique (34f) est dans le Bulletin 2004, les évaporites dans l'OFR 91-11A.

| Famille | Modèle(s) USGS | Contextes | Hôtes | Métaux contenus (fraction du minerai) | Minerai médian | Statut de la calibration |
|---|---|---|---|---|---|---|
| Porphyre Cu(-Au) | 17, 20c | arc (fort), arrière-arc, collision | granite, volcanites | Cu 0,54 %, Au 0,2 g/t, Ag 2 g/t | 144 Mt | médianes Cu et tonnage **vérifiées** (17) |
| Épithermal Au-Ag | 25a–c | arc, arrière-arc, rift, point chaud | volcanites | Au 5 g/t, Ag 100 g/t | 0,8 Mt | provisoire |
| Sn-W de granite | 15a–c | collision, craton | granite, métamorphique | Sn 1,3 %, W 0,3 % | 0,24 Mt | Sn et tonnage **vérifiés** (15b) |
| Filons polymétalliques | 22c | collision, arc, rift | métamorphique, granite, grès, calcaire | Pb 6 %, Zn 4 %, Ag 300 g/t, Cu 0,2 % | 0,2 Mt | provisoire |
| Sulfures massifs volcanogènes | 24a, 28a | ophiolite, arrière-arc, arc | basalte, ultrabasique | Cu 1,6 %, Zn 2 %, … | 1,3 Mt | Cu et tonnage **vérifiés** (24a) |
| Or orogénique en quartz | 36a | collision, craton | métamorphique, basalte | Au 16 g/t | 30 000 t | **vérifié** (36a) |
| Pb-Zn type Mississippi | 32a–b | plate-forme, marge passive | calcaire | Pb 2 %, Zn 6 % | 5 Mt | provisoire |
| Cu sédimentaire | 30b | rift, plate-forme | grès, marnes | Cu 2 % | 20 Mt | provisoire |
| Skarn Fe-Cu | 18b | arc, collision | calcaire au contact du granite | Fe 45 %, Cu 0,5 % | 3 Mt | provisoire |
| Fer rubané | 34a, 28b | craton | métamorphique | Fe 50 % | 165 Mt | **vérifié**, moyenne arithmétique Superior + Algoma |
| Fer oolithique | 34f | plate-forme, marge passive | marnes, grès, calcaire | Fe 30 % | 20 Mt | provisoire |
| Fer latéritique | — | craton, point chaud, ophiolite ; climat tropical | basalte, ultrabasique | Fe 45 % | 5 Mt | provisoire |
| Sel et gypse | 35a.3, 35a.5 | rift, arrière-arc, marge passive, plate-forme ; **requiert des évaporites** | évaporites | NaCl 90 %, gypse 5 % | 50 Mt | provisoire |
| Charbon | — | plate-forme, bassins ; climat humide | grès, marnes | charbon 90 % | 50 Mt | provisoire ; un paléoclimat adressé serait préférable |
| Placers aurifères | 39a | aval des sources d'or (lissage d'environ 60 km), écoulement, faible relief | alluvions | Au 0,2 g/t | 1,1 Mt | **vérifié** (39a) |
| Placers stannifères | 39e | aval du Sn-W | alluvions | Sn 0,3 % | 0,3 Mt | provisoire |

**Conversion des médianes.** Les médianes vérifiées sont tirées de l'annexe B du Bulletin 1693, qui donne des moyennes de log10 (médiane = 10^moyenne). Cette conversion est la nôtre. La version HTML contient des erreurs d'OCR ; une relecture sur le PDF imprimé est recommandée.

**Distributions.** Le Bulletin indique que tonnages et teneurs suivent des distributions log-normales. Pour certains minéraux industriels, l'OFR 92-437 les donne plutôt normales.

**Tirage (implémenté).** Il se fait par cellule du LOD 10 et par famille :

1. **Nombre** : `N ~ Poisson(λ)`, avec `λ = densité · prospectivité · aire / 10⁴ km²`, obtenu par inversion de la CDF au point `u = key(…, "count", lod10, cellule)`.
2. **Tonnage** : `minerai = médiane · 10^(σ·Φ⁻¹(u))`, avec σ provisoire entre 0,65 et 0,8.
3. **Teneur** : facteur commun `10^(0,25·Φ⁻¹(u))`, appliqué à tous les métaux du gisement. Un gisement polymétallique garde ainsi une seule masse hôte.
4. **Profondeur du toit** : log-normale autour d'une médiane par famille.
5. **Position** : choix pondéré entre les 128 × 128 enfants du LOD 3, par course exponentielle `argmin(−ln u / w)`. Les poids `w` sont la prospectivité interpolée. Le choix est indépendant de l'ordre.

**Accessibilité médiévale (implémenté, provisoire).** L'extension verticale vaut `(minerai / 2,7 t/m³)^(1/3)`. La profondeur atteignable vaut `40 m + 0,5 × rugosité régionale`, pour tenir compte des galeries d'exhaure en relief. La part accessible est `clip((atteignable − toit) / extension, 0, 1)`. Le récupérable médiéval est `contenu × accessible × rendement de fonte`. Le zinc et le tungstène ne sont pas récupérés comme métaux au Moyen Âge.

Un gisement est **affleurant** si son toit est à moins de 5 m. Il a un **chapeau de fer** (gossan) s'il est sulfuré, affleurant et hors climat glacé.

**Mesuré** sur le monde de référence : 7 989 gisements. Les plus nombreux sont les filons polymétalliques (2 386), l'or orogénique (1 262), le Sn-W (1 063) et les placers aurifères (1 060). On compte 14 gisements de sel, conditionnés aux évaporites.

### 5.3 Ce qui reste proposé

- Des provinces métallogéniques nommées, comme objets avec un identifiant, au lieu de champs continus seulement.
- Des distributions de σ issues des modèles quantitatifs (Singer & Menzie 2010, non relu).
- Des placers transportés le long du réseau hydrologique (§7.3).

## 6. Ressources biologiques et agricoles (implémenté)

### 6.1 Climat mensuel et eau

- **Mois** : convention Orogen (`terrain_hydrology_balance`). La température est harmonique, avec un pic en juillet pour les champs « été nord ». Les pluies sont égales à l'intérieur des semestres avril–septembre et octobre–mars.
- **Bilan hydrique** : `monthly_water_balance` (neige, sol de 150 mm, nappe), avec ETP d'Oudin. On en tire ETR, écoulement et neige par mois. Rien n'est recalculé de façon concurrente.

### 6.2 Production primaire et céréales

- **Production primaire nette, modèle Miami** (Lieth 1975, constantes relues dans Adams et al. 2004) : `NPP = min(3000/(1+e^{1,315−0,119T}), 3000·(1−e^{−0,000664P}))`, en g MS/m²/an. Elle est répartie sur les mois au prorata de l'ETR des mois où T ≥ 5 °C.
- **Saison de croissance**, de type GAEZ : mois où T ≥ 5 °C et ETR ≥ 0,5 ETP.
- **Céréales** : trapèzes de type Ecocrop, avec 0 hors des limites absolues, 1 dans l'optimum, interpolation linéaire entre les deux et facteur le plus limitant (Hijmans, Recocrop). Deux profils sont évalués, et le meilleur est gardé par cellule :
  - tempéré (blé, orge, seigle) : saison 5/12/20/28 °C, pluie 250/500/1100/2000 mm, mois le plus froid ≥ −30 °C ;
  - chaud (millet, sorgho) : saison 16/22/30/36 °C, pluie 250/450/900/1600 mm.
  - L'aptitude climatique est multipliée par un facteur de sol (types pédologiques) et un facteur de relief.
  - **Classes GAEZ v4** sur la part du rendement atteignable : VS ≥ 80 %, S 60–80 %, MS 40–60 %, mS 20–40 %, vmS < 20 %, NS.
- **Rendement médiéval** par rapport récolte/semence : `rapport = 1 + (5 − 1) · aptitude`, puis `brut = semence · rapport`, avec une semence provisoire de 0,15 t/ha.
  - Repère vérifié : en Angleterre, la moyenne des quatre grains est de 3,7 pour 1200–1249 et de 4,7 pour 1250–1499 (Slicher van Bath 1963). Le maximum de 5 est donc une borne haute médiévale.
  - Le **net** est `brut − semence`. La **production potentielle** est `net × surface arable × 0,6`, pour un assolement triennal ; cette part est provisoire.
- **Surface arable naturelle** : `terres × relief × climat × (1 − 0,6·organique) × (1 − neige pérenne)`. C'est un potentiel, pas une surface cultivée : le défrichement est un delta Chronos.

### 6.3 Pâture et transhumance

- **Fourrage** : `NPP mensuelle × part herbacée × 0,25`. Le coefficient de 0,25 est l'efficacité de prélèvement d'un parcours naturel pâturé toute la saison (NDSU R1810).
- **Part herbacée** : `herbe de Render v2 × (1 − 0,8 · couvert ligneux)`. L'herbe de Render v2 est déjà réduite par la neige saisonnière : un pâturage enneigé n'est pas pâturé.
- **Capacité** : UGB/km² par mois, avec une UGB de 250 kg (FAO) qui ingère environ 3 % de son poids par jour, soit 228 kg MS/mois. La valeur de 3 % est lue sur la table NDSU pour une vache de 1 000 lb ; son application à 250 kg est la nôtre. La capacité annuelle « avec foin » et celle du mois le plus maigre sont exportées.
- **Transhumance** : fourrage d'été des hauteurs à plus de 900 m dans un rayon d'environ 150 km, multiplié par le manque de fourrage d'hiver en plaine.

### 6.4 Bois

- **Accroissement** : `NPP · 0,4 (part ligneuse) · 0,5 (part utilisable) / 0,55 t/m³`.
  - Repères : environ 2,5 t MS/ha/an pour le taillis de chêne, frêne et aulne (Evans 1984, cité par le TCV) ; densités basales de 0,57–0,58 pour le chêne, le hêtre et le frêne, 0,51 pour le bouleau (FAO/GIEC).
  - Pour une NPP de 1 200 g/m², la formule donne 2,4 t/ha/an, soit 4,4 m³/ha/an.
- **Stock initial** : `accroissement × 60 ans × couvert`. La durée de 60 ans est provisoire.
- **Répartition du même stock**, sans double compte :
  - part bois d'œuvre : `smooth(10, 25 m, hauteur de canopée Render v2)` ;
  - part résineuse : `needle` de Render v2.
- **Calendrier** :
  - mois d'abattage : feuillage de Render v2 inférieur à 0,35, c'est-à-dire repos végétatif ;
  - mois de débardage sur neige : neige de Render v2 supérieure à 0,5.

### 6.5 Faune, pêche et eau

- **Gibier** : `D = 10^4,23 · W^−0,75` individus/km², avec W en grammes (Damuth 1981, consommateurs primaires). Pour une masse de référence de 80 kg, D ≈ 3,6/km². La densité est modulée par `NPP/1000` (borné à 2). Le prélèvement durable provisoire est de 15 % par an.
- **Fourrures** : potentiel = couvert ligneux × froid hivernal.
- **Pêche continentale** : 42 kg/ha/an de plaine inondable, soit 4,23 t/km² (Crul 1992, d'après Welcomme 1985).
  - La part de plaine inondable par cellule est un proxy : écoulement × faible rugosité × 3 %.
  - Le pic de capture suit d'un mois le pic d'écoulement.
  - Contrôle possible : Welcomme 1985 donne `C = 0,03 · A^0,97` (t/an, bassin en km²).
- **Pêche côtière** : plateau inférieur à 200 m, 2 t/km²/an × (0,5 + chaleur), provisoire. L'accès est réduit de moitié les mois de glace (Render v2). Pauly & Christensen 1995 situent le besoin en production primaire des pêcheries de plateau au-delà de 20 % ; la valeur retenue reste à calibrer.
- **Eau** : écoulement annuel (mm et m³) du bilan partagé.

### 6.6 Saisonnalité : ce qui est mensuel et ce qui ne l'est pas

| Mensuel (12 valeurs) | Annuel ou date unique | Non saisonnier |
|---|---|---|
| NPP, fourrage et UGB, neige au sol (Render v2), feuillage (Render v2), glace (Render v2), écoulement, pêche continentale (pic de crue), pêche côtière (glace), sel marin (mois chauds et secs) | mois de récolte, mois d'abattage et de débardage, capacité du mois maigre | gisements, roches, prospectivité, pierre, argile, sable, stock de tourbe, couvert potentiel, surface arable |

Le viewer montre quatre vues par saison (déc.–fév., mars–mai, juin–août, sept.–nov., en mois calendaires nord), comme les vues « Seasonal climate ». Les saisons de l'hémisphère sud sont inversées dans les valeurs.

## 7. Sel, tourbe, fer des marais, matériaux

### 7.1 Sel

Le sel a trois sources :

- **Sel gemme** : gisements évaporitiques (§5.2).
- **Sel marin solaire** (flux mensuel) : `aire de marais salant × 0,05 t/m²` par mois où T ≥ 15 °C et P < ETP. L'aire de marais salant est une côte basse et peu accidentée ; tout est provisoire.
- **Sources salées** (proposé) : flux attaché à un gisement évaporitique profond près d'une faille.

Les méthodes médiévales attestées sont le sel gemme de Wieliczka-Bochnia, exploité depuis le XIIIᵉ siècle (UNESCO 32), et la saumure bouillie de Droitwich (source secondaire). Le marais salant est l'usage côtier courant.

### 7.2 Tourbe et fer des marais

- **Tourbe** : climat froid à tempéré, humide (P/ETP > 0,8–1,2), basses terres, pédologie organique. Le volume vaut `aire × 1,5 m` (provisoire). La régénération est quasi nulle à l'échelle historique.
- **Fer des marais** : zones humides × roches riches en fer en amont (basalte, granite, ultrabasique), 0,4 t/km²/an (provisoire). C'est un flux qui se renouvelle.

### 7.3 Matériaux

Ce sont des potentiels de 0 à 1 :

- **pierre de taille** : 0,9 calcaire, 0,8 grès, 0,7 granite, 0,5 métamorphique, 0,4 basalte ;
- **chaux** : calcaire ;
- **ardoise** : 0,6 × métamorphique ;
- **argile** : marnes et pédologie argileuse ;
- **sables et graviers** : grès, sols sableux, proximité de la côte.

Les stocks ne sont pas limitants au rythme médiéval. Seuls l'**accès**, la **qualité** et le **transport** comptent. Ils se décident au LOD 0, dans la tuile (§11).

**Placers (proposé).** La version actuelle lisse les sources d'or et d'étain sur environ 60 km. La cible suit le modèle de `surface_context.route_context` : transporter la masse érodée des gisements le long du graphe receveur de l'hydrologie à 240 m, au sein des unités de propriété des tronçons.

## 8. Apport de Render v2

Render v2 (`backend/terrain_render_v2.py`, miroir NumPy du WGSL) fournit un **évaluateur de milieu** sans réglage d'affichage. Il dépend de la position, de la graine, du relief, des températures et pluies des deux saisons, du sable, de la berge et de la date. Il est sans état et déterministe. Les ressources l'appellent au centre de chaque cellule du LOD 10, avec une empreinte égale à la taille de cellule. Les mosaïques de 2,4 km et 670 m y sont filtrées ; seuls les termes de climat, de relief et de province restent.

| Champ ou layer Render v2 | Ce qu'il fournit | Usage ressources | Remplace / complète | Limite |
|---|---|---|---|---|
| `woody` | potentiel ligneux naturel ≤ 0,985 (humidité potentielle, porte thermique, peuplements) | **couvert ligneux potentiel** (× (1 − neige pérenne)) | **remplace** l'ancien couvert calculé par ETR/ETP (doublon supprimé) | proxy d'apparence. À l'échelle de 30 km, la pente est faible et la part rocheuse quasi nulle. Le rendu final mélange aussi le couvert historique (`surface_material`) et le curseur forêt, que les ressources ignorent volontairement |
| `height` | hauteur de canopée de 2 à 32 m | part bois d'œuvre du stock | **complète** le bois | proxy, sans dendrométrie |
| `needle` | part résineuse | part résineuse du bois | **complète** | proxy |
| `leaf(date)` | feuillaison | mois d'abattage | **remplace** la règle « T < 5 °C » | phénologie d'apparence |
| `grass(date)` | tapis herbacé, déjà réduit par la neige | part herbacée mensuelle de la pâture | **remplace** « 1 − 0,8·arbres » | proxy d'apparence |
| `snow(date)` | couverture neigeuse continue (froid, apport, rétention, pente) | débardage sur neige ; pâture enneigée ; vue saisonnière | **complète** le stock de neige (mm) du bilan hydrique, qui reste la grandeur physique | pas d'accumulation historique |
| `perennial_snow` | neige pérenne | exclusion des terres arables, du bois et de la pâture | **complète** | heuristique d'apparence |
| `ice(date)` | potentiel de gel (`smooth(−1,8, −8, T)`) | accès réduit à la pêche côtière | **complète** | pas de banquise physique |
| `moisture`, `potential` | humidité locale et potentielle (indice de type Köppen) | non utilisés : le bilan hydrique partagé fait foi | — | — |
| `mineral`, `tone` | **teinte** de province de 180 et 690 km (noir, brun, beige, rouge) | **non utilisé** : ce n'est pas une lithologie | à terme, alimenter cette teinte par la lithologie ressources (§12) | couleur seulement |
| `sand`, `clay`, `organic`, `rock` | proxys d'apparence bornés, la roche venant de la pente | non utilisés : la pédologie et la lithologie font foi | — | dépendent de la pente locale |
| Provenance glaciaire côtière (`terrain_glacier.py`) | proximité d'une source glaciaire plausible dans un rayon de 80 km | proposé : exclure les ressources sous glace pérenne côtière | complète la neige pérenne | heuristique documentée |
| Sols et pédologie (`terrain_soil.py`, `terrain_pedology.py`, inchangés dans le merge) | 7 mélanges pédologiques ; sable, argile et humus | facteur de sol des céréales ; tourbe (organique) ; argile et sable | entrée directe | carte d'apparence régionale (environ 40 km), pas un relevé |
| Diagnostics `render-v2-substrate`, `-moisture`, `-vegetation`, `-snow` | mêmes valeurs que l'évaluateur | lecture visuelle croisée | — | — |
| Calendrier (curseur Saison, `date ∈ [0,1)`) | une seule date pilote la phase saisonnière | mois `m` ↔ `date = (m + 0,5)/12` | même convention | les hémisphères suivent leurs propres températures |

Les audits Render v2 (sur 40 km, sur sols ouverts réels, sur le détail du sol, sur l'hydrologie et sur les pôles) traitent la qualité visuelle. Ils n'apportent pas de nouvelle grandeur physique. On en retient les règles applicables à la réinjection (§12) :

- une couleur moyenne stable entre les zooms ;
- pas de bruit seuillé ;
- un sol détaillé sous la végétation ;
- la distinction entre teinte de province et lithologie.

**Règle de séparation.** Chaque champ Render v2 utilisé est stocké avec le préfixe `v2_` et marqué « proxy » dans les légendes du viewer. Aucune ressource ne lit `parse_settings` ni le curseur de forêt. Un changement d'apparence ne change donc jamais un stock.

## 9. Contrat d'interface neutre

Ce contrat doit pouvoir servir de module commun quel que soit le dépôt hôte. JSON, unités SI ou nommées, entiers pour les quantités engagées.

```text
ResourceLayer {
  id: "wood_stock_m3"                  # identifiant ASCII stable
  label_fr, unit: "m3", quantum: 1      # entier exporté = round(valeur / quantum)
  nature: potential | stock | flow | calendar
  regime: exhaustible | regenerable | annual | quasi_fossil
  aggregation: sum | area_mean | none
  decided_at_lod: 10 ; refined_to_lod: 3 | 0
  seasonal: none | monthly[12]
  status: derived | synthetic | imported | unavailable ; proxy_of?: "render_v2.woody"
}
CellValue { cell_id (u64), lod, value (int si engagé) }
Deposit {
  id: "<famille>:<owner_cell hex>:<k>", family, owner_cell (LOD 10),
  position_cell, position_lod (3), ore_t, grade_factor, depth_top_m,
  contained {métal: t|kg}, medieval_accessible_share, medieval_recoverable {métal: t|kg},
  outcrop, gossan, status
}
RegionSummary { region_id: { natural: {layer: {value, quantum, unit}}, deposits: [id…] } }
HistoricalDelta (Chronos) {
  date, region_or_cell, kind: extraction | clearing | felling | abandonment | regrowth,
  deposit_id?, metal?, amount (entier, même quantum), cumulative
}
Provenance { source_id, status, fingerprint (sha256 du code + identité de l'atlas) }
```

- **Implémenté** : `region_summary`, les dictionnaires `Deposit`, `UNITS`, `QUANTUM`, `EXTENSIVE`, le cache identifié par sha256 du code et de l'atlas.
- **Proposé** : sérialisation versionnée et schéma JSON formel, fichiers d'échange Chronos, empreinte intégrée au `canon_id` de Chronos.

## 10. Couplage avec Chronos

Chronos lit les **agrégats régionaux**. Une région est une union de cellules du LOD 10 ; ses totaux sont des sommes d'entiers. Chronos les modifie par des **deltas datés** et ne réécrit jamais l'état naturel.

Correspondance avec les indices actuels de Chronos (`food`, `water`, `timber`, `iron`, `stone`, `salt`) :

| Indice Chronos | Remplacé par |
|---|---|
| `food` | `cereal_net_t_year` + `lsu_year_km2` (cheptel) + `game_harvest_kg_year` + pêche, en t/an d'équivalent grain à définir |
| `water` | `water_yield_m3` + accès aux rivières (hydrologie) |
| `timber` | `wood_stock_m3`, `wood_flux_m3_year`, `timber_share` |
| `iron` | `recoverable_iron` + `bog_iron_t_year` |
| `stone` | `stone_ashlar`, `lime`, `slate` |
| `salt` | `recoverable_salt` + `sea_salt_month_t` |

**Extraction et épuisement.** Pour un gisement : `restant = contenu − extrait_cumulé`, borné à 0 (`deposit_remaining`). Le coût par tonne de minerai suit `c(E) = c0 · (1 + E / accessible)²` (`extraction_cost`, proposé) : on exploite d'abord le minerai superficiel et riche.

Exemple de lecture, avec des valeurs provisoires :

1. Un filon polymétallique de 200 000 t à Ag 300 g/t contient 60 t d'argent.
2. Avec 60 % accessible et un rendement de fonte de 0,7, le récupérable médiéval est de 25,2 t.
3. 300 ans à 50 kg/an prélèvent 15 t, soit 60 % du récupérable.
4. Le coût unitaire a alors été multiplié par (1 + 0,6)² = 2,56.

Chronos peut déclencher l'abandon de la mine quand ce coût dépasse la valeur locale de l'argent.

**Forêt.** La règle de stock de Chronos (§5.4 de sa conception) devient `S(t+1) = max(0, S + r·S·(1 − S/K) − E)`, avec `K` le stock initial. On choisit `r = 4·accroissement/K`, si bien que le rendement maximal durable `rK/4` égale l'accroissement. Par exemple, avec un accroissement de 4,5 m³/ha/an et K = 270 m³/ha, r ≈ 0,067/an. Un prélèvement supérieur à cet accroissement fait décroître le stock, puis s'effondrer la régénération (`forest_stock`).

**Défrichement cohérent à toute date.** Le défrichement cumulé d'une cellule parente se répartit suivant un **ordre canonique** de ses enfants : aptitude décroissante, ajustée par une clé adressée, puis identifiant (`clearing_order`). La surface défrichée à la date t est un **préfixe** de cet ordre (`cleared_cells`). Elle est donc monotone dans le temps, exacte en surface, et identique quelle que soit l'échelle qui la lit (testé). L'abandon suit l'ordre inverse d'un second ordre canonique (proposé).

**Ce que Chronos ne fait pas.** Il ne change ni le relief, ni le climat, ni la lithologie (conception Chronos §5.4). Un stock historique ne régénère pas un gisement lorsqu'une tuile est rechargée.

## 11. Intégration Neural Earth – Chronos – Magna Urbis

| Niveau | Produit par | Consommé par | Moment |
|---|---|---|---|
| Atlas Orogen (≈ 19,5 km) | Neural Earth (Orogen) | ressources monde | génération du monde |
| LOD 10, autorité ressources | Neural Earth (`compile_world`) | viewer, Chronos (bootstrap), descentes | une fois par monde, cache disque |
| Régions (unions de cellules du LOD 10) | Chronos définit les régions ; Neural Earth fournit les sommes (`region_summary`) | Chronos (pressions, stocks, flux, droits) | bootstrap, puis à chaque bloc de temps |
| Deltas historiques | **Chronos** (extraction, défrichement, coupe, abandon) | Neural Earth (lecture naturel + delta), Magna Urbis | par bloc canonique (décennie ou année) |
| LOD 3 (240 m) : positions de gisements, budgets aréaux | Neural Earth (`descend_budget`, `_place_in_cell`) | tuiles, Chronos en raffinement local (vallée, site) | à la demande, par parent |
| LOD 0, tuile 40 km / 30 m | Neural Earth, raffinement de tuile (aujourd'hui `surface_context.py` dans Magna Urbis) | Magna Urbis (établissements, carrières, glaisières, bois, eau, terres), textures | à la demande par tuile |
| LOD ≤ 0 : textures et apparence | Neural Earth (Render v2) et le moteur de textures Godot | affichage | rendu |
| Cités et établissements | Magna Urbis | Chronos (sites), affichage | à la demande |

**Lecture « naturel + delta »** : une tuile datée lit les deltas de Chronos agrégés à son parent, puis les répartit par les mêmes règles exactes (ordre canonique, LOD 3 puis LOD 0). Une forêt coupée en 1350 reste coupée dans toute vue de 1400.

## 12. Réinjection dans le choix des textures

Doctrine de `GROUND_MATERIAL_MAPPING.md` : « les règles fixent le possible, le hasard stable choisit » ; aucune invention sans information. La couche ressources fournit des **informations déclarées** qui manquaient (O1, O3, O5). Elle ne fabrique jamais une texture spéciale sans source.

| Information ressources | Code | Effet autorisé sur le sol visible | Ce qui ne change pas |
|---|---|---|---|
| Lithologie à 8 classes (métamorphique, ultrabasique, évaporites) | O1 | variantes de roche (schiste feuilleté, serpentinite), parents de R13–R20, teinte de la province minérale Render v2 alimentée par les fractions réelles | la proportion de roche exposée (N1) |
| Gisement affleurant sulfuré avec chapeau de fer | O1 | emprise elliptique orientée par `structureAzimuth`, de taille proportionnelle au tonnage ; teinte d'oxyde (type **B11**, M13) | pas de teinte rouge hors emprise ; aucun gisement non affleurant visible |
| Évaporites affleurantes et bassin fermé aride ; marais salant déclaré | O1/O4 | croûte saline M15 et gypse M16 : **nouveaux types** à ajouter au catalogue de 90 types, qui n'en a pas | l'aridité seule ne suffit pas |
| Tourbe (part de tourbière, stock) | O3 | M10 : **H02** et B02 assombris, nouveau type « tourbe exposée » proposé | un TWI élevé seul ne suffit pas |
| Volcanisme de point chaud ou d'arc, épithermal | O5 | altération hydrothermale près des gisements épithermaux (M20), qualifiée | pas de soufre ni de ponce sans déclaration |
| Delta Chronos : carrière, halde, mine | O9 | G01/G06 et R* frais dans l'emprise déclarée | jamais sans événement daté |
| Défrichement, culture | — | Render v2 lit le couvert « naturel + delta » | l'agriculture reste exclue de Render v2 (sa spécification) |

**Points d'insertion.**

- **Magna Urbis** (`godot-render/tools/ground_surface.py::scores_for_chunk`) :
  - groupe B : B11 reçoit `oxyde × chaud`, B02 et H02 reçoivent `tourbe` ;
  - groupe R : les poids des parents viennent des fractions conditionnées ;
  - aucun nouveau tirage : la palette reste tirée une fois par type (`ground_textures.gd`).
- **Neural Earth** (Render v2) : un champ `mineral` alimenté par la lithologie (proposé) ; un masque d'emprises de gisement à 240 m, filtré par l'empreinte du pixel.
- **Non implémenté.** Ces modifications touchent les shaders et le catalogue ; elles demandent une validation visuelle sur les tuiles réelles, selon le protocole Render v2.

## 13. Extension fantasy (implémenté, désactivée par défaut)

- **Activation** : `compile_world(..., fantasy=True)`, ou `TERRAIN_RESOURCES_FANTASY=1` pour le viewer.
- **Contenu** : une famille de démonstration, « métal stellaire », dans des contextes de point chaud et de collision.
- **Isolation** : domaine de clés distinct (`resources/fantasy/...`) et statut `synthetic`. Le test `test_fantasy_is_isolated` vérifie qu'activer la fantasy ne change aucun octet des autres couches ni des autres gisements.
- Le catalogue `NeuralEarth-Resources-Fantasy.md` reste la référence pour l'étendre.

## 14. Layers informatives du viewer (implémenté)

121 vues `orogen-resources-*`, plus 4 vues fantasy si elle est activée, en six groupes du menu : géologie, gisements, agriculture et élevage, forêt-faune-pêche-eau, matériaux-sel-tourbe, saisons.

Elles empruntent le chemin des diagnostics Orogen : tuiles PNG, carte et globe, activation par `orogen_layers`.

- **Légende** (`web/terrain_resources.js`) : rampe ou classes, unité, échelle logarithmique éventuelle, statut, mention « proxy ».
- **Inspecteur** : un clic affiche les valeurs de la cellule du LOD 10 et les gisements à moins de 60 km, via `/api/resources/<seed>/inspect`.
- **Affichage** : aux LOD ≥ 10, valeur des cellules d'autorité. Plus fin, jusqu'au LOD −6 : descente avec bruit et conservation (§16). La vue monde (aperçu 2 048 × 1 024) reste bilinéaire entre centres du LOD 10.

`terrain_orogen_layers.py` n'est pas modifié, car il fait partie de l'identité de l'atlas Orogen. L'enregistrement se fait dans `terrain_server.py`. **Ce fichier, `index.html` et `terrain_toolbar.js` font partie de l'identité d'implémentation de `terrain_manifest.py` : leur modification invalide les caches de tuiles neuronales**, comme toute modification de ces fichiers. Voir `docs/layers.md`.

## 15. Mesures et validation

Machine locale de Gregory, Windows 11, RTX 3090. Les ressources n'utilisent que le CPU.

| Mesure | Valeur | Commande |
|---|---|---|
| Compilation monde LOD 10 (652 × 1 304 cellules), monde de référence | 63–65 s, dont environ 34 s pour les 12 évaluations Render v2 | script de mesure local avec `compile_world`, et premier affichage viewer |
| Rechargement depuis le cache disque | 1,0 s | `world_resources` |
| Rendu d'une vue monde 1 024 × 512 | médiane 0,17 s, maximum 0,74 s sur 121 vues | idem |
| Tuile de gisements (304² échantillons) | 0,03 s | idem |
| Inspecteur | 0,7 s | idem |
| Tests | 18 tests OK en environ 57 s (dont 4 pour la descente) | `python -m unittest tests/python/test_terrain_resources.py` |
| Viewer, première compilation dans le serveur (requêtes de tuiles concurrentes) | 157 s | serveur local, port 8775, `TERRAIN_PRELOAD_MODELS=0` |
| Viewer, premier affichage depuis le cache disque | 11 s jusqu'aux 24 tuiles prêtes | Edge headless via Playwright |
| Viewer, changement de vue (24 tuiles, 1 280 × 760, carte) | médiane 2,4 s, maximum 6,5 s (gisements) | idem, avec captures d'écran |

**Vérification du viewer (mesurée).**

- Les 121 vues ont été parcourues dans Edge headless. Chacune a ses tuiles servies sans erreur et affiche sa légende.
- Menu : six groupes de 20, 39, 11, 14, 9 et 28 boutons.
- La carte, le globe (contexte tectonique, gisements) et l'inspecteur au clic ont été vérifiés.
- Captures : `docs/images/resources-forage-jja.png` (pâture juin–août, avec légende) et `docs/images/resources-deposits-lod6.png` (gisements au LOD 6).
- L'appréciation visuelle de la lisibilité reste à faire par Gregory.

Ce que vérifient les tests :

- vecteurs de clés identiques à Chronos ;
- aires sphériques fermées et emboîtées ;
- identifiants et parents ;
- `apportion` exact et indépendant de l'ordre ;
- déterminisme à l'octet ;
- bornes et partitions ;
- gisements cohérents avec la lithologie et le contexte, positions descendantes de la cellule propriétaire ;
- stocks égaux à la somme des gisements ;
- descente exacte et indépendante de l'ordre ;
- agrégats régionaux exacts ;
- isolation de la fantasy ;
- delta historique monotone et exact ;
- champs Render v2 consommés sans recalcul ;
- descente : moyennes des enfants égales au parent entre LOD affichés (continu, borné, mélange), déterminisme et indépendance à l'ordre, absence de saut entre tuiles voisines, bruit indexé par la position et non par la tuile.

Les tests utilisent un atlas synthétique de 64 × 128 au LOD 14, où la descente est bornée à 2^7 enfants par axe (`placement_lod`).

## 16. Descente d'échelle avec bruit et conservation (implémenté)

**Problème corrigé.** La première version affichait à tout zoom une interpolation bilinéaire de la grille du LOD 10 (30,72 km) : de gros pixels, aucun détail. C'était contraire au principe de Neural Earth, qui produit des bruits et des détails procéduraux à toutes les échelles. Désormais, chaque champ est une fonction évaluable à tout LOD affiché et cohérente entre LOD (`backend/terrain_resource_detail.py`).

**Définition.** La valeur est calculée sur la grille canonique des tuiles d'affichage : 256 échantillons, un halo de 24, pas de `30·2^L` m. Au LOD `L`, elle dérive de la valeur au LOD parent `P = min(10, L+3)`. Chaque cellule parente couvre 8 × 8 enfants.

```text
brut(x)   = P̃(x) · exp( m(x) + σ · Σ_o a_o · n_o(x) )
valeur(x) = brut(x) · (1 + ε_b · φ_b(x))      # par bloc b = cellule parente
```

- **`P̃` (surface parente).** C'est l'interpolation bilinéaire du réseau parent, après une itération pycnophylactique `P* = P + (P − S·P)`. Ici S est la moyenne de cellule de la surface bilinéaire, soit le noyau [1, 6, 1]/8 par axe. La surface obtenue a déjà des moyennes de cellule proches du parent (Tobler 1979).
- **`m(x)` (modulation locale).** Elle ne lit que des entrées canoniques du LOD L :
  - le DEM neuronal de la tuile et ses dérivées : pente, position en fond de vallée ou sur crête, anomalie d'altitude, exposition nord ;
  - la proximité des rivières sources, lue dans le service hydrologique avec des options fixes (mois annuel, sans détail ni cosmétique).

  Les coefficients sont fixés par profil (`PROFILES`, calibration proposée) : arable, pâture, forêt, humide, rocheux, bas-fond, froid, saison, écoulement, minéral, neutre, marin. La variation à l'échelle du parent appartient au parent : un passe-haut (σ = 2,5 échantillons) la retire.
- **Bruit.**
  - Octaves `o = L+1 … P` de bruit de valeur quintique, adressé par `(seed, couche, LOD de l'octave, cellule du réseau)` avec la clé de Chronos.
  - La longitude boucle. Amplitude `2^{0,6(o−P)}`, octave de Nyquist amortie à 0,25.
  - Les octaves plus grossières vivent dans les parents : chaque LOD est le passe-bas des LOD plus fins.
- **Conservation.**
  - Première correction multiplicative par bloc, lissée entre centres de blocs (bilinéaire).
  - Correction finale **exacte et continue aux bords de blocs** : `1 + ε_b·φ_b`, où φ est une bosse nulle au bord et de moyenne 1, `(π/2)² sin(πu) sin(πv)`. Si `ε_b < −0,4`, on revient au rapport simple, qui reste exact.
  - Indices bornés (fractions, aptitudes, prospectivités, mois) : même schéma, par décalage en logit résolu par bisection.
  - Mélanges (roches, contextes) : ajustement proportionnel itératif par classe, avec la même bosse.
  - Résultat :
    - la moyenne des 64 enfants, pondérée par l'aire (cos φ), égale la valeur du parent ;
    - pour les champs continus et bornés, l'égalité est exacte à l'arrondi float32 du stockage près ;
    - pour les mélanges, l'écart est d'environ 10⁻³, et la somme des fractions reste à 10⁻³ de 1.
- **Absence de couture entre tuiles.**
  - Les valeurs parentes sont toujours lues dans l'intérieur de la tuile qui les possède, jamais dans un halo.
  - Tous les noyaux tiennent dans le halo de 24 échantillons, avec une couronne de blocs de 8 : gradient, Gaussiennes de rayon 3 et 7,5, rivières jusqu'à 8 échantillons, passe-haut.
  - Les données de chaque bloc sont donc identiques dans toutes les tuiles qui le voient.
- **Zéros canoniques.** Un parent nul impose des enfants nuls, par non-négativité et conservation. Une limite nette subsiste donc là où le LOD 10 vaut exactement 0. Pour cette raison, le fond de la vue « gisements » est une prospectivité bilinéaire non conservée, signalée comme telle.
- **Extrapolation côtière.** Dans les cellules marines du LOD 10, les couches terrestres prennent la valeur de la cellule terrestre la plus proche ; la pêche côtière fait l'inverse. Une côte fine située dans une cellule d'autorité marine a ainsi une valeur parente. La moyenne de bloc porte sur les échantillons terrestres du DEM fin.

**Couches particulières.**

| Couche | Traitement |
|---|---|
| Roches | Mélanges continus avec : contacts déformés par un warp multi-octave ; strates (calcaire, grès, marnes, évaporites, métamorphique) en bandes parallèles à l'azimut structural, lui-même parallèle aux ceintures orogéniques (gradient des contextes collision, arc, rift et arrière-arc) ; intrusions et coulées en taches ; roches dures sur crêtes et pentes, roches tendres en fonds |
| Stocks contenus et récupérables | Somme de noyaux gaussiens normalisés, un par gisement, de largeur max(emprise, 0,75 échantillon) : l'intégrale égale la masse du gisement à tout LOD |
| Gisements | Position au LOD 0 tirée par clé dans la cellule du LOD 3 ; formes par famille (ci-dessous) ; halo de prospectivité ; anneau de chapeau de fer pour un sulfuré affleurant ; sous 1,5 pixel, symbole de taille écran constante |
| Mois de récolte (cyclique) | Cellule la plus proche après une déformation de domaine adressée, pour éviter la grille de 30 km ; pas de conservation revendiquée |

Formes des gisements par famille :
- **filons** orientés selon la structure, longueur ∝ minerai^0,45 ;
- **amas** ellipsoïdaux de 40 à 300 m d'épaisseur ;
- **lentilles stratiformes** d'allongement 5:1 ;
- **placers** sur les rivières sources ou, à défaut, sur les fonds de vallée plats (pente < 4°), sur 0,1 à 2 km.

**Réalisme mesuré.** Monde de référence, DEM neuronal réel, point (−1 900 km, −3 800 km), script local `realism.py`.

| Contrôle | Résultat |
|---|---|
| Conservation parent / moyenne des enfants (écart relatif max, avant stockage float32) | NPP : 2·10⁻¹⁵ (LOD 0 → 3 et 5 → 8) ; aptitude céréalière : 0 ; mélanges de roches : 5·10⁻⁴ à 3·10⁻³ |
| Ordres de grandeur (conservés par construction) | Au LOD 0 : NPP 319–611 g/m²/an (p5–p95) autour d'une moyenne héritée de 452 ; pierre de taille 0,46–0,69 |
| Couplage local au relief : corrélation de Spearman des variations à moins de 5 échantillons | pierre/pente : +0,46 au LOD 2, +0,41 au LOD 0 ; neige/anomalie d'altitude : +0,30 au LOD 2 |
| Corrélation brute sur toute la tuile | faible (\|ρ\| ≤ 0,2), car la variance vient surtout des LOD parents |
| Fractalité : pente spectrale β du log-champ, P(k) ∝ k^−β | 2,3 à 3,2 à tous les LOD testés (7, 4, 2, 0) : comportement auto-similaire de type mouvement brownien fractionnaire |

Contrôle de grille : rapport du gradient médian sur les bords des blocs de 8 au gradient médian ailleurs. Un rapport de 1 signifie qu'aucun bord de bloc ne ressort.

| LOD | Après la bosse continue | Avant |
|---|---|---|
| 7 | 1,5 | 1,8–1,9 |
| 4 | 0,43–0,49 | 1,1–1,2 |
| 2 | 0,49–0,61 | — |
| 0 | 0,47–0,82 | — |

Un rapport inférieur à 1 traduit un léger « coussin » au centre des blocs. Il ne se voit pas sur les captures, mais la mesure est donnée. Au LOD 7, le grain résiduel vient du DEM de 3,84 km et de l'octave de Nyquist.

**Captures avant/après.** `docs/images/resources-avant-apres-geo-rock.png`, `-mat-stone.png` et `-season-snow-djf.png`. En haut l'ancien affichage bilinéaire, en bas la descente, sur les mêmes tuiles aux LOD 7, 4, 2 et 0.

`docs/images/resources-gisements-zoom.png` montre les emprises d'un porphyre, d'un filon, d'un fer oolithique et d'un placer aux LOD 3, 1 et 0. Dans ce relief montagneux, le placer n'a que quelques pièges plats.

**Performance (mesurée, serveur local, une tuile de 256², en secondes).**

Avant : chaque tuile recalculait le champ sur le monde entier et reconstruisait les gisements en Python, soit 0,05 à 0,26 s par tuile, sans cache.

Après :

| Cas | Temps par tuile |
|---|---|
| Tuile en cache (PNG en mémoire, plus cache navigateur d'un an) | 0,015–0,03 s |
| Tuile fine d'un champ continu, entrées déjà présentes | 0,03–0,5 s |
| Mélange de roches | 0,3–1,5 s |
| Gisements | 0,05–0,6 s |
| Première tuile d'un emplacement | jusqu'à 10 s, dominées par la génération neuronale des tuiles physiques et la construction du réseau hydrologique, comme pour Render |

**Changement de vue dans le viewer** (Edge headless, 1 280 × 760, cinq couches à chacun des LOD 10, 7, 4, 2 et 0, temps jusqu'à la fin du chargement) :

| Mesure | Médiane | Étendue |
|---|---|---|
| Première visite (tuiles neuronales du disque, ressources à calculer) | 4,5 s | 1,0 à 44 s ; la valeur la plus haute correspond au premier passage au LOD 2, avec génération des tuiles physiques |
| Revisite (caches chauds) | 2,4 s | 1,0 à 8,7 s |
| Ancien affichage bilinéaire, vue monde seulement | 2,4 s | — |

Les annulations 409 lors des changements rapides sont celles du viewer, voulues.

Ce qui a changé :
- les champs d'autorité sont précalculés par couche (LRU de 24) ;
- les gisements sont des tableaux, filtrés vectoriellement par emprise ;
- l'intérieur des tuiles est stocké en float32, dans un budget de 768 Mo ;
- les tableaux mensuels intermédiaires sont retirés après compilation : le monde en mémoire passe de 1,27 Go à environ 0,3 Go.

## 17. Limites et décisions ouvertes

1. **Calibration.** Densités de gisements, σ des log-normales, semence, part assolée, pêche côtière, sel solaire, épaisseur de tourbe : provisoires. Les médianes USGS marquées « vérifiées » restent à relire sur le PDF.
2. **Résolution de l'autorité.** Le LOD 10 (30,72 km) est plus grossier que l'atlas (19,5 km). Le LOD 9 (15,36 km) suivrait mieux la source, pour environ 4 fois le coût (la limite de `monthly_water_balance` est d'environ 1 M cellules terrestres).
3. **Métrique** plane dans l'atlas pour les distances tectoniques ; monde de diamètre par défaut seulement.
4. **Hydrologie** : les placers et la pêche continentale devraient suivre les unités de propriété des tronçons (`multiscale-hydrology.md` §2), plutôt qu'un lissage et un proxy de plaine inondable.
5. **LOD 3** : les budgets entiers (export Chronos) sont répartis avec des intensités interpolées ; le relief latent à 240 m n'est branché que dans la descente d'affichage (§16), pas encore dans `descend_budget`.
10. **Descente** : coefficients des profils et σ du bruit proposés ; ils sont calibrés sur un seul point du monde de référence. Couplage local mesuré pour la pierre et la neige seulement. Le couplage aux rivières dépend du réseau source (pas des affluents de 240 m). Les LOD négatifs n'ajoutent que du bruit.
6. **Paléoclimat** du charbon et des évaporites : un champ adressé existe pour les évaporites seulement.
7. **Chronos** : la sérialisation des deltas et l'entrée de l'empreinte des ressources dans le `canon_id` restent à faire.
8. **Textures** : toutes les entrées O1, O3 et O5 sont proposées, sans implémentation dans les shaders.
9. **Viewer** : les modifications de `index.html`, `terrain_toolbar.js` et `terrain_server.py` changent l'identité d'implémentation des tuiles neuronales. Il faut décider si les layers ressources doivent passer par un module chargé sans entrer dans cette identité.

## 18. Références

Seules les références relues pendant cette étude sont listées. Les valeurs non relues sont signalées dans le texte.

- Cox, D.P. & Singer, D.A. (éds.), 1986. *Mineral deposit models*. USGS Bulletin 1693. doi:10.3133/b1693 (modèles descriptifs, annexe B grade-tonnage, log-normalité).
- Bliss, J.D. (éd.), 1992. *Developments in mineral deposit modeling*. USGS Bulletin 2004. doi:10.3133/b2004 (modèle 34f, fer oolithique).
- Orris, G.J. & Bliss, J.D. (éds.), 1991. *Some industrial mineral deposit models*. USGS OFR 91-11A. doi:10.3133/ofr9111A (35a.3 sel stratifié, 35a.5 gypse). OFR 92-437 pour les modèles grade-tonnage.
- USGS SIR 2010-5070, *Mineral deposit models for resource assessment*, chapitres A–R. doi:10.3133/sir20105070.
- Lieth, H., 1975. Modeling the primary productivity of the world. In *Primary Productivity of the Biosphere*, Springer. Constantes relues dans Adams, White & Lenton, 2004, *Ecological Modelling* 177:353–391. doi:10.1016/j.ecolmodel.2004.03.014.
- Damuth, J., 1981. Population density and body size in mammals. *Nature* 290:699–700. doi:10.1038/290699a0.
- Slicher van Bath, B.H., 1963. *Yield ratios, 810–1820*. A.A.G. Bijdragen 10, Wageningen (p. 16).
- Fischer, G. et al., 2021. *Global Agro-Ecological Zones v4 – Model documentation*. FAO. doi:10.4060/cb4744en. FAO ECOCROP et Hijmans, Recocrop.
- FAO, Plant Production and Protection Paper 162 (glossaire TLU de 250 kg). Meehan & Sedivec, NDSU Extension R1810.
- TCV, *Conservation Handbooks*, « General information about coppicing » (d'après Evans 1984). FAO FRA 2005, appendice 5 (densités basales GIEC).
- Welcomme, R.L., 1985. *River fisheries*. FAO Fish. Tech. Pap. 262. Crul, 1992, CIFA Occ. Pap. 16. Pauly, D. & Christensen, V., 1995. *Nature* 374:255–257.
- Loisel, J. et al., 2014. *The Holocene* 24(9):1028–1042. doi:10.1177/0959683614538073. Bricker, Newell & Simon, 2003, USGS OFR 2003-346 (fer des marais).
- UNESCO, liste du patrimoine mondial n° 32, mines de sel de Wieliczka et Bochnia.
- Tobler, W.R., 1979. Smooth pycnophylactic interpolation for geographical regions. *Journal of the American Statistical Association* 74(367):519–530. Principe de conservation des moyennes par région ; notre variante est discrète, en une itération.
- Dwarf Fortress wiki, « Stone layers », « Vein », « Ore » : couches et hôtes de minerais, à titre de comparaison ludique. Azgaar's Fantasy Map Generator, v1.124.0, « Goods » : placement non déterministe par biome et hauteur ; nous retenons l'adressage par clés.
- Projet : `docs/design/multiscale-hydrology.md`, `docs/design/render-v2*.fr.md`, `docs/methods/render-v2-glacier-provenance.md`, Chronos `conception_generateur_histoire_multiechelle_v3.md` §5.4, §10, §15, `docs/reviews/vision-entites-sociales.md`, Magna Urbis `docs/GROUND_MATERIAL_MAPPING.md` et `docs/GROUND_TEXTURE_MAPPING_V1.md`.
