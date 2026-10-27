# Ressources naturelles multi-échelles — où trouver les ressources

Date : 27 octobre 2026. Branche locale `resources-layers` de Neural Earth, rebasée sur `origin/main` `051dd1f`, qui contient le merge Render v2 `d6cbb78`. Destinataires : Neural Earth, qui produit, Chronos, qui consomme et porte le temps, et Magna Urbis, qui consomme localement et construit les cités.

## 0. Statut et lecture

**R3h, 27 octobre 2026 : la présentation courante est décrite au §20.** Une couche par ressource indique les gisements récupérables avec les techniques fixes du scénario. Les anciens écrans « Stock contenu » et « Récupérable médiéval » des §1–19 sont des descriptions historiques et des contrats internes, plus des choix du menu. Les potentiels/prospectivités sont regroupés en bas sous **Diagnostic** : intensité servant au placement, pas inventaire de ressources.


Ce document distingue trois statuts :

- **Implémenté** : code présent dans `backend/terrain_resources.py` (autorité), `backend/terrain_resource_layers.py` (couches, paquets GPU), `backend/terrain_resource_gpu.py` (miroir NumPy de l'évaluateur), `web/terrain_resources_gpu.js` (évaluateur WGSL), `web/terrain_resources.js`, et testé par `tests/python/test_terrain_resources.py`, `tests/browser/test_terrain_resources_gpu.cjs` et `tools/verification/verify_resources_gpu_parity.cjs`.
- **Mesuré** : valeur obtenue sur le monde Orogen en cache (seed `7985403534069300490`, style `continents`), avec la commande indiquée.
- **Proposé** : conception non encore codée, ou calibration provisoire.

**Mise à jour R3e, 27 octobre :** le monde initial Orogen a désormais un diamètre de **4 000 km** ; les mesures actuelles sont au §19. Les tableaux de performance des §15–16 sont conservés comme référence historique sur le monde de **12 732 km**, et ne décrivent pas le nouveau monde.

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
- **Bornes (implémenté)** : le réseau conserve son pas physique de 30,72 km et son origine. Ses dimensions, les aires rognées, la période en longitude, les distances tectoniques et les entrées Render v2 des ressources suivent les bornes de l’atlas. Ces bornes entrent dans les identités des caches de l’autorité et de l’environnement. Validation actuelle : sphère de 4 000 km ; monde plan non validé.

### 3.2 Hasard adressé (implémenté)

- `key(seed, parts)` est la fonction de `chronos-core/src/keys.rs` : FNV-1a sur les parties, séparateur `0xff`, puis splitmix64. Le test `test_keys_match_chronos_rust` compare cinq valeurs produites par la compilation de `keys.rs` avec rustc 1.99.
- **Par cellule** : `splitmix(key(seed, [domaine…, "lod<L>"]) XOR cell_id)`, puis `splitmix(… XOR k)` pour le k-ième tirage.
- Aucun état aléatoire n'est partagé et aucun `default_rng` n'est utilisé. Le résultat ne dépend ni de l'ordre de calcul, ni de la fenêtre demandée.
- Le bruit de valeur des provinces lithologiques est aussi adressé par cellule, au LOD 13 (245,76 km), puis interpolé. Un même point donne donc la même valeur, quelle que soit la grille qui l'échantillonne.

### 3.3 À quel LOD chaque fait est décidé

| LOD (pas) | Faits décidés | Origine de l'information | Calcul | Statut |
|---|---|---|---|---|
| Atlas Orogen (≈ 19,5 km) | Distances aux frontières, contextes tectoniques, lithologie à la résolution de l'atlas | `crust`, `boundaries`, `convergence`, `hotspot`, `backArc`, `uplift`, climat sur deux saisons | à la compilation monde | implémenté |
| **10 (30,72 km)** | **Autorité** : fractions de contexte et de roche, prospectivités, climat mensuel, bilan hydrique, NPP, céréales, pâture, bois, gibier, pêche, sel, tourbe, matériaux ; **nombre, tonnage et teneur des gisements** ; budgets entiers par cellule | atlas échantillonné, bilan hydrique partagé, évaluateur Render v2 | une fois par monde, dans un processus séparé, 27 à 41 s publiés par étapes, puis cache disque (§15.1) | implémenté |
| 11 et plus (≥ 61 km), régions Chronos | Sommes exactes des cellules du LOD 10 ; une région est une union de cellules du LOD 10 | agrégation | à la demande | implémenté (`aggregate`, `region_summary`) |
| 9 à 4 (15,36 km – 480 m) | Budgets entiers : sommes des enfants au LOD 3. **Champs affichés : évaluateur GPU par pixel, bruit adressé, zones, conservation des moyennes du parent à une tolérance mesurée (§16)** | agrégation ; relief des blocs natifs grossiers (LOD 4 à 7) ou des tuiles physiques | GPU, par pixel | implémenté |
| **3 (240 m)** | **Position des gisements** ; budgets aréaux répartis exactement depuis le LOD 10 | intensités interpolées du LOD 10, aire exacte des enfants, facteur canonique optionnel (relief latent à 240 m, proposé) | à la demande, par cellule parente | implémenté (`descend_budget`, `_place_in_cell`) |
| 2 à 0 (120–30 m) | Champs affichés modulés par le DEM neuronal et l'eau de la tuile (§16) ; emprises de gisements (filons, amas, lentilles, placers, halo, chapeau de fer) ; zones de roches à contacts organiques, strates le long de l'azimut structural | DEM neuronal, hydrologie | GPU, par pixel | implémenté (viewer) |
| 0 (30 m), tuile 40 km | Répartition exacte de chaque bloc de 240 m entre ses 64 cellules ; carrières, glaisières et tourbières locales ; géologie fine de `surface_context.py` conditionnée | champs de tuile (`surface_context.py`) | à la demande par tuile | proposé (note Magna Urbis) |
| −1 à −6 (15 m – 47 cm) | Champs affichés : même évaluateur ; contacts de zones et emprises de gisements continuent de s'affiner par le warp ; le relief amplifié de la tuile module les champs. Apparence : textures, teintes d'oxyde, croûtes de sel, tourbe visible | même chaîne ; aucun nouveau stock | GPU ; rendu | implémenté (champs), proposé (textures, §12) |

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
- **Réalisation (géologie réalisée, §16.3)** : la table et les évaporites donnent des parts *attendues* `E_k`. La géologie est réalisée par `classe(x) = argmax_k [log E_k + b_k + 1,6·N_groupe + 0,7·N_k]`, avec un bruit de province par groupe (plutonique, volcanique, métamorphique, sédimentaire ; 123 à 983 km) et un bruit par roche (61 à 491 km), allongés le long de l'azimut structural pour les ceintures. Les parts **canoniques** de chaque cellule du LOD 10 sont les parts d'aire de cette réalisation sur 4 × 4 échantillons ; les prospectivités et les gisements les utilisent, et l'affichage évalue la même fonction. Les biais `b_k` font suivre aux parts mondiales les parts attendues. Constantes proposées.

**Mesuré**, en part des terres (réalisé / attendu) : granite 21,2/21,3 %, grès 20,2/18,4, métamorphique 17,8/18,1, argilo-marneux 16,2/14,9, basalte 12,8/12,9, calcaire 10,7/12,6, ultrabasique 0,9/1,0, évaporites 0,1/0,7. 77 % des cellules terrestres portent une seule roche (avant la réalisation, la part dominante médiane était de 0,27, avec 5,3 classes effectives).

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
5. **Position** : choix pondéré entre les 128 × 128 enfants du LOD 3, par course exponentielle `argmin(−ln u / w)`. Les poids `w` sont la prospectivité interpolée, multipliée par l’intensité du district. Les centres et le routage des marques sont décrits au §19.2. Le choix est indépendant de l'ordre.

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

111 vues `resources-*`, plus 4 vues fantasy si elle est activée, en six groupes du menu : géologie (20), gisements (50), agriculture et élevage (11), forêt-faune-pêche-eau (14), matériaux-sel-tourbe (9), saisons (7). 16 vues sont des proxys Render v2, 7 sont saisonnières.

Elles sont calculées **sur le GPU**, comme Render v2, sur la carte et le globe (§16). Le serveur ne sert qu'un paquet par monde et par couche ; les tuiles PNG Python ne restent qu'en repli sans WebGPU et pour l'aperçu monde (bilinéaire, 2 048 × 1 024).

- **Légende** (`web/terrain_resources.js`) : rampe ou classes, unité, échelle logarithmique éventuelle, statut, mention « proxy », « détail calculé sur GPU » ; pour les mélanges, rappel de la tolérance de parts.
- **Saison** : curseur continu pour les 7 vues saisonnières ; le mois affiché suit le calendrier nord ; seul un tampon de paramètres est réécrit, sans nouveau paquet.
- **Inspecteur** : un clic affiche les valeurs de la cellule du LOD 10 et les gisements à moins de 60 km, via `/api/resources/<seed>/inspect`.

`terrain_orogen_layers.py` n'est pas modifié, car il fait partie de l'identité de l'atlas Orogen. L'enregistrement se fait dans `terrain_server.py`. **Ce fichier, `index.html` et `terrain_toolbar.js` font partie de l’identité d’implémentation de `terrain_manifest.py` : leur modification invalide les caches de tuiles neuronales** (`terrain_renderer.js` et les modules ressources n’en font pas partie). Voir `docs/layers.md`.

## 15. Mesures et validation historiques (12 732 km)

Machine locale de Gregory, Windows 11, RTX 3090. L'autorité, la réalisation des roches et les budgets sont calculés sur le CPU (NumPy) ; l'affichage dans le navigateur (WebGPU).

### 15.1 Calcul de l'autorité sans bloquer le viewer (implémenté)

**Problème corrigé (mesuré par le coordinateur).** La première requête `/api/resources/<seed>/gpu` attendait 221 s, synchrone dans la requête HTTP : `compile_world` tournait dans le thread de la requête, et le navigateur n'affichait rien. Le cache disque (767 Mo) était invalidé par toute modification du compilateur, car sa clé était le hachage du texte source.

**Ce qui est fait.**

- **Processus de compilation dédié** (`backend/terrain_resource_compile.py`). Il est lancé dès que le viewer ouvre un monde Orogen (`/api/resources/<seed>/status`), hors du GIL du serveur. Le serveur lui écrit les couches d'atlas utiles, puis suit son fichier d'état. Un plantage natif de la compilation ne tue plus le serveur.
- **Étapes publiées.** Après chaque étape, le processus réécrit un instantané des champs déjà calculés. Une couche est servie dès que les champs qu'elle lit existent ; sinon `/gpu` répond `202` avec l'étape, l'avancement et une estimation.

  | Étape | Contenu | Couches disponibles ensuite |
  |---|---|---|
  | `tectonics` | contextes, roches attendues, azimut structural, biais de la géologie réalisée, climat mensuel, bilan hydrique | roches, contextes, eau, écoulement |
  | `environment` | environnement Render v2, biologie | NPP, céréales, pâture, bois, faune, pêche, saisons |
  | `geology` | parts canoniques des roches, matériaux, fer des marais | fractions de roches, matériaux |
  | `deposits` | prospectivités, gisements, stocks | gisements, stocks |

- **Parallélisme déterministe** (`backend/terrain_resource_workers.py`). L'environnement Render v2 (6 tâches de 2 mois) et le placement des gisements tournent dans des processus de travail. Le résultat est identique au calcul en série (empreinte des champs égale, testé).
- **Calculs allégés, résultats inchangés** :
  - hachage du bruit en 6 appels au lieu de 8 par maille ;
  - orientations du bruit aligné évaluées seulement là où leur poids est positif ;
  - Render v2 évalué seulement sur les terres et le plateau (> −200 m), car les valeurs de mer profonde ne sont jamais lues ;
  - tirage de Poisson limité aux cellules d'intensité non nulle ;
  - préfixes de clés mémorisés.
- **Calculs allégés, résultats changés** (d'où `COMPILER_VERSION = resources-compiler-3`) :
  - parts canoniques des roches sur 2 × 2 échantillons (LOD 9) au lieu de 4 × 4 ;
  - biais calibré sur une cellule terrestre sur huit.
- **Cache** (`<runtime>/resources/resource-layers-v3`) :
  - npz compressé, sans les 18 champs de stocks, recalculés au chargement comme sommes exactes des gisements : 150 Mo, plus 3,4 Mo de gisements et 22 Mo d'environnement Render v2, au lieu de 767 Mo ;
  - chargement : environ 1,4 s ;
  - écriture : environ 10 s, après que le monde est servi.
- **Clé du cache** :
  - monde : `cache_key` de l'atlas, ou hachage du relief, **et bornes physiques de l’atlas** ;
  - graine et option fantasy ;
  - `terrain_resources.COMPILER_VERSION` ;
  - hachage de `terrain_render_v2.py`, une entrée externe.

  L'environnement Render v2 a son propre cache (`V2_INPUTS_VERSION`, même hachage). Après une modification du compilateur, il est réutilisé.
- **Règle de version.** Incrémenter `COMPILER_VERSION` dès qu'une sortie canonique de `compile_world` change (valeurs, champs, gisements, clés). Incrémenter `V2_INPUTS_VERSION` si les entrées données à Render v2 changent. Une modification d'affichage (couches, WGSL, miroir, couleurs) ne touche ni l'une ni l'autre.
- **Affichage pendant le calcul** (`web/terrain_resources.js`). La légende montre une barre, l'étape et une estimation, par exemple « Calcul des ressources du monde : environnement Render v2 et biologie (étape 2/4, ~12 s) ». La carte garde le relief sous la couche, et non un fond vide. Une erreur s'affiche (« Couche indisponible : … »). L'inspecteur indique le calcul en cours.

### 15.2 Temps jusqu'au premier pixel (mesuré)

Commande : `node tools/benchmarks/benchmark_resources_first_pixel.cjs [--spawn PYTHON] [--seed S] [--clear-cache DIR]`.

Le script démarre un serveur froid, ouvre le viewer sur le monde Orogen, puis choisit les 10 couches l'une après l'autre dans le menu. Il chronomètre le temps jusqu'à ce que le paquet soit lié au GPU et qu'une image soit dessinée. Dans tous les cas, le monde s'ouvre en 11,4 à 12,6 s.

| Couche (ordre du clic) | Monde neuf, graine 1234 | Même monde après modification du compilateur (environnement Render v2 en cache) | Monde en cache disque, serveur redémarré | Monde déjà chargé |
|---|---|---|---|---|
| NPP (premier clic, juste après l'ouverture) | 25,3 s | 12,3 s | 1,2 s | 0,5 s |
| Roches | 1,3 s | 1,0 s | 0,9 s | 0,8 s |
| Contextes | 1,1 s | 3,9 s | 1,0 s | 0,7 s |
| Aptitude céréalière | 0,4 s | 0,6 s | 0,5 s | 0,6 s |
| Pierre de taille | 7,0 s | 0,5 s | 0,7 s | 0,7 s |
| Neige | 1,5 s | 1,5 s | 1,4 s | 1,3 s |
| Stock de bois | 0,8 s | 0,4 s | 0,6 s | 0,3 s |
| Mois de récolte | 0,3 s | 0,3 s | 0,8 s | 0,2 s |
| Gisements | 10,8 s | 10,2 s | 0,9 s | 0,3 s |
| Stock de cuivre | 0,3 s | 0,5 s | 1,1 s | 0,3 s |

Étapes du processus de compilation pendant l'ouverture du monde (secondes depuis son lancement) :

| Cas | tectonique | environnement | géologie | gisements |
|---|---|---|---|---|
| Monde neuf | 12,1 | 23,2 | 31,1 | 41,1 |
| Après modification du compilateur | 11,1 | 14,3 | 21,7 | 31,0 |

Hors du serveur, sans le viewer, la compilation complète prend 27 s (tectonique 7 s, environnement 8 s, géologie 5,5 s, gisements 6,5 s), contre 206 s avant.

**Objectifs.**

- **Monde déjà chargé, moins de 2 s** : tenu pour les 10 couches (0,2 à 1,4 s).
- **Monde neuf après une modification de code, moins de 10 s** : **non tenu** au premier clic.
  - Les roches, les contextes et l'eau arrivent environ 12 s après l'ouverture du monde.
  - Les couches de l'environnement arrivent vers 23 s (12 s quand l'environnement Render v2 est en cache).
  - Les gisements arrivent vers 41 s.
  - Pendant l'attente, la légende montre l'étape et l'estimation, et la carte reste visible.
  - Le reste à gagner est surtout l'environnement Render v2 (environ 8 s hors charge, 11 s pendant la génération des tuiles) et le placement des gisements. Pistes : un portage GPU de l'évaluateur, ou une autorité plus grossière d'abord (proposé).

### 15.3 Plantage du serveur (exit 139)

- **Cause : non établie.** Le serveur précédent, sur le port 8775, est mort d'une violation d'accès sans trace Python. À ce moment, il exécutait :
  - le solveur de seuils PyTorch (CUDA) dans un thread du serveur, depuis supprimé ;
  - des calculs NumPy concurrents ;
  - la génération des tuiles neuronales.

  La mémoire validée de la machine était presque épuisée (serveur de Gregory, 19 Go, et le mien, 14 Go).
- **Ce qui est fait** :
  - plus aucun code ressources n'utilise PyTorch ni de thread de calcul dans le serveur ;
  - la compilation tourne dans un processus séparé, une par monde, et un plantage y reste confiné ;
  - `faulthandler` est activé, pour qu'un prochain plantage laisse une pile Python dans le journal du serveur.
- **Pas de nouveau plantage** pendant les mesures du §15.2 (5 serveurs froids, plus le serveur laissé sur 8775).

### 15.4 Autres mesures

| Mesure | Valeur | Commande |
|---|---|---|
| Paquets GPU | 1,7 Mo (continu), 20 Mo (saisonnier), 6,8 à 8,8 Mo (gisements, stocks), 17 à 20 Mo (roches, contextes) | `output/resources-gpu/packages/` |
| Tests Python des ressources | 22 tests OK en environ 70 s | `python -m unittest tests/python/test_terrain_resources.py` |

Ce que vérifient les tests Python :

- vecteurs de clés identiques à Chronos ;
- aires sphériques fermées et emboîtées ;
- identifiants et parents ;
- `apportion` exact et indépendant de l'ordre ;
- déterminisme à l'octet ;
- bornes et partitions ;
- gisements cohérents avec la lithologie et le contexte, positions descendantes de la cellule propriétaire ;
- stocks égaux à la somme des gisements ;
- descente des budgets exacte et indépendante de l'ordre ;
- compilation parallèle (processus de travail) identique à la compilation en série, étapes publiées dans l'ordre, environnement Render v2 en cache donnant le même monde ;
- agrégats régionaux exacts ;
- isolation de la fantasy ;
- delta historique monotone et exact ;
- champs Render v2 consommés sans recalcul ;
- miroir GPU (§16.5) : fixture de parité égale au miroir, bruit indexé par la position et non par la tuile, géologie affichée égale aux parts canoniques à tout LOD, roches en unités connexes, parts réalisées valides et déterministes, moyennes des champs continus.

Les tests de l'autorité utilisent un atlas synthétique de 64 × 128 ; ceux du miroir, un réseau synthétique de 32 × 20 cellules du LOD 10.

## 16. Affichage multi-échelle sur GPU (implémenté)

**Problème corrigé.** La version précédente (`terrain_resource_detail.py`, supprimé) descendait chaque champ sur le CPU et servait une tuile PNG Python par vue : 0,03 à 1,5 s par tuile, un grain uniforme et des mélanges de roches en confettis. L'affichage suit désormais l'architecture de Render v2 : un évaluateur WGSL calcule chaque pixel dans le navigateur, à partir du DEM, de ses Gaussiennes et de l'hydrologie déjà présents sur le GPU, et d'un **paquet** servi une fois par monde et par couche. Aucune tuile n'est calculée en Python pour une vue.

Les budgets canoniques ne changent pas : stocks, tonnages, budgets entiers et tout ce que Chronos consomme restent calculés exactement en NumPy (`terrain_resources.py`, §3.4, §10). L'affichage conserve les moyennes du parent **à une tolérance mesurée** (§16.3, §16.4).

### 16.1 Architecture

| Élément | Fichier | Rôle |
|---|---|---|
| Paquet | `terrain_resource_layers.py:gpu_package`, route `/api/resources/<seed>/gpu?mode=` | en-tête JSON, puis champs du LOD 10 en demi-flottants (1 couche ; 12 pour une couche saisonnière ; K + 2 pour les roches et les contextes : log-parts attendues biaisées ou fractions, puis cos 2θ et sin 2θ de l'azimut structural), gisements (12 mots chacun) triés par cellule du LOD 10, avec la table `(début, nombre)` de chaque cellule. Cache serveur de 6 paquets par monde, cache navigateur de 6 |
| Évaluateur WGSL | `web/terrain_resources_gpu.js` | cœur (hachage, bruit, champs, géologie réalisée, contexte dominant, gisements), puis deux entrées : la passe *shade* des tuiles physiques (LOD ≤ 3, et 8 à 10) et le fragment des blocs natifs grossiers (LOD 4 à 7, comme Render v2) |
| Miroir NumPy | `backend/terrain_resource_gpu.py` | mêmes opérations, float32 et hachages uint32 ; référence des tests de parité et de la réalisation canonique des roches |
| Géologie réalisée | `terrain_resources.realize_rocks` (canon) et `terrain_resource_gpu.realized_class` (même fonction) | parts canoniques des roches mesurées sur la fonction que le GPU affiche (§16.3) |
| Compilation | `terrain_resource_compile.py` (processus), `terrain_resource_workers.py` (tâches parallèles) | autorité calculée hors du serveur, publiée par étapes ; `/gpu` répond 202 tant qu'une couche attend (§15.1) |
| Renderer | `web/terrain_renderer.js` | liaisons 11 à 15 (champ, réservée, paramètres, gisements, seaux) dans la passe *shade* et dans le pipeline grossier ; un changement de paquet re-shade les tuiles physiques et relie les blocs grossiers |
| Légende, curseur de saison, inspecteur | `web/terrain_resources.js` | charge le paquet de la couche choisie ; le curseur réécrit seulement 704 octets de paramètres |
| Repli sans WebGPU | `terrain_server.py` | tuiles PNG bilinéaires du LOD 10, sans détail (ancien chemin) |

**Adressage.** Chaque pixel est un échantillon d'indice entier global `(gi, gj)` sur le réseau `30·2^L` m du LOD affiché, origine au centre de la carte. Sur une tuile physique, `L` est celui de la tuile ; sur un bloc grossier, `L` est un cran plus fin que l'empreinte du pixel à l'écran (`dpdx`) : le détail suit le zoom entre deux LOD de blocs, sans marches d'un pixel d'échantillon.

**Hasard adressé.** Bruit de valeur quintique sur le réseau de l'octave `o` (pas `30·2^o` m) : nœud `(i, j)` haché par `lowbias32(i·0x9e3779b1 ⊕ lowbias32(j + 0x85ebca77) ⊕ sel)`, sel = `lowbias32(seed32 + couche·0x632be5ab + (o + 64)·0x27d4eb2d + composante·0x165667b1)`, `seed32 = key(seed, ['resources', 'gpu']) mod 2^32` (clé de Chronos). La valeur ne dépend que de la position : identique dans toute tuile, tout ordre de requête et à tout LOD qui affiche cette octave. **Plancher d'octave** : un LOD n'affiche que les octaves de longueur d'onde ≥ 4 échantillons (`o ≥ L + 2`) ; l'octave de 2 échantillons créait un grain de pixel.

### 16.2 Champs continus

```text
v(x) = P*(x + w) · exp( t(x) + σ·n(x) − σ²/2 )        (champs positifs)
v(x) = borne(logit⁻¹( logit(P*) + t(x) + σ·n(x) ))     (fractions, indices, mois : décalage en logit entre les bornes physiques)
```

- **`P*`** : interpolation bilinéaire du LOD 10 après une itération pycnophylactique (`P* = 2P − S·P`, S = [1, 6, 1]/8 par axe, Tobler 1979), lue à une position déformée `x + w`. Le warp `w` a des octaves 8 à 10 (7,7 à 31 km), d'amplitude 0,25 longueur d'onde : les contours du LOD 10 deviennent organiques sans déplacer la masse de plus d'une demi-cellule.
- **`n`** : bruit fbm d'octaves 7 à 9 (3,8 à 15 km), normalisé, σ par profil (0,08 à 0,45). Il n'y a **pas de bruit plus fin** : sous 4 km, le détail vient du relief et de l'eau.
- **`t`** : termes de relief du DEM de la tuile, atténués jusqu'à 75 % entre les LOD 1 et 5 (la rugosité d'un texel grossier n'est pas un modelé et donnait un grain), mêmes définitions que Render v2 : pente passe-haut (pente fine − pente des Gaussiennes larges), fond de vallée et crête (TPI = (Gaussienne petite − grande)/(6 r), seuils ±0,10), exposition nord, anomalie d'altitude, eau (couverture fluviale de l'hydrologie, mip 2). Coefficients par profil (`PROFILES`, proposés) : arable, pâture, forêt, humide, rocheux, bas-fond, froid, saison, écoulement, minéral, neutre, marin. Exemple, arable : pente −1/6 par degré, vallée +1, crête −0,8, eau +0,3.
- **Couches saisonnières** : 12 couches mensuelles dans le paquet ; le GPU interpole linéairement entre les deux mois qui encadrent la date du curseur. Le curseur est continu ; seul le tampon de paramètres change.
- **Mois de récolte** (cyclique) : chaque échantillon prend le mois d'une des quatre cellules qui l'entourent, choisie avec la probabilité de son poids bilinéaire en comparant un bruit lisse uniformisé (octaves 8 à 10, quantiles tabulés) aux poids cumulés. Chaque cellule règne en moyenne sur une aire de cellule, en taches organiques, sans mélange de mois ni marche d'escalier de 30 km (la version « cellule la plus proche après warp » en dessinait).
- **Bas de l'échelle** : la couleur se fond dans le fond neutre sur les premiers 6 % de la plage, pour qu'un champ qui atteint 0 n'ait pas de bord net.

**Tolérance mesurée** (monde de référence, 240 cellules terrestres tirées au hasard, LOD 5 = 32 × 32 échantillons par cellule, sans termes de relief, script local `real_measure_cont.py`) :

| Couche | Σ affiché / Σ autorité | Rapport par cellule p10 – p90 | Écart absolu moyen / plage |
|---|---|---|---|
| NPP | 0,996 | 0,88 – 1,11 | 1,9 % |
| Aptitude céréalière | 0,998 | 0,93 – 1,06 | 1,1 % |
| Rendement céréalier | 0,986 | 0,90 – 1,09 | 3,1 % |
| Stock de bois | 1,004 | 0,88 – 1,20 | 1,5 % |
| Densité de gibier | 1,002 | 0,90 – 1,12 | 1,5 % |
| Part de tourbe | 1,004 | 0,94 – (cellules quasi nulles) | 2,6 % |

Les termes de relief s'ajoutent à cette tolérance : leur moyenne par cellule n'est pas nulle (une cellule de vallées est relevée). Ce biais n'est pas corrigé ni mesuré sur le DEM réel (limite, §17).

### 16.3 Roches : géologie réalisée ; contextes tectoniques

**Des provinces, pas des confettis.** Les essais successifs (seuils de quantiles par cellule sur des champs de zone, avec ou sans parts parentes « durcies ») donnaient des confettis : l'autorité mettait cinq à huit roches dans chaque cellule de 30 km, et des seuils qui imposent ces parts à l'intérieur de chaque cellule fabriquent des taches de quelques kilomètres. La solution retenue inverse l'ordre : **la géométrie des roches est définie d'abord, à toutes les échelles, et les parts canoniques du LOD 10 en sont mesurées.**

```text
classe(x) = argmax_k [ log E_k(x + w) + b_k + β₁·N_g(k)(x + w) + β₂·N_k(x + w) ]
```

- **`E_k`** : part *attendue* de la roche k, issue des champs Orogen (contextes tectoniques × table `LITHOLOGY`, évaporites selon l'aridité et la paléo-aridité, §5.1), au LOD 10, interpolée et lue à une position déformée.
- **`N_g`** : bruit de **province**, partagé par les roches d'un groupe : plutonique (granite), volcanique (basalte), métamorphique (métamorphique, ultrabasique), sédimentaire (calcaire, grès, marnes, évaporites). Octaves 12 à 15 (123 à 983 km), décroissance 1 : les grandes provinces dominent. β₁ = 1,6.
- **`N_k`** : bruit de la roche dans son groupe, octaves 11 à 14 (61 à 491 km), β₂ = 0,7.
- **Azimut structural** : les groupes et roches qui forment des ceintures (métamorphique, ultrabasique, roches sédimentaires) utilisent un bruit étiré ×2 le long de l'azimut structural θ (parallèle aux ceintures orogéniques : normale au gradient lissé des contextes collision, arc, rift, arrière-arc), par mélange de quatre orientations entières du réseau. Granite et basalte sont isotropes.
- **`w`** : warp d'octaves `L + 2` à 11 (0,15 λ) : les contacts deviennent organiques quand on zoome, l'intérieur d'une unité reste d'une seule roche.
- **`b_k`** : biais par roche, calibré (6 itérations sur un échantillon par cellule, une cellule terrestre sur huit) pour que les parts mondiales réalisées suivent les parts attendues.

**Canon et affichage, une seule fonction.** `terrain_resources.realize_rocks` évalue cette fonction (miroir NumPy) sur 2 × 2 échantillons du LOD 9 par cellule terrestre (4 × 4 au LOD 8 jusqu'à `resources-compiler-2` ; réduit pour le temps de calcul, §15.1) ; les parts obtenues sont les fractions `rock_*` canoniques, que les prospectivités et les gisements utilisent. Le GPU évalue la même fonction à chaque pixel, à tout LOD : l'affichage concorde avec le canon par construction, aux échantillons et aux warps fins près. Il n'y a plus de seuils à résoudre. Les parts attendues restent disponibles (`rock_expected_*`).

**Mesuré** (monde de référence ; script local `real_measure2.py`, 240 cellules terrestres tirées au hasard) :

| Contrôle | Résultat |
|---|---|
| Cellules terrestres d'une seule roche (2 × 2 échantillons) | 85 % ; part dominante p10 0,75 |
| Parts mondiales réalisées / attendues | granite 21,3/21,3 %, grès 19,7/18,4, métamorphique 17,8/18,1, marnes 16,8/14,9, basalte 12,8/12,9, calcaire 10,6/12,6, ultrabasique 0,9/1,0, évaporites 0,1/0,7 |
| Affichage au LOD 6 contre parts canoniques (TV par cellule) | moyenne 0,019, p90 0,067, max 0,41 (avec 4 × 4 échantillons : 0,008, 0,032, 0,11) |
| Affichage au LOD 4 | moyenne 0,020, p90 0,068, max 0,42 |
| Paires d'échantillons voisins de roches différentes, LOD 4 (480 m) | 0,37 % (5,2 % avec les seuils de quantiles) |
| Réalisation dans la compilation | 5,5 s (parts, en processus de travail) + 1,8 s (biais), §15.1 |

**Contextes tectoniques.** Leurs fractions sont des appartenances floues (noyaux de distance aux frontières de plaques, plancher océanique de 2 %), pas des parts d'aire. Rendues en zones, trois appartenances de 30 % donnaient un semis de taches de 10 km (essai mesuré et vu sur capture, écarté). Ils sont affichés comme **contexte dominant** de l'interpolation déformée, à tout LOD : ceintures continues, contacts organiques, aucune part revendiquée. Le contexte océanique est retiré des cellules terrestres pour cet affichage.

### 16.4 Gisements et stocks

- **Gisements** : position au LOD 0 tirée par clé dans la cellule du LOD 3 (inchangé). Forme par famille : filon orienté selon la structure (longueur ∝ minerai^0,45), amas ellipsoïdal, lentille stratiforme 5:1, placer dans les fonds de vallée. Halo de prospectivité, anneau de chapeau de fer pour un sulfuré affleurant ; sous 1,5 pixel, symbole de taille écran constante. Le GPU parcourt les seaux des cellules du LOD 10 à ±1 à ±3 cellules selon le zoom.
- **Stocks** contenus et récupérables : somme de noyaux compacts elliptiques normalisés par l’aire sphérique, un par gisement. Seule leur largeur change avec le LOD (§19.3). L’intégrale retrouve la masse à la tolérance de quadrature mesurée ; les budgets canoniques restent des sommes exactes d’entiers.

### 16.5 Parité CPU/WGSL et tests

- `tools/verification/build_resources_gpu_fixture.py` écrit `tests/fixtures/resources_gpu_parity.json` depuis le miroir : paquets synthétiques (continu positif, borné, saisonnier, géologie réalisée et contexte dominant, cyclique, gisements, stocks) et termes de relief.
- `tools/verification/verify_resources_gpu_parity.cjs` compile le cœur WGSL réel et compare (adaptateur WebGPU réel).
- `tests/browser/test_terrain_resources_gpu.cjs` charge le renderer réel, une tuile synthétique et quatre paquets.
- `tests/python/test_terrain_resources.py`, classe `ResourceGpuMirrorTests`.

**Mesuré** (RTX 3090, Chrome) :

| Contrôle | Résultat |
|---|---|
| Parité, 8 configurations, 114 cas | 0 désaccord ; erreur relative max 6·10⁻⁷ (continus, stocks), classes de la géologie réalisée et du contexte dominant et mois (sélection de cellule) identiques |
| Navigateur | mêmes pixels pour un même paquet ; la saison change l'image (écart moyen 8,3/255) ; zones cohérentes (0,7 % de paires de pixels voisins de couleurs différentes) ; quatre paquets distincts |
| Miroir | fixture = miroir ; bruit indexé par la position ; géologie affichée = parts canoniques à tout LOD (TV moyenne < 0,05) ; roches en unités connexes, pas en confettis ; parts réalisées valides et déterministes ; moyennes continues |

### 16.6 Performance comparée à Render v2

Même machine, même page, même renderer, horodatages GPU injectés autour du renderer non modifié (`tools/benchmarks/benchmark_resources_gpu.cjs`, paquets réels du monde de référence, 32 tuiles de 256² par mesure, médiane de 5).

| Mesure (RTX 3090, Chrome, médianes) | Render v2 | Ressources |
|---|---|---|
| Cuisson d'une tuile physique de 256² (passes flou + *shade*), temps GPU, à 30, 120 et 1 920 m | 0,86 – 0,97 ms | 0,11 – 0,14 ms (continus, saisons, mois, gisements, stocks) ; 0,15 – 0,16 ms (contextes) ; 0,20 – 0,26 ms (roches) |
| Même cuisson, temps mural par tuile (soumission + fin du travail) | 1,8 – 2,6 ms | 0,57 – 0,68 ms |
| Bloc natif grossier en plein écran 1 920 × 1 080, temps GPU | 0,72 ms | 0,76 – 1,02 ms ; 1,48 ms (contextes) ; 3,0 ms (roches) |
| Changement de couche (envoi du paquet au GPU) | — | 0,8 – 1 ms (continus) ; 4,6 – 5,6 ms (gisements) ; 10 – 10,5 ms (roches, contextes, saisons) |
| Taille du paquet (une fois par monde et couche) | — | 1,7 Mo (continus) ; 6,8 – 8,8 Mo (gisements) ; 17 Mo (roches) ; 20 Mo (contextes, saisons) |

Les ressources cuisent une tuile 3 à 8 fois plus vite que Render v2 : l'évaluateur ne fait qu'un champ par pixel, sans matériaux. Les roches coûtent le plus (8 bruits de roche et 4 bruits de province par pixel, dont certains orientés) : 3 ms par image plein écran sur bloc grossier, soit 4 fois Render v2, encore dans le budget d'une image à 60 Hz.

**Changement de vue dans le viewer** (même script de capture, 1 280 × 760, carte ; la durée inclut une attente fixe de 0,8 s) : sur une vue déjà chargée, changer de couche de ressources prend une médiane de 0,9 à 1,4 s selon le LOD, soit l'attente fixe plus le premier paquet ; le premier passage sur une couche de roches attend la compilation de l'autorité. Ces temps ne se comparent pas directement à Render v2 : le script atteint d'abord la vue en Render v2, ce qui inclut la génération des tuiles neuronales (7 à 120 s selon le LOD).

### 16.7 Captures et auto-évaluation

Captures : `docs/images/resources-gpu-<couche>.jpg`, Render v2 en haut, la couche en bas, sur les mêmes vues aux LOD 10, 7, 4, 2, 0 et −2 (monde de référence, point fixe de la carte, juillet sauf la neige en janvier). `resources-gpu-gisements.jpg` : une autre vue, centrée sur un porphyre cuprifère affleurant (−2 199 km, 8 662 km), aux LOD 9, 6, 3, 1, −1 et −3, avec les gisements, le stock de cuivre et les roches.

Auto-évaluation, capture par capture (zones lisibles ? artefacts ? lien au relief ?) :

| Couche | Ce qui fonctionne | Ce qui reste faible |
|---|---|---|
| Roches (`geo-rock`) | provinces de 100 à 400 km aux LOD 10 et 7, deux grandes unités et des rubans le long de la structure au LOD 4, contact organique au LOD 2, une seule unité aux LOD 0 et −2 : la taille des unités est cohérente entre les échelles ; aucun bord de cellule | au LOD 10, la carte du monde reste chargée (provinces de 3 à 10 pixels) ; la teinte ne suit pas le relief (pas de roche dure sur les crêtes) |
| Contextes (`geo-context`) | ceintures continues le long des marges et des sutures, contacts organiques, aucune tache | contexte dominant seulement ; la ceinture d'arc reste large et uniforme |
| NPP (`bio-npp`) | gradient régional lisible, relief bien intégré aux LOD 2 à −2 | au LOD 4, la capture montre un grain en petits blocs de 2 à 3 pixels qu'un rendu isolé du même bloc ne reproduit pas (cause non isolée, probablement l'échelle de présentation du viewer) ; marbrure du bruit de 4 à 15 km aux LOD 7 et 4 |
| Aptitude céréalière | masques nets mais sans marches de 30 km (fondu en bas d'échelle) ; vallées favorisées au LOD 2 | grandes plages neutres (montagnes), sans information à ces zooms |
| Pierre de taille | fort couplage au relief (crêtes claires, fonds sombres), lisible à tous les LOD fins | plages grises nettes au LOD 7 là où la roche n'a pas de pierre |
| Neige (janvier) | suit l'altitude et l'exposition, crêtes blanches, fonds gris | aux LOD 10 et 7, gros aplats ; limite pluie-neige un peu floue au LOD 4 |
| Stock de bois | massifs forestiers continus, lisière fondue | intérieur sombre et peu contrasté ; rampe trop foncée |
| Mois de récolte | taches organiques sans escalier de 30 km, mois non mélangés | couleurs cycliques criardes ; contacts courbes mais arbitraires (pas liés au relief) |
| Gisements, stock de cuivre | symbole visible au LOD 3, ellipse du corps à −3, halo du stock qui grossit avec le zoom | au LOD 1 et −1, symbole très petit ; le disque du stock a un bord net aux LOD −1 et −3 (troncature du noyau) |

Erreurs du viewer pendant les captures : délais d'attente de Render v2 (chargement neuronal), annulations 409 voulues, et quelques erreurs 500 du serveur sur l'aperçu monde quand la mémoire validée de la machine était presque épuisée (serveur de Gregory et le mien ensemble). Aucune erreur WebGPU ni de compilation de shader.

## 17. Limites et décisions ouvertes

1. **Calibration.** Densités de gisements, σ des log-normales, semence, part assolée, pêche côtière, sel solaire, épaisseur de tourbe : provisoires. Les médianes USGS marquées « vérifiées » restent à relire sur le PDF.
2. **Résolution de l'autorité.** Le LOD 10 (30,72 km) est plus grossier que l'atlas (19,5 km). Le LOD 9 (15,36 km) suivrait mieux la source, pour environ 4 fois le coût (la limite de `monthly_water_balance` est d'environ 1 M cellules terrestres).
3. **Métrique** plane dans l'atlas pour les distances tectoniques, avec le pas physique du monde courant. Sphère de 4 000 km validée au §19 ; monde plan non validé.
4. **Hydrologie** : les placers et la pêche continentale devraient suivre les unités de propriété des tronçons (`multiscale-hydrology.md` §2), plutôt qu'un lissage et un proxy de plaine inondable.
5. **LOD 3** : les budgets entiers (export Chronos) sont répartis avec des intensités interpolées ; le relief latent à 240 m ne module que l'affichage (§16), pas encore `descend_budget`.
6. **Paléoclimat** du charbon et des évaporites : un champ adressé existe pour les évaporites seulement.
7. **Chronos** : la sérialisation des deltas et l'entrée de l'empreinte des ressources dans le `canon_id` restent à faire.
8. **Textures** : toutes les entrées O1, O3 et O5 sont proposées, sans implémentation dans les shaders.
9. **Viewer** : les modifications de `index.html`, `terrain_toolbar.js` et `terrain_server.py` changent l'identité d'implémentation des tuiles neuronales. Il faut décider si les layers ressources doivent passer par un module chargé sans entrer dans cette identité.
10. **Affichage GPU** (§16) :
    - les termes de relief déplacent la moyenne d'une cellule (une cellule de vallées est relevée) ; ce biais n'est ni corrigé ni mesuré sur le DEM réel ;
    - profils de relief et σ du bruit proposés, réglés à l'œil sur un seul monde ;
    - roches : la réalisation est un bruit structuré par les contextes, pas une carte géologique ; les évaporites sont sous-représentées (0,1 % au lieu de 0,7 %), la calibration des biais ne converge pas pour une classe aussi rare ; β et octaves réglés à l'œil sur un seul monde ;
    - temps : un monde neuf n'affiche ses premières couches qu'après 12 s (roches, contextes), 23 s (environnement) et 41 s (gisements) ; objectif de 10 s non tenu (§15.2) ;
    - contextes : unité dominante, pas de parts ;
    - les blocs natifs grossiers (LOD 4 à 7) n'ont pas de texture de rivières : le terme « eau » y est nul ;
    - paquets de 2 à 20 Mo, chargés à chaque changement de couche (cache de 6) ;
    - sans WebGPU, repli PNG bilinéaire sans détail.
11. **Tests** : dans `tests/python`, `test_terrain_navigation` a un échec (`fully_warmed`, préchauffage des graphes CUDA) sans lien établi avec les ressources, non vérifié sur une copie propre ; la suite complète (672 tests) n'a pas été menée à terme (plus d'une heure, interrompue).

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


## 19. Monde de 4 000 km et R3e — implémentation et mesures du 27 octobre

### 19.1 Périmètre implémenté

Le profil initial `orogen` demande `world_diameter_km=4000` dans `terrain_generation.py` et dans le repli des contrôles du viewer. Les mécanismes de géométrie existants sont conservés. Une recette enregistrée reste une recette explicite ; ce changement ne réécrit pas ses réglages.

Sur cette sphère, les bornes sont ±6 283,185307 × ±3 141,592654 km. L’autorité LOD 10 passe de **1 304 × 652 = 850 208** à **410 × 206 = 84 460 cellules**, soit 9,94 % du réseau initial. L’atlas Orogen reste à 2 048 × 1 024 : le changement de diamètre ne réduit pas sa résolution en pixels, mais son pas devient 6,136 km. L’environnement Render v2 consommé par les ressources suit le rayon et les coordonnées du nouvel atlas.

La texture GPU interpole entre les centres réels des cellules rognées. La longitude se replie sur la circonférence physique, et non sur le nombre entier de colonnes multiplié par 30,72 km. Le détail procédural non périodique s’atténue sur les deux cellules voisines de la jonction ; le champ d’atlas reste périodique. Les distances des emprises de gisements se replient sur la même circonférence. Les bornes sont présentes dans le paquet, les entrées des processus de travail et les deux identités de cache.

`COMPILER_VERSION = resources-compiler-6` : la version 4 couvre les nouvelles entrées géométriques ; la réalisation finale des districts change les propriétaires, les positions et les clés de placement. La version 5 a servi à mesurer une réalisation intermédiaire trop dispersée, écartée après inspection des regroupements. `V2_INPUTS_VERSION` reste inchangée : les nouvelles bornes différencient directement son cache. Les classes de roches de ce petit monde restent très dominées par le grès ; ce n’est pas une calibration géologique multi-mondes.

### 19.2 Potentiels et districts implémentés ; paramètres proposés

**Potentiel par métal.** Onze couches `resources-dep-potential-<métal>` couvrent fer, cuivre, étain, plomb, zinc, argent, or, sel, gypse, charbon et tungstène. Le métal stellaire n’est exposé qu’en mode fantasy. Pour un métal m :

```text
potentiel_m = Σ_f [ densité_f × prospectivité_f × tonnage_médian_f × teneur_f,m ] / 10 000
```

Les familles, médianes et teneurs sont celles de `FAMILIES` et `PLACERS`, issues des modèles déjà documentés au §5.2. Le facteur 1 000 convertit tonnes en kilogrammes pour or, argent et métal stellaire. La grandeur est une **dotation typique pondérée par km²**, pas l’espérance mathématique d’une loi lognormale, ni un inventaire de gisements découverts. Son domaine est [0, somme des contributions maximales des familles]. La rampe logarithmique a des bornes de couleur provisoires ; elle ne définit pas les bornes physiques. Le champ continu utilise le profil minéral GPU, avec son miroir NumPy et deux configurations de parité (cuivre, fer). Les zones proviennent des prospectivités ; elles ne suivent pas le tirage des sites.

**Hiérarchie.** Provinces permissives (prospectivités), puis candidats de districts adressés au LOD 12 (123 km), puis gisements au LOD 3. Deux pour cent des candidats par famille sont actifs, avec au moins un candidat actif dans le domaine. Leur intensité est elliptique, orientée par l’azimut structural de la ceinture. Les écarts-types vont de 10–30 km dans l’axe à 4–30 km en travers ; les bassins évaporitiques sont isotropes. Les placers ont également des districts. L’occupation de 2 %, ces dimensions et le plancher de fond de 0,05 sont **proposés pour la calibration**, bien que le mécanisme soit implémenté. Ils ne sont pas des constantes tirées de la littérature.

Le nombre initial de marques d’une famille est toujours le tirage de Poisson adressé du §5.3. La répartition par plus forts restes déplace **ce même nombre entier** vers les districts, pondérés par la prospectivité et l’aire. Chaque marque garde son identifiant, son `source_cell`, son minerai, sa teneur et ses métaux contenus ; `owner_cell` désigne sa nouvelle cellule propriétaire. Les clés de placement incluent l’identifiant de la marque. `district_id` et `district_center_m` rendent l’appartenance inspectable dans les données. Les cellules partielles ne placent aucun centre hors du monde. L’accès médiéval est recalculé avec le relief du nouveau propriétaire : le récupérable peut donc changer, sans changer le contenu.

Ce choix est inspiré par la séparation entre domaine permissif, modèles teneur-tonnage et estimation du nombre de gisements de [Singer & Menzie, 2010](https://academic.oup.com/book/42025). Le regroupement spatial est discuté dans [USGS OFR 93-23](https://pubs.usgs.gov/of/1993/0023/report.pdf). L’introduction de [Cox & Singer, 1986](https://pubs.usgs.gov/bul/b1693/html/bull1nzi.htm) distingue explicitement gisements individuels et données parfois agrégées en districts : notre district est un regroupement spatial, et n’ajoute pas un second stock aux marques qu’il regroupe.

### 19.3 Affichage des stocks implémenté

Même somme de gisements à tous les LOD, pour le contenu comme pour le récupérable. Pour chaque emprise de demi-axes a, b, la longueur de support devient :

```text
d = 135 km × smoothstep(LOD 2, LOD 6)
A = max(2a, 3 pixels, d)
B = max(2b, 3 pixels, d/3)
K(x,y) = 3 / (π A B Z) × max(0, 1 − u²/A² − v²/B²)²
valeur = Σ_g masse_g × K_g × 10⁶     [t/km², ou kg/km²]
```

L’ellipse suit l’orientation de l’emprise. Les LOD 10–6 élargissent le stock jusqu’à un écart-type de 45 km dans l’axe (support 135 km), ou trois pixels au très grand angle ; les LOD 5–3 révèlent les amas et les LOD ≤2 reviennent aux emprises. Les filons conservent leur allongement quand ils deviennent résolus. Les symboles et géométries de la couche « Gisements discrets » restent disponibles.

`Z` est l’intégrale du noyau contre l’aire sphérique, avec rognage aux pôles. Sa marginale en latitude est intégrée par huit points de Gauss-Legendre, dans le miroir et le WGSL. Il n’y a ni champ de stock interpolé pour remplacer les sites à petite échelle, ni terme de relief modifiant leur masse. Le support compact a une valeur et une dérivée nulles au bord. Un fondu de couleur sur la couverture du noyau évite que l’échelle logarithmique durcisse visuellement cette fin de support ; ce fondu ne change pas la densité. L’évaluateur parcourt tous les seaux qui intersectent le support, sans dupliquer les lignes polaires ni les colonnes repliées.

**Tolérance mesurée de l’intégrale** : erreur relative maximale **0,01164 %** contre une quadrature adaptative indépendante, sur 27 cas (LOD 10, 7, 6, 5, 4, 3, 2, 0, −2 ; équateur, latitude intermédiaire, centre à 300 m du pôle). Test borné à 0,02 %. C’est une tolérance float32 et quadrature, pas une égalité bit à bit d’un raster de pixels. Les budgets et sommes par propriétaire sont, eux, exactement entiers. Le masque terre/mer du viewer reste une présentation : il ne sert pas à intégrer un stock.

### 19.4 Mesures sur la référence 4 000 km

Monde seed 42, profil `orogen`, GPU activé, RTX 3090, Chrome, serveur de ce worktree sur **8775**. Numba est redirigé vers `E:/TerrainDiffusionRuntime/numba-cache`, sans écriture dans la jonction amont. Aucun test de noyaux CUDA nécessitant l’arrêt d’un autre serveur n’a été lancé.

Les durées du premier pixel utilisent le critère existant de `benchmark_resources_first_pixel.cjs` : paquet lié au GPU, puis deux images `requestAnimationFrame`. Elles ne chronomètrent pas une lecture de pixel GPU. Les captures ci-dessous vérifient séparément que le viewer a effectivement rendu la couche et que les tuiles proches sont des sorties du décodeur.

| Mesure | Ancienne référence 12 732 km (§15) | 4 000 km, tâche A |
|---|---|---|
| Compilation pendant l’ouverture, environnement neuf | 41,1 s | 13,60 s |
| Étapes tectonique / environnement / géologie / gisements | 12,1 / 23,2 / 31,1 / 41,1 s | 5,72 / 7,61 / 11,04 / 13,57 s |
| Ouverture du viewer | 11,4–12,6 s | 21,20 s ; seconde ouverture 19,86 s |
| Premier passage NPP / roches | 25,3 / 1,3 s, monde neuf | 2,854 / 2,131 s |
| Dix couches sur monde chaud | 0,2–1,4 s | 0,166–0,317 s |

La réduction de l’autorité est mesurée ; l’ouverture globale n’est pas accélérée dans cet essai. Les coûts fixes de l’atlas, des aperçus et du viewer restent présents. Les deux dépassements de 2 s du premier passage sont conservés dans les résultats ; ils ne sont pas présentés comme des mesures sur monde chaud.

**R3e final, version 6 (mesuré).** Compilation 12,62 s, avec environnement Render v2 en cache ; étapes 4,59 / 4,91 / 8,33 / 12,55 s, cache durable prêt à 13,70 s. À comparer aux 31,0 s historiques avec environnement en cache, et non aux 41,1 s du cas entièrement neuf. Les graines des campagnes historiques diffèrent : cette comparaison est indicative, pas une expérience contrôlée à relief identique.

| Premier pixel, monde chaud | Secondes |
|---|---:|
| Potentiel cuivre | 0,499 |
| Potentiel fer | 0,384 |
| Stock cuivre | 0,383 |
| Stock fer | 0,816 |
| Ensemble des 13 couches du benchmark | 0,291–0,816 |

Commande : `node tools/benchmarks/benchmark_resources_first_pixel.cjs --port 8775 --seed 42 --json output/resources-gpu/first-pixel-4000-r3e-final-warm.json`. Les 13 couches sont affichées, zéro erreur JavaScript remontée, ouverture du viewer en 15,346 s. Au premier passage après recompilation, les quatre nouvelles vues ont pris 1,603 / 0,500 / 0,167 / 0,183 s ; l’ouverture a pris 21,873 s. L’objectif de moins de 2 s sur monde chaud est tenu sur cette campagne.

### 19.5 Budgets et regroupements mesurés

Comparaison du cache 4 000 km de la tâche A (702 gisements) avec la version finale (703). Un site peut contenir plusieurs métaux : les nombres de la table ne se somment pas. Les unités sont des tonnes, sauf or et argent en kilogrammes.

| Métal | Nombre avant → après | Contenu avant → après |
|---|---:|---:|
| Fer | 41 → 41 | 1 196 220 427 → identique |
| Cuivre | 496 → 496 | 42 867 557 → identique |
| Plomb | 469 → 469 | 26 395 082 → identique |
| Zinc | 469 → 469 | 17 648 514 → identique |
| Argent | 589 → 590 | 228 869 786 → 228 870 035 kg |
| Or | 153 → 154 | 1 726 244 → 1 728 234 kg |
| Charbon | 14 → 14 | 3 485 635 794 → identique |
| Étain, sel, gypse, tungstène | 0 → 0 chacun | 0 → 0 chacun |

Le gisement orogénique supplémentaire (1 990 kg d’or, 249 kg d’argent) vient de la correction du champ géologique périodique des bornes ; les teneurs/tonnages des autres marques sont conservés. À champs permissifs identiques, le test reconstruit les tirages initiaux et vérifie **exactement** les identifiants, comptes, minerais et contenus de toutes les marques après routage. Les stocks et récupérables par propriétaire sont aussi comparés exactement aux sommes de gisements.

**Le récupérable change réellement**, car les galeries ont désormais le relief du district d’arrivée : cuivre 3 847 205 → 2 273 612 t (−40,9 %), fer 63 778 402 → 52 443 764 t (−17,8 %), plomb 12 932 191 → 13 424 584 t, argent 79 532 424 → 75 697 403 kg, or 763 512 → 572 261 kg, charbon 165 838 620 → 120 875 144 t. Zinc : 0 inchangé. Ce changement est documenté et n’est pas présenté comme une conservation du récupérable entre versions.

Les **703 gisements occupent 125 districts**, de 1 à 29 sites ; 31 districts sont des singletons. Distance plane médiane au plus proche gisement du même métal, avec repli longitudinal : cuivre **108,26 → 3,97 km**, fer **393,30 → 27,07 km**, or **167,36 → 8,11 km**, charbon **775,56 → 28,77 km**. Cela mesure le regroupement, sans valider géologiquement les dimensions proposées.

### 19.6 Validation et captures relues

- **Python : 54 tests réussis**, 112,998 s, suites `test_terrain_resources.py` (27), `test_terrain_server_world.py` (9), `test_terrain_tile_rendering.py` (16), `test_terrain_web_assets.py` (2). Inclut centres et placement déterministes, requêtes en ordre inverse, appartenance au propriétaire, budgets exacts, conservation des marques, bornes de tous les potentiels et isolation fantasy. Les bornes s’appliquent également à l’évaluateur continu GPU, indépendamment de la rampe de couleur.
- **CPU/WGSL : 216 cas, 10 configurations, zéro désaccord**, erreur relative maximale normalisée **7,88·10⁻⁷** ; classes, mois et contextes identiques. Commandes : `python tools/verification/build_resources_gpu_fixture.py`, puis `node tools/verification/verify_resources_gpu_parity.cjs`. Les paquets synthétiques incluent des cellules longitudinales partielles, des échantillons extérieurs au domaine, deux potentiels effectivement composés depuis les familles et des stocks aux LOD 10 à −2.
- **Renderer réel : un test navigateur réussi**, répétition exacte, différence saisonnière moyenne 24,30/255 ; aucune erreur de shader détectée par ce test. `node --test tests/browser/test_terrain_resources_gpu.cjs`.
- **Captures : 30 images sources**, 410 × 260, assemblées en quatre planches ; 5 LOD × (Render v2, potentiel, stock) × 2 métaux. `node tools/verification/capture_resources_gpu.cjs --port 8775`. Même caméra et mêmes tuiles dans chaque comparaison ; zéro tuile attendue au déclenchement. LOD 10/7/4 : source `conditioning-preview` ou bloc natif courant du viewer ; LOD 2/0 : source **`decoder`**, vérifiée par le script. Les vignettes proches ne sont donc pas un agrandissement de l’aperçu monde.

Planches relues visuellement, Render v2 en haut :

- [Potentiel cuivre](images/resources-gpu-potential-copper.jpg)
- [Potentiel fer](images/resources-gpu-potential-iron.jpg)
- [Stock cuivre](images/resources-gpu-stock-copper.jpg)
- [Stock fer](images/resources-gpu-stock-iron.jpg)
- [Jonction des longitudes : Orogen, Render v2 et ressources](images/resources-gpu-wrap.jpg), `node tools/verification/capture_resources_wrap.cjs`.

| Planche | Constat après lecture | Limite restante |
|---|---|---|
| Potentiel cuivre | domaine continental continu au LOD 10, régions cohérentes aux LOD 7/4, relief lisible aux LOD 2/0 | le monde très gréseux donne un potentiel assez uniforme ; marbrure trop présente au LOD 7 |
| Potentiel fer | provinces régionales et larges zones compatibles avec les roches sédimentaires, sans sites isolés ; détail de relief conservé au zoom proche | contraste régional limité dans le continent choisi, et texture grossière à moyenne échelle |
| Stock cuivre | districts visibles à grande échelle ; regroupements allongés au zoom moyen ; corps résolu au zoom proche | la plus grosse lentille choisie devient une bande qui remplit la vue au LOD 0 ; les très petits sites restent sous-pixel |
| Stock fer | concentrations regroupées, plusieurs corps au LOD 4 et une lentille au LOD 2 ; même contenu total du modèle | 41 sites seulement : le monde conserve de grandes plages sans stock ; un grand corps couvre la vue LOD 0 |

**Jonction observée.** La planche de jonction assemble les bandes droite puis gauche du viewer, à 3 840 m/pixel, y = 2 300 km. La croûte Orogen se raccorde. Une cassure verticale du détail de terrain reste visible dans **Render v2** ; elle se retrouve dans le relief sous les ressources. La période physique du champ ressource est vérifiée séparément par les tests, mais cette capture ne permet pas d’affirmer que toute la présentation est sans couture. Cette faiblesse non ressource est signalée ; elle n’a pas été réécrite dans cette tâche.

**Faiblesses et proposé.** Le placement et les dimensions des districts, les rampes et les profils de relief demandent une calibration sur plusieurs mondes. Le monde choisi ne fournit aucun gisement d’étain, sel, gypse ou tungstène : leurs couches existent et leurs bornes sont testées, leur apparence réelle n’est pas validée ici. Les potentiels sont des dotations typiques, leurs valeurs ne sont pas des réserves. Les champs de stock sont conservatifs à la tolérance de quadrature ; une somme des couleurs ou de pixels ponctuels n’est pas un export de tonnage. Le gros grès dominant, les aperçus et le temps d’ouverture global restent des limites visibles du monde courant ; aucune réécriture de géométrie, d’Orogen ou de Render v2 n’a été faite. Les anciennes limites de conservation des champs continus (§16.2) restent applicables aux potentiels, distinctement des noyaux de stock.


## 20. R3h — localiser les ressources avec des techniques fixes

### 20.1 Implémenté : lecture de la carte et quantités

Le menu offre **Fer, Cuivre, Étain, Plomb, Zinc, Argent, Or, Sel, Gypse, Charbon, Tungstène**, puis le **Métal stellaire** seulement si `TERRAIN_RESOURCES_FANTASY=1`, et **Tous les gisements**. Il n’expose plus les couches de contenu ou une deuxième série « récupérable ». Le contenu reste interne pour calculer teneurs et budgets. Tous les potentiels par métal et toutes les prospectivités par famille viennent en dernier dans **Diagnostic** ; leurs deux légendes, structurée et texte, expliquent leur rôle de placement.

La quantité cartographiée est le **récupérable initial** à techniques fixes, après l’accès par profondeur/relief et le coefficient de récupération. Aucun progrès technique n’est simulé. Or, argent et métal stellaire restent exportés en kg ; les tailles de symboles convertissent ces kg en tonnes physiques, pour garder la même échelle entre ressources. Le total de « Tous » additionne ces masses physiques, sans ajouter le minerai ni le contenu une seconde fois. Les réserves d’un site multirésource apparaissent dans chaque couche concernée ; les comptes par ressource ne se somment donc pas.

**Interprétation Zn/W (provisoire).** Zinc = équivalent Zn dans un minerai triable, notamment utilisable en cémentation du laiton ; tungstène = équivalent W dans du minéral triable, dont l’usage médiéval reste **non validé**. Ces deux couches ne promettent pas la production de métal raffiné. Coefficients fixes proposés : Zn 0,30 et W 0,25, en plus de l’accès ; ils remplacent les zéros du précédent scénario « métal fondu ». Cette convention doit être validée comme règle de jeu, surtout pour W.

Chaque couleur désigne une famille. Pour une masse récupérable M en tonnes :

```text
rayon_px² = 9 + 5 × clamp(log10(max(M, 1)), 0, 8)
```

L’aire du disque est donc affine en log10(M), bornée : diamètre 6 px jusqu’à 1 t, 8,7 px à 100 t, 10,8 px à 10 000 t et 14 px à partir de 100 Mt. Un contour sombre de 1 px le sépare du terrain. La taille suit les pixels CSS de la caméra, y compris sur des tuiles plus fines conservées en cache ; un symbole peut rester plus grand lors d’un suréchantillonnage de tuile. Le shader choisit la plus grande masse à chaque recouvrement, indépendamment de l’ordre des seaux. Les petites occurrences occultées se séparent au zoom ; chaque couche ressource reste sélectionnable. Il n’y a pas de regroupement avec compteur.

Le fondu vers l’emprise commence lorsque `sqrt(a×b)/mpp` dépasse le rayon du symbole et finit au double : pas de bascule brutale pour un filon étroit. Amas et lentilles ont un axe transversal courbe, les filons suivent la structure et les placers la pente régionale, avec une modulation bornée par la vallée. Le contour continue à suivre le corps résolu. Les teneurs tirées sont plafonnées pour que la somme des fractions minérales ne dépasse pas 100 % (tolérance des quanta d’arrondi). Les teneurs tirées sont plafonnées pour que la somme des fractions minérales ne dépasse pas 100 % (tolérance des quanta d’arrondi). La coloration ne modifie aucun tonnage. L’inspection par **clic** près du symbole donne famille, ressource, récupérable avec unité, teneur, profondeur du toit et longueur × largeur de l’emprise ; ses centres adressés sont ceux du GPU. Le rayon de recherche suit le zoom et la couche filtre les sites avant de limiter la réponse. Le repli CPU filtre aussi les ressources et utilise la même échelle de symboles ; ses emprises détaillées ne sont pas validées.

### 20.2 Implémenté : populations locales, évaporites et déterminisme

Aux modèles de grands corps s’ajoutent fer des marais, petit fer sédimentaire, petits filons Pb–Ag/Cu, sources salées/saumures, carrières de gypse, petits filons Sn–W et affleurements de charbon. Les placers Au et Sn ont une population plus dense et des médianes abaissées : respectivement 25 000 et 15 000 t de matériau. Il ne s’agit pas de multiplier les anciens corps géants. Les nouveaux sites ont des toits médians de 1–4 m et des médianes de 3 000–30 000 t ; les petites teneurs récupérables peuvent être arrondies à zéro au quantum d’export et sont alors absentes de la couche correspondante.

La rareté des évaporites venait notamment d’une réalisation catégorielle trop peu convergée et d’une porte utilisant les quatre échantillons de roche d’une cellule de 30,72 km. La calibration passe à 12 itérations amorties, un échantillon toutes les quatre cellules terrestres. Le prior lithologique conserve la même formule de paléo-bassin. Les évaporites utilisent désormais les hôtes **attendus continus** pour l’intensité de placement, et un seuil graduel de 0 à 0,003 de fraction attendue, pour éviter de transformer une cellule rarement échantillonnée en interdiction géologique ou en rectangle de carrières. Cette fraction n’est pas une preuve de saumure exploitable partout dans le bassin.

`COMPILER_VERSION` passe de **resources-compiler-7** à **resources-compiler-12** suivant §15.1 ; 8 à 11 étaient des essais locaux, non livrés. Le compteur de Poisson n’est plus tronqué à 40 lorsque la moyenne est élevée dans une grille d’export grossière. Les tirages gardent les domaines adressés et la seed ; chaque marque conserve sa clé, son propriétaire et son district. La position utilise une CDF des poids des enfants dans l’ordre canonique des indices, tirée par la clé de la marque. Les CDF sont réutilisées entre occurrences d’un même propriétaire, avec mémoire de travail bornée ; ni leur cache, ni la taille des lots, ni l’ordre des requêtes ne changent le résultat. Les centres de districts inactifs sont exclus une fois par famille du calcul d’attribution. Les budgets de contenu et de récupérable restent des sommes entières exactes des sites par propriétaire. Le GPU garde un index de seaux LOD 10 (début, nombre), et parcourt uniquement ceux intersectant le symbole ou l’emprise : aucun parcours mondial par pixel. La palette passe à 32 entrées pour ne pas confondre les nouvelles familles.

### 20.3 Proposé : calibration historique et plages de densité

**Points d’ancrage vérifiés, pas inventaire exhaustif.** Le [cadre archéologique du Forest of Dean, Hoyle 2017, p.144](https://historicengland.org.uk/images-books/publications/research-framework-forest-of-dean-district/forest-of-dean-glos-research-framework-2017/) rapporte 25–30 forges au milieu du XIIIe siècle, 43 en 1270 et 60 en 1282. Ce sont des installations de traitement ; plusieurs peuvent exploiter un même gisement. Les [comptes du district de 2019–2020, p.2](https://www.fdean.gov.uk/media/102do2gn/fod-draft-statement-of-accounts-2019-20_final.pdf) donnent une surface moderne de 526 km². Diviser ces comptes par cette surface donne **475–1 141 installations pour 10 000 km²**, calcul indicatif **provisoire** : les limites médiévales et modernes ne coïncident pas. C’est un ordre de grandeur de région spécialisée, pas une densité de minerais du monde entier.

La [notice officielle de Droitwich, Historic England 1020256](https://historicengland.org.uk/listing/the-list/list-entry/1020256) recense cinq puits à saumure au Domesday et plus de 1 000 t/an de sel, puis environ 1 500 t/an vers 1400. Elle confirme aussi la présence côtière et intérieure de cette activité dès la fin du Xe siècle. Il serait trompeur d’en déduire une densité de gisements sans aire de bassin inventoriée. Une plage régionale de **5–100 sources/sites pour 10 000 km²** est donc **provisoire**, et les salines marines restent le flux `sea_salt`, distinct de cette population de saumures et de sel gemme.

Les paramètres ci-dessous sont des occurrences pour **10 000 km² à prospectivité 1**, avant regroupement en districts et accès médiéval. Le nombre réel suit l’intégrale `densité × prospectivité × aire`, pas la densité maximale multipliée par toutes les terres. Les plages sont **proposées, provisoires** pour tous les modèles : les modèles USGS teneur-tonnage ne sont pas des inventaires médiévaux de petites exploitations. Le fer se concentre ensuite dans des régions de district plus denses que la moyenne mondiale. Une calibration archéologique comparable par bassin, famille et période reste à faire.

| Famille | Paramètre implémenté / 10 000 km² | Plage de calibration proposée, provisoire |
|---|---:|---:|
| `porphyry_cu` | 0.15 | 0.0375–0.6 |
| `epithermal_au_ag` | 0.6 | 0.15–2.4 |
| `sn_w_granite` | 0.5 | 0.125–2 |
| `polymetallic_vein` | 0.8 | 0.2–3.2 |
| `vms` | 0.4 | 0.1–1.6 |
| `orogenic_au` | 0.5 | 0.125–2 |
| `mvt_pb_zn` | 0.15 | 0.0375–0.6 |
| `sediment_cu` | 0.1 | 0.025–0.4 |
| `skarn_fe_cu` | 0.3 | 0.075–1.2 |
| `bif_iron` | 0.08 | 0.02–0.32 |
| `ironstone` | 0.25 | 0.0625–1 |
| `laterite_fe` | 0.3 | 0.075–1.2 |
| `evaporite_salt` | 0.2 | 0.05–0.8 |
| `coal` | 0.1 | 0.025–0.4 |
| `bog_iron` | 25 | 6.25–100 |
| `local_ironstone` | 30 | 7.5–120 |
| `small_pb_ag` | 35 | 8.75–140 |
| `salt_spring` | 20 | 5–80 |
| `gypsum_quarry` | 100 | 25–400 |
| `small_sn` | 25 | 6.25–100 |
| `small_coal` | 20 | 5–80 |
| `placer_au` | 30 | 7.5–120 |
| `placer_sn` | 25 | 6.25–100 |
| `fantasy_star_metal` | 0.02 | 0.005–0.08 |

Ces plages ne constituent pas des intervalles de confiance historiques. Les coefficients d’accès, les tonnages, le nombre de sources salées, le rendement des petites carrières, les dimensions des districts et l’utilité du tungstène restent **provisoires**. Le modèle vise toutes les tailles d’occurrences représentées par ces populations ; il ne peut garantir un recensement exhaustif des sites réels.

### 20.4 Rattrapage documentaire R3g (ad8c6ba), implémenté et mesures archivées

**Ceintures métallifères.** Les tracts tectoniques physiques (`tract_*`) remplacent la somme de faibles appartenances floues comme contrainte minérale. Une union pondérée des contextes est passée dans un seuil lisse au carré, multipliée par les hôtes et par des provinces fertiles adressées de 123/246 km. Les placers dérivent des sources primaires, diffusées régionalement et limitées par l’eau et la rugosité. L’échelle des potentiels métalliques est **linéaire** (borne de couleur 12 % de la borne physique théorique) ; ces couches sont maintenant diagnostiques.

**Emprises organiques.** Les anciennes vues de densité conservée utilisaient cinq lobes pondérés (0,13/0,23/0,28/0,23/0,13), un axe courbe et la pente régionale pour les placers ; le noyau était normalisé en aire sphérique. Elles restent internes au miroir et aux tests d’intégrale, et ne sont plus proposées comme cartes de ressources. Les vues R3h suivent les axes et formes par corps avec un contour et le fondu du §20.1. L’intégrale d’une couleur ou d’un symbole n’est pas un export de masse.

**Jonction longitudinale.** Les champs utilisent la circonférence physique et les centres rognés ; le détail non périodique s’atténue à la jonction. Le correctif R3g de `_wrap_edt` étend le halo jusqu’au support significatif des ceintures, afin qu’une frontière tectonique de l’autre côté de l’antiméridien reste visible. Les distances des dépôts se replient également sur la période physique. Les tests de périodicité concernent les ressources ; la cassure de relief Render v2 signalée en §19.6 reste une limite distincte et n’est pas déclarée corrigée par R3h.

**Contraste mesuré R3g, archives locales** `output/resources-r3g/contrast-before-controlled.json` et `contrast-after.json`, seed 42. Pour le cuivre, la part du potentiel intégrée dans les 10 % de terres les plus dotées passe de **20,56 % à 75,33 %** ; la part de terres sous 1 % de la borne physique passe de **10,07 % à 76,02 %**. Pour le fer : **28,39 % à 93,96 %**, et **33,19 % à 92,06 %**. Ce sont des métriques du diagnostic avant/après R3g, pas de nouveaux taux de gisements R3h ; les densités modifiées interdisent de les présenter comme mesures actuelles.

### 20.5 Mesuré R3h : validation, comptes et captures

**Référence :** Orogen, seed 42, sphère 4 000 km, 14 126 403 km² terrestres, RTX 3090, serveur de ce worktree sur 8775. Comparaison avec `ad8c6ba` (`resources-compiler-7`). Aucun processus sur 8765 n’a été arrêté. Les erreurs transitoires de mémoire Windows des essais intermédiaires ont imposé des captures séquentielles avec redémarrage de 8775 entre ressources ; les résultats finaux ci-dessous sont ceux de la campagne achevée.

| Ressource | Sites présents avant | Sites présents après | Récupérable > 0 avant | Récupérable > 0 après |
|---|---:|---:|---:|---:|
| Fer | 11 | 2070 | 8 | 2066 |
| Cuivre | 110 | 3278 | 105 | 2957 |
| Étain | 18 | 2645 | 18 | 2608 |
| Plomb | 99 | 3419 | 97 | 3410 |
| Zinc | 99 | 3419 | 0 | 3379 |
| Argent | 154 | 3486 | 146 | 3479 |
| Or | 63 | 1059 | 60 | 881 |
| Sel | 0 | 387 | 0 | 386 |
| Gypse | 0 | 1659 | 0 | 1658 |
| Charbon | 2 | 461 | 1 | 461 |
| Tungstène | 8 | 1812 | 0 | 1527 |

**Sites distincts : 198 → 11817.** Un site multirésource compte dans plusieurs lignes ; seule la colonne récupérable > 0 détermine les symboles de la couche. Métal stellaire absent du scénario sans fantasy.

**Évaporites mesurées :** 0.683 % des roches terrestres réalisées, pour 0.742 % attendus (contre environ 0,1 % réalisés dans le retour initial). Le prior physique n’a pas été gonflé pour forcer ce résultat.

**Densité régionale mesurée :** disques sphériques de 10 000 km² centrés sur les sites canoniques à 240 m, récupérable > 0, distance de corde via `cKDTree`. Maxima : fer **1061**, sel **108**, gypse **400**. La réduction des densités locales Fe et saumures corrige l’essai initial qui dépassait 4 000 sites de fer par disque. Le maximum fer est du même ordre que le repère Forest of Dean (475–1 141 installations normalisées) ; ce rapprochement reste provisoire et ne transforme pas les forges recensées en gisements certains.

**Compilation avec environnement V2 en cache : 13.41 s**, contre **12,50 s** dans le cache de référence antérieur (+7.3 %). Étapes cumulées : 5.21 / 5.45 / 8.36 / 13.28 s ; cache durable à 14.51 s. La comparaison est indicative : même monde et environnement en cache, mais la référence antérieure n’a pas été recompilée simultanément sous une charge système contrôlée. Les essais R3h intermédiaires à 43,64 / 33,98 / 19,74 s ne sont pas les performances finales.

**Premier pixel, monde chaud, 13 couches : 0.166–0.400 s**, ouverture 21.135 s, 0 erreur JavaScript. Avant : 0,151–0,383 s, ouverture 19,953 s. Critère identique du script : paquet GPU lié puis deux images `requestAnimationFrame`, sans lecture de pixel ni attente de toutes les tuiles.

**Validation finale :** 46 tests Python réussis (suites demandées), parité NumPy/WGSL 252 cas dans 11 configurations sans désaccord ; erreur normalisée maximale 2,35·10⁻⁶, ou 3,88·10⁻⁷ absolu pour les symboles. Le test navigateur du renderer réel réussit et reproduit exactement son image. Les régressions couvrent aussi le paquet partiel vide, les unités kg/t, les recouvrements indépendants de l’ordre, le fondu, les grandes moyennes de Poisson et les teneurs industrielles bornées.

**Captures :** 50 images sources (5 ressources × 5 LOD × terrain/couche), dimensions 410 × 260, assemblées en cinq planches. Les LOD 10/7/4 utilisent la source courante `conditioning-preview`, les LOD 2/0 exigent `decoder`, avec zéro tuile attendue à la capture. Caméra identique dans chaque paire. Chaque ressource est nettement visible aux cinq zooms sur les planches relues :

- [Cuivre](images/resources-gpu-sites-copper.jpg)
- [Fer](images/resources-gpu-sites-iron.jpg)
- [Sel](images/resources-gpu-sites-salt.jpg)
- [Gypse](images/resources-gpu-sites-gypsum.jpg)
- [Tous les gisements](images/resources-gpu-sites-points.jpg)

Sources PNG et états détaillés : `output/resources-gpu/r3h-captures/`. Commande complète : `node tools/verification/capture_resources_gpu.cjs --port 8775` ; la variante `--resources copper` emploie les identifiants anglais (`copper`, `iron`, `salt`, `gypsum`, `points`), et `--append` permet une campagne séquentielle avec redémarrage de 8775 entre ressources.

**Faiblesses et proposé :** les districts peuvent rester très concentrés, et certains contacts de populations héritent de la grille permissive LOD 10. Les petits symboles se recouvrent au grand angle, sans compteur ; le zoom et les couches individuelles servent à les séparer. Les aperçus grossiers ont moins de relief que les tuiles du décodeur. La jointure de relief Render v2 reste distincte du correctif de ressources. Les densités, l’interprétation Zn/W, les coefficients de récupération et la géologie n’ont pas de validation archéologique multi-mondes. Les images montrent la présence, pas un inventaire historique vérifié ni l’intégrale de masse d’un raster.

## 21. Compilation accélérée, sans changement du canon — resources-perf

**27 octobre 2026, référence `e2046f93bfe1b844bd8a8b838d6742b79e04a7ed`.**
Le premier affichage et le changement de couche sont accélérés ; **l’objectif de compilation complète à 1,5 s n’est pas atteint**. Aucun fichier de `web/`, aucune palette, aucune définition de couche ou légende n’a été modifié.

### 21.1 Protocole et profil avant modification

Deux vrais atlas Orogen du profil par défaut, sphère de **4 000 km**, seeds **42** et **20261027** (nouvelle seed), ont été conservés avant modification dans `E:/TerrainDiffusionRuntime/resources/perf-inputs/`. Grille d’autorité : **410 × 206**, LOD 10. Les atlas, modèles, caches et sorties de profilage restent hors Git ; les empreintes de régression sont dans `tests/fixtures/resources-e2046f9-*.json`.

Matériel : RTX 3090 ; calcul canonique CPU NumPy/SciPy, Numba disponible dans le venv fourni. `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` pour les mesures. Le port utilisé est exclusivement **8776**. Une tentative initiale a échoué par manque de mémoire engagée Windows ; ses temps ne sont pas repris. Le serveur nécessitait `NUMBA_CACHE_DIR=E:/TerrainDiffusionRuntime/numba-cache/resources-perf` pour éviter les tentatives de cache dans la jonction upstream non modifiable.

Chronométrage de chaque étape, sans `cProfile`, compilateur initial avec ses workers de processus :

| Étape, durée propre (s) | Seed 42 | Seed 20261027 |
|---|---:|---:|
| Tectonique, lithologie attendue, calibration, climat et eau | 5,211 | 4,887 |
| Environnement Render v2 et biologie | 1,463 | 1,290 |
| Géologie réalisée et matériaux | 2,584 | 2,428 |
| Gisements et stocks | 5,483 | 5,238 |
| **Total compilation** | **14,771** | **13,873** |
| Deuxième compilation dans le même processus, sans cache monde | 14,490 | 14,921 |
| Lecture du cache durable, reconstruction des stocks comprise | 0,479 | 0,447 |

Profil `cProfile` initial **sériel**, pour voir les fonctions plutôt que l’attente des sous-processus. Temps cumulés, donc **non additionnables** :

| Fonction (s cumulées) | Seed 42 | Seed 20261027 |
|---|---:|---:|
| `compile_world` | 13,744 | 14,073 |
| `world_deposits` | 5,346 | 6,159 |
| `place_positions` | 1,818 | 2,356 |
| `_placement_distribution` | 1,474 | 2,006 |
| `tectonic_contexts` | 2,055 | 2,321 |
| `_wrap_edt` (six distances) | 1,379 | 1,538 |
| `lithology` | 1,305 | 1,282 |
| `calibrate_rock_bias` | 1,360 | 1,137 |
| `realize_rocks` | 1,388 | 1,058 |
| `value_noise` (tous les appels) | 1,810 | 1,561 |
| `render_v2_fields` | 0,939 | 0,824 |
| `_deposit` | 1,015 | 1,052 |

Le profil ne reproduit donc pas 8 s de calcul Render v2 pur : ici, les processus, imports, publications de snapshots et autres charges distinguent le temps perçu du temps des équations. Les gisements, la préparation à résolution atlas et la géologie sont également déterminants.

### 21.2 Changements implémentés

- **Échantillonnage exact des nœuds utiles.** Les distances et les voisinages tectoniques conservent l’atlas complet, mais les équations de contexte et lithologie ne sont évaluées qu’aux nœuds lus par les quatre voisins bilinéaires de l’autorité. Même ordre des opérations et même conversion float32 ; aucune sous-résolution du canon.
- **Tables temporaires réutilisées.** Les douze itérations de calibration géologique partagent leurs warps et bruits fixes. Les biais et interpolations restent recalculés à chaque itération ; aucun biais approximatif réutilisé entre mondes.
- **Render v2 mensuel factorisé.** `terrain_resource_environment.py` calcule une fois les termes indépendants de la date, puis les douze saisons. À 30,72 km, les deux amplitudes des mosaïques fines sont exactement nulles : leurs hashes ne sont plus calculés. Les équations utiles reprennent le miroir CPU indépendant, testé également à 30 et 480 m. Un bake de tuile, dépendant de sa résolution et éventuellement de son affichage, ne remplace pas l’autorité.
- **NPP publié en premier.** Une étape `climate` publie l’indice de productivité exact dès les seuls champs de climat et hauteur disponibles, avant les distances tectoniques. Sa valeur est vérifiée contre celle de la compilation finale. Les autres couches attendent leurs vrais champs ; aucun stock provisoire n’est envoyé.
- **HTTP et compilation découplés en mémoire.** Un thread de compilation sérialisé conserve les snapshots immuables en mémoire. Plus d’export d’atlas, de spawn du compilateur ou de relecture de snapshots pour les requêtes. Un pool résident de quatre threads traite les six EDT, qui relâchent le GIL. La petite géologie reste sérielle : la distribuer entre threads s’est révélée plus lente. Le CLI isolé et le map de processus restent disponibles en repli/diagnostic.
- **Placement des gisements.** CDF optionnelle fusionnée Numba, CPU, sans `fastmath`, sans réduction parallèle ni nouvel aléa ; repli NumPy avec `TERRAIN_RESOURCES_FAST=0` ou en l’absence de Numba. Suppression des grosses copies de champs dans l’interpolation, décodage groupé des positions, calcul par lots du district le plus proche et sommes entières vectorisées des stocks. Les draws et leur ordre restent inchangés.
- **Cache compact compatible.** Même paire NPZ/JSON, tableaux utiles uniquement, stocks dérivés reconstruits, JSON compact, DEFLATE niveau 1, publication atomique par fichiers temporaires uniques. Exemple seed 42 : tableaux **19,47 Mo** contre **69,46 Mo** sans compression. Décompression des membres NPZ en quatre threads : environ 0,07 s contre 0,15 s en série, sans changer les octets des tableaux. La sauvegarde est hors du chemin critique du premier affichage.

### 21.3 Mesures après et comparaison

Mesure pure : aucun cache d’autorité ni environnement v2 injecté ; Orogen est déjà fourni comme entrée. « Processus froid » inclut imports et lecture de cet atlas, puis **une seule** compilation. Il ne comprend ni la génération d’Orogen, ni la deuxième compilation, ni `cProfile`, ni la sauvegarde. Les caches de fichiers du système ne sont pas vidés. Le cache de code Numba existe ; une toute première JIT sans ce cache a donné 8,71 s de compilation sur seed 42, contre 6,11 s au deuxième appel de cet essai.

| Mesure (s) | Avant 42 | Après 42 | Avant 20261027 | Après 20261027 |
|---|---:|---:|---:|---:|
| Compilation initiale, chemin de production de chaque version | 14,771 | 5,993 | 13,873 | 5,664 |
| Recompilation même processus, sans cache monde | 14,490 | 4,916 | 14,921 | 4,747 |
| Processus neuf, imports + atlas + compilation, référence **sérielle** | 10,138 | 6,639 | 11,325 | 6,286 |
| Cache durable, ancien chargeur / nouveau chargeur parallèle | 0,479 | 0,219–0,229 | 0,447 | 0,219–0,233 |
| Écriture durable, hors chemin critique | 1,370 | 0,709 | 1,335 | 0,707 |

La ligne processus neuf utilise le meilleur chemin sériel initial comme comparaison supplémentaire : compilation seule initiale 9,422 / 10,632 s, deuxième appel 9,513 / 9,141 s. Elle n’attribue pas artificiellement le coût de spawn à l’import du serveur. Les anciennes valeurs `cold_process_wall_s` des premiers fichiers `before*.json` chronométraient tout le lot (warm + profilage compris) ; **ce ne sont pas des temps froids valides**. Le benchmark corrigé conserve ce total sous `benchmark_batch_wall_s`. Le tableau de lecture disque suppose le compilateur déjà importé, comme après le calcul de l’identité dans le serveur : un processus lecteur entièrement neuf a pris **0,640 s**, imports compris, avant deux lectures à 0,229 / 0,225 s. Ce cas ne respecte pas 0,3 s.

| Étape après (durée propre, s) | Seed 42 | Seed 20261027 |
|---|---:|---:|
| Productivité initiale | 0,094 | 0,090 |
| Tectonique et préparation | 1,571 | 1,581 |
| Environnement et biologie | 0,213 | 0,211 |
| Géologie | 0,991 | 0,895 |
| Gisements et stocks | 3,115 | 2,878 |

Le profil après donne `world_deposits` 2,759 / 2,850 s, `realize_rocks` 0,845 / 0,729 s, `family_prospectivities` 0,728 / 0,681 s, et `place_positions` 0,766 / 1,019 s. `cProfile` ne mesure pas toutes les fonctions des threads EDT ; leurs temps restent inclus dans l’étape chronométrée au mur.

**Navigateur réel, seed 42**, ressources et environnement v2 absents d’un répertoire de cache dédié ; serveur initial redémarré depuis la référence pour la mesure avant :

| Mesure (s) | Avant | Après |
|---|---:|---:|
| Ouverture du monde | 21,415 | 18,183 |
| Premier NPP après ouverture | 1,733 | **0,722** |
| Premier paquet de chaque autre couche (12 couches, compilation éventuelle comprise) | 0,166–2,116 | 0,166–0,617 |
| Sélection répétée, 13 couches déjà chargées | non mesurée | **0,050–0,067** |
| Compilation complète sous charge du viewer | 17,08 | **9,62** |
| Lecture depuis le disque après redémarrage du viewer | non mesurée | **0,83** |
| Erreurs JavaScript | 0 | 0 |

Le critère du script reste paquet GPU lié + deux `requestAnimationFrame`, **pas une lecture de pixel ni une attente de toutes les tuiles**. Le serveur peut commencer les ressources pendant l’ouverture ; les 0,722 s ne signifient donc pas compilation complète en moins d’une seconde. Le gain d’ouverture n’est pas attribué au seul compilateur : caches terrain et charge de la machine interviennent.

Le script ouvre `gpu=1` : le profil effectif du viewer est ici `orogen--g2d683499c4cc36059bc4305c`, qui active les six étapes Orogen GPU, contrairement au profil `orogen` utilisé par le benchmark pur et ses empreintes. Les comparaisons avant/après restent faites sur les mêmes entrées à l’intérieur de chaque protocole. La tentative navigateur de la seed fraîche 20261027 a expiré après **300 s avant ouverture du monde** ; aucune mesure « premier pixel » n’en est déduite. Cela reste une faiblesse de validation du parcours complet sur cette seed, distincte de ses compilations et équivalences d’autorité réussies.

**Objectifs :** premier affichage < 1 s et changement chaud ≤ 0,3 s atteints dans cette mesure seed 42 ; lecture du cache pur autour de 0,22 s, mais **0,83 s sous charge du viewer après redémarrage**, donc l’objectif disque n’est pas acquis pour ce parcours. Ce second essai navigateur ouvre le monde en 19,475 s, affiche NPP en 0,616 s après ouverture, répète les sélections en 0,050–0,067 s, avec zéro erreur JavaScript. **Compilation complète ≤ 1,5 s, et stretch < 0,5 s : non atteints**, y compris dans le processus chaud. La JIT initiale, le GIL autour des objets de gisements, les bruits géologiques, les prospectivités et les EDT restent des coûts à réduire. Le calcul canonique CUDA n’a pas été introduit ; il faudrait une preuve supplémentaire d’identité des classes et budgets avant de l’activer.

### 21.4 Reproduction et canon

Depuis la racine, avec le Python fourni et les variables de threads ci-dessus :

```powershell
$py = 'D:/Workspace/Self/RPG/infinite_map/.venv/Scripts/python.exe'
$env:NUMBA_CACHE_DIR = 'E:/TerrainDiffusionRuntime/numba-cache/resources-perf'
# Les snapshots sont exclusivement produits avec le code épinglé.
& $py tools/benchmarks/benchmark_resources_compile.py --prepare
& $py tools/benchmarks/benchmark_resources_compile.py --reference --snapshot --label reference
& $py tools/benchmarks/benchmark_resources_compile.py --label after
# Ne jamais vider le cache partagé ; isoler un répertoire pour le viewer.
$env:TERRAIN_RESOURCES_CACHE = 'E:/TerrainDiffusionRuntime/resources/perf-viewer-new-run'
$env:TERRAIN_PRELOAD_MODELS = '0'
& $py -u backend/terrain_server.py --port 8776
# Dans un second terminal :
node tools/benchmarks/benchmark_resources_first_pixel.cjs --port 8776 --seed 42 --repeat --json output/resources-perf/viewer.json
```

`--reference` charge avec `git show` les modules canoniques de `e2046f9`, sans écrire dans `.git` ni remplacer les sources du worktree. Pour recréer les entrées sur une autre machine, exécuter `--prepare` depuis cette révision : les digests sémantiques des tableaux d’entrée doivent correspondre ; le SHA du conteneur ZIP n’est qu’une trace de l’archive initiale et dépend de ses métadonnées.

**Identité canonique :** `COMPILER_VERSION = resources-compiler-12` conservé. Tous les tableaux, y compris les flottants et classes, tous les budgets, unités et dictionnaires complets de gisements sont comparés aux empreintes initiales. Les deux mondes sont recompilés deux fois : CDF Numba, puis repli NumPy. Tolérance constatée pour les champs compilés : **zéro octet différent**, plus stricte que 10⁻⁵ relatif. Aucun hasard ni réduction GPU dans l’autorité. Les entrées externes absentes entraînent un skip explicite du test complet ; les tests environnement/étape précoce/cache restent autonomes.

Validation finale : **50 tests Python réussis** (suites `test_terrain_resources.py`, `test_terrain_server_world.py` et nouveau `test_terrain_resource_equivalence.py`), dont job non bloquant et aller-retour cache ; parité NumPy/WGSL **252 cas / 11 configurations, zéro désaccord**, erreur normalisée maximale 2,35·10⁻⁶ sur le stock, classes et mois exacts. Les rapports détaillés et profils sont sous `output/resources-perf/`, la livraison Git sous `output/resources-delivery/resources-perf.bundle` ; aucun push.
