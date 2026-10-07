# Bootstrap natif World Builder

Implémentation du 2026-10-07 : `terrain_bootstrap.py` et `native/terrain_bootstrap/`.
Source géographique : `World::generate` du dépôt MIT
[`dunkean/world-builder-rs`](https://github.com/dunkean/world-builder-rs), commit
`009efe18c457757da12ffc8a6215806efb87f012`, version
`prototype-0.11.0-hybrid`. Voir l'étude des fonctions et limites dans
`TERRAIN_BOOTSTRAP_WORLD_BUILDER_STUDY.md`.

Le bridge utilise uniquement les altitudes physiques `world.cells[*].height_m`
de l'atlas CPU. Il n'appelle ni `World::sample`, ni le viewer, ni le moteur CUDA
de l'upstream, ni les recettes historiques de `terrain_macro.py`, ni un générateur
de ville. Le détail NN et le LOD restent dans leurs chemins runtime existants.

## Géographie et sélection déterministe

Le rayon est 6 371 000 m, la gravité 9.81 m/s² et la résolution 256 cellules par
face, soit 393 216 cellules. L'érosion est désactivée dans cette première source
coarse ; les côtes et le relief initial viennent des plaques et du fBm sphérique
de World Builder. Les paramètres du sampler fin sont mis à zéro.

| Style | Plaques | Échelle | Fragmentation | Fraction océan demandée | Critère natif sur les terres positives |
| --- | ---: | ---: | ---: | ---: | --- |
| gondwana | 3 | 0.35 | 0.12 | 0.65 | plus grosse masse ≥85 % des terres |
| continents | 14 | 0.85 | 0.85 | 0.65 | plus grosse masse ≤65 %, au moins 3 masses >5 % des terres chacune |
| earthlike | 14 | 1.0 | 0.8 | 0.71 | plus grosse masse ≤75 %, au moins 2 masses >5 % |
| archipelago | 28 | 3.0 | 1.6 | 0.82 | plus grosse masse ≤15 %, au moins 100 composantes |

Les critères utilisent huit voisins du cube-sphère et les aires sphériques des
cellules. Ils concernent `height_m > 0`, pas le masque hydrologique de l'upstream.
La première tentative utilise exactement la seed u64 demandée, zéro compris.
Au besoin, 11 seeds de remplacement sont dérivées par SplitMix64, uniquement
pour sélectionner une sortie du générateur upstream qui respecte le style.
Ce mixage ne génère aucun terrain. Si aucun candidat ne convient, le bridge
échoue et indique les 12 topologies ; il ne remplace pas silencieusement le style.

`requested_seed_u64`, `selected_seed_u64`, `selected_attempt`, `attempts`,
`native_config`, `canonical_height_topology` et `native_world_id` sont persistés.
Une seed demandée différente du seed de disposition est donc explicite et
vérifiable. La sélection est un contrat de géométrie coarse, pas une reproduction
de la géographie terrestre historique. La seed zéro avec l'ancien preset
continents 8/0.75/0.5 a échoué sur les 12 candidats ; ce preset a été remplacé.

## Altitude, projection et lecture

Le raster stocké est float32 1024×512, équirectangulaire, aux centres des pixels.
Longitude `atan2(z,x)`, axe Y nord→sud. Reconstruction bilinéaire entre centres
de cellules natives, avec voisins inter-face. Une bande d'une cellule combine
continûment les reconstructions adjacentes : un simple changement de face
dominante produisait une petite discontinuité, désormais couverte par les tests
natifs d'arêtes et de coins.

ETOPO fournit uniquement les distributions conditionnelles mesurées d'altitudes
terrestres positives et de profondeurs positives. Aucun emplacement géographique
ETOPO n'est copié. Les quantiles de référence et du raster procédural sont
pondérés par aire sphérique. Deux fonctions linéaires par morceaux, monotones,
font correspondre 257 quantiles ; zéro reste zéro, rang et signe sont conservés.
Les nœuds bruts et les nœuds cibles, unités et SHA256 ETOPO sont dans la receipt.
Cette adaptation n'ajoute pas de plateau continental physique ni de modèle
géologique d'hypsométrie ; elle adapte les mètres au conditionnement terrestre.

La convention finale est **mer si altitude <0**. Les bassins intérieurs négatifs
de World Builder sont donc aussi négatifs dans ce raster. `physical_ocean` garde
en parallèle le masque hydrologique par cellule native la plus proche, à titre
diagnostique ; il n'influence pas l'interprétation du conditionnement NN.

Les lectures prennent deux axes en mètres sur le domaine
`[-20 000 km,+20 000 km] × [-10 000 km,+10 000 km]`, Y négatif au nord.
La longitude est périodique ; les lectures au-delà des pôles sont bornées au
premier/dernier rang. Les mêmes coordonnées interrogent le même raster persisté,
indépendamment de l'ordre, du découpage, de la caméra et du LOD. Elles ne
réensemencent pas un bruit par requête.

## Source, build et cache

Prérequis : checkout voisin `../world-builder-rs` au commit indiqué, Rust/Cargo,
NumPy/Rasterio et `terrain-diffusion/data/global/etopo_10m.tif`. Le bridge dépend
du crate local par chemin ; il ne contient pas une copie vendue des sources.
Leurs droits restent définis par le `LICENSE` MIT du checkout upstream.

Le premier accès vérifie le commit, l'absence de modifications suivies ou non
suivies dans les sources utilisées, et les SHA256 récursifs des sources natives,
des manifests et du lockfile bridge. Si nécessaire, il compile en release avec
`cargo build --locked`, cible explicite `native/terrain_bootstrap/target`.
La receipt locale contient la version Cargo, le SHA256 des sources et du binaire.
Avant chaque nouvelle génération, les sources natives, le Python importé et
ETOPO sont revérifiés contre l'identité capturée. L'exécutable est copié dans
un répertoire privé, puis sa copie est contrôlée par SHA256 avant lancement.
Une recompilation concurrente du binaire partagé ne peut donc pas changer le
générateur utilisé après cette vérification. Un changement de source/binaire
durant la vie du processus demande explicitement son redémarrage.
`verify_implementation_identity(expected=None)` effectue ces contrôles sans
cache ni recompilation et vérifie aussi le binaire partagé. Son argument peut
être l'identité du manifest ; sans argument, il utilise l'identité déjà
initialisée. Le vérificateur ne masque donc pas une mutation de fichiers
derrière le LRU d'`implementation_identity()`.
La génération et les tests mentionnés ci-dessous sont CPU uniquement.

La namespace cache inclut aussi le source Python, la version NumPy, ETOPO, les
deux résolutions, les bornes, la projection/convention et le seed/style demandés.
Le cache se trouve par défaut sous
`E:/TerrainDiffusionRuntime/heightmap-bootstrap/world-builder-heightmap-v2/`
sur Windows ; `TERRAIN_BOOTSTRAP_CACHE` permet de changer sa racine.
Chaque NPZ contient les altitudes brutes/finales, le masque physique et la
receipt complète. Toute forme, dtype, namespace, clé ou empreinte incohérente
invalide la lecture. Un ZIP tronqué invalide aussi la lecture. Les écritures
emploient un remplacement atomique après flush/fsync. Les tableaux chargés
sont exposés en lecture seule.

Les anciens caches/custom worlds ne sont pas réutilisés : version, provenance
et données diffèrent. Les sources upstream et ETOPO restent nécessaires pour
vérifier l'identité, même si un NPZ existe. Une machine sans checkout voisin
doit installer ce prérequis avant de générer ou vérifier le bootstrap.

## Vérification réellement exécutée

```powershell
C:/Users/grego/.cargo/bin/cargo.exe test --release --locked --manifest-path native/terrain_bootstrap/Cargo.toml
.venv/Scripts/python.exe -W error::ResourceWarning -m unittest test_terrain_bootstrap_native -v
```

Deux tests Rust réussis : continuité des douze arêtes et huit coins sur un atlas
à valeurs volontairement dissemblables ; conservation du champ constant et
des centres intérieurs. Neuf tests Python réussis sur le vrai binaire : les
quatre styles seed0, provenance, mètres/signe/rang, distributions pondérées,
lecture aux centres, wrap longitude et clamp polaire, ordre/découpage des
requêtes, intégrité/relecture/corruption de cache, entrées invalides et rejet
des changements de source/exécutable pendant la vie du processus. Une
génération réelle seed `18446744073709551615` avec Rayon 1 puis 4 threads utilise
les dimensions de production : atlas natif 256²×6 cellules, raster 1024×512.
Les deux exports donnent exactement les mêmes 2 097 152 octets d'altitude et
524 288 octets de masque océan, la même sélection de candidats, le même ID natif
et la même configuration physique. Le test ne se limite plus à un atlas/raster
réduit. Après cette extension, les neuf tests Python ont de nouveau réussi en
7.647 s sur cette machine. Aucun source de génération Python/Rust n'a changé
pour cette extension ; les identités et octets des heightmaps restent stables.
Les durées de génération sont diagnostiques et exclues de l'identité demandée.

La couverture multi-seed et les cartes comparatives sont produites séparément
par `verify_terrestrial_bootstrap.py`, sans inference NN dans son mode CPU.
