# Vérification du bootstrap terrestre

7 octobre 2026. Les quatre styles utilisent le vrai atlas physique de
`world-builder-rs`, révision `009efe18c457757da12ffc8a6215806efb87f012`, puis une
conversion monotone des distributions terre/profondeur vers ETOPO. Aucun
générateur de ville ni ancien modèle Macro ne participe à cette vérification.
Les hauteurs du NN sont conservées intégralement : aucun masque de côte du
parent n'est appliqué au DEM appris.

## Résultat CPU effectivement mesuré

32 mondes : styles `gondwana`, `continents`, `earthlike`, `archipelago`, seeds
`0, 1, 13, 42, 97, 2026, 1234567, 4294967297`. Chaque raster signé est
1024 × 512, projection équirectangulaire, coordonnées physiques sur
40 000 × 20 000 km. Les statistiques utilisent les aires exactes des bandes
sphériques ; les composantes ont quatre voisins et un raccord de longitude.

| Style | Terre, aire planétaire | Plus grande masse / terre | Masses ≥ 0,1 % de la planète | Composantes totales |
|---|---:|---:|---:|---:|
| Gondwana | 32,90–34,97 % | 88,32–98,79 % | 1–8 | 47–199 |
| Continents | 34,10–34,95 % | 32,22–61,94 % | 6–11 | 27–62 |
| Earthlike | 27,11–28,90 % | 32,49–60,30 % | 6–13 | 46–72 |
| Archipelago | 17,59–17,85 % | 5,14–12,65 % | 41–52 | 286–348 |

Les 32 rasters satisfont aussi le critère morphologique large de leur style,
réévalué indépendamment après reconstruction et conversion en mètres.
L'acceptation du générateur utilise des cellules du cube avec huit voisins ;
elle n'est pas présentée comme identique à cette topologie raster.
Les petites composantes comprennent des îlots et des fragments de raster.
Ces nombres ne garantissent pas un nombre exact de continents.
Le critère de style s'applique à l'aire native/sphérique : dans le domaine plan,
Archipelago seed `4294967297` a une plus grande masse de 16,46 % de la terre,
au-delà du seuil natif de 15 %. Ce cas est conservé comme limite de projection.

Les contrôles portent sur le signe terre/mer brut après conversion, la finitude,
l'immuabilité des arrays, l'identité des cinq champs physiques, le découpage
des requêtes et leur ordre. Le décalage d'une circonférence produit une erreur
de hauteur nulle ; l'écart de part et d'autre du raccord à ± 1 cm reste inférieur
à 0,000489 m. La hauteur physique est transmise sans clipping.

Le rapport distingue le raster brut reconstruit de l'atlas natif et le raster
physique recalibré. Les coordonnées d'ETOPO ne sont pas utilisées comme une
carte de ce monde : seules les distributions conditionnelles d'altitude et de
profondeur, pondérées par l'aire sphérique, sont reprises. Cela impose une
hypsométrie commune aux styles, et ne démontre ni géologie locale réaliste ni
relief appris à 30 m.

Le viewer montre un globe déroulé en coordonnées planes, avec latitude
limitée aux bords du parent. Les cellules polaires occupent davantage d'aire
à l'écran que sur la sphère. Les mesures à poids uniforme sont également
conservées, sans les confondre avec une fraction planétaire :

| Style | Terre sur le domaine plan, poids uniforme |
|---|---:|
| Gondwana | 26,29–41,15 % |
| Continents | 26,75–44,06 % |
| Earthlike | 21,58–37,64 % |
| Archipelago | 15,26–19,98 % |

L'aire au-dessus de 2000 m représente 5,48–5,49 % de la terre dans les quatre
styles, conséquence de la calibration hypsométrique commune. Les régions
négatives classées hors océan physique représentent 0,11–2,38 % de la planète
selon le monde. Cette différence inclut des bassins sous le datum et des
cellules de reconstruction côtière ; le masque d'océan est une classification
native au plus proche, utilisée ici comme diagnostic.

Artefacts finaux CPU v2 : `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json`,
`comparison-s*.png`, `local-candidates.json`, et un `height.npz`/`metrics.json`
par monde. Les reçus contiennent les hashes des sources, de l'exécutable,
d'ETOPO et des deux rasters float32.

## Écart aux statistiques du checkpoint

Les cinq champs physiques sont échantillonnés sur 256 × 128 coordonnées par
monde. L'altitude reçoit exactement une racine signée, puis chaque canal reçoit
les moyennes/écarts-types réellement lus dans le config du checkpoint,
indices `[0,2,3,4,5]`. Natural est comparé aux mêmes coordonnées et seeds.
`conditioning-diagnostic.npz` conserve les valeurs physiques, les axes et les
constantes utilisées.

| Canal | Aire sphérique avec \|z\| > 4, terrestre v2 | Natural |
|---|---:|---:|
| Altitude encodée | 0 % | 0 % |
| Température | 4,40–6,54 % | 0 % |
| Seasonality BIO4 | 2,95–3,36 % | 4,45–4,87 % |
| Précipitation BIO12 | 0,119–0,171 % | 0,181–0,267 % |
| Variabilité BIO15 | 0,434–0,686 % | 0,019–0,063 % |

Le plus grand écart absolu de température vaut 12,55 écarts-types du checkpoint.
Avec des poids uniformes sur le domaine plan, jusqu'à 21,98 % des échantillons
de température dépassent quatre écarts-types ; la projection amplifie ainsi
le poids des régions polaires dans ce diagnostic.
Ce contrôle signale notamment les entrées froides polaires et en haute altitude ;
il ne les corrige pas par clipping. Une normalisation correcte ne garantit
pas que toutes les entrées sont dans la distribution d'apprentissage. Les
sondes NN polaires servent à mesurer cette limite sur les vraies sorties.

## Vérification NN et portée de la preuve

`validate_terrestrial_nn.py` lit les vrais contributeurs du checkpoint pour
chaque point coarse, puis exporte coarse, latents et DEM natif pour les sites
locaux. Les seeds restent des entiers exacts ; les manifestes et les hashes
lient chaque sortie au même parent vérifié par CPU.

Les cartes de sondes sont des points stratifiés, affichés au plus proche. Elles
ne sont ni un raster complet du monde appris, ni des moyennes LOD, ni une base
valide pour compter les continents du NN. Les distances de côte sont limitées
aux côtes visibles dans le crop ; elles valent `null` si une côte est absente.

Les sites sont sélectionnés sur le parent avant toute inférence, sans score
esthétique : côte de rugosité médiane, côte à faible relief, plaine et altitude
élevée médiane. Le nom historique `mountain-median` dans les IDs désigne un
rang d'altitude dans la partie haute du parent ; aucune pente ou crête n'entre
dans cette sélection. Il est conservé dans les reçus existants pour préserver
leur traçabilité, et ne certifie pas une morphologie de montagne.
Pour une côte, une bissection sur le sampler physique centre le crop sur un
vrai passage par zéro. La comparaison distingue le parent directement
échantillonné et l'interpolation du vrai lattice de conditionnement à 7680 m.
Le NN n'est comparé à aucune image colorée et aucun signe changé n'est masqué.

## Bruit d'altitude et déplacement réel des côtes

L'étude de bruit avec le climat terrestre v1, avant sa correction de seed u64,
compare quatre fenêtres réelles de coarse NN, seed 0, une par
style, à quatre valeurs du bruit d'altitude `cond_snr[0]` : 0,5 ; 0,2 ; 0,1 ;
0,05. Les quatre canaux climatiques restent à 0,5, les poids sont inchangés.
Chaque fenêtre couvre 491,52 km sur 64 × 64 cellules de 7680 m. Les distances
ci-dessous comparent les passages par zéro interpolés linéairement sur les
arêtes, dans les mêmes coordonnées physiques, avec le passage appris le plus
proche. Elles ne sont pas une correspondance géologique de segments.

| Style, seed 0 | Médiane parent → côte NN, bruit 0,5 | Médiane, bruit 0,05 | Signe changé, bruit 0,5 → 0,05 |
|---|---:|---:|---:|
| Gondwana | 18,33 km | 3,06 km | 7,1 → 1,0 % |
| Continents | 23,74 km | 4,57 km | 7,2 → 1,1 % |
| Earthlike | 7,33 km | 2,38 km | 2,9 → 0,5 % |
| Archipelago | 6,87 km | 1,83 km | 1,9 → 0,3 % |

La fenêtre large établit un déplacement réel des côtes. L'absence de côte dans
un petit crop de 7,68 km n'est donc pas interprétée comme une disparition de
toute la côte du monde. Diminuer ce paramètre réduit le bruit ajouté à
l'altitude et renforce le contrôle du parent. Même à 0,05 les côtes ne sont pas
verrouillées : la médiane reste kilométrique dans ces coarse patches. Cette
étude motive l'altitude à 0,05 ; la validation finale du climat v2 est distincte
et conserve quatre canaux climatiques à 0,5.

Reçus complets et planches :
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-final-nn/coast-noise-study/coast-noise-report.json`,
`*-noise-board.png`, arrays `.npz` signés et manifestes par variante. Les
manifestes contiennent les cinq valeurs de bruit réellement exécutées.

La comparaison native v1 sur trois fenêtres de 15,36 km confirme une baisse
du signe changé avec 0,05 : Earthlike seed 0, 42,2 → 18,8 % ; Earthlike seed 42,
49,7 → 26,2 % ; Gondwana seed 0, 42,9 → 12,6 %. Ce gain n'est pas uniforme
pour toutes les distances : Earthlike seed 0 a une médiane parent → NN de
1,41 → 1,82 km, tandis que son p95 diminue de 6,25 → 3,36 km et la médiane
inverse NN → parent de 5,51 → 2,68 km. Les sorties marines natives peuvent
rester très proches de zéro ; aucune bathymétrie apprise fidèle n'est promise.

## Relief comparé à empreinte identique

Quatre fenêtres natives de 15,36 km, seed 0, comparent les mêmes coordonnées
à altitude-noise 0,05 / 0,1 / 0,5 et à Natural inchangé : haute altitude médiane
Earthlike, plaine Earthlike, haute altitude médiane Archipelago et Continents. Les trois
variantes terrestres ont 100 % de terre sur les quatre fenêtres. Les mesures
sur leur intersection terrestre, sur la terre propre à chaque variante et
sur le masque fixe du parent sont donc identiques ici ; aucun changement de
masque ne peut expliquer leurs différences de pente.

| Fenêtre | Relief p95−p05, 0,05 / 0,5 | Pente p95, 0,05 / 0,5 | Écart-type du détail, 0,05 / 0,5 |
|---|---:|---:|---:|
| Earthlike haute altitude médiane | 54,1 / 48,8 m | 0,0333 / 0,0527 | 4,20 / 6,11 m |
| Archipelago haute altitude médiane | 446,8 / 549,8 m | 0,4350 / 0,6621 | 40,65 / 66,51 m |
| Continents haute altitude médiane | 150,6 / 125,8 m | 0,2155 / 0,1701 | 19,42 / 14,03 m |
| Earthlike plaine | 55,5 / 104,6 m | 0,0799 / 0,1965 | 4,68 / 9,66 m |

Le détail est la hauteur moins son lissage gaussien de sigma 1000 m, mesuré
sur les pixels terrestres intérieurs. Les pentes utilisent le pas réel de 30 m
et excluent les pixels adjacents à la mer ou au bord du crop. À 0,05, les pentes
p95 des sites élevés Earthlike et Archipelago baissent de 36,8 et 34,3 % par
rapport à 0,5 ; Continents augmente de 26,7 %. Les écarts-types du détail y
retiennent respectivement 68,7 %, 61,1 % et 138,4 %. La pente p95 de la plaine
ne retient que 40,66 %, soit une baisse de 59,34 % ; son détail ne retient que
48,47 %. Le réglage réduit donc matériellement certains détails locaux.
Le site Earthlike élevé est lui-même doux : sa pente p95 à 0,05 est 0,0333 et
son relief p95−p05 seulement 54,1 m sur 15,36 km. Ces échantillons peuvent être
des hauts plateaux ou des versants peu accidentés ; ils ne constituent pas
une certification des crêtes ou massifs, et ne démontrent pas l'absence de
lissage excessif sur tous les terrains fortement accidentés.

À ces coordonnées, Natural n'a que 2,8 % de terre pour Earthlike haute altitude et
aucune terre pour Continents haute altitude ; ces baselines ne sont pas comparables
comme des terrains terrestres. Archipelago haute altitude a 100 % de terre avec
Natural : son relief p95−p05 est 426,2 m, sa pente p95 0,5317 et son détail
43,07 m, contre 446,8 m / 0,4350 / 40,65 m pour le bootstrap à 0,05.

Le rapport conserve les spectres de puissance radiaux du coarse en mètres,
du lowfreq latent converti en mètres, des quatre features latentes normalisées
et du DEM natif. Leur calcul retire un plan, applique une fenêtre Hann et
enregistre les bandes de fréquence et la normalisation. Ces spectres portent
sur le contexte entier, et ne sont pas présentés comme des spectres de terre
lorsque Natural y contient de la mer. Les arrays bruts permettent de refaire
les autres métriques sans nouvelle inférence.

Artefacts comparatifs :
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/matched-relief/relief-comparison.json`
et quatre `*-relief-board.png`. Chaque planche compare les quatre véritables
sorties NN avec palette et lumière fixes. La décision actuelle conserve
l'altitude-noise à 0,05 pour les hauteurs et la navigation avec ces limites
de lissage explicites ; elle ne certifie ni les montagnes de tous les mondes
ni le climat appris comme un moteur physique.

Le script exact utilisé pour les 16 inférences comparatives est conservé sous
`matched-relief/comparer-gpu-source.py`, SHA256
`873747617734dff6a54cf6fd9dbc3dea3f1052aec6cd402499ad19f92b0903a9`, identique
au `comparer_sha256` du reçu. Il a été récupéré à l'identique puis vérifié,
avant les enrichissements CPU de détail et de spectre natif.

## Sélection CPU complémentaire de relief fort et de p99

`select_terrestrial_relief_sites.py` prépare huit sites supplémentaires avant
toute inférence NN : un maximum de relief local par style, seed 0, et un site
de hauteur p99 par style, seed 42. Le relief local est la vraie étendue de
hauteur du sampler physique sur une grille 3 × 3 couvrant 15,36 km ; la
recherche exclut les pôles, exige neuf échantillons terrestres et un centre
au-dessus de 500 m. Elle choisit le maximum mesuré, sans résultat NN ni score
esthétique. La hauteur p99 est le quantile pondéré par l'aire sphérique des
centres non polaires dont la même empreinte reste terrestre.

| Style | Centre du site de relief fort | Étendue locale 3 × 3 | Centre p99 | Étendue locale du p99 |
|---|---:|---:|---:|---:|
| Gondwana | 3635 m | 1584 m | 4718 m | 88 m |
| Continents | 3458 m | 975 m | 4677 m | 94 m |
| Earthlike | 3325 m | 580 m | 4689 m | 500 m |
| Archipelago | 2070 m | 1098 m | 4673 m | 150 m |

Le p99 peut encore être un plateau ; le relief fort peut être un versant
plutôt qu'un massif complet. Le JSON conserve les coordonnées et les règles
exactes : `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/relief-stress-sites.json`.
Les huit empreintes ont ensuite été réellement exécutées sur CUDA à 512 ×
512 pixels, avec les trois bruits d'altitude 0,05/0,1/0,5 : 24 exports complets
sur les sources finales après durcissement des contrôles d'admission. Les 24 DEM sont finis et restent entièrement
terrestres ; les masques de terre propre, parent et intersection des trois
variantes sont donc identiques. La comparaison de relief n'est ici pas
confondue par un changement terre/mer.

Les valeurs ci-dessous sont celles de 0,05 ; les trois dernières colonnes
sont les ratios 0,05 / 0,5. La pente est le p95 des pixels terrestres intérieurs,
en m/m ; le détail est l'écart-type du résidu après filtre gaussien de sigma
1000 m. Il ne décrit pas une fidélité absolue au terrain réel.

| Site | Relief p95−p5 (m) | Pente p95 | Ratio relief | Ratio pente | Ratio détail |
|---|---:|---:|---:|---:|---:|
| Gondwana relief fort | 2085.4 | 0.978 | 1.060 | 0.826 | 0.829 |
| Continents relief fort | 867.2 | 0.657 | 0.614 | 0.843 | 0.753 |
| Earthlike relief fort | 599.2 | 0.443 | 1.112 | 1.029 | 0.830 |
| Archipelago relief fort | 842.8 | 0.416 | 0.757 | 0.716 | 0.415 |
| Gondwana p99 | 412.4 | 0.418 | 0.687 | 0.708 | 0.658 |
| Continents p99 | 158.6 | 0.183 | 0.843 | 0.910 | 0.997 |
| Earthlike p99 | 898.8 | 1.410 | 0.975 | 1.055 | 1.055 |
| Archipelago p99 | 585.8 | 0.641 | 0.760 | 0.788 | 0.716 |

Le bruit 0,05 conserve donc du relief important aux quatre sites réellement
choisis pour leur variation locale. Il réduit toutefois le détail de
l'Archipelago relief fort de 58,5 % sur cette mesure, malgré 843 m de relief
p95−p5 et une pente p95 de 0,416. La perte de détail n'est pas universellement
inférieure à 50 %, et les résultats ne justifient pas de promettre un gain
universel de relief. Le p99 Earthlike conserve même un détail légèrement
supérieur à 0,5, alors que les autres p99 sont plus lissés.

Le JSON conserve les spectres radiaux du height coarse, du lowfreq latent,
des quatre features latentes et du DEM natif, avec plan retiré, fenêtre Hann
et unités explicites. Ce sont les empreintes entières co-enregistrées ; leur
légende ne les appelle pas « land-only ». Les métriques de hauteur et de pente,
elles, utilisent les masques terrestres déclarés. La pluie, BIO4 et BIO15
restent non négatifs sur ces huit nouveaux sites à 0,05 ; cela n'efface pas
les sorties polaires négatives mesurées ailleurs.

Reçu : `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final-hardened-relief-stress/relief-comparison.json`.
Ses 24 manifestes et SHA256 NPZ sont attestés contre les sources et checkpoint
actuels dans `cpu-attestation.json`. Les huit `*-relief-board.png` montrent
les trois sorties véritables à palette et lumière constantes. La planche
`high-local-relief-comparison-overview.png` regroupe les quatre reliefs forts. Le comparer
figé et ses helpers sont sauvés dans `source_snapshot/` ; son SHA256 est
`38c3e8ea2fc987cd6128ea3e820150f4be6b4bc99786c4e1036815553ce2d3e5`.


## Audit final du climat v2 et du NN

Le reçu final utilise le climat v2, son seed complet u64 et le bruit
`[0.05, 0.5, 0.5, 0.5, 0.5]`. Il comprend 21 DEM natifs de 512 × 512 pixels
à 30 m, soit une fenêtre de 15,36 km : deux côtes, une plaine, une haute altitude médiane
et un contexte polaire par style, avec un deuxième contexte polaire Earthlike
seed 42. Les arrays coarse et latents sont également conservés pour chaque
site. Les 21 DEM sont finis ; aucune valeur ne dépasse 20 km en valeur absolue.

Les huit côtes ordinaires restent présentes dans le crop NN. Le signe changé
représente 4,1–25,8 % du crop, et la médiane de distance parent → NN est
0,274–1,927 km. La côte polaire Earthlike seed 42 a 18,2 % de signes changés
et une médiane de 2,291 km. Les distances inverses et leurs queues peuvent
être plus grandes : elles figurent intégralement dans le JSON. Les plaines
et sites élevés ordinaires gardent leur signe terrestre. Les quatre sites de
haute altitude médiane ont des hauteurs NN allant, selon le site, de 1210 à
1732 m ; la planche ombrée montre un relief local souvent doux, sans certifier
la morphologie des massifs du monde.

Huit mondes, quatre styles × seeds 0/42, ont aussi 32 points coarse NN chacun.
Sept ne changent aucun signe parmi ces points ; Continents seed 42 en change
un, soit 3,125 % des points en poids uniforme ou 1,831 % avec les poids
sphériques. Un sondage supplémentaire de 512 points sur Earthlike seed 0
mesure 0,391 % de signes changés en poids uniforme et 0,438 % avec les poids
sphériques. La terre sur ce sondage passe de 27,65 % du parent à 27,21 % du NN
avec les poids sphériques ; cela ne représente pas une carte complète apprise.

Les sorties climatiques NN brutes ne satisfont pas toutes les contraintes
physiques. Quatre des cinq crops polaires ont de la précipitation négative :
100 % des pixels Gondwana seed 0, Earthlike seed 0 et Archipelago seed 0 ;
83,75 % du crop côtier polaire Earthlike seed 42. Les minima sont respectivement
−406, −330, −210 et −66,6 mm/an. Le crop de haute altitude Gondwana seed 42 est lui aussi
entièrement négatif en précipitation. Sur les 512 points coarse Earthlike seed 0,
16,80 % ont une précipitation négative, minimum −472 mm/an, et un point a un
BIO15 négatif, −6,59 %. BIO4 reste non négatif dans ces sondes.

Ces prédictions ne sont ni cachées ni corrigées par clipping. Elles limitent
l'interprétation du climat appris, même lorsque les entrées physiques sont
valides et les hauteurs finies. Aucun seuil esthétique ou seuil de déplacement
côtier n'est enregistré : `numerical_failures: []` signifie seulement que les
contrats numériques ont passé, et ne constitue pas une acceptation physique
du climat ou une garantie de fidélité exacte des côtes.

Reçus finaux :
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final-hardened/neural-report.json`
et `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final-hardened-earthlike-global512/neural-report.json`.
Les NPZ, leurs SHA256, manifestes de checkpoint et planches sont adjacents ;
`high-elevation-quality-board.png` regroupe les quatre crops élevés sans changer
leurs couleurs ni leur exposition. L'enrichissement des mesures planes et des
diagnostics climatiques a relu les mêmes arrays sauvegardés sur CPU.
Les 21 exports de stages ont également passé une vérification des manifestes
contre les sources et le checkpoint actuels, et une comparaison des SHA256
NPZ avec leurs reçus. `final-hardened/cpu-attestation.json` enregistre les huit
identités de monde vérifiées. Les stages finaux ont été réexécutés sur CUDA
après le durcissement final des contrôles de profil, précision et hauteur bootstrap, et du
statut HTTP des erreurs internes d'admission ; ce ne sont pas des
artefacts anciens auxquels on aurait seulement attribué de nouveaux hashes.
Le sondage Earthlike de 512 points a passé les mêmes contrôles actuels de
sources/checkpoint et de SHA256 NPZ. Les 16 exports comparatifs historiques
conservent leur attestation correspondant à la version antérieure du serveur
et leur script GPU exact ; ils ne sont pas présentés comme des reçus de la
version finale durcie. Les copies des harnesses et leurs SHA256 sont sauvées
avant chaque exécution fraîche dans `source_snapshot/`.

Les trois reçus `final-hardened*` ont également été comparés aux reçus
`post-admission-*`, qui restent conservés. Tous les champs numériques de leurs
NPZ sont identiques octet par octet, avec les mêmes formes et dtypes : les 21
exports locaux et leurs comparaisons de source, les huit sondages de 32 points,
le sondage Earthlike de 512 points et les 24 variantes de relief fort/p99.
Les champs de manifeste ont changé avec les sources durcies ; aucune identité
octet par octet des archives NPZ complètes n'est revendiquée. Les métriques de
côte, relief et climat ci-dessus restent donc les mêmes. Les vérifications CPU
des manifestes actuels, SHA256 NPZ et arrays comparés figurent dans chaque
`cpu-attestation.json`, avec huit identités de monde pour le reçu principal,
une pour Earthlike 512 points et 24 pour les trois bruits des huit sites de
stress. Le script d'attestation est conservé dans
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final_hardened_cpu_attest.py`.
Les planches de synthèse ont été recopiées des reçus précédents après cette
vérification des arrays ; les planches individuelles ont été rendues par les
exécutions fraîches.

Le contrôle Natural final traverse la vraie admission `server.get_world(42,
'natural')`, puis échantillonne la tuile LOD2 −14/10 sur CUDA. Il ferme le premier
monde, vide ses caches RAM et ceux du pipeline partagé, et reconstruit un deuxième
monde serveur avec les mêmes poids et les fenêtres coarse persistées.
Les deux paquets de 391 444 octets (DEM float32 304 × 304 et climat float32
5 × 33 × 33) sont identiques au baseline pré-bootstrap et entre eux, erreur
maximale 0. La seconde passe relit quatre fenêtres disque : aucun nouveau forward
coarse ni nouvelle fenêtre réseau ; base et decoder recomputent respectivement
16 et 25 forwards. Le manifeste enregistré est celui du serveur admis, précision
`bf16`, pas l'identité générique d'une exportation de recherche.
Reçu final avec snapshot du harness, compteurs et manifeste :
`E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/natural-server-replay-final.json`.

## Reproduction

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe verify_terrestrial_bootstrap.py --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu
.\.venv\Scripts\python.exe validate_terrestrial_nn.py --cpu-report E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final-hardened --expanded-sites --global-width 8 --seed 0 --seed 42 --pixels 512
.\.venv\Scripts\python.exe validate_terrestrial_nn.py --cpu-report E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final-hardened-earthlike-global512 --skip-local --global-width 32 --style earthlike --seed 0
.\.venv\Scripts\python.exe compare_terrestrial_relief.py --cpu-report E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/matched-relief
.\.venv\Scripts\python.exe select_terrestrial_relief_sites.py --cpu-report E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/relief-stress-sites.json
.\.venv\Scripts\python.exe compare_terrestrial_relief.py --cpu-report E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/final-hardened-relief-stress --sites-json E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/relief-stress-sites.json --skip-natural --pixels 512
.\.venv\Scripts\python.exe verify_natural_runtime_reference.py --output E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/natural-server-replay-final.json
```

La commande NN demande un créneau GPU exclusif. `--plan-only` prépare les sites
et le rapport sans charger le GPU. `--style`, `--site-id`, `--pixels` et
`--global-width` permettent un sous-ensemble explicite. Aucun poids de modèle
ou code upstream n'est modifié par ces programmes.
