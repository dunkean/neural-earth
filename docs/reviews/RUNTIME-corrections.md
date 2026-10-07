# Runtime — décisions après review

Quatre blocs : navigation, inférence NN, initialisation continentale, cache
exact des statistiques. Reviews Astra high et Opus 5.5 par CLI conservées dans
ce dossier. Les résultats antérieurs restent historiques avec leurs sources.

Reviews finales : [Astra high r2](RUNTIME-FINAL-astra-r2.md) et
[Opus 5.5 CLI r1](RUNTIME-FINAL-opus-r1.md), ACCEPT WITH LIMITATIONS, sans
P0/P1. Opus autorise le défaut expérimental avec gates de documentation :
les corrections F1–F3 ci-dessous ont été appliquées après cette lecture.
Le smoke réel sans `profile` sélectionne A4 puis revient à natural, PASS.

| Dernière review Opus | Traitement final |
|---|---|
| F1 / F3 : résumé et chronomètres obsolètes | Bilan runtime actualisé avec tous les reçus finaux, premier aperçu froid distinct du second LOD 11, overlap des tests CPU et attente en file sans attribution causale. Côte polaire défavorable publiée. |
| F2 : défaut non enregistré | Décision A4 expérimentale inscrite dans le suivi, la validation et le document A4. Sections B1–B5 conservées comme snapshots historiques. |
| F4 : ratio ETOPO trop précis | Inventaires non appariés et sous-échantillonnage expliqués ; facteur observé sans prétendre au ratio des distributions planétaires. « Complet » signifie les records échantillonnés dédupliqués. |
| F5 : tests de coordonnées absents | Deux tests CPU PASS : axes et centres des paires, demi-pixel et orientation nord/sud du vrai raster synthétique. Aucun code de diagnostic modifié. |
| F6 : ancien comparer non archivé | Source historique non reconstructible déclarée ; chaînes report/NPZ/metrics/manifest conservées, nouveau comparer et replay final liés séparément. |
| Scope du cas polaire | Supplément CPU `polar-scope.json` : 22,95 % / 21,92 % des paires échantillonnées au-delà de 60°. Ce n'est pas un taux d'échec NN. |

Les demandes de préparation NN mondiale complète, de timing prolongé LOD 2
et de bornes climatiques indépendantes de la seed restent des gates pour ces
revendications futures. Elles ne sont pas présentées comme réalisées par ce
défaut d'initialisation expérimental.

| Bloc | Retour retenu | Correction / décision |
|---|---|---|
| Navigation | Ancêtres redondants sur une vue partiellement détaillée | Couverture par cellule ; une tuile plus fine déjà présentée évite ses ancêtres. |
| Navigation | Réception avant présentation | Barrière après soumission au renderer ; test rAF volontairement suspendu. |
| Navigation | Ancêtres masquant l'image native initiale | Soustraction des rectangles de protection et ajustement des UV. |
| Navigation | Pan annulant du travail natif utile | Conservation des calculs déjà admis qui intersectent encore la vue ; les nouveaux calculs fins attendent la barrière. |
| Navigation | Tests de pan incomplets, Opus r3 P2 | Assertions de queue au niveau courant, annulation hors vue, étape 3 retenue, aucun nouveau calcul fin, aucun redémarrage des requêtes conservées, stabilisation sans nouvel abort ; tests VM et Chrome passent. |
| NN | Harness tolérant malgré une revendication d'identité | Option `--require-exact` : valeurs et octets ; tests incluant zéro signé. |
| NN | Provenance et restauration de fonctions | Hash du NPZ de référence et des modules importés ; restauration du même objet fonction. Supplément CPU des reçus historiques, sans réécrire leurs preuves. |
| NN | Branche NaN/Inf non exercée | Tests de préparation sur CPU, batches 1/4/9/16, entrée initiale et fusionnée ; paramètres et calcul runtime inchangés. |
| NN | Opus r2 : prédiction stub nulle | Stub non nul, comparaison au helper upstream et mutation retirant le terme de prédiction : le test rejette cette mutation. |
| NN | Opus r2 : échec de benchmark mal enregistré | Reçus atomiques running/failed/complete, échantillon fautif conservé, gates explicites vérifiés sous `-O`. |
| NN | Opus r2 : hashes importés et supplément ambigu | Ajout des sources app/macro/world/manifest et upstream. Supplément distingue 92 comparaisons indépendantes de 44 auto-comparaisons ; aucune exécution de tests inventée. L'archive du harness intermédiaire des paires manque et cette limite est déclarée. |
| NN | LOD2 variable et GPU du bureau actif | Résultats favorables et défavorables conservés ; gain apparié indicatif, aucune garantie de latence stable. |
| NN | Cache par `id(t)` | Pas de collision reproduite : les closures retiennent `t`, le cache est vidé au rebuild. Durcissement non nécessaire à cette livraison. |
| A4 | Opus v1 P1 : bande côtière presque plate | V2 utilise une pente non nulle, une rampe terrestre adaptée à l'altitude et une pente sous-marine plus marquée. Rejeu des quatre coordonnées originales. |
| A4 | Sommets écrêtés en plateaux | Saturation exponentielle douce au-dessus de 70 % de la borne source. |
| A4 | Sonde sparse présentée comme topologie mondiale | Retrait des aires et composantes de cette sonde ; 512 points appris, explicitement distincts d'un monde complet. |
| A4 | Entrée analytique 30 m différente de l'entrée NN | Planches séparées : analytique, lattice de conditionnement 7,68 km interpolée pour affichage, DEM appris 30 m. |
| A4 | Guard de sources incomplet | Vérification de tous les fichiers d'entrée et de visualisation inscrits dans le reçu CPU. |
| A4 | Topologie testée seulement sur sa calibration | Contrôle à résolution et positions distinctes ; petites îles résiduelles documentées. |
| A4 | Climat trop certifié | Support naturel dépendant de la seed déclaré heuristique ; diagnostics de clipping par latitude et altitude. Pas de certification des bornes d'entraînement inconnues. |
| A4 | Opus r2 : seulement des côtes favorables | Inventaire échantillonné de 14 983 / 14 461 paires sur lattice réelle, cas minimum/médiane par seed exportés au NN. Minimum de saut axial, pas de norme du gradient. Environ 1 % des paires échantillonnées restent très peu marquées. |
| A4 | Opus r2 : distributions et compression | Sauts côtiers médians 183/189 m contre 60 m ETOPO rééchantillonné ; plaines basses rares, compression des crêtes et gradients documentés. Limites de réalisme, sans modification de formule v2 ni certification des marginals terrestres. |
| Statistiques | Éviter le recalcul après redémarrage | Cache exact, identité par contenu, intégrité, écriture atomique et fallback ; neuf tests GeoTIFF CPU. |
| Statistiques | Opus r2 : ordre des sources et dépendance GDAL | Schema 2 lie aussi l'ordre des rasters et GDAL. Tests d'invalidation et d'échec aux quatre étapes d'écriture ; nouvelle identité terrain documentée. |
| Attribution visuelle | Astra : A4 étiqueté naturel dans le cartouche | Nom du profil actif lorsque la latitude n'est pas affichée ; recapture navigateur corrigée conservée séparément. |
| Provenance navigateur | Astra r2 : conversion des floats entiers par JSON.stringify | Capture supplémentaire de la réponse `/api/world` brute avec SHA ; reconstruction Python canonique réussie. Les anciens reçus ne prétendent pas disposer de cette réponse brute. |

Limites conservées : coût des quatre transferts utiles lors d'un pan continu,
échec permanent bloquant une étape, GPU du bureau non contrôlé pour les timings,
identité numérique vérifiée sur la configuration RTX3090 actuelle. La planète
NN complète demande une préparation distincte ; le premier aperçu mondial
reste son conditionnement. Le moteur Rust n'est pas lié : seule une adaptation
portable de ses formules de côte est utilisée. Aucun post-traitement procédural
n'est appliqué au DEM appris.
