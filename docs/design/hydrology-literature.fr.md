# Hydrologie multi-échelle : revue ciblée de la littérature et des projets

Date : 21 octobre 2026. Cette revue accompagne la [conception de la stack](multiscale-hydrology.md). Elle sélectionne les méthodes utiles à Neural Earth ; elle ne constitue ni une revue systématique exhaustive, ni une reproduction des résultats publiés. Aucun chiffre de performance externe n'est présenté comme une mesure de notre intégration.

Le problème est spécifique : le monde est fini, sphérique ou plan, mais son relief neural détaillé arrive à la demande. Il faut préserver les relations amont-aval sans connaître tout le DEM fin. Les méthodes qui supposent un DEM global déjà disponible et celles qui construisent tout le terrain autour des rivières apportent des briques ; aucune ne s'intègre telle quelle.

## Drainage, bassins et continuité spatiale

| Référence primaire | Apport | Décision pour Neural Earth |
| --- | --- | --- |
| [Barnes, Lehman et Mulla, 2014 — Priority-Flood](https://arxiv.org/abs/1511.04463), *Computers & Geosciences*, DOI 10.1016/j.cageo.2013.04.024 | Inondation ordonnée depuis des exutoires, traitement des dépressions, variantes pour directions et bassins. File simple à l'intérieur des dépressions. | Retenir un noyau déterministe compilé avec Numba, avec ordre total des égalités. Conserver le sol brut séparément de la surface utilisée pour router. Le remplissage ne signifie pas que toutes les cuvettes deviennent des lacs. |
| [Barnes, 2016 — Parallel Priority-Flood](https://arxiv.org/html/1606.06204v2), DOI 10.1016/j.cageo.2016.07.001 | Chaque domaine produit des labels et seuils ; un graphe plus large détermine ensuite la solution commune. | Retenir le principe des conditions aux limites partagées. Le papier ne démontre pas qu'un halo suffit, ni que l'on peut ignorer les domaines inconnus. Notre solution hiérarchique sera une adaptation contrainte par les parents, et non le drainage exact d'une planète entière décodée à 30 m. |
| [Eilander et al., 2021 — Iterative Hydrography Upscaling](https://hess.copernicus.org/articles/25/5287/2021/), DOI 10.5194/hess-25-5287-2021 | Changement d'échelle d'une hydrographie connue, avec préservation des connexions et paramètres de chenal. | Retenir la connectivité et les grandeurs sous-maille comme contrats de LOD. Écarter l'idée que l'IHU puisse inventer notre réseau fin : il procède du fin vers le grossier. |

Conséquence de conception : calculer une autorité globale bon marché sur la source finale, puis des régions canoniques avec des traversées de frontière partagées. Une tuile d'affichage ne peut pas décider seule de son débit entrant, de son exutoire ou du niveau d'un lac. Des zones jamais visitées ne doivent pas apporter moins d'eau que les zones déjà chargées.

## Lacs, dépressions et endoréisme

| Référence primaire | Apport | Décision pour Neural Earth |
| --- | --- | --- |
| [Barnes, Callaghan et Wickert, 2020 — Depression Hierarchy](https://esurf.copernicus.org/articles/8/431/2020/), DOI 10.5194/esurf-8-431-2020 | Représentation des cuvettes emboîtées, de leurs seuils et de leurs connexions. | Retenir une hiérarchie explicite. Éviter de confondre « zone sous une altitude » et lac connecté. |
| [Barnes, Callaghan et Wickert, 2021 — Fill–Spill–Merge](https://esurf.copernicus.org/articles/9/105/2021/), DOI 10.5194/esurf-9-105-2021 | Distribution des volumes dans cette hiérarchie, débordement et fusion des dépressions remplies. | Adapter à un bilan annuel puis saisonnier parcimonieux. Ce n'est ni un modèle de pluie-débit complet ni un solveur hydraulique de rivière ; évaporation, infiltration et climat sont des choix supplémentaires de notre modèle. |

Lacs ouverts, lacs fermés et cuvettes sèches sont nécessaires. L'implémentation locale de Magna Urbis constitue une référence pratique, mais son calcul à l'équilibre simplifié et ses itérations fixes ne valent pas preuve de convergence générale. Une seule identité de lac doit contrôler niveau, contour, routage et rendu, y compris quand son emprise traverse plusieurs régions. Une cuvette continentale sous zéro ne devient pas automatiquement de la mer.

## Formes fluviales, relief et LOD

| Référence primaire | Apport | Décision pour Neural Earth |
| --- | --- | --- |
| [Génevaux et al., 2013 — Terrain Generation Using Procedural Models Based on Hydrology](https://perso.liris.cnrs.fr/eric.galin/Articles/2013-river-networks.pdf), DOI 10.1145/2461912.2461996 | Réseau géométrique hiérarchique et terrain composé autour de primitives hydrologiques. | Retenir la séparation entre structure, trajectoire et surface. Remplacer tout le générateur neural par un terrain construit depuis les rivières sortirait de notre objectif. |
| [Cordonnier et al., 2016 — Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion](https://doi.org/10.1111/cgf.12820) | Couplage relief et drainage à grande échelle. | Référence pour la cohérence géomorphologique et le contexte d'Orogen. Ne pas ajouter une seconde simulation tectonique et d'érosion globale après l'inférence. |
| [Peytavie et al., 2019 — Procedural Riverscapes](https://diglib.eg.org/items/555e713d-c93f-4c7c-987e-674f7153d34d), DOI 10.1111/cgf.13814 | Composition du lit et de l'eau sur un terrain, avec formes de rivière et animation procédurale. | Retenir des sections de lit et effets d'eau dépendant de la morphologie. Notre adaptation n'annonce pas l'implémentation complète de leur système d'animation. |
| [Dou et al., 2024 — A unified LOD model for river network and DEM](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1397798/full), DOI 10.3389/fenvs.2024.1397798 | Simplification conjointe de données vectorielles et du relief pour éviter leur désaccord. | Retenir l'exigence d'accord rivière/terrain pendant la simplification ; conserver les renderers actuels, sans introduire leur représentation TIN. |
| [Davis, 2026 — Plate-Local River Generation](https://arxiv.org/html/2610.00301v1), prépublication arXiv:2610.00301 | Calcul régional à la demande et composition d'un profil de chenal descendant avec le terrain. | Retenir l'idée de domaines de calcul bornés. Le réseau ne traverse explicitement pas les plaques : cette restriction exclut une adoption comme architecture des bassins de Neural Earth. L'intégration avec un générateur appris y reste prospective. |

La difficulté qui reste à notre charge est le conflit entre topologie conservée, relief neural libre et incision limitée. Aucun des articles sélectionnés ne garantit simultanément ces trois propriétés dans Neural Earth. Le plan propose des corridors, des profils partagés et une couche physique de lit, avec détection des incompatibilités avant publication. Un essai intégré devra mesurer la qualité réelle de ce compromis.

Pour les méandres, la première solution reprend les offsets continus et contraintes de vallée du Rust local de Magna, avec une phase attachée à un tronçon stable. La migration historique répétée ne sera pas relancée au zoom. La sinuosité doit dépendre de la pente, du confinement et de la taille du cours d'eau ; un bruit ou une spline appliqués uniformément à une trace D8 ne suffisent pas.

## Climat, débit, étiage et navigabilité

- **[Oudin et al., 2005, présentation de la méthode par INRAE](https://webgr.inrae.fr/eng/tools/hydrological-models/evapotranspiration-model)** : estimation parcimonieuse de l'évapotranspiration potentielle. Adaptation possible aux données de température et à une convention astronomique explicite. Ce calcul ne remplace pas un bilan de sol ou une calibration de bassin.
- **[Hock, 2003 — Temperature index melt modelling in mountain areas](https://doi.org/10.1016/S0022-1694(03)00257-9)** : justification d'une fonte simplifiée liée aux températures positives. Retenir pour le déphasage saisonnier neige/débit, sans prétendre simuler l'évolution des glaciers.
- **[USGS — mesures indirectes et relation de Manning](https://wwwrcamnl.wr.usgs.gov/sws/fieldmethods/Indirects/rantz.htm)** : estimation d'écoulement sous des hypothèses de régime uniforme. Utile pour une approximation de section mouillée sur des tronçons adaptés ; insuffisant pour remous, lacs, marées ou garanties de navigation.

Neural Earth fournit des statistiques BIO et, avec Orogen, deux demi-années. Elles ne déterminent pas une chronique météorologique. On distinguera donc débit moyen, minimum mensuel climatologique et mois secs des véritables quantiles d'étiage ou périodes de retour, que ces données ne permettent pas d'estimer. La navigabilité sera une classification saisonnière de worldbuilding incluant les obstacles et la continuité d'itinéraire.

## Projets disponibles et stratégie de réutilisation

| Projet | Briques pertinentes | Choix |
| --- | --- | --- |
| **Magna Urbis local**, `../city_generator`, fichiers courants au HEAD `1d245aa` ; sous-module embarqué `08f3e61` | Rust : flood, plats, bilan des lacs, profils, offsets continus, courbes Hermite ; TS : migration et morphologie ; bridge : lits et rendu | Adapter des méthodes ciblées avec provenance GPL, sans recopier l'orchestrateur fini n×n, les exutoires aux bords ou les IDs de cellule comme contrats mondiaux. |
| [PyFlwDir](https://github.com/Deltares/pyflwdir), MIT | Tableaux NumPy/Numba, bassins, accumulation, ordres, vecteurs et IHU | Référence d'implémentation légère ; adaptation de fonctions attribuées si utile. Les conventions d'exutoire doivent être remplacées. |
| [RichDEM](https://github.com/r-barnes/richdem), GPL-3.0, et [code FillSpillMerge](https://github.com/r-barnes/Barnes2020-FillSpillMerge) | Références de Barnes pour drainage et dépressions | Référence algorithmique et petits cas de comparaison ; éviter d'imposer une nouvelle compilation C++ au runtime. Vérifier la licence précise de chaque fichier repris. |
| [pysheds](https://github.com/pysheds/pysheds), GPL-3.0 | Drainage et délimitation de bassins | Référence indépendante pour quelques cas finis ; pas besoin d'installer une seconde pile de production. |
| [meanderpy](https://github.com/zsylvester/meanderpy), Apache-2.0 | Modèle cinématique de migration fondé sur les courbures amont, méandres et recoupements | Source de comparaison morphologique. Son solveur itératif et ses dépendances de visualisation ne vont pas dans le chemin interactif. |
| [mapgen4 / Red Blob Games](https://github.com/redblobgames/mapgen4), Apache-2.0 | Organisation compacte de graphes, maillages et rendu procédural | Inspirations de structure et de lisibilité. La construction du relief n'est pas celle de notre runtime neural. |
| [Plate-local river generation](https://github.com/Davismk3/plate-local-river-generation) | Trois démonstrateurs C++17 et leur prépublication | Étudier le calcul borné ; pas d'intégration directe ni d'attribution automatique d'une licence d'après celle du papier. |

Toute reprise effective enregistrera le commit, les notices, les fonctions concernées et leurs modifications. L'existence d'un projet ne démontre pas sa performance sur nos mondes, sa compatibilité avec nos coutures ou la qualité de nos lacs.

## Synthèse retenue

La stack est une hybridation spécifique : drainage global et hiérarchie de dépressions sur la source finale ; contrats spatiaux hérités ; affinement du thalweg sur les champs neuraux canoniques ; géométrie continue inspirée de Magna ; terrain et eau conditionnés ensemble ; bilan climatique conservatif et propriétés dérivées. Les garanties portent sur ce modèle construit. Le drainage exact d'un DEM neural global inconnu n'est pas revendiqué.

La revue des méthodes, la lecture des sources locales et les critiques Opus sont trois formes de preuve distinctes. Le modèle théorique est documenté ; la qualité visuelle, le coût des dépendances neurales et la fréquence des conflits de relief devront être établis pendant l'implémentation autorisée.
