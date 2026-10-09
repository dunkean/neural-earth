# Neural Earth — conception du moteur de régions

Cette note conserve les orientations prospectives en français. Le runtime livré est décrit dans [architecture.md](../architecture.md). Les audits, benchmarks, reviews et comptes rendus historiques sont archivés hors du dépôt.

## Unité de calcul

Le moteur doit traiter une région comme un graphe partagé de dépendances coarse, latentes et décodées. Plusieurs caméras demandant le même terrain doivent réutiliser ce travail. Les données physiques et leur provenance doivent rester indépendantes de l'apparence.

La caméra publie un besoin spatial, un niveau de détail et un epoch. L'ordonnanceur donne la priorité à la couverture visible, puis à son raffinement, puis aux voisins. Les parents restent affichés jusqu'à l'arrivée de leurs remplaçants. L'annulation retire les travaux devenus inutiles aux frontières de calcul prévues.

## Portabilité envisagée

Une future voie d'inférence ONNX Runtime Web/WebGPU pourrait déplacer les réseaux dans un worker du navigateur. Elle doit porter le bruit déterministe, les dépendances spatiales, le solveur, le post-traitement et les caches, au-delà du seul export des U-Net. Le prototype dans `webgpu/` reste une étude de faisabilité.

Un agent natif optionnel pourrait utiliser ONNX Runtime CUDA comme référence de portage, puis TensorRT si les mesures justifient cette dépendance. Le runtime public actuel reste Python/PyTorch CUDA ; ces propositions ne décrivent pas une compatibilité déjà livrée.

## Contrats à préserver

- Même seed, mêmes coordonnées et paramètres : champ reproductible dans le contrat numérique validé.
- Pas de réinitialisation aléatoire par requête ou par caméra.
- Séparation entre résolution physique de la source et échantillonnage affiché.
- Conservation des unités des cinq champs de conditionnement.
- Identités distinctes pour les sources, les poids, les réglages physiques et l'apparence.
- Vérification des halos, des frontières de fenêtres, du méridien et des régions polaires.
- Référence CPU indépendante pour les palettes, matières et méthodes portées.

## Pistes scientifiques

L'hydrologie cohérente à plusieurs échelles, la conservation contrôlée des structures d'érosion dans le raffinement neuronal et l'adaptation aux climats polaires restent des pistes de recherche. Une palette de biome ou une neige visuelle ne suffisent pas à valider ces propriétés.

Le détail sous 30 m doit être évalué comme un raffinement expérimental. Il ne doit pas être présenté comme une résolution apprise ou mesurée supérieure à celle du checkpoint.

## Validation de futurs ports

Comparer les régions identiques, les mêmes fenêtres et leurs recouvrements. Mesurer séparément le démarrage, la première région, les régions avec dépendances réutilisées, les transferts, la mémoire et le rendu. Une moyenne de débit d'inférence ne remplace pas une mesure de latence de navigation complète.

Références : [InfiniteDiffusion](https://arxiv.org/abs/2512.08309), [upstream](https://github.com/xandergos/terrain-diffusion), [Orogen](https://github.com/raguilar011095/planet_heightmap_generation).
