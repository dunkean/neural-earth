# Réglages interactifs de génération

Le panneau **Paramètres de génération** permet de comparer les sources de
conditionnement du NN sur la même région. Les modifications restent un
brouillon jusqu'à **Appliquer paramètres**. L'application conserve la caméra ;
un changement de seed conserve les réglages appliqués. Les paramètres appliqués
figurent dans le lien de la page et sont reproductibles après redémarrage.

Pour comparer : appliquer A, cliquer **Mémoriser A**, modifier et appliquer B,
puis utiliser **Afficher A / Afficher B**. **Rétablir le profil par défaut**
restaure les paramètres du profil sélectionné. Le préréglage **Naturel · référence**
sélectionne le chemin original du checkpoint. L'indication d'étage affiché
distingue les entrées, le coarse appris, les latents et le DEM.

| Réglage | Effet et plage |
| --- | --- |
| Altitude | Natural original, atlas terrestre natif, ou Natural continental expérimental |
| Climat | Natural ou climat terrestre par latitude, indépendamment de l'altitude |
| Continents | Gondwana, continents séparés, terrestre, archipel ; utilisé par les sources terrestres |
| Force continentale | Tout nombre fini ; à zéro le signal d'altitude Natural est conservé exactement ; les valeurs hors de 0–1 extrapolent le mélange |
| Rayon de lissage | Écart type σ du filtre gaussien, tout nombre fini ≥ 0 km ; ce n'est pas un diamètre de continent |
| Fréquence, cinq canaux | Tout multiplicateur fini, sans plafond ; zéro produit un bruit constant et un signe négatif inverse les coordonnées du bruit |
| Octaves, cinq canaux | Tout entier ≥ 1, sans plafond ; le changement ne recalcule pas les tables de quantiles du bruit |
| Bruit permis (SNR), cinq canaux | Toute amplitude relative finie > 0, sans plafond ; réduire ancre davantage le réseau aux entrées |
| Biais terre / mer | 0–1 ; poids océanique égal à 1−valeur dans la distribution ETOPO ; 1 conserve seulement les terres |

Les commandes inactives sont désactivées : fréquence et octaves d'altitude et
biais terre/mer pour l'atlas natif ; force et lissage hors du mode continental.
Les valeurs sont conservées si la source est changée. Le biais 0,5 reprend
exactement la table de quantiles historique ; les autres valeurs construisent
une distribution pondérée déterministe à partir d'ETOPO, entre −60° et +60°.

L'interface est en anglais. Tous les réglages numériques se saisissent librement
avec le point décimal (`0.125`) ou la notation scientifique (`1e-6`), même si le
navigateur est configuré en français. La validation de l'API conserve les mêmes
contraintes mathématiques que l'interface, sans les anciennes plages arbitraires.

## Mode continental expérimental

Le prototype remplace doucement la composante à grande échelle du signal
Natural dans l'espace signé `u = sign(h) × sqrt(abs(h))` attendu par le NN :

```text
u_modifié = u_natural + force × [Gaussσ(u_atlas) − Gaussσ(u_natural)]
h_modifié = sign(u_modifié) × u_modifié²
```

La composante macro est calculée sur une grille mondiale fixe 1 024 × 512,
avec longitude périodique et latitude bornée. Le signal Natural régional reste
présent. Le climat Natural est finalisé après le changement d'altitude, avec
son gradient thermique ; le climat terrestre utilise aussi l'altitude finale.
L'encodage signé racine carrée est appliqué une seule fois à l'entrée du NN.

Ce contrôle est destiné aux expériences décrites dans
[l'enquête sur la géomorphologie](NN_GEOMORPHOLOGY_INVESTIGATION.md).
Il ne prouve pas que le modèle sait conserver ce signal sous une nouvelle
continentalité. Le NN peut déplacer les côtes. Il n'y a pas de masque après
inférence pour les remettre dans la silhouette de l'atlas. Les poids, le solveur,
les fenêtres et les batches NN sont ceux du runtime existant.

## API et isolation

`GET /api/generation/schema` fournit les valeurs par défaut et les bornes.
`GET /api/world?seed=42&world_profile=natural&generation=<JSON encodé>` valide
les réglages et renvoie `generation_settings`, `generation_profile` et
`world_identity`. Les réglages peuvent être partiels ; les valeurs manquantes
reprennent les valeurs par défaut du profil de base.

Une configuration personnalisée porte le nom `<profil>--g<24 hex>`, calculé
sur le JSON canonique versionné. Un reçu atomique dans
`E:/TerrainDiffusionRuntime/generation-settings` permet de résoudre ce nom après
redémarrage. Le nom peut ensuite être utilisé comme `world_profile` dans les
requêtes de tuiles, caméra, aperçu et préparation. Le profil sémantique reste
disponible séparément dans les métadonnées. Un reçu incohérent est refusé.

Les réglages canoniques figurent dans le manifeste. Les fenêtres coarse,
heightmaps physiques, images et jobs utilisent l'identité personnalisée.
Deux configurations d'une même seed ne partagent pas leurs résultats.
Les paramètres exactement égaux aux valeurs par défaut reprennent le profil
de base, sans créer de reçu ni de dépendance native pour Natural.
Les reçus de réglages n'ont pas de quota global ; les quotas de caches du
runtime existant restent applicables.

## Vérification

Les tests CPU couvrent la validation, la persistance, la falsification, le
recouvrement de fenêtres, les champs physiques et l'isolation de l'API et des
caches. Les cinq canaux d'entrée upstream restent identiques octet par octet
lorsque seul le SNR est changé, et lorsque la force continentale est zéro.
Le navigateur est testé avec un transport simulé pour l'application, A/B,
la conservation de caméra, les liens et les espaces de requêtes.

Les scripts `verify_natural_runtime_reference.py` et
`verify_generation_runtime.py` effectuent séparément des vérifications CUDA
bornées du chemin réel du serveur, serveur HTTP arrêté. Les reçus datés se
trouvent sous `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/`.
Ces vérifications ne constituent pas une mesure de qualité visuelle mondiale
ni une certification de latence.

Validation de cette implémentation, le 7 octobre 2026 : **121 tests Python
passent**, ainsi que les vérifications Node de LOD, raffinement et fraîcheur.
Le replay CUDA Natural de la région sauvegardée est identique octet par octet
pour les 391 444 octets d'altitude et de climat, y compris après recréation du
monde. Deux variantes personnalisées passent l'admission réelle et produisent
des champs finis : bruit d'altitude 0,2 au LOD 3, et Natural continental force
0,8 au LOD 2 avec coarse, base et décodeur effectivement exécutés.

Le navigateur réel, avec WebGPU et le serveur local, vérifie l'application,
A/B, le maintien de caméra, le rechargement, la remise à zéro et les requêtes
de heightmaps personnalisées, sans erreur JavaScript. Les reçus sont
`generation-controls-natural-reference.json`, `generation-controls-runtime.json`,
`generation-controls-browser.json` (transport simulé) et
`generation-controls-live-browser.json` (serveur réel).
