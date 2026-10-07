# Runtime UI — Astra HIGH, revue r3

Date : 2026-10-07. Relecture ciblée de la correction P1 signalée par Opus
sur l'image native initiale, avec non-régression des corrections Astra r1/r2.
Sources en lecture seule ; transport et renderer simulés ; aucun serveur,
aucune inférence ni allocation GPU pour cette revue.

## Verdict : ACCEPT WITH LIMITS

Le P1 de protection de l'image initiale est fermé. Aucun P0/P1/P2 restant
identifié dans le périmètre UI examiné.

`index.html:31` limite la protection au mode relief et à la cible LOD 0,
quand l'image initiale existe. Le plan de rendu reçoit ces bornes.
`terrain_lod.js:6` soustrait le rectangle protégé sans chevauchement des
morceaux restants ; les UV sont recalculées depuis leurs coordonnées monde.
`terrain_lod.js:24` n'applique cette soustraction qu'aux parents de repli :
une tuile exacte LOD 0 peut remplacer l'image initiale.

Le choix d'étape examine la partie visible de chaque cellule. Une cellule
entièrement couverte par l'image ne requiert aucun ancêtre ; une cellule
partiellement découverte conserve le parcours de raffinement nécessaire.
La construction des intérêts dans `index.html:72` emploie la même règle
de couverture initiale. Le viewport totalement protégé passe directement
aux requêtes natives, tout en gardant l'image visible.

## Preuves exécutées

PASS : `test_terrain_lod.cjs`, `test_terrain_freshness.cjs`,
`test_terrain_refinement.cjs`, `verify_progressive_zoom.cjs` et
`verify_navigation_mock.cjs`.

Le test progressif vérifie désormais :

- LOD 4 reçu pendant un rAF retenu, sans départ de LOD 3 avant libération.
- Ordre 4 → 3 → 2 → 1 → 0, avec tous les LOD rendus au plus égaux au
  niveau précédent lors du lancement du suivant.
- Une vraie requête LOD 4 retenue pendant le changement de caméra,
  au moins une annulation observée, puis aucune requête fine obsolète après fit.
- Une caméra entièrement dans l'image initiale : seulement des requêtes
  LOD 0 et aucun parent superposé à la zone protégée.
- Aucune requête LOD 12 et aucune demande supplémentaire à la revisite chaude.

Résultat observé : `nativeImageProtected:true`, `heldRAFBlocksRefinement:true`,
`cancelAborts:1`, `hotRefreshRequests:0`, 60 requêtes au total.

Contrôle indépendant Node : cinq positions du rectangle protégé, dont
intersection partielle, couverture totale et absence d'intersection ; somme
des aires restantes exacte, UV finies et correctement proportionnées. Une
bande découverte d'une unité monde suffit à imposer LOD 4, alors qu'une vue
entièrement protégée demande directement LOD 0. PASS.

Empreintes du test progressif exécuté :

```text
index.html
02d10e6cc495634d2477574362940c7ff8fd278aec10aacd8e5e48c95c8cba8e
terrain_lod.js
6cd853396f5c3480ba54ffddadabe08a9db0f8efc3f759474302b49f99c5d84f
verify_progressive_zoom.cjs
6152f0aa887a91e4070a408465c0ab6671bc94ccf361cd2cda9e0aa39374372e
```

## Limites

Les garanties portent sur le plan et la soumission du dessin client, pas sur
la présentation physique à l'écran. La géographie, la fidélité NN et les
performances GPU ne sont pas mesurées par les tests UI. Le budget grand
viewport reste conforme au test simulé ; cela ne mesure pas la RAM/VRAM
globale. Les deux P2 Astra précédents restent fermés.
