# Runtime UI — Astra HIGH, revue r4

Date : 2026-10-07. Relecture du P1 Opus concernant le déplacement pendant
une requête native, puis non-régression des barrières progressives et de
l'image native initiale. Sources en lecture seule ; aucun serveur ni travail
GPU réel lancé par cette revue.

## Verdict : ACCEPT WITH LIMITS

P1 fermé. Aucun P0/P1/P2 supplémentaire identifié dans les changements
examinés. Les limites de rendu simulé et de présentation physique restent
les mêmes que dans r2/r3.

`index.html:78` conserve les intérêts des tuiles cibles déjà présentes ou
en cours, même lorsqu'une bande nouvellement découverte ramène le niveau
global à une étape plus grossière. `index.html:83` conserve aussi les
requêtes intermédiaires plus fines déjà admises : même époque, tuile encore
visible, niveau compatible avec la cible et absence de préchargement.
`fetchTile` stocke désormais la tâche dans le vol (`:92`), ce qui permet
ce filtrage. Les intérêts transmis au serveur contiennent ces travaux.

La règle ne crée pas de nouveau travail fin dans une bande froide : elle
maintient uniquement le travail déjà en cours et les tuiles déjà prêtes.
Les tâches hors vue, d'ancienne époque ou plus fines que la nouvelle cible
au dézoom perdent toujours leurs intérêts. Les préchargements qui deviennent
réellement visibles au niveau cible peuvent normalement devenir utiles.

Le marquage de présentation est resserré (`index.html:53`–`:56`) : seules
les tuiles possédant une image soumise au Canvas2D ou une texture incluse
dans les rectangles GPU sont marquées. Une simple entrée du plan sans
texture effectivement disponible ne débloque plus le raffinement.

## Vérifications exécutées

Les cinq tests passent : `test_terrain_lod.cjs`, `test_terrain_freshness.cjs`,
`test_terrain_refinement.cjs`, `verify_progressive_zoom.cjs`,
`verify_navigation_mock.cjs`.

- Le test VM de couverture partielle traverse les étapes 4/3/2/1 et ne
  demande que les ancêtres du secteur découvert. Il vérifie aussi la
  conservation simultanée des vols utiles LOD 0/1/2/3.
- Le test Chrome retient les réponses LOD 0, déplace la caméra de 16 km
  vers une bande froide et vérifie que **deux** vols encore visibles restent
  non annulés et présents dans les intérêts serveur. La vue termine ensuite.
- Le rAF retenu empêche toujours le passage 4 → 3. Le parcours nominal
  conserve l'ordre `[4,3,2,1,0]` et la couverture précédente avant chaque
  nouvelle étape.
- Le changement vers fit est marqué avant le clic ; aucune requête fine
  supplémentaire n'est lancée après ce point, la tuile grossière annulée
  n'entre pas dans le cache et le rendu n'expose pas de détail historique.
- L'image initiale est maintenant contrôlée pendant que des paquets natifs
  sont réellement retenus, avec pending > 0 et des requêtes non vides :
  aucun parent ne recouvre le rectangle protégé, uniquement du LOD 0 demandé.
- Une VM indépendante du script inline vérifie qu'une tuile GPU absente
  du renderer conserve `presented:false`, puis passe à true lorsqu'elle
  participe aux rectangles soumis. Des vols intermédiaires préchargés,
  hors vue et d'ancienne époque sont bien annulés.

Le reçu progressif observé contient `panRetainedFlights:2`,
`heldRAFBlocksRefinement:true`, `nativeImageProtected:true`,
`hotRefreshRequests:0`, 99 requêtes. Le compteur d'annulations est cumulé
sur plusieurs actions ; il ne doit pas être interprété comme celui du
seul changement vers fit. Le test large viewport reste sous son budget
avec 138 560 656 octets au parcours natif adapté à LOD 1.

Empreintes exécutées :

```text
index.html
dfb4a87de8fa0edf4dbac3e627cef1e58343c2e80712380b54d7878de21e48aa
terrain_lod.js
6cd853396f5c3480ba54ffddadabe08a9db0f8efc3f759474302b49f99c5d84f
verify_progressive_zoom.cjs
2cbd3e9d4e38219397f02950bcffe7dade6884b5477b323367d1d18ea822d341
```

Ce verdict clôt le défaut d'annulation lors du pan. Il ne prétend pas
mesurer une latence HTTP/NN réelle, la présentation matérielle des étapes,
ni une préparation globale LOD 11. Ces contrôles d'intégration appartiennent
au coordinateur disposant du serveur et du créneau GPU.
