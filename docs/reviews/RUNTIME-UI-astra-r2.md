# Runtime UI — Astra HIGH, revue r2

Date : 2026-10-07. Relecture des corrections de
`RUNTIME-UI-astra-r1.md`. Sources en lecture seule ; aucun serveur, aucune
inférence et aucun test GPU matériel. NN et macro A4 restent hors périmètre.

## Verdict : ACCEPT WITH LIMITS

Les deux P2 de r1 sont corrigés. Aucun P0/P1/P2 restant identifié dans ce
bloc UI. Les limites ci-dessous concernent la portée des preuves, sans
correctif bloquant supplémentaire.

### P2 fermé — Barrière de dessin avant raffinement

`index.html:92` crée désormais les tuiles avec `presented:false`.
`terrain_lod.js:29` les exclut de la couverture permettant d'avancer une
étape. `index.html:55` marque les tuiles utilisées après l'appel au rendu,
puis `index.html:56` programme la reprise du raffinement. Ce dernier réveil
évite de rester bloqué si le timer a déjà constaté une couverture non dessinée.

`verify_progressive_zoom.cjs:44` retient les callbacks rAF après déplacement
de la caméra, attend la réception du LOD 4 et l'absence de transferts, puis
vérifie qu'aucune requête LOD 3 ou plus fine n'est partie. Après libération
des frames, le parcours se termine dans l'ordre `[4,3,2,1,0]`. Test exécuté
et PASS, avec `heldRAFBlocksRefinement:true` et zéro demande à la revisite
chaude. Les assertions du planner couvrent aussi `presented:false/true`.

### P2 fermé — Ancêtres inutiles en couverture partielle

`index.html:71` vérifie les parents plus fins de chaque cellule cible et
`index.html:78` exclut ces cellules de la génération d'ancêtres d'étape.
`test_terrain_refinement.cjs` charge le vrai script inline : une moitié
couverte par LOD 1 et une moitié froide ne demandent que le parent LOD 4 de
la moitié froide. Test exécuté et PASS.

Vérification indépendante supplémentaire, sans fichier source modifié :
VM du script inline avec caméra LOD 0, coordonnées x négatives, moitié
gauche déjà couverte à un niveau plus fin et moitié droite nécessitant
l'étape courante. Pour chaque étape 4, 3, 2 et 1, les seuls intérêts
d'ancêtre sont `[[étape,-1,0]]`, sans demande pour la moitié gauche.
Les quatre cas passent.

## Vérifications exécutées sur les corrections

- `node test_terrain_lod.cjs` : PASS.
- `node test_terrain_freshness.cjs` : PASS.
- `node test_terrain_refinement.cjs` : PASS.
- `node verify_progressive_zoom.cjs` : PASS.
- `node verify_navigation_mock.cjs` : PASS, viewport 5760 × 3240,
  cache natif de 138 560 656 octets sous son budget, adaptation à LOD 1,
  changement de monde/mode/backend et absence de détail historique au dézoom.
- VM indépendante de couverture partielle négative aux étapes 4/3/2/1 : PASS.

Empreintes du test progressif exécuté :

```text
index.html
a549942032714d3508e0b0563b65e40973dc2acb1e56b9521fc46ec3e3cfc35b
terrain_lod.js
0d8e37082ef06c656c425ca11c9f7684c2220164abbd07aca1bdc2b7bc2fe42b
verify_progressive_zoom.cjs
563ae8eab7999c62a90fa6d9c32a996df5ca5f3aff36426481dda114032ae0fd
```

## Limites conservées

La barrière démontre qu'un frame utilisant la couverture intermédiaire est
soumis par le client avant le démarrage de l'étape suivante. Elle ne prouve
pas le moment de présentation physique par le GPU, le compositeur ou l'écran.
Le navigateur utilise un renderer et un transport simulés ; la documentation
signale déjà cette limite. Le nouveau test de couverture partielle mérite
d'être ajouté à la liste des vérifications documentées lors du bilan final.

Le plafond LOD 11, les barrières de couverture du viewport, l'annulation des
intérêts abandonnés, les gardes d'époque, les expirations et revalidations,
le maintien des parents et l'ordre des étapes restent cohérents à la lecture
et avec les tests. Le budget validé est celui du cache client ; ce n'est pas
une mesure globale de RAM/VRAM. Aucun résultat ici ne valide la vitesse NN,
les sorties numériques optimisées, les formes continentales A4 ou leur
promotion comme profil par défaut.
