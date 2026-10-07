# Runtime UI — Astra HIGH, revue r1

Date : 2026-10-07. Périmètre : bloc UI terminé (`index.html`,
`terrain_lod.js`, les quatre tests JS et `docs/RUNTIME_OPTIMIZATION.md`).
Revue en lecture seule des sources ; seul ce rapport est ajouté. Aucun serveur
démarré, aucune inférence, aucun test GPU matériel. Les modifications NN et
macro A4 en cours ne sont pas évaluées comme des livrables terminés.

## Verdict initial : CHANGES REQUIRED

Aucun P0/P1 trouvé. Deux P2 reproduits dans le vrai script client, chargé dans
une VM Node avec transport/rAF simulés. Le parcours nominal satisfait l'ordre
4 → 3 → 2 → 1 → 0 et le plafond LOD 11, mais les deux garanties ci-dessous
nécessitent une correction et des tests ciblés.

### P2 — La présence en cache ne garantit pas un dessin avant le calcul suivant

**Localisation initiale :** `index.html:49`, `index.html:71`,
`index.html:89`, `index.html:90` ; `terrain_lod.js:28`.

À réception d'une tuile, `draw()` programme un rAF. Le raffinement utilise
immédiatement toutes les tuiles du cache, tandis qu'un timer peut relancer
`refresh()` après 50 ms. Rien ne lie l'ouverture du niveau suivant à
l'exécution du dessin précédent. Si le rAF est suspendu ou retardé, les
requêtes LOD 3 peuvent donc partir sans dessin du LOD 4 ; le même cas existe
entre les autres étapes.

**Reproduction sans GPU :** charger le script inline jusqu'aux événements dans
une VM ; retenir les callbacks de `requestAnimationFrame` ; caméra LOD 0 avec
`W=H=128`, `cx=cy=3840`, `mpp=30` ; placer la tuile LOD 4 `(0,0)` en cache,
appeler `draw(); refresh()`. Résultat observé :

```text
NO_DRAW_BARRIER {"draws":0,"active":3,"requested":[[3,0,0]]}
```

**Correction proposée :** suivre pour chaque nouvelle tuile son utilisation
dans un frame soumis par `drawFrame`, et faire attendre le raffinement tant
que les nouvelles couvertures requises ne sont pas passées par ce dessin.
Déclencher ensuite le rafraîchissement. Tester un rAF retenu entre réception
LOD 4 et départ LOD 3, puis entre étapes suivantes. Cette garantie concerne
la soumission du dessin client ; elle ne démontre pas la présentation
physique sur l'écran.

### P2 — Une couverture partielle par des parents plus fins régénère des ancêtres

**Localisation initiale :** `index.html:73`–`index.html:75`.

`refinementLod` reconnaît correctement un parent plus fin pour choisir
l'étape globale. Mais la construction des requêtes ne vérifie que l'absence
de la tuile au niveau cible. Dès qu'une autre partie du viewport impose LOD 4,
elle ajoute aussi des requêtes LOD 4 pour les cellules déjà couvertes par
LOD 1/2/3. Cela gaspille calcul/transfert et contredit la promesse de sauter
les ancêtres des régions déjà couvertes lors d'une revisite partielle.

**Reproduction sans GPU :** caméra LOD 0, `W=4352`, `H=256`, `cx=65280`,
`cy=3840`, `mpp=30`, soit un viewport `[0,0,130560,7680]`. Précharger huit
tuiles LOD 1 `(tx=0..7, ty=0)`, couvrant toute la partie visible du premier
parent LOD 4. Laisser la bande suivante froide puis appeler `refresh()`.

```text
PARTIAL_CACHE {"active":4,"requested":[[4,0,0],[4,1,0]]}
```

Seule `(4,1,0)` est nécessaire à cette étape. Le cas symétrique existe aux
étapes LOD 3/2 lorsque des parents plus fins sont déjà disponibles.

**Correction proposée :** avant d'ajouter un ancêtre d'étape pour une cellule
cible, vérifier la couverture de cette cellule par tous les niveaux entre
la cible et l'étape. Ajouter un test mixte chaud/froid et négatif en
coordonnées, sans seulement vérifier la revisite entièrement native.

## Vérifications et autres constats

Exécutés et PASS :

- `node test_terrain_lod.cjs`.
- `node test_terrain_freshness.cjs`.
- `node verify_progressive_zoom.cjs` : ordre `[4,3,2,1,0]`, aucune requête
  LOD 12, LOD 3 volontairement bloqué, zéro requête à la revisite chaude.
- `node verify_navigation_mock.cjs` : viewport 5760 × 3240, adaptation
  native à LOD 1, cache 138 560 656 octets sous le plafond, modes/profils,
  annulation/navigation et absence de détail historique au dézoom.

Le plafond LOD 11 est partagé par le calcul du LOD, le fit et le zoom ; les
parents automatiques sont eux aussi plafonnés. Les barrières de **couverture
en cache** du viewport fonctionnent : une cellule encore dépourvue d'un
parent suffisant retient l'étape globale. Le problème du premier P2 est
distinct : cette couverture peut ne pas encore avoir été dessinée.

Les requêtes hors intérêts sont abandonnées ; les changements de monde ou
de backend ont des garde-fous d'époque. Le retour au dézoom filtre le détail
plus fin que la cible dans le plan de rendu. Le budget utilise un LOD adapté,
une marge et une éviction ; le test grand viewport passe. Cela ne constitue
pas une mesure de mémoire GPU réelle ou de mémoire totale du processus.

Les aperçus expirent même sans nouvelle télémétrie ; les LOD 1/2 non finaux
sont revalidés ; le remplacement conserve l'ancienne tuile jusqu'à réception
et protège la nouvelle texture lors de la destruction de l'ancienne. Les
revalidations sont sérialisées tout en laissant partir la couverture
manquante. Le préchargement attend que les tuiles cibles soient présentes.

Les tests existants ne prouvent pas l'affichage physique intermédiaire, la
latence NN réelle ni la géographie finale. Le test progressif vérifie qu'un
parent a été dessiné au départ d'une étape, mais ne simule pas un rAF retenu.
Ces limites doivent rester explicites dans la documentation et le bilan.

## Suivi

Les deux reproductions et corrections proposées ont été transmises à
l'agent principal, qui prépare les corrections. Le verdict ci-dessus porte
sur le snapshot initial. Une vérification des corrections doit être ajoutée
avant de clore ces deux P2.
