# Render v2 — état sauvegardé le 23 octobre 2026

Branche demandée par l'utilisateur : `codex/render-v2-snapshot-2026-10-23`,
dans Neural Earth et Magna Urbis. Il s'agit de l'état actuel demandé pour
commit/push, pas d'une déclaration de finition photoréaliste.

## Production présente

Un seul Render v2, saison et diagnostics associés, matériaux partagés
WGSL/NumPy/GLSL, couvert multiscalaire, conservation des plages et de
l'éclairage du relief, composition saisonnière de neige/eau/glace, et masque
régional des rivières. Le plancher annuel de neige corrige la plaine EF devenue
grise en été. Empreinte de production NE :
`2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3`.

Validation de cet état : 77 tests Python de rendu réussis ; typecheck CG réussi.
La correction annuelle possède 27 captures réelles ciblées, avec témoins
DFC/arides inchangés. Les preuves CG et leurs limites figurent dans
`../city_generator/docs/methods/render-v2.md` depuis la racine commune RPG.

## Travail non intégré ou non terminé

- Les essais de clasts, de couronnes stochastiques et de neuf volumes ont été
  rejetés visuellement. Leurs outils ne remplacent pas le shader de production.
- Le prototype de sol par exemples CC0 a montré un progrès ciblé. Son extension
  herbe/litière, son intégration de production et le port CG restent à terminer.
- Les véritables icebergs ont été isolés par ablation sur des tuiles marines.
  Leur étirement dans la carte polaire reste à résoudre sans déplacer les
  objets entre projections. Les couronnes proches restent stylisées.
- Le couvert coarse doit encore être jugé à emprise identique au fin ; ses
  grandes formes hivernales viennent de la contribution forestière, pas du
  grain du sol. Le matériau des rivières reste simple.
- La campagne finale de 990 images n'est pas terminée. Le run `89516…`,
  495 images principales et 90 hydrologiques, est rejeté après la découverte
  du défaut de neige annuelle. Aucun verdict n'est transférable à la correction.
- La couverture native atteint 20 des 24 combinaisons strictes demandées ;
  quatre plateaux AF/CFB/DFC/POLAR restent manquants. Les recherches locales
  supplémentaires ne prouvent pas leur absence dans le monde entier.
- Les limites de DEM 30 m, d'hydrologie source et de filtre de profondeur des
  lacs sont décrites dans le document d'implémentation.

Les captures, gros téléchargements, caches et données générées restent dans
les répertoires ignorés. Les scripts d'expérience peuvent nécessiter ces
intrants locaux ; leur présence dans la branche ne signifie pas qu'ils sont
intégrés ou validés. Le serveur HTTP déjà ouvert doit être relancé pour charger
le dernier miroir NumPy annuel ; le prochain démarrage charge le code sauvegardé.

Voir [l'implémentation et ses limites](render-v2-implementation.fr.md),
[le protocole de référence](render-v2-reference-protocol.fr.md) et
[les références sur les sols](render-v2-soil-literature.fr.md).
