# Validation de l’implémentation terrain

> Snapshot historique, antérieur au remplacement complet des initialisations
> Earth/Macro A1–A4. Aucun de ces profils n'est désormais actif. Voir
> [TERRAIN_BOOTSTRAP_RESTART.md](TERRAIN_BOOTSTRAP_RESTART.md) pour l'état courant.

Snapshot du 7 octobre 2026, RTX 3090 24 Gio, Windows, Python 3.12 / PyTorch 2.11 CUDA 12.8. Les cinq blocs sont décrits dans [le suivi](AUDIT_IMPLEMENTATION.md) ; les décisions et corrections de review sont conservées dans [reviews](reviews/).

**Suivi ultérieur livré le même jour :** A4 v2 est maintenant le défaut
expérimental d'initialisation, avec natural conservé. LOD 12 est exclu du client,
le zoom affine 4/3/2/1/0 ; optimisation de préparation NN et cache exact livrés.
Le [bilan runtime actuel](RUNTIME_OPTIMIZATION.md) conserve les nouveaux reçus
et les [reviews finales Astra/Opus](reviews/RUNTIME-corrections.md). Huit régions
A4 NN et une sonde sparse ont été vérifiées, avec échec local côtier polaire
explicitement publié. Ce n'est pas une planète NN entièrement préparée.
Les résultats B1–B5 ci-dessous restent historiques et liés à leurs sources.

Clôture technique : reviews [Astra high](reviews/B5-astra-r1.md) et [Opus 5.5 high CLI](reviews/B5-opus-r1.md) **ACCEPT WITH LIMITATIONS**. La [review Opus ciblée](reviews/B5-opus-r2.md) a conduit aux derniers correctifs de mesure ; leurs sources, captures et reçus ont été vérifiés par Astra. Le [ledger de corrections](reviews/B5-correction-r1.md) distingue les défauts clos des limites conservées. Cette acceptation technique ne certifie pas les portes produit encore ouvertes.

## Livrables et décision produit

Le chemin actif conserve le conditionnement `natural` du checkpoint épinglé, SNR 0,5 et batch latent 16. Il dispose de fenêtres NN instrumentées, d’un cache coarse appris persistant, d’une préparation mondiale incrémentale, d’une navigation bornée jusqu’à 30 m/pixel et de provenance explicite pour chaque LOD. Les NN actifs tournent sur CUDA ; le navigateur WebGPU rend les altitudes reçues.

| Porte de l’audit | Résultat | Décision |
|---|---|---|
| E1 référence / macro | Corpus attesté : 14 sites CPU × 4 variantes ; 8 sites NN × 4. Trois sites 1024². A0 HTTP témoin exactement préservé. La côte A3 sélectionnée reste trop plate. | Référence conservée ; nouvelle macro non promue, validation visuelle ouverte. |
| E2 NN WebGPU | Trois réseaux exportés et crop complet exécuté. NCHW corrige un défaut NHWC ; crop navigateur max 4,7927 m, CPU diagnostique 0,5101 m. | **NO-GO**, seuil 1 m dépassé ; placement, capture et résidence intégrale non certifiés. |
| E3 scheduler/navigation | Aucun écart ajouté par le scheduler sur les crops rejoués ; revisites mesurées et segments CPU caméra/dessin courts sur parcours chaud. Dépendance batch/partition antérieure subsiste ; mesures insuffisantes pour le débit froid durable. | Intégration utilisable ; objectif produit « ultra rapide à froid » non acquis. |
| E4 coarse mondial | 6 160 fenêtres terminées, apron climatique inclus ; 439,60 s, 707 266 560 octets de payload NPY ; aperçu appris relu sans NN. | Préparation persistante fonctionnelle ; nouvelle seed mondiale immédiate non acquise. |

E5–E8 (distillation, raccord procédural, hydrologie, glaciers) restent la phase suivante demandée par l’utilisateur. Les mips exacts du DEM natif couvrent uniquement les LOD 1 et 2 lorsque tous leurs enfants sont présents ; le coarse mondial à 7,68 km et ses moyennes physiques restent des approximations apprises.

## Mesures et snapshots exécutés

Les rapports JSON portent les identités et/ou hashes des sources exécutées. Le générateur CUDA est figé ; le client et le harness ont reçu les corrections indiquées ci-dessous. Les poids existaient localement ; premier téléchargement absent, température et répétitions multi-session ne sont pas mesurés. Les temps HTTP excluent la présentation GPU.

| Scénario | Mesure | Rapport / snapshot | Portée |
|---|---|---|---|
| Premier crop natif HTTP, seed `20261007092` | 5,747 s | `final-natural-http.json` ; serveur `b3e9c5e3fc37` | Miss physique avec premier chargement des poids de cette session ; un seul cas. |
| Crop natif voisin HTTP | 0,484 s | Même rapport HTTP | Réutilise dépendances et poids ; pas indépendant. |
| Revisites HTTP, 9 hits | 11–42 ms | Même rapport HTTP | Lecture cache ; les headers de durée NN sont historiques, le wall est la latence réelle. |
| Première vue native navigateur, seed 42 | 5,208 s | `browser-navigation.json` ; client `04ddf644dac2` | Historique, 15 patches à 30 m ; lire headers cache pour le degré de froid. |
| Revisite de cette vue native navigateur | 240 ms | `browser-prepared-world.json` ; client `04ddf644dac2` | Historique, cache déjà présent ; pas une preuve de NN froid <250 ms. |
| Revisite native, client corrigé | 224 ms | `browser-final-client.json` ; client `e063156f1a04` | Cache déjà présent ; intervalle rAF 16,8–16,9 ms pendant sauts clavier, instrument historique. |
| Vue native avec harness continu final | 208 ms, 15 hits | `browser-continuous-pan.json` ; harness `75689d9fb02e`, client `e063156f1a04` | Une revisite chaude ; pas de garantie <250 ms générale ou de NN froid. |
| Pans continus ±6 km/s, 2 s chacun | Environ 12 km ; dessin CPU p95 0,6 / 0,4 ms, caméra 0,1 ms | Même rapport continu final | Segment CPU mesuré, borne inférieure du travail main-thread ; exécution GPU et présentation non mesurées. |
| Couverture des pans continus | Samples pending 0, fallback LOD 0 ; stabilisation finale 28 ms chacun | Même rapport continu final | Préchargement désactivé, champs entièrement depuis le cache ; cadence compositor estimée ~60 Hz, 0 intervalle >1,5× période. |
| Zoom monde puis natif | 200 ms ; 8 hits, tous patches decoder au settlement | Même rapport continu final | Attente corrigée qui confirme les besoins recalculés et leur couverture ; cache chaud. |
| Déplacements par touches, annoncés 6 km/s, 2 s | p95 intervalle 16,8 ms ; handler 0,2 ms | `browser-navigation.json` ; client `04ddf644dac2` | Sauts de 3 km toutes les 500 ms ; ni pan continu ni coût du dessin. |
| Déplacements après préparation globale | Intervalle rAF 16,8–16,9 ms ; handlers 0,1–0,3 ms | `browser-prepared-world.json` ; client `04ddf644dac2` | Sauts clavier historiques ; pas un benchmark de débit soutenu. |
| Aperçu et renderer prêts, run chaud | 1,147 s | `browser-prepared-world.json` ; client `04ddf644dac2` | Readiness seulement, pas première présentation utile. |
| Vue mondiale initiale entièrement reçue | 3,475 s avant préparation ; 1,746 s après | `browser-navigation.json` / `browser-prepared-world.json` ; client `04ddf644dac2` | Les deux vues étaient entièrement conditioning-preview ; la promotion corrigée est mesurée séparément. |
| Préparation mondiale coarse | 439,60 s, 6 156 nouvelles / 6 160 prévues | `coarse-world.json` ; monde `40d8825e5453` | Quatre fenêtres existaient ; poids chargés ; aucune vue concurrente pendant ce run. |
| Reconstruction mondiale LOD 12 apprise | 8,294 s, zéro forward supplémentaire | Même rapport coarse mondial | Miss physique ; fusion/lecture/agrégation coûteuses, pas mip préfabriqué. |
| Relecture après redémarrage | 12,1 ms, zéro forward supplémentaire | `coarse-world-restart-hit.json` ; monde `40d8825e5453` | Cache physique hit ; exclut reconstruction. |

Rapports : [HTTP](<E:/TerrainDiffusionRuntime/audit-implementation/final-natural-http.json>), [navigateur initial](<E:/TerrainDiffusionRuntime/audit-implementation/browser-navigation.json>), [navigateur après préparation](<E:/TerrainDiffusionRuntime/audit-implementation/browser-prepared-world.json>), [client final](<E:/TerrainDiffusionRuntime/audit-implementation/browser-final-client.json>), [coarse mondial](<E:/TerrainDiffusionRuntime/audit-implementation/coarse-world.json>), [capture native finale](<E:/TerrainDiffusionRuntime/audit-implementation/browser-final-client-native.png>). Le premier scénario historique `initial-world-preview` du rapport initial mesure en réalité la vue entièrement reçue, pas la première image utile. Le benchmark courant sépare ces événements. Les deux premiers runs navigateur précèdent la dernière correction client ; ils conservent leurs hashes historiques.

Le test réel de promotion a détecté une rafale de revalidations : une tuile occupait le GPU, les suivantes conservaient leur aperçu puis attendaient le backoff. Le [rapport échoué à 180 s](<E:/TerrainDiffusionRuntime/audit-implementation/browser-promotions-pre-fix.json>) reste conservé. Le client sérialise maintenant les revalidations des tuiles déjà affichées, tout en laissant jusqu’à quatre transferts pour la couverture manquante. Le [rejeu corrigé](<E:/TerrainDiffusionRuntime/audit-implementation/browser-promotions.json>) remplace toute la vue par le coarse appris en 25,443 s sans aucun NN supplémentaire. Certaines tuiles avaient déjà été promues par le run précédent et cinq probes HTTP : ce temps est une preuve de correction, pas une latence propre depuis un monde froid. Le test JavaScript vérifie ce partage de la file.

| Rapport navigateur | Hash client `index.html` abrégé | Statut et couverture |
|---|---|---|
| `browser-navigation.json` | `04ddf644dac2` | Historique avant correction ; première vue mondiale entièrement conditioning-preview. |
| `browser-prepared-world.json` | `04ddf644dac2` | Historique avant correction ; vue mondiale encore entièrement conditioning-preview malgré le cache coarse complet. Vue native revisitée depuis le cache. |
| `browser-promotions-pre-fix.json` | `04ddf644dac2` | Échec de promotion complète après 180,948 s. |
| `browser-promotions.json` | `e063156f1a04` | Client corrigé, toute la vue coarse-area-mean ; cache partiellement réchauffé. |
| `browser-final-client.json` | `e063156f1a04` | Client corrigé ; 24 patches mondiaux coarse-area-mean ; région native entièrement depuis le cache. Instrument de déplacement par touches encore historique. |
| `browser-continuous-pan-pre-settle-fix.json` | `e063156f1a04` | Harness `6fba68338f99` ; pan valide, mais fausse déclaration de settlement sur le zoom ; historique conservé. |
| `browser-continuous-pan.json` | `e063156f1a04` | Harness `75689d9fb02e` ; toutes les baselines et fins natives confirmées, adapter NVIDIA Ampere, captures liées par SHA. |

Le [run continu final](<E:/TerrainDiffusionRuntime/audit-implementation/browser-continuous-pan.json>) déplace la caméra à chaque rAF et mesure le vrai `drawFrame` avec un wrapper temporaire. Ce segment CPU exclut uploads, layout différé et sampling de diagnostic ; c’est une borne inférieure du travail main-thread. La gate bout-en-bout <16,7 ms incluant GPU et présentation reste non mesurée. La cadence rAF en Chrome headless est celle du compositor, pas le refresh d’un écran. Le préchargement est désactivé ; pending compte les tuiles du LOD demandé encore absentes, pas les zones vides grâce aux parents de secours. Les samples enregistrent aussi leurs LOD rendus et la part de patches de secours. Le second pan part du centre mondial après le zoom, pas de la fin du premier. Un autre client serveur existait : le run ne prouve pas une isolation matérielle.

Le premier reçu continu déclarait à tort le zoom settled avec huit tuiles attendues. Ses temps de settlement sont retirés du bilan. Le dernier helper force un recalcul pour la position finale, attend l’epoch correspondante puis vérifie à nouveau et retourne le snapshot effectivement couvert. Les deux baselines natives, la transition et les deux fins de pan sont désormais confirmées `pending=0`, LOD 0, source decoder. Le backend WebGPU est exigé ; adapter, paths et SHA des captures, attribution synchrone des réponses au scénario et hashes du harness sont dans le reçu.

La [relecture après redémarrage](<E:/TerrainDiffusionRuntime/audit-implementation/coarse-world-restart-hit.json>) vérifie identité inchangée, payload fini et provenance apprise. Les 12,1 ms mesurent uniquement le hit disque ; les durées de génération conservées dans ses headers restent historiques.

## Fidélité et vérifications

- 62 tests Python : manifest, macro, fenêtres, climat, navigation, préparation et mips finaux ; tous passent.
- Tests JavaScript des réponses tardives et promotions, LOD et mock navigateur 5760×3240 : tous passent. Cette grande vue adapte son affichage natif au LOD 1 pour respecter le budget ; elle n’est pas une preuve matérielle 30 m sur cette surface.
- Cache coarse v3 : cinq fenêtres pondérées restituées exactement, quatre hits disque, zéro NN ; scheduler activé/désactivé exactement identique et replay dans le même cache vivant exact.
- Contrôle HTTP pré-audit : 391 444 octets strictement identiques pour une tuile LOD 0, seed `20261007092`, `(-14,10)`, hauteur et cinq climats. Pas une preuve exhaustive pour toute navigation.
- Le profil BF16/batch 16 antérieur à B2 montre déjà 0,210693 m de variation entier/sous-crops sur un cas local entièrement marin. Ce n’est ni une borne générale ni une validation des côtes.
- `TERRAIN_CANONICAL_LATENTS=1`, batch scalaire expérimental : 0,096–0,324 m entier/sous-crops, 2,04–11,90 m face au batch 16 sur quatre seeds. **NO-GO**, profil distinct, non activé par défaut.
- WebGPU : dix tests Node, test de corruption des poids externes, 304 initializers base comparés à l’inline avec lecture bornée, 56 fichiers tenseurs vérifiés, crop CPU numérique PASS. Les mesures navigateur existantes restent liées aux scripts réellement exécutés ; le durcissement final de provenance n’a pas été présenté comme un nouveau run GPU.

Le corpus [E1 final](<E:/TerrainDiffusionRuntime/reference-b1r2-final/e1_report.json>) atteste ses sources et artifacts. L’identité du générateur inclut maintenant les 107 sources Python locales de référence, dont tous les modules du package NN. Les contrôles GPU coarse/scheduler ont été rejoués après cet ajout manifest, serveur arrêté : [coarse persistant](<E:/TerrainDiffusionRuntime/coarse-persistence-fidelity.json>) et [scheduler](<E:/TerrainDiffusionRuntime/window-scheduler-fidelity.json>) passent avec les hashes actuels, restitution exacte et aucun NN en relecture. Le mock large écran a aussi été rejoué sur le client final : 1 434 requêtes, 36 publications caméra, budget respecté, aucune superposition de détail fin au dézoom.

Planches A0–A3 à échelles communes : [côte macro](<E:/TerrainDiffusionRuntime/reference-b1r2-final/comparisons/macro_coast_seed0/dem-shaded.png>), [montagne macro](<E:/TerrainDiffusionRuntime/reference-b1r2-final/comparisons/macro_mountain_seed0/dem-shaded.png>), [plaine macro](<E:/TerrainDiffusionRuntime/reference-b1r2-final/comparisons/macro_plain_seed0/dem-shaded.png>). La lisibilité du relief local naturel n’établit pas une géographie continentale crédible ; la vue mondiale naturelle reste très fragmentée. La macro continentale est un livrable expérimental à examiner, pas un résultat esthétique acquis.

## Limites acceptées et suite prioritaire

Les budgets sont de 192 Mio côté cache navigateur et renderer, 512 Mio par monde foreground et 64 Mio pour le monde de fond, quota coarse 2 Gio par monde et quota physique configurable. Ils ne constituent pas une borne stricte de toute la VRAM NN ou de toutes les anciennes identités disque. Une seule préparation mondiale est active ; une file saturée demande une reprise manuelle. Les publications de fenêtres sont sérialisées dans un process, pas entre plusieurs serveurs écrivains.

La dernière review Opus distingue d’autres P2 non bloquants : flags cuDNN/TF32 non explicitement épinglés dans l’identité et replay sous pression VRAM non mesuré ; promotion de previews après restart dépendant d’une préparation ou d’un monde chargé ; double vérification de la persistance ; extension du coarse hors du monde aux LOD 4–6 et coût associé ; possible concurrence Windows sur des PNG, non reproduite. Ces limites restent ouvertes. L’API `read_mip` relative au grid est volontairement exclue du chemin serveur global ; ses gros blocs demandent une empreinte complète contenue dans le grid et peuvent être refusés. La préparation actuelle persiste les sources pondérées, pas une pyramide globale préfabriquée.

La validation progressive garde l’ancienne tuile visible, passe après la couverture manquante et recule jusqu’à 30 s. Une probe occupée peut encore retarder la promotion. L’identité est un snapshot au démarrage : redémarrer après modification de code. Sa granularité conservatrice invalide parfois un cache NN après un simple changement serveur.

Les prochains gains doivent viser la reconstruction/fusion des vues mondiales déjà apprises, le débit base/decoder pendant un parcours froid prolongé, puis le premier nœud divergent du base NCHW. La macro demande une nouvelle itération côtière et une validation visuelle avant tout changement du défaut. Aucun raccord procédural n’a été introduit dans les mètres du DEM.
