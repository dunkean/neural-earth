# Implémentation de l’audit terrain — suivi

> Historique des premiers blocs de l'audit. Les initialisations Earth/Macro A1–A4
> citées ici ont été retirées. Le bootstrap actuellement livré, ses reviews et
> ses mesures sont dans [TERRAIN_BOOTSTRAP_RESTART.md](TERRAIN_BOOTSTRAP_RESTART.md).

Objectif : génération crédible et navigation du monde plat de 40 000 × 20 000 km jusqu’au DEM neuronal natif de 30 m. Le raccord au moteur procédural, le sous-détail, l’hydrologie mondiale et les glaciers restent la phase suivante.

La référence qualitative est `natural`, checkpoint `9ef8030cb805b433b98ec25c5dddefbac07a9e26`, bruit conditionnant 0,5. Une macro expérimentale possède une identité distincte ; sa validation numérique ne vaut pas approbation visuelle.

## Suivi livré après les cinq blocs : runtime et A4

Le client ouvre désormais **A4 v2 comme initialiseur expérimental par défaut** ;
`natural` reste accessible dans le sélecteur. Reviews finales
[Astra high r2](reviews/RUNTIME-FINAL-astra-r2.md) et
[Opus 5.5 CLI](reviews/RUNTIME-FINAL-opus-r1.md) : ACCEPT WITH LIMITATIONS,
sans P0/P1. La navigation plafonne au LOD 11 et présente 4/3/2/1/0 au zoom.
LOD 3 retire environ 0,7 s de préparation ; le gain LOD 2 demeure variable.

Le corpus A4 final comprend huit régions NN et 512 points coarse mondiaux.
Le cas polaire faible seed 0 reste défavorable : 7,04 % de terre NN contre
45,84 % d'entrée interpolée. L'aperçu mondial est du conditionnement ; aucune
planète NN complète n'est revendiquée. Mesures navigateur, défaut A4 puis
retour natural et limites : [bilan runtime](RUNTIME_OPTIMIZATION.md),
[A4](CONTINENTAL_BOOTSTRAP.md), [corrections](reviews/RUNTIME-corrections.md).
Les tableaux B1–B5 ci-dessous décrivent leur snapshot historique, antérieur à
ce suivi ; leurs timings et anciens LOD ne décrivent pas le nouveau client.

## Cinq blocs maximum

| Bloc | Travail | Responsable | Validation |
|---|---|---|---|
| B1 | Référence, manifest complet, corpus, exports et compositeur macro A1/A2/A3 | Sol high | Contrats physiques, répétabilité, planches côte/montagne/plaine ; Astra high + Opus 5.5 CLI |
| B2 | Fenêtres NN canoniques, dépendances, instrumentation et cache coarse appris | Sol xhigh | Entier/sous-crops, permutations, reprise, absence de fusion partielle ; deux reviews |
| B3 | Navigation, préparation mondiale incrémentale, provenance et cache intégré | Coordination | Tests HTTP et navigateur, zoom/pan/téléportation, fidélité naturelle ; deux reviews |
| B4 | Export et preuve navigateur WebGPU des réseaux et du crop | Sol xhigh | Exécution réelle, pas de repli caché, erreurs, mémoire et temps ; deux reviews |
| B5 | Corrections des reviews, corpus NN et mesures finales | Équipe | Rejouer les contrôles pertinents, statut explicite de chaque porte E1–E4 ; deux reviews |

Les agents partagent les fichiers avec des périmètres d’écriture séparés. Les charges GPU sont séquencées. Les fichiers initiaux ont été copiés dans `E:/TerrainDiffusionRuntime/audit-implementation/baseline` avant l’intégration.

## Reviews et décisions

Les prompts, sorties et corrections sont conservés dans `docs/reviews/`. Opus a été invoqué avec le modèle explicite `claude-opus-5-5`, confirmé dans les reçus CLI. Astra utilise `gpt-6-astra` en effort `high`. Les deux reviews finales acceptent avec limites ; les corrections finales du benchmark et leurs reçus sont vérifiés par Astra et évalués dans [le ledger B5](reviews/B5-correction-r1.md).

## État

| Bloc | État courant | Résultat / porte restante |
|---|---|---|
| B1 | Reviews Astra/Opus r1/r2, corrections appliquées | 10 tests CPU ; corpus final attesté `reference-b1r2-final` : 56 entrées, 32 exports NN sur 8 sites. Dépendances U-Net ajoutées à l'identité ; 107 sources locales. A3 reste expérimental. |
| B2 | Reviews Astra/Opus r1/r2 puis Astra r3 accepté avec limites | Relecture GPU finale exactement égale, zéro NN, 4 hits disque ; scheduler sans écart ajouté et replay identique exact. Comptabilité multi-instance et quota corrigés. Profil scalaire latent NO-GO. |
| B3 | Double review r1/r2 accepté avec limites, correction de promotion finale | Navigation jusqu'à 30 m, préparation automatique après aperçu, validation progressive sans effacer l'affichage, mips DEM et provenance. Client corrigé : revisite 224 ms ; ancien instrument clavier/cadence clairement historique. |
| B4 | Opus r2 et Astra r3 accepté avec limites ; E2 NO-GO | Crop JS CPU diagnostique 0,51 m ; crop navigateur NCHW 4,79 m > seuil 1 m. Conditionnement, reporting et provenance corrigés ; pas de déploiement du NN navigateur. |
| B5 | Astra et Opus r1 accepté avec limites ; correctifs finaux vérifiés | Corpus attesté, paquet naturel exact, contrôles GPU rejoués, 6 160 fenêtres mondiales en 439,6 s / 707 MB ; promotion corrigée et pan continu mesuré. |

## Résultats mesurés

Le [bilan final](VALIDATION_FINAL.md) sépare les résultats sur sources figées, les preuves historiques et les portes produit non atteintes. La préparation mondiale finale termine les 6 160 fenêtres avec apron, dont 6 156 nouvelles, en 439,60 s pour 707 266 560 octets de NPY. Une tuile mondiale LOD 12 prend encore 8,294 s à reconstruire sans NN supplémentaire. Le test navigateur réel a détecté puis validé la correction d'une rafale de revalidations ; les validations sont maintenant sérialisées tandis que la couverture manquante garde jusqu'à quatre transferts.

- GPU RTX 3090 : `coarse-persistence-fidelity.json` restitue exactement les sept canaux pondérés, sans appel NN lors de la relecture. Cinq fenêtres seulement ; ce n'est pas une couverture mondiale.
- `window-scheduler-fidelity.json` : aucune différence ajoutée par le scheduler. Le profil local antérieur à B2 diffère de 0,210693 m entre entier et sous-crops sur un cas ; ce n'est ni une borne générale ni une preuve de causalité dans le code upstream.
- Profil latent scalaire expérimental : écart entier/sous-crops 0,096–0,324 m et écart à la référence batch 16 de 2,04–11,90 m sur quatre seeds. La porte de fidélité échoue ; pas de promotion.
- Premier essai HTTP naturel : 5,69 s pour le premier crop natif, 0,47 s pour son voisin, 9–33 ms pour les revisites cache. Les poids sont présents localement ; ces échantillons ne prouvent pas le débit d'un pan soutenu.
- Navigateur historique : intervalle rAF p95 16,8–16,9 ms pendant des déplacements par touches ; ce n'est pas le coût du dessin. Le run final fait deux pans continus ±6 km/s pendant 2 s : segment CPU dessin p95 0,6 / 0,4 ms, caméra 0,1 ms, aucun intervalle >1,5× la période compositor estimée. Tous les transferts sont des hits, baseline/fins natives couvertes, samples sans pending ; le préchargement est désactivé et un autre client serveur existe. La gate GPU/présentation <16,7 ms et le débit NN froid durable restent non mesurés. Aucun temps de readiness n'est présenté comme une première présentation utile <250 ms.
- WebGPU ORT 1.30 : le défaut de la première Conv a été isolé au chemin NHWC (16,89 d'écart). NCHW explicite ramène cette Conv à 3,34e-6. Le coarse et le decoder passent leurs forwards réels contre ORT CPU ; le base conserve un écart de 0,2546. Le crop corrigé termine 75 forwards avec et sans option de capture mais conserve 4,7927 m d'écart, supérieur à la porte 1 m. Le même JS avec backend CPU diagnostique donne 0,5101 m. L'activation effective de capture, la résidence de tous les nœuds et une côte véritable ne sont pas certifiées.

Les rapports techniques résident dans `E:/TerrainDiffusionRuntime/`. Chaque essai final doit porter les hashes des sources réellement exécutées. Les résultats initiaux sont conservés et identifiés comme tels. Une optimisation fidèle, une nouvelle macro et une expérience de port numérique ne sont pas interchangeables.

La comparaison finale HTTP naturelle sur la seed `20261007092`, LOD 0, tuile `(-14,10)`, restitue exactement les 391 444 octets de la référence pré-audit : hauteur et cinq climats à écart zéro. Cela porte sur un paquet fixe, pas sur tous les historiques de navigation. Les tests finaux actuels passent : 62 Python, fraîcheur/promotions JavaScript, LOD et mock navigateur 5760×3240 ; cette très grande vue adapte son LOD à 1 pour rester dans le budget.

## Décisions de périmètre

Les recommandations de review sur les fichiers vides, les poids de fusion, la couverture aux bords, le conditionnement JS, la provenance, le quota et la cohérence entre lecteurs sont acceptées et corrigées. Les demandes de plus grands crops sont appliquées à trois sites 1024², avec le reste du corpus à 256². Le raccord procédural, la simulation glaciaire, la distillation et un runtime multi-GPU restent hors de cette phase. Aucun défaut d'un backend expérimental n'autorise une dégradation silencieuse de la référence CUDA.

L'identité actuelle est volontairement conservatrice et inclut les sources du serveur ; une modification de serveur peut donc invalider des fenêtres dont les NN n'ont pas changé. Les quotas sont par profil physique et par monde coarse, sans quota global automatique des anciennes identités. Ces limites de réutilisation et de stockage sont déclarées, pas assimilées à des gains de performances.

La préparation est globale et unique : ouvrir un autre monde remplace le travail de fond actif ; la pause est globale. Les fenêtres terminées restent persistées. Une saturation de la file peut nécessiter de relancer la préparation. La validation des aperçus recule progressivement jusqu'à 30 s et passe après les tuiles manquantes ; une probe occupée ne bloque pas une réponse cache, mais peut différer la promotion pendant un quantum GPU.

Les reviews Opus contiennent un paragraphe automatique sur des connecteurs Calendar/Drive/Sentry. Aucun n'est nécessaire à cette tâche ; ce paragraphe est ignoré. Le nettoyage récursif d'un ancien dossier scratch a été rejeté par la revue automatique avec le seul motif « blocked by policy » ; il reste sur E: et n'entre pas dans le corpus attesté.
