# Revue indépendante : distillation du base
*Claude Opus 5.5, effort xhigh, 12/10/2026. Lecture seule, rien n'a été exécuté.*

## 1. Diagnostic hiérarchisé

**H1 (prioritaire) : les élèves sont sous-optimisés.**
*Faits.*
- Au pas 15000 du native160×3, la MSE x0 du canal0 de première phase est de 0,152 en TRAIN et 0,149 en VAL. Il n'y a pas d'écart train/val : c'est un sous-apprentissage, et les données ne sont pas le facteur limitant actuel.
- Les runs utilisent Adam à 1e-5–2e-5, un batch de 8 fenêtres (4 par phase) et durent 500 à 4000 pas.
- Diagnostic canal0×16 : les mises à jour ont un RMS de 2e-4 à 9e-4 sur des poids MP de RMS≈1. La fonction n'a presque pas bougé (MSE 0,14217→0,14201).
- Adam fait un pas d'environ LR par élément. Sur des poids normalisés, 1e-5 est minuscule.

*Limites.* Aucune courbe longue à LR élevé n'existe. Je n'ai pas vérifié le LR d'entraînement du teacher (`consistency_base_192-3.cfg`). La capacité n'est donc démontrée ni suffisante ni insuffisante.

**H2 : l'objectif et les métriques sont mal alignés sur la morphologie.**
*Faits.*
- La phase 1 (t≈1,56, entrée ≈ bruit pur) est un générateur en un pas.
- Les amplitudes des élèves valent 0,63–0,82 : c'est le rétrécissement typique d'une MSE. Les termes énergie/bande ajoutent du bruit incohérent, pas des incisions.
- Le PTQ FP8 a une MSE canal0 de 0,03–0,06, mais garde corr>0,99 et rappel≈0,93.
- La QAT baisse le canal0 de 20–26 %, mais sa MAE est pire au pas 1000. La MSE canal0 ne juge donc pas la morphologie.

*Hypothèse.* En montagne, la MAE LOD0 est proche de la MAE LOD3 (47/51 m) avec corr 0,998. L'erreur serait surtout à grande échelle (canal4), canal que la normalisation par variance sous-pondère.

**H3 : les élèves partent loin de la fonction teacher.**
*Fait.* Plus l'élève part proche du teacher, meilleure est sa morphologie : QAT topologie intacte (corr>0,99), puis quart-rang (0,88), puis native160 et champ pleine largeur (≈0,85). Le champ, la condition dense et l'attention locale changent la fonction calculée. Passer de 128 à 192 canaux n'a rien apporté, mais les deux étaient sous-entraînés : aucun plafond n'est démontré.

**Précision.** Elle n'est une cause que pour la piste FP8 :
- plaines LOD3 : corr 0,42–0,82 et rms 1,2–1,9, contre ≈0,96–0,97 et ≈0,99 en BF16 ;
- côte humide, coarse original : rappel LOD0 0,03 au PTQ, puis 0.

Le plancher BF16 (0,1–1,8 m) met la précision hors de cause pour les élèves.

## 2. Ce que montrent les oracles et contrôles, et ce qu'ils ne montrent pas

*Ils montrent :*
- le wrapper et le blending sont corrects ;
- l'incision se perd dans le canal0 de phase 1 ;
- la phase 2 s'apprend : champ + première phase teacher donne 36/29/41 m, corr 0,997.

*Le QAT au checkpoint 500* (montagnes coarse4, LOD0 14,9/14,8/36,8 m contre 27/20/52 au PTQ) prouve qu'un court entraînement proche du teacher améliore le relief fin. Au pas 1000, la MAE remonte : 47/28/75 m en LOD0, et 51/39/65 m en LOD3 contre 35/28/41 au 500. C'est une dérive grande échelle que la loss ne voit pas. Les plaines LOD3 et la côte originale ne sont réparées à aucune étape.

*Ils ne montrent pas :*
- qu'une régression ne peut pas apprendre le canal0 ;
- une cause unique ;
- un drainage réel (le rappel TPI n'en est pas un) ;
- les raccords des familles récentes ;
- un gain de vitesse FP8.

## 3. Critique de la conduite expérimentale

- Des gardes ambitieuses (canal0÷2 à 1000 pas) ont arrêté des runs à LR minuscule.
- Plusieurs variables ont changé à la fois d'une famille à l'autre.
- Les décisions ont reposé sur des scores agrégés et sur la MSE canal0.
- Aucun balayage de LR adapté aux couches MP n'a été fait.
- Batch de 8 fenêtres seulement, et EMA demi-vie 500 sur un run de 1000 pas.
- Les trois mondes VAL montagne ont été réutilisés à chaque décision.
- La sélection s'est faite au dernier pas, alors que le QAT500 était meilleur en montagne.

## 4. Stratégie principale : compression progressive depuis le teacher exact, avec une KD longue et bien optimisée

- **Architecture et géométrie** : topologie native du teacher (fenêtres 64/pas 32, T2, blending, 58 scalaires, attention, skips). Les raccords et la sémantique des conditions sont ainsi hérités. Cible 128×2 : c'est la seule réduction native mesurée à ≈2× (1,726 s contre 3,462 s, 1024², 5090, poids aléatoires).
- **Initialisation** :
  - poids teacher exacts, avec des portes multiplicatives appliquées après la normalisation MPConv ;
  - canaux classés par importance sur TRAIN ;
  - suppression du 3ᵉ bloc par une porte résiduelle, en corrigeant la normalisation `mp_sum` (à vérifier dans `mp_layers.py`) ;
  - contrôle d'identité exacte avec toutes les portes à 1.
- **Déroulé** : annelage des portes de 1 à 0 sur 20k pas, compactage, puis consolidation sur 40–60k pas. Conserver un checkpoint intermédiaire de taille ≈160×3.
- **Données** : banques TRAIN actuelles, sans changement. Sampling 25/15/60, trois coarse, 50 % d'états de seconde phase on-policy (le code QAT est réutilisable).
- **Loss** :
  - x0 par phase ;
  - canaux 0–3 en MSE normalisée, première phase ×2 (pas de ×16) ;
  - canal4 en Huber, exprimé en mètres ;
  - KD des 37 blocs sur les canaux conservés, avec un poids décroissant ;
  - ratio d'amplitude par bande, à faible poids.
- **Optimisation** : AdamW, β2 0,99. LR fixé par le contrôle de la section 5 (1e-4, 3e-4 ou 1e-3). Warmup 1k, LR constant pendant l'annelage, puis cosinus jusqu'à 10 %. Batch effectif de 32 fenêtres (16 par phase). EMA demi-vie ≈2k ; évaluer les poids bruts et l'EMA.
- **Durée** : ≈80k pas. Les logs donnent 0,13 s/pas (native160, batch 8) et 0,59 s/pas (QAT avec KD et on-policy). Le coût à batch 32 doit être mesuré, pas extrapolé en promesse. La 5090 entraîne, la 4090 évalue et rend les planches.

## 5. Contrôle minimal falsifiable

Reprendre le native160×3 au pas 15000. Entraîner la phase 1 seule, teacher-forcée, batch 32, 3k pas à LR 1e-5 puis à 3e-4 (et 1e-3 si c'est stable). Mesurer la MSE canal0 en TRAIN et VAL, et la cohérence spectrale élève/teacher par bande.

- **H1 confirmée** si, à 3e-4, la MSE TRAIN baisse d'au moins 30 %, que VAL suit et que la cohérence monte dans la bande d'incision.
- **H1 réfutée** si TRAIN reste vers 0,14 : il y a alors un plafond de représentation ou d'objectif. Dans ce cas, abandonner la réduction de largeur et ajouter, sur la phase 1 seulement, un terme distributionnel (discriminateur sur les features gelées du teacher).

## 6. Arrêt, sélection, raccords, performance

- **Arrêt** : aucun arrêt qualité avant 20k pas, sauf NaN, divergence ou identité cassée. Ensuite, arrêter si deux évaluations espacées de 5k n'améliorent pas les portes de plus de 2 %.
- **Portes de qualité** (5 cas × 3 coarse × LOD3/LOD0), mesurées par rapport au BF16 et au QAT500 :
  - cohérence canal0 au moins égale au PTQ ;
  - amplitude par bande entre 0,9 et 1,1 ;
  - plaines LOD3 : corr ≥0,9 et rms entre 0,9 et 1,1 ;
  - côte : rappel non effondré, terre/mer stable ;
  - thalwegs vérifiés par accumulation D8 sur les hauteurs décodées, pas seulement par TPI.
- **Sélection visuelle** : une candidate est testable visuellement si toutes les portes passent, même avec une MAE montagne au-dessus de 35 m. Garder le meilleur checkpoint, pas le dernier. Faire un A/B aveugle contre le teacher, sur le même coarse et avec la même lumière.
- **Raccords** : seulement sur la finaliste, dans la chaîne app à froid, avec décalages de 32. Mondes 101/303/404 utilisés une seule fois.
- **Performance** : viser ≈2× sur l'étape base, d'après les coûts mesurés. En dessous de 128×2, le temps plafonne vers 1,5 s à cause des surcoûts hors réseau. Le FP8 n'a apporté aucun gain mesuré.