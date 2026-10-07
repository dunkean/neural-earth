# Runtime NN — Astra HIGH, revue r1

Date : 2026-10-07. Périmètre : différences terminées entre le snapshot
`E:/TerrainDiffusionRuntime/runtime-lod-optimization/before` et les sources
actuelles `terrain_inference.py`, `terrain_nn_constants.py` ; harnais
`benchmark_runtime_lod.py`, `benchmark_runtime_progression.py` ; artefacts
baseline, batched-inputs, final et progression. Les raccords macro A4 présents
dans le diff ne sont pas validés comme un bloc macro terminé.

Sources en lecture seule. Aucun serveur démarré, aucun nouveau calcul CUDA.
Contrôles locaux limités à la lecture d'artefacts NumPy et à des opérations
PyTorch explicitement CPU.

## Verdict : ACCEPT WITH LIMITS

Aucun P0/P1/P2 de correction numérique ou de durée de vie des caches trouvé
sur le chemin de production examiné. L'optimisation LOD 3 a une preuve
mesurée et une explication concrète. Un gain stable LOD 2 n'est **pas** acquis
par les séries actuelles ; ce verdict ne ferme pas l'objectif de performance
LOD 2. Les mesures appariées en cours devront être examinées séparément.

## Revue du calcul et des caches

`terrain_inference.py:286` prépare en une fois le batch latent effectivement
choisi par InfiniteTensor. Le découpage des dépendances, les frontières des
batches, l'ordre des contextes, les offsets des seeds et l'accumulation des
fenêtres ne sont pas changés. La conversion BF16 des samples précède toujours
leur normalisation ; les expressions du bruit, de `x_t` et de la prédiction
gardent leur ordre. La moyenne climat reste une réduction 2 × 2 indépendante
pour chaque fenêtre. Le traitement des NaN reproduit le comportement de
l'ancien chemin GPU, lequel appelait le traitement par fenêtre de batch un.

L'assemblage NumPy du bruit suit exactement l'ordre des contextes, avec les
mêmes appels `gaussian_noise_patch`. Il remplace les multiples transferts
H2D par un transfert du batch, sans augmenter le batch modèle ni le compléter
artificiellement. Les sorties sont redistribuées dans cet ordre ; la fusion
reste dans InfiniteTensor. Le chemin CPU historique est conservé.

Le cache `_terrain_latent_constants` (`terrain_inference.py:298`) appartient
au monde et est invalidé par `_build_latent` (`:341`). Sa clé `id(t)` est
suffisante dans les appelants actuels : les tenseurs de temps sont retenus
par les closures de construction du stage pendant toute la vie de celui-ci.
Les moyennes, écarts-types, histogrammes et poids y sont constants ; leur
reconstruction invalide le cache. Ce raisonnement ne vaut pas pour un futur
appel arbitraire avec un temps temporaire ou des constantes mutées sans
reconstruction.

Les nouveaux caches de `terrain_nn_constants.py` conservent le résultat des
expressions scalaires complètes, notamment division puis multiplication du
facteur de concaténation. Ils ne réassocient pas les arrondis BF16. Les clés
des poids incluent valeurs, dtype et device ; les références conservées dans
`_weights` empêchent la réutilisation des identifiants employés par les caches
dérivés. Le noyau constant de resampling dépend des canaux, du mode, du
facteur, du dtype et du device. Les opérations avec gradients reviennent
aux fonctions originales. Les caches sont globaux au processus, mais leur
cardinalité reste finie pour les architectures/dtypes/devices fixes utilisés
ici ; ils ne stockent aucun résultat dépendant de la seed ou des coordonnées.

Le passage par `mp_layers` dans `_mp_forward` assure que les alias consultent
bien ces helpers préparés. Les remplacements et leur restauration couvrent
les aliases utilisés. Les poids des réseaux, la précision, le solver, les
tailles des fenêtres et les modèles restent inchangés dans ce bloc.

## Fidélité vérifiée indépendamment

Lecture CPU des NPZ, comparaison de dtype, shape et **octets** de chaque
tableau, plus contrôle de finitude :

- baseline → batched-inputs : 24 tableaux, zéro différence.
- baseline → final : 24 tableaux, zéro différence.
- progression-baseline → progression-final : 20 tableaux, zéro différence.

Les 24 tableaux représentent 12 cas LOD 3/2, chacun avec hauteur et cinq
plans climat. Ils incluent premier usage et réutilisations des mêmes trois
sites/seed avec stores neufs ; ce ne sont pas 12 sites géographiques distincts.
La progression ajoute les cinq étapes pour deux seeds avec réutilisation
du store. Les histogrammes des batches et les nombres de forwards modèle
restent identiques entre avant/après pour les cas comparés.

Contrôle CPU supplémentaire sous `torch.inference_mode()` : helpers sum,
concat et resample comparés à leurs originaux, FP32/BF16, batchs 1/4/16,
poids par défaut, scalaire et liste, puis keep/down/up/up_bilinear. Les
premiers appels et les hits de cache sont identiques bit à bit. Aucun
tenseur CUDA alloué par cette vérification.

Les sources lues correspondent exactement aux empreintes de `final.json` :

```text
terrain_inference.py
03fd47dd5c2e6219628b42669865929f58f029350aa3d8cd320d57d83386632c
terrain_nn_constants.py
2d2465089ba7b0e1634e12f2624c83b6833644594f52d9e1f6c077ee3113b87d
```

La fidélité est établie pour ces cas et ce runtime ; elle ne démontre pas
l'identité sur toute combinaison de matériel, profil, batch expérimental
ou entrée possible. Les gates de référence existants restent pertinents.

## Lecture des performances

Le harnais crée un store NN neuf par cas, conserve les modèles résidents,
désactive le service des mips finaux, synchronise CUDA autour du temps mural
et isole une relecture du même store. Il sépare le premier cas par LOD des
cinq cas suivants. Les compteurs distinguent forwards modèle et batches de
stages : 80 forwards coarse correspondent à quatre fenêtres × 20 steps,
pas à 80 fenêtres coarse. Les durées inclusives de stages et modèles ne
doivent pas être additionnées comme des postes exclusifs.

| Série | Médiane LOD 3 | Médiane LOD 2 | Portée |
|---|---:|---:|---|
| baseline | 2,978 s | 4,593 s | 5 cas chauds, stores neufs |
| batched-inputs | 2,272 s | 3,731 s | 5 cas chauds, stores neufs |
| final | 2,117 s | 5,356 s | 5 cas chauds, stores neufs |

La baisse LOD 3 finale est d'environ 29 % sur cette série. Le LOD 2 final
va de 3,692 à 5,658 s : annoncer le seul résultat initial favorable comme
gain final stable serait incorrect. Le coût decoder de 25 forwards demeure.
Les deux parcours progressifs mesurés ont des LOD 3/2 plus rapides après
optimisation, mais deux seeds, sans répétitions ni maîtrise de l'activité
graphique du bureau, ne suffisent pas à estimer une distribution de latence.
Les LOD 1/0 autour de 10 ms dans ces parcours réutilisent le même store ;
ce ne sont pas des temps d'inférence indépendante à froid.

Les premiers artefacts et le harnais progression disent « Exclusive GPU ».
L'information disponible signifie seulement serveur Terrain arrêté et pas
d'autre worker NN Terrain ; l'activité graphique du bureau est incontrôlée.
Le harnais LOD actuel formule cette limite explicitement et capture un état
GPU avant/après. Ces instantanés ne constituent pas une mesure continue de
la charge pendant chaque échantillon. Aucun de ces benchmarks ne mesure
HTTP, le rendu, le délai d'une navigation complète ni un viewport entier.
Les p95 sur cinq échantillons sont des quantiles descriptifs interpolés,
pas une estimation robuste du p95 utilisateur.

Au moment de cette revue, le premier couple LOD 2 apparié complet donne
4,271 s avant et 3,945 s après (deux cas chauds seulement). Il confirme que
des résultats favorables existent, sans lever à lui seul la variabilité
observée. Les autres couples en cours ne sont pas présumés réussis.

## Complément — deux ordres appariés et bilan numérique complet

Les quatre processus appariés sont maintenant complets et ont été relus,
ainsi que `docs/RUNTIME_LOD_OPTIMIZATION.md` et `numerical-summary.json`.
Les quatre temps chauds par version donnent :

| Version | Paire 1 | Paire 2, ordre inversé | Médiane regroupée |
|---|---|---|---:|
| Avant | 4,225451 ; 4,315752 s | 5,276899 ; 4,699797 s | 4,507774 s |
| Final | 4,173656 ; 3,715996 s | 3,882240 ; 3,776479 s | 3,829359 s |

Le gain regroupé est de 15,05 %, avec des couples favorables dans les deux
ordres. Ce sont quatre valeurs par version sur deux sites chauds de runtime,
avec stores NN neufs, pas quatre sites indépendants. Le bureau reste actif
et la série finale défavorable de 5,356 s reste conservée. La nouvelle
documentation présente correctement ces limites : gain LOD 2 observé sur
les couples, sans garantie stable de latence ou de p95.

Vérification CPU indépendante répétée sur tous les groupes du bilan :
empreintes SHA-256 des sept NPZ conformes au reçu, dtype/shape et octets de
chaque array identiques à la référence correspondante. **92 arrays vérifiés**
(24 + 24 + 6 + 6 + 6 + 6 + 20), aucun écart, y compris pour les zéros signés.
Le verdict **ACCEPT WITH LIMITS** est maintenu. Le bloc dispose maintenant
d'une preuve favorable appariée au LOD 2, toujours limitée à ce corpus et à
ces conditions de mesure.
