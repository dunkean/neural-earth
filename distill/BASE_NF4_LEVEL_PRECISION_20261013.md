# Base NF4 : précision du gain et correction de niveau

Le 70k de secours reste inchangé dans l'application. Ce travail prépare un
nouvel essai séparé ; il ne constitue pas encore une amélioration acceptée
du modèle, de sa vitesse ou de son grain.

## Résultats acquis

**Gain de sortie FP32.** Sur sept captures TRAIN, promouvoir directement
`out_gain` change déjà le calcul de `gain/sqrt(fan_in)` : le biais des trois
montagnes augmente de 6,39 / 9,50 / 5,86 m sans changer la valeur nominale.
Le prototype avec référence BF16 fixe reproduit **exactement** les deux
étapes, leurs entrées, les latents fusionnés et les hauteurs décodées au départ.
Un incrément de 1/16 de cran BF16 permet ensuite des déplacements de niveau
d'environ 1,1 à 5,0 m en montagne. Cela supprime le grand cran scalaire ; les
convolutions et activations restent BF16, donc le calcul n'est pas continu
au sens mathématique. Aucun réglage unique n'a encore été calibré ou validé.

**Attribution sur 32 mondes TRAIN indépendants**, nouveaux bruits, trois coarses,
même emprise LOD3/0. TT désigne teacher puis teacher ; ST élève puis teacher ;
TS teacher puis élève ; SS élève puis élève. Biais absolu moyen LOD0 en mètres :

| Catégorie | ST | TS | SS |
|---|---:|---:|---:|
| Plaine désertique |0,141|0,895|0,945|
| Plaine tempérée |0,309|0,542|0,733|
| Côte humide complexe |1,988|0,552|2,498|
| Plaine basse/mer |0,118|0,071|0,119|
| Montagne humide |12,102|9,430|14,273|

Il faut différencier **les deux étapes** pour corriger le niveau. Les erreurs
ne s'additionnent pas systématiquement ; elles varient selon le terrain.

**Objectif de niveau isolé.** Résidu en racine signée q, pondéré par le jacobien
du teacher détaché ; flou σ=8 latents, plus basse fréquence C0–C3. Cible TT
indépendante de l'élève. Aucun passe-haut ni terme de côte dans cet essai.
Les 37 scalaires globaux reçoivent zéro gradient **auxiliaire** ; leur gradient
principal reste disponible. Ce n'est pas un gel arbitraire aux valeurs teacher.

**Planchers TRAIN**, 4090, mêmes 32 mondes/bruits : BF16 contre FP32 du base seul,
solver/fusion et decoder BF16 inchangés. RMS basse fréquence en mètres :
désert 0,951 ; tempéré 0,799 ; côte 0,633 ; plaine/mer 0,156 ; montagne 26,329.
Ces valeurs normalisent l'objectif, elles ne certifient pas le plancher 5090
ni une qualité sur les mondes VAL/FINAL.

**Backward réel 5090**, cinq strates : gradients finis à travers les deux
étapes, teacher sans gradient, scalaires auxiliaires protégés ; pic mémoire
autour de 7,1 Go et 3–5 s par probe complète. Les gradients auxiliaires sont
beaucoup plus forts que ceux deKD : leur coefficient doit être calibré.
Cinq tests CPU vérifient l'identité à la référence, le gradient du gain et
les invariants de l'objectif/protection des scalaires.

## Calibration terminée et paire lancée

Calibration 5090 sur 32 régions terminée, après 500 pas Adam à **LR zéro**.
Identité exacte de tous les poids vérifiée ; les moments sont nouveaux, aucune
reprise exacte d'Adam 70k n'est revendiquée. Coefficient figé : 2,156151e−5.
Pic alloué 10,2 Go, 119 s pour les 32 probes après les pas à taux nul. Avant
plafonnement, 138 couples région/tenseur dépassaient un ratio préconditionné
de 10. Avec le plafond global de 0,5, aucun ; maximum observé 7,20.

Smoke GPU terminé : A/B avec λ=0 ont exactement les mêmes poids, EMA, Adam,
RNG, pertes et normes. Trois batchs réels de huit fenêtres confirment l'identité
du snapshot scalar-QAT et du runtime aux convolutions NF4 préparées ; le gain
FP32 est conservé. 21 tests CPU passent, incluant les règles du catalogue live
et les protections des captures de validation.

La paire 70k→80k est lancée sur 5090 : A garde KD seul ; B ajoute la basse
fréquence TT aux deux étapes et protège les 37 scalaires contre ce seul
gradient auxiliaire. A reprend les nouveaux moments à 500 ; B les construit
avec son propre objectif à taux nul. Montée de LR sur 500 updates, puis 3e−6,
EMA demi-vie 500. Blocs A/B entrelacés de 2500 updates, raw/EMA conservés à
72,5k / 75k / 77,5k / 80k. Évaluations sur 4090, coarse4/2, images en vrac
dans `_base`. Les deux LOD des nouveaux rapports ont la même emprise ; le
proxy TPI résumé correspond à 480 m aux deux LOD. Le grain et les incisions
restent des critères séparés ; aucun bénéfice de qualité n'est encore établi.

## Preuves et données

Racine locale : `/home/delete/data/distill/`.

- `eval/native-nf4-scalar-precision-train-probe.json`
- `eval/native-nf4-level-train-attribution-5090/{contract,report}.json`, `arrays.npz`
- `eval/native-nf4-level-train-floor-4090.json`
- `eval/native-nf4-level-gradient-probe-5090.json`
- Calibration : `eval/native-nf4-level-gradient-calibration-5090.json`
- Moments nouveaux : `ckpt/native-nf4-level-init/adam500.pt`
- Smoke définitif : `eval/native-nf4-level-paired-final-smoke-5090.json`
- Paire : `ckpt/native-nf4-level-70k-80k/`
- Job durable : `base-nf4-level-paired-70k-80k`

Réserve disque : les états complets inactifs de NF4 ont été copiés vers
`/mnt/d/workspace/distill-archive/`, vérifiés par SHA256 avant remplacement
par liens. Aucune copie unique n'est supprimée ; les poids/données de la paire
active restent dans le système de fichiers Linux. Reçu :
`preparation/nf4-level-reserve-cold-archive.json`.

Les instantanés 70k/100k, les paquets vérifiés et les captures précédentes sont
conservés. Aucun poids base expérimental n'est ajouté à Git.

## Premier témoin visuel, avant les résultats B

A à72k a été figé depuis un état complet atomique, sans toucher au trainer.
Une copie70k avec gain FP32 à sa référence reproduit mathématiquement le
point initial et permet les mêmes métriques sur la même emprise. Les deux
évaluations4090 sont terminées : sept capturesVAL, coarse4, decoder original.
La MAE moyenneLOD0 passe de11,11 à9,63m ; cela masque des différences :
les plaines désertique/tempérée régressent de2,60→3,12m et1,40→2,38m, alors
que les trois montagnes passent de26,30→20,87 /13,04→12,89 /32,51→26,32m.
Ce témoin montre qu'un redémarrage KD seul peut améliorer certaines montagnes
et aggraver des plaines : **aucun bénéfice de B n'est encore démontré**.
Le grainLOD3 reste visible sur la planche ; la qualité n'est pas acceptée.

Planches et alternativesCSV/JSON/TXT dans `_base` :

- `native-weight-nf4levelstart-coarse4-070000.*`
- `native-weight-nf4levelaearly-coarse4-072000.*`

Comparaison appariée : `eval/native-nf4-level-a-early-matched-comparison.json`.
A a depuis terminé son premier bloc à72,5k ; B construit ses propres moments
à taux nul avant ce même bloc. Le [paquet avec scalaireFP32](NF4_SCALAR_PACKING.md)
a aussi été audité sur4090 :149,654742Mo tout compris, champs et decoder exacts
par rapport au prototype avant export. Le paquet du candidat retenu devra être
exporté et réaudité sur5090.
