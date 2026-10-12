# Base NF4 : précision du gain et correction de niveau

Le70k de secours reste inchangé dans l'application. Ce travail prépare un
nouvel essai séparé ; il ne constitue pas encore une amélioration acceptée
du modèle, de sa vitesse ou de son grain.

## Résultats acquis

**Gain de sortie FP32.** Sur sept capturesTRAIN, promouvoir directement
`out_gain` change déjà le calcul de `gain/sqrt(fan_in)` : le biais des trois
montagnes augmente de6,39 /9,50 /5,86m sans changer la valeur nominale.
Le prototype avec référenceBF16 fixe reproduit **exactement** les deux
étapes, leurs entrées, les latents fusionnés et les hauteurs décodées au départ.
Un incrément de1/16 de cranBF16 permet ensuite des déplacements de niveau
d'environ1,1 à5,0m en montagne. Cela supprime le grand cran scalaire ; les
convolutions et activations restentBF16, donc le calcul n'est pas continu
au sens mathématique. Aucun réglage unique n'a encore été calibré ou validé.

**Attribution sur32mondesTRAIN indépendants**, nouveaux bruits, trois coarses,
même empriseLOD3/0. TT désigne teacher puis teacher ; ST élève puis teacher ;
TS teacher puis élève ; SS élève puis élève. Biais absolu moyenLOD0 en mètres :

| Catégorie | ST | TS | SS |
|---|---:|---:|---:|
| Plaine désertique |0,141|0,895|0,945|
| Plaine tempérée |0,309|0,542|0,733|
| Côte humide complexe |1,988|0,552|2,498|
| Plaine basse/mer |0,118|0,071|0,119|
| Montagne humide |12,102|9,430|14,273|

Il faut différencier **les deux étapes** pour corriger le niveau. Les erreurs
ne s'additionnent pas systématiquement ; elles varient selon le terrain.

**Objectif de niveau isolé.** Résidu enracine signéeq, pondéré par le jacobien
du teacher détaché ; flouσ8latents, plus basse fréquenceC0–C3. CibleTT
indépendante de l'élève. Aucun passe-haut ni terme de côte dans cet essai.
Les37scalairesglobaux reçoivent zéro gradient **auxiliaire** ; leur gradient
principal reste disponible. Ce n'est pas un gel arbitraire aux valeurs teacher.

**Planchers TRAIN**,4090, mêmes32mondes/bruits : BF16contreFP32 du base seul,
solver/fusion et decoderBF16 inchangés. RMS basse fréquence en mètres :
désert0,951 ; tempéré0,799 ; côte0,633 ; plaine/mer0,156 ; montagne26,329.
Ces valeurs normalisent l'objectif, elles ne certifient pas le plancher5090
ni une qualité sur les mondesVAL/FINAL.

**Backward réel5090**, cinq strates : gradients finis à travers les deux
étapes, teacher sans gradient, scalaires auxiliaires protégés ; pic mémoire
autour de7,1Go et3–5s par probe complète. Les gradients auxiliaires sont
beaucoup plus forts que ceux deKD : leur coefficient doit être calibré.
Cinq testsCPU vérifient l'identité à la référence, le gradient du gain et
les invariants de l'objectif/protection des scalaires.

## En cours / suite

Calibration5090 sur32régions, précédée de500pasAdam à **LRzéro** pour mesurer
le préconditionnement. Les poids doivent rester exactement identiques à70k ;
les moments sont nouveaux, aucune reprise exacte d'Adam70k n'est revendiquée.
Comparer ensuite deux bras issus du même70k, avec témoinsλ0 et états conservés.
Ne lancer la nouvelle paire longue qu'après vérification des gradients par
tenseur et du smoke. Le grain et les incisions restent des critères séparés.

## Preuves et données

Racine locale : `/home/delete/data/distill/`.

- `eval/native-nf4-scalar-precision-train-probe.json`
- `eval/native-nf4-level-train-attribution-5090/{contract,report}.json`, `arrays.npz`
- `eval/native-nf4-level-train-floor-4090.json`
- `eval/native-nf4-level-gradient-probe-5090.json`
- Calibration en cours : `eval/native-nf4-level-gradient-calibration-5090.json`
- Moments nouveaux prévus : `ckpt/native-nf4-level-init/adam500.pt`

Les instantanés70k/100k, les paquets vérifiés et les captures précédentes sont
conservés. Aucun poids base expérimental n'est ajouté àGit.
