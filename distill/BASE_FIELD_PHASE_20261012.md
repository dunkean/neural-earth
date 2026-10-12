# Base : vallées humides et reprise du champ

Le premier CNN distillé avait supprimé les vallées d'érosion. Les montagnes
humides avec vallées encaissées restent donc un critère de sélection explicite,
en plus des quatre cas rares historiques. Les critères numériques de relief et
de pluie servent à choisir des scènes teacher ; ils ne prouvent pas à eux seuls
la présence d'un réseau fluvial. Les rendus teacher doivent aussi être inspectés.

Les données sont sous `~/data/distill/crops/native-wet-mountains-v1` et
`native-wet-mountain-train-extension-v1`. L'entraînement échantillonne 25 % de
montagnes humides, 15 % des anciens cas rares et 60 % de scènes générales, avec
les branches coarse original, coarse4 et coarse2. Il dispose de quinze mondes
montagneux TRAIN ; les trois mondes VAL620001–620003 et les mondes finaux réservés
101/303/404 restent exclus du TRAIN. Les cibles proviennent du base teacher sur
le coarse propre à chaque branche.

Le champ pleine largeur192×3, deux phases, s'est arrêté au pas2037 sur son
garde-fou, avant les6000 prévus. À2000, coarse4 en montagne humide donne encore
125–147 m d'erreur moyenne décodée, avec une amplitude TPI8 de0,66–0,82 : les
vallées restent atténuées. Ce modèle n'est pas accepté. Rapports et états sous
`~/data/distill/ckpt/field-full-width-pilot`; les anciens paliers restent conservés
à l'identique sur D: avec liens et SHA vérifiés, et l'état complet2037 reste sous
Linux pour la reprise.

Le contrôle causal à1500 (`eval/field-oracles1500.json`) donne29–41 m et une
corrélation TPI8 de0,997 lorsque la seconde passe reçoit le premier champ teacher.
Cela désigne le premier champ comme défaut principal, sans prouver une incapacité
architecturale. Le champ teacher reconstruit256 possède un anneau externe32
partiellement fusionné ; seul son centre192 est vérifié contre la cible exacte.

Après revue réelle Opus5.5 (`review/opus-field-oracle-guard2000-20261012`), la
reprise bornée teste des coordonnées périodiques de fenêtres : seize canaux
Fourier, deux projections additives initialisées à zéro. Les poids bruts2037,
moments Adam existants, RNG et EMA sont conservés ; les nouveaux paramètres ont
leur propre état Adam. La loss du premier champ passe de0,5 à1,0. La seconde
passe reçoit parfois le champ teacher détaché (50 %, puis25 %) pendant le TRAIN.
Les comparaisons et la sélection utilisent exclusivement les deux passes
autonomes. Les raccords supervisés comparent également deux chaînes autonomes.

Commandes : `python -m distill.field_phase_train --smoke --output <proof>`, puis
`python -m distill.field_phase_train --smoke-root <proof>`. La5090 entraîne,
la4090 exécute `python -m distill.field_phase_watch` tous les500 pas. Budget4000
nouveaux pas, contrôle intermédiaire2000 ; arrêt sur régression importante des
vallées ou baisse répétée de leur amplitude. Photos réelles, CSV, JSON et texte
sont déposés à plat dans `~/data/distill/_base/field-phase-*.png` ; diagnostics
des lignes sous `field-attention-lines-phase-*.png`.

Il reste à valider qualité, autres cas rares, performances au repos et véritables
raccords froids dans l'application. La cohérence entre champs décalés est un
diagnostic préalable, pas cette validation finale. Aucune accélération acquise
n'est annoncée et le base original reste le défaut de l'application. Aucun poids
base expérimental n'est ajouté à Git.

## Contrôle causal des canaux au1000

La comparaison des cinq canaux du premier champ (`field_first_features.py`)
montre qu'au500 le canal0 concentre environ80% de l'erreur normalisée des
quatre features. Cette erreur diminue4–5% par rapport à l'EMA2037, mais la
géométrie finale autonome reste insuffisante : au1000, montagne/coarse4 donne
150/130/109m, corrélation TPI8 de0,848/0,854/0,896.

`field_channel_oracles.py` remplace séparément des canaux initiaux par leurs
cibles teacher enregistrées, sans changer les poids ni les bruits. Sur les
trois mondes VAL coarse4, remplacer le seul canal0 ramène l'erreur décodée à
56/28/66m et la corrélation à0,993–0,994. Remplacer les trois autres features
ou la hauteur seule corrige beaucoup moins le terrain. C'est un contrôle causal
du défaut, pas une performance du modèle autonome ni une acceptation.
L'anneau teacher externe32 reste partiellement fusionné dans ce contrôle.

Résultats externes : `eval/field-first-features500.json` et
`eval/field-channel-oracles1000.json`, accompagnés de textes. Les photos
autonomes au1000 sont dans `_base/field-phase-<coarse>-001000.png`.
Ces données sont transmises à Opus dans `review/opus-field-channel0-20261012`.
La recette du producteur actif reste figée ; toute nouvelle stratégie doit
attendre son garde-fou ou son budget et conserver son état de reprise.
