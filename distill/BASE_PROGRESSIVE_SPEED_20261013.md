# Programme lancé : accélérer le base par réduction progressive

Autorisation du 13 octobre : lancer les entraînements et faire reviewer chaque
grande étape par **Opus 5.5**. B75 publié reste disponible pour l'intégration.
La nouvelle cible est le gain de calcul à qualité visuelle proche, avec poids
compatibles avec le budget de téléchargement. Aucun gain final n'est déclaré.

## Premier palier en cours

Job `base-progressive160-anneal20k-5090`, données et checkpoints sous
`~/data/distill/ckpt/native-progressive160-v1/anneal/`.

- Départ teacher original BF16, identité exacte vérifiée avant réduction.
- **20 000 mises à jour** : extinction graduelle des canaux192→160, trois blocs,
  attention, conditionnement58, fenêtres64/32 et **deux passes** conservés.
- Les kernels projetés, normalisations encoder/embedding et facteurs du concat
  évoluent graduellement. À l'extrémité, leur sémantique correspond au réseau
  physiquement plus étroit ; les petites différences numériques BF16 liées
  aux dimensions des opérations sont mesurées séparément.
- Teacher original figé ; supervision de vitesse, gradients et états x0 sur
  les cinq canaux, plus features dans les coordonnées retenues. Un auxiliaire
  basse fréquence TT traverse les deux étapes et protège le niveau des terres
  et mers. Pas de loss d'énergie seule qui récompense du grain inventé.
- Données TRAIN indépendantes : 25%montagnes humides, environ15%cas critiques,
  reste général, coarse original/4/2 ; moitié des batchs sur l'état courant de
  l'élève. Réserve mémoire CPU bornée à96 régions.
- Gain de sortie ancréFP32 ; gains d'embedding gelés durant ce premier palier.
  LR warmup1000 vers1e-5, décroissance vers3e-6 ; EMA demi-vie500 ; clipping1.
  L'auxiliaire est plafonné à25% des gradients bruts et préconditionnésAdam.

La **5090 entraîne**. Le job
`base-progressive160-evaluation-review-4090` évalue sur la4090 tous les2000pas
avec coarse4 et coarse2 : plaines désertiques/tempérées, côte humide, plaine
basse/mer et trois montagnes humides à vallées incisées. Teacher BF16 recalculé
sur le même GPU et le même bruit. PNG/CSV/JSON/TXT dans `_base`, en vrac :
`base-progressive160-<step>-<coarse>.*`. Les images de progression agrandissent
les champs natifs32/256px ; elles ne remplacent pas les planches finales512px.

## Fin de palier et revue automatique

Le suivi attend la fin du trainer, puis matérialise le réseau160, vérifie les
kernels, mesure le changement numérique sur cinq champs TRAIN et exporte un
checkpoint compact. Ce réseau compte **178 138 245 paramètres**. Les temps
d'inférence et les vrais raccords froids seront mesurés sur les poids entraînés,
sur les deux axes et halos, avec coarse4 et decoder livré communs.

Le suivi rédige `STAGE1_REPORT.md` avec les preuves et lance le CLI
`claude-opus-5-5 --effort xhigh`, lecture seule et outils pré-autorisés. Il
conserve la revue **complète**, le transcript et l'identité du modèle sous
`~/data/distill/review/opus-progressive160-stage1/`. Une erreur termine le
contrôle concerné ; aucun redémarrage silencieux. L'avis doit être analysé avant
de fixer la récupération suivante.

## Paliers suivants

1. Récupération du réseau160, ordre de grandeur30kpas, ajustée avec la revue.
   Nouvelle revue Opus sur qualité, vitesse et raccords.
2. Réduction graduelle160→128 en conservant les trois blocs/deux passes, puis
   récupération. Ordre de grandeur20k+30kpas, avec revue entre grands paliers.
   Environ100kpasses au total pour cette proposition ; les recettes suivantes
   ne sont pas encore figées, les avis et images peuvent les modifier.
3. QAT/stockage du réseau retenu, réel benchmark CPU/GPU selon intégration,
   lieux rares/réservés, raccords canoniques et ordinaires, bonnes planches512px
   et nouvelle release. Objectif initial : environ2× mesuré, sans promettre
   que la baisse de FLOPs se convertira exactement en baisse de latence.

La suppression de blocs puis la distillation vers une passe restent des
étapes ultérieures, après un modèle plus étroit jugé correct.

## Contrôles de lancement

32batchs TRAIN pour choisir les canaux/têtes ; trois contrôles source exacts ;
backward à trois amplitudes de réduction ; kernels compactés exacts ; drift
BF16 explicite ; EMA correctement reliée à ses propres paramètres ; quatre
vrais pas avec état courant et auxiliaire TT plafonné. Pic du smoke13,2Go.
Trois testsCPU couvrent normalisations des canaux conservés, identité source,
chargeur compact et rejet des états encore en cours de réduction.

Le premier smoke au début du calendrier avait des gradients nuls parce que
la réduction était sous le seuilBF16. Le contrôle a été répété à des amplitudes
non nulles, avec gradients finis sur les quatre pas. Le premier seuil de
forward compact1e-5 a été dépassé de peu : des kernels exacts et un budget
numérique séparé ont remplacé l'exigence d'un forward GPU identique.
