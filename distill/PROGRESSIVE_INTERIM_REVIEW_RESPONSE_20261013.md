# Suite retenue après la revue intermédiaire Opus5.5

Avis complet : [REVIEW_OPUS_PROGRESSIVE_INTERIM_20261013.md](REVIEW_OPUS_PROGRESSIVE_INTERIM_20261013.md).
Identité `claude-opus-5-5`, effortxhigh, session
`c4ed0524-f9d9-4f10-b2c2-d49e4a11d605`, sortie0. Les preuves originales et
les copies du rapport/complément avec SHA vérifiés sont dans
`~/data/distill/review/opus-progressive160-interim6000-retry/`.

## Décision

Terminer le retrait192→160 à20k sans changer son code ni sa recette. La
projection est correcte ; les pertes et métriques ne prouvent pas une limite
de capacité. Le retour visuel positif à4k et la tolérance au bruit restent
pris en compte, avec l'objectif actif d'accélérer les trois réseaux.

À20k, comparer raw etEMA, y compris leurs maîtres FP32 dans `latest.pt`, avant
de choisir le départ de récupération. Le compacteur accepte désormais
`--weights raw|ema --keep-master` : l'inférence conserve ses poids BF16 mais
`recovery_master` garde les sous-incréments FP32. Il retire le vieil optimizer
large du payload compact ; ce sera une nouvelle phase, pas une fausse reprise
de ses moments. Les noyaux et champs seront vérifiés sur les vrais poids20k.

## Corrections de récupération

- Superviser le canal4 de x0 en mètres aux deux passes, sans flou ni simple
  objectif de puissance. Calibrer la contribution sur les gradients TRAIN,
  pas uniquement sur les valeurs numériques de loss.
- Renforcer le chemin TT à deux passes : hauteur pleine résolution et côte
  près de0m, en plus du niveau basse fréquence. Recette de départ proposée
  par Opus : auxiliaire tous les2pas, plafond de gradients0.5, à vérifier
  avant le vrai lancement.
- Réduire/décroître le featureKD pendant la redistribution des canaux. Le
  poids0.045 proposé est une base à recalibrer ; le ratio des losses n'est
  pas le ratio de leurs gradients.
- Garder25% de montagnes humides et augmenter les cas côtiers/plaines basses
  vers20%, avec coarse2 et4. Journaliser des moyennes par phase, état courant
  et catégorie sur100pas : les lignes actuelles sont aliasées.
- Redémarrer les moyennes à largeur fixe, comparer les demi-vies500/2000,
  puis sélectionner sur les planches, niveaux, côtes, relief et raccords.

Les fonctions physiques sont préparées dans
`native_progressive_recovery_loss.py` et couvertes par quatre testsCPU :
unités réelles, deux passes, erreur de position malgré une puissance identique,
et sens du gradient mer/terre. Le trainer de récupération et sa calibration
GPU ne sont pas encore lancés.

## Contrôles et nuances

Un contrôle borné sur4090 a réussi : quatre batchs sur/hors politique,
loss/gradients nuls et quatre pasAdam sans modifier le teacher ; cinq champs
TRAIN TT frais exactement identiques, avec gradients des anciennes et
nouvelles pertes à0. Preuve :
`~/data/distill/eval/native-progressive160-zero-identity-4090.json`.
Cela complète l'identité déjà prouvée ; ce n'est pas le contrôle2k proposé
par Opus et ne permet pas de conclure à la convergence d'un élève réduit.

La sélection par groupes64 hors attention est confirmée dans l'ancien
`native_compression.py`. Elle sera séparée de la sélection des têtes avant
une nouvelle réduction de largeur ; le fichier verrouillé du run actif
reste inchangé. Comparer les projections initiales surTRAIN avant adoption.

Nuance à l'avis : les normes inférieures au seuil dans les lignes journalisées
ne prouvent pas que le clipping n'a jamais agi entre ces lignes. La nouvelle
journalisation lèvera cette ambiguïté.

## Priorité de vitesse

Le160 seul donne seulement1.15–1.34× dans le test architectural. Après
récupération, priorité à moins de blocs et/ou une passe réellement distillée
vers la sortie du teacher à deux passes. La largeur128 vient ensuite si les
mesures la justifient, avec la sélection corrigée. Ces gains futurs ne sont
pas annoncés comme mesurés.

À20k : compléter le banc avec batch4, comparer B75 au même protocole, mesurer
raccords/recouvrements et temps de chaîne. La revue finale du palier reste
prévue. Les seuils proposés par Opus servent à comparer les défauts ; le
jugement visuel de l'utilisateur prime. Decoder puis coarse suivront, avec
les mêmes exigences de mesure et des poids intégrables.
