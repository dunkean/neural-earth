# Suite du base NF4 après100k : diagnostic et stratégie

L'[avis complet d'Opus5.5 xhigh](REVIEW_OPUS_NF4_FINAL_BIAS_20261012.md)
est conservé sans condensation. Il confirme que la perte passe-haut de hauteur
`h=q|q|` offre une échappatoire : réduire le niveau q réduit aussi l'amplitude
du grain en mètres. Cela n'isole pas à lui seul la cause expérimentale de B,
mais explique pourquoi la perte précédente ne protège pas le niveau.

## Vérifications effectuées

1. **D0, lecture CPU de46états** : pas d'initialisationCUDA, aucune mise à jour.
   LesEMA A/B à42,5k/45k/47,5k diffèrent uniquement par le scalaire BF16
   `out_gain` parmi les scalaires ; à50k un `emb_gain` diffère aussi.
   A vaut−0,1953125 et B−0,1943359375 : B a une **amplitude plus faible**
   d'un cranBF16 (les gains sont négatifs). Le run principal présente aussi
   des sauts de biais aux changements de cran, mais dérive entre ces sauts :
   le gain ne peut pas expliquer tout le problème.
2. **Ablation causale TRAIN**, sans training : les poids deB50k restent fixes,
   seul son `out_gain` est remplacé temporairement par celui deA50k. Sur les
   trois montagnes, le biaisLOD0 monte de **36,84 /53,01 /32,01m**. Sur les
   quatre plaines/côtes, il monte de0,15 à0,97m. Une montagne passe de−32,11
   à+20,90m : le cran est trop grossier pour corriger finement le niveau.
   Cette expérience démontre l'effet causal du scalaire, pas une amélioration
   générale ni une calibration finale. Elle utilise sept capturesTRAIN,
   avec LOD3/0 mesurés sur la même emprise physique.

Preuves : `eval/native-nf4-scalar-audit-20261013.json/.csv` et
`eval/native-nf4-outgain-causal-train-probe.json`.

## Programme retenu

1. **Tester le gain de sortie enFP32**, sans modifier les autres poids ni la
   géométrie native. Vérifier l'identité exacte avec70k à la valeur de départ,
   puis sa réponse à de petits incréments. Cela vise le saut discret démontré ;
   le stockage supplémentaire d'un seul scalaire est négligeable. Le prototype
   reste séparé de70k et de ses fichiers packés validés.
2. **Identifier le biais restant surTRAIN** : attribution des deux étapes avec
   niveau, basse fréquence et décodage, plus régressions gain/offset par canal.
   Calibrer surTRAIN seulement. Un réglage unique de gain n'est pas supposé
   suffisant, notamment pour les mondes202/404.
3. **Corriger le niveau avant le grain**. Cible basse fréquence : teacher à
   deux étapes complètes, sur les mêmes bruits neufs et conditions. Utiliser
   des résidus enq pondérés par le jacobien du teacher détaché ; inclureC0–C3
   pour protéger aussi le niveau aprèsdecoder. Pas de passe-haut dans ce test.
4. **Comparer à un témoin issu du même70k** : l'étatAdam70k n'existe pas,
   donc ce serait une réinitialisation explicite, pas une reprise exacte.
   Scalaires protégés contre les gradients auxiliaires, contrôleλ=0 exact,
   journal de gradients par tenseur, mêmes tirages et GPU par paire. Ne pas
   remettre arbitrairement les gains du teacher : cela changerait le point
   de départ70k. Mesurer d'abord les conséquences du gel ou duFP32.
5. **Grain ensuite**, séparément, avec cible même entrée pour la deuxième
   étape et métriques enq/à niveau égalisé. Le passe-haut apparié seul peut
   aussi lisser une bande mal corrélée ; la protection des incisions et du
   spectre doit être mesurée. Conserver les cas défavorables202/404.

Je retiens le diagnostic d'Opus, mais pas une promesse de résultat ni ses
durées estimées. Les essais de gain, les gradients et leurs coûts doivent être
mesurés avant un nouveau long run. Le70k compact reste un candidat de secours
testable, avec ses défauts publiés. Aucun nouveau training n'est encore lancé,
aucun poids servi n'est modifié par les probes.
