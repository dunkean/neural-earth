# Base : supervision des fenêtres du premier champ

L'essai de coordonnées périodiques s'est arrêté au garde-fou au2037 nouveaux
pas, après évaluation2000. Sur les trois montagnes humides VAL/coarse4,
erreur décodée154/128/89m, corrélation0,839/0,856/0,917 et amplitude
0,848/0,876/0,753 : résultat autonome toujours insuffisant. Tous ses états
sont conservés ; les anciens paliers et l'état complet du précédent pilote
sont archivés sur D: avec liens Linux et empreintes vérifiées. Le dernier état
complet de cet essai reste sous Linux pour la reprise.

Le contrôle par canaux au1000 indique que remplacer uniquement le premier
canal0 par sa cible teacher rétablit une corrélation de0,993–0,994, contre
0,85–0,90 pour la chaîne autonome. Au dernier état2037, son erreur normalisée
n'a baissé que10–13% sur les neuf branches montagneuses VAL. Cela satisfait
la condition d'Opus pour essayer une supervision des fenêtres séparées, plutôt
que prolonger ce même essai sans changement. Sources externes :
`eval/field-first-features-final.json`, `eval/field-channel-oracles1000.json`,
`review/opus-field-channel0-20261012/OPUS_REVIEW.md`.

L'explication proposée par Opus reste une hypothèse. La fusion de quatre
sorties déterministes est une cible de régression valide ; elle n'est pas
intrinsèquement ambiguë. De même, l'identité avec le teacher du contrôle64
initial ne prouve pas celle des poids modifiés après entraînement. Ce nouvel
essai teste l'utilité de cibles intermédiaires par fenêtre et de sorties
distinctes, sans présenter ces explications comme une cause démontrée.

`FieldHeadsBase` garde le tronc192×3 commun aux deux phases. Quatre petites
convolutions de sortie copiées de celle du parent produisent les cinq canaux
de première phase ; la seconde conserve la sortie originale. Les classes
correspondent aux parités mondiales des origines de fenêtres64/pas32, période64.
Chaque tête vise les sorties non fusionnées de ses fenêtres enregistrées parmi
les49 captures. Les régions sans cible sont masquées. La fusion utilise les
poids triangulaires natifs fixes et doit reconstruire exactement la cible
teacher centrale192. Elle intervient **après** les arrondis BF16 des sorties
de fenêtres, comme dans le teacher. Aucun mélange appris ne peut contourner
les contrôles. Le tronc n'est pas exécuté quatre fois.

Reprise des poids bruts, EMA, Adam et RNG du parent2037. Nouvelles têtes copiées
séparément depuis les états brut et EMA ; moments Adam frais pour elles. LR
des paramètres hérités3e-5, têtes1e-4, warmup100, clip1, EMA demi-vie500.
Loss précédente conservée, plus MSE par tête1.0 normalisée par les variances
TRAIN sur cinq canaux. Forcing25%, sampling25% montagne/15% anciennes rares/
60% général et trois coarse, compteur global continué à4074. Recadrages x/y
toujours comparés sur deux chaînes autonomes.

Budget3000 nouveaux pas, tests GPU exacts avant lancement. Au1000, le canal0
fusionné doit améliorer sa MSE d'au moins15% sur chaque montagne/coarse4.
Au2000 : corrélation≥0,95, erreur≤80m, rappel des vallées≥0,80,
amplitude≥0,90, et baisse de la MSE des têtes sur toutes les branches. Les
planches autonomes et métriques sont publiées tous500 dans
`~/data/distill/_base/field-heads-<coarse>-<step>.*` ;5090 entraîne,4090 évalue.

Acceptation finale inchangée : qualité des vallées et des autres paysages,
mondes finaux réservés, chaîne app avec coarse4/decoder distillés identiques
des deux côtés, raccords froids et performances réellement mesurées. Aucun
poids base expérimental dans Git ; base original toujours actif par défaut.
