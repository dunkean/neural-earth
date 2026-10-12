# Synthèse pour décision — distillation du base

12 octobre 2026. [Rapport complet](BASE_EXPERIMENTS_REVIEW_20261012.md), [Astra high](REVIEW_ASTRA_BASE_20261012.md), [Opus5.5 xhigh, avis complet](REVIEW_OPUS_BASE_20261012.md). Les propositions ci-dessous ne sont pas lancées.

## Ce que les deux avis changent

Les deux reviewers recommandent de **partir du teacher original et de le comprimer progressivement**, en conservant ses conditions, attention, fenêtres et deux passes. Changer simultanément cette géométrie et la taille du réseau a trop éloigné les élèves de la fonction originale. Ils déconseillent de repartir sur un grand champ une passe à ce stade.

Ils critiquent aussi les arrêts à500/1000 pas : ces pilotes montrent que leurs recettes ne donnent pas une récupération rapide, pas que la compression est impossible. Les losses globales ont parfois progressé pendant que les vallées disparaissaient. La MSE du canal0 explique une partie du défaut mais ne suffit pas pour sélectionner un bon relief.

Le checkpoint **QAT500 est un résultat à préserver** : sur trois montagnes humides/coarse4, MAE LOD0≈15/15/37m, contre47/28/75m au1000, avec des formes nettement plus fidèles que les anciens élèves comprimés. Il reste du bruit au LOD3 et des défauts de côtes. C'est une candidate pour un test visuel, pas un base accéléré : le FP8 est encore simulé en BF16.

## Là où ils diffèrent

| Choix | Astra | Opus |
|---|---|---|
| Compression | Demi-rang sur42 convolutions internes du décodeur ; largeur et blocs conservés | Réduction progressive jusqu'au natif128×2, avec canaux et blocs supprimés graduellement |
| Apprentissage | Superviser aussi les champs fusionnés des deux passes, avec gradient traversant leur composition | Entraînement nettement plus long, plus gros batch et LR adapté aux couches MP |
| Budget proposé |30k mises à jour | Environ80k mises à jour |
| Gain étayé aujourd'hui | Indice~1,18× par fenêtre, hors chaîne complète | Coût d'architecture~2× sur champ1024² ; poids candidats non entraînés |
| Risque | Faible compression, bénéfice de vitesse limité | Compression plus forte, récupération des vallées incertaine |

## Mon interprétation

La sous-optimisation mise en avant par Opus est **plausible, pas démontrée comme cause principale**. Des runs plus longs existent et deux LR plus élevés ont déjà été essayés, mais leurs pilotes ont été arrêtés tôt. Des erreurs TRAIN proches de VAL peuvent venir de l'optimisation, de la capacité ou de l'objectif ; elles ne départagent pas ces causes.

Astra identifie une asymétrie réelle des losses. Je ne retiens pas comme établi le « canal hauteur sous-pondéré » évoqué par Opus : le code ajoute justement des termes hauteur normalisés par de petites variances. Leur effet doit être vérifié sur la composition autonome, plutôt que corrigé par un poids arbitraire de plus.

Enfin, conserver la géométrie native aide à préserver la sémantique, **mais ne dispense pas des vrais tests de continuité entre tuiles**. Aucune preuve d'un ancien modèle ne vaut pour le prochain.

## Ce que je recommande

1. **Conserver QAT500 pour juger le look & feel**, avec teacher et même coarse/decoder. Cela donne un repère fidèle, sans attendre le prochain long run.
2. **Faire un seul contrôle court, apparié**, sur un petit jeu fixe : départ identique, objectif opérateur simple contre objectif qui supervise aussi la chaîne autonome, puis LR prudent contre LR plus élevé. Vérifier les champs, les vallées et le bruit ; pas seulement la loss. Ce contrôle doit choisir la recette, pas devenir une nouvelle succession de diagnostics.
3. **Viser ensuite128×2 par compression progressive du teacher**, en reprenant l'idée d'Astra de superviser la composition des deux passes. C'est la voie avec un potentiel de vitesse utile déjà mesuré. Programmer une compression lente puis une récupération longue, garder brut/EMA et checkpoints intermédiaires, et arrêter sur divergence ou régression persistante plutôt que sur « canal0÷2 au1000 ».
4. **Sélectionner sur les cas critiques et les images**, puis tester les vrais raccords froids sur les deux axes et les mondes réservés. Ne pas annoncer×2 avant le benchmark des poids réellement bons.

**Décision proposée :** voie128×2 progressive pour viser environ×2, avec le contrôle court pour fixer l'objectif et le LR. La voie Astra demi-rang reste l'option prudente si la fidélité prime sur un gain qui pourrait se limiter à~20%. Aucun nouveau training n'est lancé avant la décision de l'utilisateur.
