# Analyse de l'avis complet d'Opus sur le grain

L'[avis intégral](REVIEW_OPUS_NF4_ROOT_GRAIN_20261013.md),733lignes, est
conservé sans condensation. La CLI a confirmé `claude-opus-5-5`, effortxhigh,
session `7c10ee2a-53bc-4d4b-a378-da9ef472a269`, lecture seule, succès/code0.
La paire A/B active n'est pas modifiée.

## Ce que je retiens

- Le grain existe enq ; le passe-haut enhauteur permet une dérive vers la mer.
- Une contrainte d'énergie seule ne protège pas le contenu du teacher : il
  faut mesurer la composante corrélée et la composante orthogonale séparément.
- La sonde INT8 après QAT ne juge pas la capacité d'un QAT mixte entraîné.
  Elle montre aussi des changements d'amplitudeLOD0 et de côte, à conserver
  dans les résultats plutôt que de regarder le grain seulement.
- Les limites du suivi sont réelles : peu de pixels de vallée auLOD3,
  journaux auxiliaires affichant seulement la strate montagne, évaluateur
  utilisant `cudnn.benchmark=True`. Le sampling réel reste équilibré ; les
  logs ne suffisent pas à juger les autres catégories.
- Le budget exact des `emb_linear` peut financer une quantification plus
  fine des convolutions. C'est une piste à tester, pas une validation à150Mo.

## Corrections et réserves

**Normalisation globale, pas par filtre.** `MPConv.forward` appelle
`normalize(w)` sans `dim`. Dans `mp_layers.py:10`, `vector_norm(...,dim=None)`
porte sur le tenseur entier. Le test de conservation de norme doit donc
comparer les RMS **globaux** avant/après NF4. Une renormalisation par filtre
changerait aussi les amplitudes relatives des sorties, même sans quantification.
Cette partie de la proposition d'Opus ne reproduit pas le contrat du teacher.

**Les fenêtres de bruit sont cohérentes.** La nouvelle reproduction CPU
régénère les neuf bruits, applique la fusion et compare au patch global :
égalité vérifiée. On ne peut pas supposer que la fusion moyenne neuf bruits
indépendants. Un Monte-Carlo d'arrondi indépendant par fenêtre ne prouve pas
le plancher réel ; il faudra mesurer les expressions numériques directement.

**Projections calculées par région.** Le rapportD3 utilisait un RMS non centré
et une corrélation centrée sur un recadrage. `a=cR` n'est alors pas une identité
si les moyennes de bande ne sont pas nulles. Les nouveaux calculs utilisent
le produit scalaire réel et ne multiplient pas des médianes distinctes.

**Test de direction.** Échouer pour cinq valeurs deμ ne prouve pas l'absence
d'une direction apprenable ou une limite de représentation. De plus, une
charnière quadratique de pente a un gradient nul à sa frontière : un contrôle
de premier ordre seul peut manquer son effet. Il faudra tester un petit
déplacement mesuré, vérifier les changements de codes et conserver les
garde-fous de contenu. Je ne transforme pas ce test local en preuve d'impossibilité.

## G0a reproduit, CPU uniquement

`native_grain_budget_cpu.py` vérifie lesSHA des entrées et des capturesTRAIN,
exclut les mondes réservés et porte son propreSHA.768projectionsC4,
2304projectionsC0–C3,384régressions de bruit, deux emprises32/48, trois
bandes. Aucune initialisationCUDA ni mise à jour de poids.

Sur48x48, SS contreST, bandeσ1 :

| Catégorie | RMS relatif | Pente corrélée | RMS orthogonal enq |
|---|---:|---:|---:|
| Désert |1,915|0,762|0,278|
| Tempérée |1,964|0,809|0,331|
| Côte humide |1,692|0,804|0,296|
| Plaine/mer |1,660|0,798|0,243|
| Montagne humide |1,299|0,792|0,320|

La composante orthogonale absolue est assez proche entre catégories. Mais le
R² partiel sur le bruit injecté n'explique que0 à2,5% de l'erreur résiduelle,
après contrôle du signal teacher. Cela **ne soutient pas une simple erreur de
gain d'annulation du bruit** ; un mécanisme non linéaire ou filtré reste ouvert.

Le code reproduit aussi les décalages constants duD3, avec une provenance
complète. Trois tests CPU vérifient les projections avec moyenne non nulle,
la régression partielle et les références d'énergie nulle.

Preuve : `/home/delete/data/distill/eval/native-nf4-grain-budget-cpu.json`.

## Suite immédiate

1. Finir le premier bloc comparable A/B et lire le niveau par catégorie.
2. Sur4090, séparer réseauBF16/FP32 et calcul de sortie : mêmes trigonométries
   et mêmes vitesses, puis arrondi final seul, opérations intermédiaires et
   trigonométrie. Une comparaisonFP32/BF16 complète ne mesure pas uniquement
   le dernier arrondi.
3. Comparer la PTQ du teacher en blocs512/128/64, puis RMS global conservé.
   Fixer les règles et mesurer les coûts avant d'entraîner un nouveau quantifieur.
4. Le prochain objectifHF devra protéger la pente corrélée, les incisions et
   les côtes ; son choix dépendra de ces mesures et de la paire en cours.

Les mondes202/404, les grandes plaines, la plaine très basse/mer et les montagnes
humides restent dans les décisions finales. Le rendu visuel et les raccords
ne sont pas remplacés par les moyennes de ces diagnosticsTRAIN.
