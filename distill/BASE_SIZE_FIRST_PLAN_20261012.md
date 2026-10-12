# Base : nouvelle priorité taille, programme proposé

12 octobre 2026. Programme préparé, **aucun entraînement ni nouvelle évaluation GPU lancé**. L'utilisateur donnera le top.

## Retour utilisateur qui remplace la décision précédente

- Les images QAT avec coarse2 et coarse4 conviennent visuellement ; **les vallées sont OK**. Ne pas présenter leur récupération comme le chantier principal restant pour ce checkpoint.
- Défauts résiduels : amplitude haute fréquence excessive en plaine, quelques pixels côtiers dispersés. Une correction après génération peut être acceptable, à montrer plutôt qu'à présumer.
- Priorité principale : base assez petit pour être embarqué dans la page. Cible **50–100Mo**, plafond de secours **150Mo** pour les poids du base.
- La vitesse vient ensuite. Les voies WASM CPU et WebGPU sont déjà prises en charge par l'utilisateur ; leur amélioration est différée. Aucun chantier de kernels ou de réécriture navigateur à lancer ici.
- Les images vues constituent une acceptabilité visuelle sur ces exemples ; elles ne donnent pas une garantie universelle, mais elles changent le choix pratique des critères. Les anciens seuils canal0/rappel ne doivent pas empêcher de conserver une candidate utile à l'utilisateur.

## Conséquence sur la stratégie

Le QAT500 devient un repère visuel utilisable. Conserver les formes et diminuer la taille est maintenant plus utile que continuer à chercher l'identité stricte avec le teacher ou un gain10×. La voie demi-rang choisie seulement pour un petit gain de vitesse n'est plus prioritaire.

Deux moyens différents : réduire le nombre de paramètres ; réduire le nombre de bits stockés par poids. Le second peut conserver la topologie et les vallées sans garantir une accélération. Les activations ne doivent pas nécessairement subir la même quantification que les poids : une **quantification des poids seuls** mérite d'être testée pour éviter d'imposer l'arrondi FP8 aux activations, dont la contribution au grain n'est pas encore isolée.

## Tailles vérifiées par inventaire CPU, sans inférence

Checkpoint QAT500 actuel :1 014 858 729octets. Il conserve les253 688 037 paramètres du teacher et les poids sont enregistrés enFP32. L'arrondi FP8 simulé ne rend pas ce fichier petit.

| Option de stockage / architecture | Charge utile estimée des poids | Statut |
|---|---:|---|
| QAT actuel, poids internes8bits / autres16bits |267,6Mo|Au-dessus du plafond ; pas encore exporté ainsi|
| Réseau natif entier, poids internes4bits / autres16bits |147,7Mo|Proche du plafond150 ; précision à démontrer|
| Natif128×3, interne8bits / autres16bits |125,3Mo|Secours sous150 ; pas de poids appris avec cette recette|
| Natif128×2, interne8bits / autres16bits |98,2Mo|Cible50–100 ; qualité de la nouvelle recette à démontrer|
| Natif96×3, interne8bits / autres16bits |74,1Mo|Autre cible50–100, garde les trois blocs ; non entraîné|
| Natif128×3, interne4bits / autres16bits |72,0Mo|Combine deux compressions ; risque de qualité supérieur|

Estimations **hors échelles, codebooks, métadonnées et graphe** ; ni fichiers livrés ni mesures de mémoire d'exécution. Mo=10⁶octets. Les dimensions d'embeddings768 et de bruit192 sont explicitement conservées dans cet inventaire ; laisser leurs valeurs automatiques changer aurait produit des architectures différentes des élèves natifs précédents.

Preuve : `/home/delete/data/distill/preparation/base-browser-size-options-20261012.json`. Les profils réduits sont construits sur le device `meta` : aucun forward, aucune création de données teacher ni activité GPU.

Un fichier8/4bits peut être décompressé/déquantifié au chargement pour des calculs à précision supérieure. Cela réduit le téléchargement, **pas nécessairement la mémoire vive ni le coût de calcul**. Le format de livraison doit être vérifié avec un décodeur de référence et documenter les valeurs effectives, échelles et conventions de normalisation MP.

La documentation ONNX Runtime décrit la quantification4bits automatique pourMatMul/Gather, pas une conversion automatique de toutes les convolutions de ce base. Il ne faut donc pas annoncer qu'un exportINT4 direct de ce U-Net fonctionnera tel quel. L'intégration navigateur étant différée, on peut préparer un stockage packé documenté et une reconstruction de référence, sans promettre des kernels4bits natifs. Sources : [quantification ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [WASM et WebGPU](https://onnxruntime.ai/docs/tutorials/web/).

## Programme des premières heures après le top

### 0–1h : isoler stockage et qualité

Conserver QAT500 et le teacher original comme références. Préparer des poids packés et leur décodeur de référence ; contrôler reconstruction et taille réellement écrite. Tester une quantification **des poids seuls**, d'abord8bits pour contrôle puis4bits pour approcher150Mo, avec normalisations/calculs sensibles conservés. Comparer les deux points de départ plutôt que supposer que les poids adaptés auFP8 restent optimaux pour une autre quantification.

Évaluation limitée mais contrastée sur coarse2/coarse4 : deux plaines plates, côte humide découpée, plaine basse/mer et montagne humide. Regarder l'amplitude haute fréquence, les côtes et la conservation des formes. La causalité du bruit FP8 poids/activations reste une hypothèse tant que cette ablation n'est pas faite.

### 1–3h : récupérer la qualité de l'option compacte

Si le4bits natif garde des formes exploitables, lancer une adaptation QAT courte, poids seuls, teacher original figé ; mêmes fenêtres/conditions/T2. Le but est une candidate proche150Mo, sans changer immédiatement l'architecture. Échelles/codebook peuvent être calibrés surTRAIN, jamais surVAL. Éviter les termes d'énergie qui réclament artificiellement plus de grain ; les cibles restent le teacher exact sur les mêmes entrées/coarse.

La5090 entraîne ; la4090 évalue utilement et prépare les images. Garder le meilleur checkpoint visuel et les jalons, pas seulement le dernier ; publier les PNG/CSV/JSON dans `_base`. Les critères de pilote doivent porter sur les défauts observés, pas exiger canal0÷2 en1000pas.

### 3–6h : choisir la candidate et la réduction supplémentaire

Si une candidate près150Mo convient, la conserver comme secours et préparer une réduction progressive de largeur en gardant les trois blocs, vers128×3. Son8bits est estimé125Mo ; son4bits72Mo offre la cible50–100 si la précision tient. Ne pas cumuler immédiatement suppression de blocs et forte réduction des canaux. Si le4bits natif détruit les formes malgré l'adaptation, privilégier plutôt128×3/8bits puis une réduction graduelle vers100Mo.

Ce créneau peut produire une première candidate et fixer le palier de compression ; il ne garantit pas l'achèvement de la distillation. Les instants de checkpoint dépendent du débit réellement constaté.

En parallèle limité, préparer un aperçu de correction du grain **sur zones plates seulement**, puis examiner côte/mer et vallées. Aucun flou global, aucune modification silencieuse du runtime. Une éventuelle correction doit rester cohérente aux frontières grâce au halo et être comparée avant/après.

## Livrables et décision

Premières heures : fichiers de poids réellement packés et mesurés, reconstruction de référence, checkpoints conservés, comparaisons brutes et éventuelle correction locale, courbes/métriques ciblées sur le grain et la côte. Les données et poids restent horsGit.

Pour une candidate retenue : vérifier ensuite les vrais raccords froids, deux axes/ordres de chargement, les lieux réservés et le temps d'inférence. Ni la quantification ni la géométrie native ne certifient automatiquement la continuité. Aucun gain de vitesse n'est annoncé avant mesure des poids réellement utiles.

**Orientation proposée :** tester d'abord une compression de précision à topologie intacte pour obtenir un secours≤150Mo rapidement ; poursuivre la réduction progressive vers50–100Mo si la qualité le permet. Les vallées actuellement jugées bonnes sont un acquis à préserver, pas un défaut à corriger par principe.
