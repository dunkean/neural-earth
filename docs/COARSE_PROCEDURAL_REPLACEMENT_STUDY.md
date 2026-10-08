# Remplacer le coarse appris par `world-builder-rs` ?

Note du 8 octobre 2026. Expérience isolée ; aucun remplacement activé dans le produit.

## 1. Ce que le latent attend réellement

**Oui, le moteur procédural peut remplacer le réseau coarse, s’il fournit une représentation géographique cohérente.** Le latent reçoit des statistiques de terrain, sans dépendre des poids du coarse. Le NN est une manière apprise de produire cette représentation ; un bruit arbitraire ne la remplace pas.

La chaîne actuelle est : atlas/climat → diffusion coarse → diffusion latente → décodeur. Une cellule coarse couvre **7,68 km**. Pour chaque fenêtre latente, le modèle reçoit les moyennes et les **P5 absolus** d’altitude d’un voisinage **4 × 4**, quatre moyennes climatiques centrales, un masque de données valides, cinq paramètres d’histogramme et un niveau de bruit de conditionnement. Le masque vaut actuellement 1 partout : **ce n’est pas un masque terre/mer**. Les paramètres d’histogramme valent zéro par défaut. Le bruit gaussien qui initialise la diffusion est une entrée distincte.

L’altitude utilise `u = signe(h) × √|h|`. Il faut **transformer les hauteurs avant** de calculer `moyenne(u)` et `quantile(u, 0,05)`. `√moyenne(h)` donne une autre information, particulièrement aux côtes. Le coarse NN produit initialement le canal `moyenne − P5`, puis l’inférence le reconvertit en P5 absolu pour le latent. Un adaptateur procédural doit fournir directement ce P5 absolu, avec les normalisations existantes.

Le P5 distingue une plaine d’un plateau traversé de vallées malgré une même moyenne. La disposition des cellules renseigne les gradients. Le coarse NN propose aussi des relations apprises entre climat et relief ; le remplacer retire cette proposition à grande échelle. Le latent conserve la génération des formes fines.

Un champ de bruit érodé et ancré dans un monde déterministe peut alimenter le latent **après extraction de ces statistiques**. Un gaussien indépendant utilisé comme carte coarse ne décrit ni bassin versant ni côte stable. Le bruit et le dropout à l’entraînement n’assurent pas la cohérence géographique voulue.

Sources locales : [conditionnement latent](../terrain-diffusion/terrain_diffusion/inference/world_pipeline.py), [statistiques d’entraînement](../terrain-diffusion/terrain_diffusion/training/datasets/coarse_dataset.py), [dataset latent](../terrain-diffusion/terrain_diffusion/training/datasets/h5_latents_dataset.py).

## 2. Cas pratique exécuté avec le moteur Rust

Monde Rust : seed **16116671174843089066**, atlas de 16 cellules par côté de face, huit étapes d’érosion. Côte choisie par recherche déterministe : face 1, cellule `(−80, 72)`. Comparaison sur **15,36 × 15,36 km dans la carte locale**, décodée à 30 m puis moyennée à 60 m.

Le code actuel `world-core` a échantillonné 128 × 128 cellules coarse, avec 8 × 8 points par cellule pour estimer moyenne et P5 en racine signée. Trois branches partagent checkpoint, seed, bruits latents/décodeur, climat procédural et précision BF16. Seul le producteur coarse change : NN ; statistiques procédurales ; contrôle procédural où P5 est remplacé par la moyenne. Ce dernier retire volontairement l’information de relief interne. CUDA Graphs désactivés ; modèles préchauffés et caches de tuiles vidés entre branches.

| Mesure | Source Rust | Coarse NN | Stats Rust → latent | Contrôle P5 = moyenne |
|---|---:|---:|---:|---:|
| Médiane d’altitude, m | 184 | 392 | 206 | 202 |
| Erreur RMS par rapport à Rust, m | — | 195 | **23** | 34 |
| Corrélation avec Rust | — | 0,792 | **0,987** | 0,958 |
| Désaccord terre/mer | — | 3,07 % | 2,48 % | 1,40 % |
| Pente RMS, m/m | 0,026 | 0,191 | **0,041** | 0,057 |
| Génération NN, secondes | — | 2,04 | **1,31** | 1,27 |

[Comparaison visuelle](../generated/coarse-replacement-study/comparison.png) · [mesures complètes](../generated/coarse-replacement-study/report.json).

Le remplacement conserve mieux le relief. **Il ajoute du détail** : pente RMS supérieure de 55 %. Le NN coarse propose un relief plus marqué. Sans P5, des bandes artificielles apparaissent ; le meilleur accord terre/mer ne suffit pas à valider ce contrôle.

Ces résultats mesurent la fidélité au moteur, pas une supériorité esthétique universelle. Une seule côte, un atlas peu résolu et 64 sous-échantillons par cellule ne valident ni montagnes, ni archipels, ni coutures entre faces. La source possède trois composantes terrestres contre une après remplacement : les petites formes et la côte ne sont pas garanties. Le climat est celui des proxies Rust, identique entre branches ; cet essai ne valide pas leur réalisme et ne reprend pas la côte Orogen du diagnostic précédent.

## 3. Intégration, coût et recommandation

L’API Rust `diffusion-input` existe déjà, mais elle fournit **cinq canaux destinés à l’entrée du coarse NN**, sans P5. Elle moyenne les mètres avant transformation : la brancher directement sur le latent serait incorrect. Il faut un producteur de six canaux : moyenne de racine signée, P5 absolu de racine signée, BIO1/BIO4/BIO12/BIO15. Ensuite, conserver les poids d’assemblage, halos, interpolation, normalisations et graines existants. Le test réalise ce remplacement avec un `InfiniteTensor` sans appels au réseau coarse.

Le moteur actuel n’est pas entièrement GPU : atlas, érosion globale et sampler cohérent sont CPU/Rayon ; l’érosion régionale expérimentale utilise CUDA avec repli CPU ; le rendu utilise le GPU. Le chemin cohérent évite d’activer une nouvelle région érodée au zoom, selon le code et l’API `health`.

Dans l’essai, l’atlas prend **247 ms**, les statistiques de 16 384 cellules **1,57 s CPU**, et l’échantillonnage source 60 m **53 ms**. Le passage NN baisse de 2,04 à 1,31 s, soit **36 %**, lorsque les statistiques sont déjà préparées. Sur la toute première région, les 1,57 s de préparation peuvent annuler ce gain ; elles couvrent toutefois un contexte bien plus vaste et se réutilisent. Chargement des modèles et compilation du programme expérimental sont exclus. Aucun gain GPU global n’a été mesuré.

**Je recommande un mode expérimental “moteur procédural → statistiques coarse → latent → décodeur”**, avec le NN coarse comme variante comparative. Évaluer trois terrains contrastés et des fenêtres adjacentes : détail, conservation des objets, côtes, coût à froid/à chaud. Conserver strictement côtes, lacs et drainage nécessitera une contrainte sur le résidu final : supprimer le NN coarse ne donne pas cette garantie.

Sources moteur : [adaptateur](../../world-builder-rs/crates/world-app/src/diffusion.rs), [sampler](../../world-builder-rs/crates/world-core/src/sampler.rs), [érosion CUDA](../../world-builder-rs/crates/world-erosion/src/cuda.rs). Reproduction : programme Rust dans `generated/coarse-replacement-study/probe`, puis `.venv/Scripts/python.exe generated/coarse-replacement-study/run_probe.py`. Hashes d’inférence enregistrés et inchangés pendant la mesure.
