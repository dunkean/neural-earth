# Base une passe NF4 : reprise autorisée à 20k

Instruction du 14 octobre : entraîner le candidat de moins de 100 Mo jusqu'à
20 000 pas et produire des planches sur plusieurs terrains et climats.
Les images à 500 pas restent des diagnostics, sans acceptation du modèle.

Job durable : `base-onepass-residual-nf4-20k-4090`, sur la RTX 4090.
État et poids : `~/data/distill/ckpt/native-onepass-residual-nf4-20k-v1/`.
Le contrôleur précédent a été arrêté explicitement pour cette extension.
Son dernier état complet au pas 750 est copié dans `parent.pt`, avec SHA256
dans `request.json`. Les originaux restent conservés.

La reprise conserve les maîtres FP32, Adam, EMA et RNG. Elle active le QAT NF4
dès le départ, sur les convolutions internes du réseau 160×2 à une passe,
avec le résidu appris FP32 de 200 octets. Les deux bruits explicites,
fenêtres 64/32, conditionnement, supervision du teacher à deux passes et
pertes physiques sont conservés. Le LR reprend la valeur héritée puis
décroît vers 1e-6 jusqu'au compteur 20 000, sans nouveau warmup.
Cela représente 19 250 nouveaux pas QAT après le parent 750.

Un smoke reprend réellement le parent pendant quatre pas ; un export GPU
contrôle ensuite six sorties de fenêtres exactes avant le long entraînement.
Le calcul des échelles NF4 à l'export utilise le GPU, comme le QAT : le
premier export CPU avait échoué au contrôle exact, malgré sa taille correcte.
Les anciens fichiers ne sont pas écrasés.

Snapshots brut/EMA tous les 1 000 pas et état complet de reprise à chaque
segment. Jalons visuels à 2k, 5k, 10k, 15k et 20k : brut/EMA, coarse2/coarse4,
decoder original commun, même GPU et même bruit que le teacher recalculé.
Chaque jalon exporte et audite les vrais fichiers compacts avant de les rendre.
La charge utile attendue est 84 593 780 octets ; elle est vérifiée à l'export.

Planches sous `~/data/distill/_base/` :
`base-onepass-residual-nf4-{ema,raw}-<step>-{coarse4,coarse2}.png`.
Le dossier du même nom, puis `gallery/`, contient un aperçu de six scènes et
six PNG séparés avec LOD3/LOD0 : désert aride, plaine tempérée, côte humide,
plaine basse/mer, deux montagnes humides. La planche intégrale contient aussi
une troisième montagne humide. Ce sont des rendus des champs calculés,
pas des captures du viewer avec textures de biomes.

Le contrôleur conserve les sorties d'évaluation, les hashes et les métriques.
Toute erreur interrompt le job et demande une inspection avant reprise.
20k fixe le budget autorisé ; ce compteur ne garantit pas l'acceptation
visuelle, les raccords ni une accélération de la chaîne complète.

## Prolongation autorisée tant que le terrain progresse

Instruction supplémentaire : poursuivre au-delà de20k tant qu'il y a du progrès.
Job suiveur : `base-onepass-residual-nf4-adaptive-4090` ; il attend la réussite
du job20k, sans interrompre ni modifier ses contrats. Les blocs suivants font
5000pas, sans plafond de compteur prédéfini. Chaque bloc garde le réseau,
le budget84,6Mo, les maîtres/Adam/EMA/RNG, et produit les mêmes quatre planches
brut/EMA × coarse2/coarse4 à partir des vrais paquets exportés et audités.

La politique dans `adaptive/policy.json` compare les mêmes scènes/GPU/LOD.
Score relatif équilibré entre les cinq catégories de terrain : erreur d'altitude,
erreur de position topographique, grain haute fréquence et désaccord terre/mer.
Poids45/35/5/15%, renormalisés quand un terme est absent. Un gain d'au moins
0,5% compte comme progrès ; aucune catégorie ne peut perdre plus de10% en
altitude ou morphologie, ni une scène côtière plus de2points de désaccord.
Brut etEMA sont comparés ; la meilleure version admissible est conservée.
Ces critères sont des proxies automatiques, sans prétendre à une acceptation
visuelle. Ils évitent de prolonger sur la seule baisse de loss TRAIN.

Les jalons10k/15k/20k initialisent la tendance. Deux blocs consécutifs sans
améliorer le meilleur état confirment le plateau ; une régression matérielle
arrête la prolongation. `adaptive/{run-status,history,best}.json` trace toutes
les décisions et pointe les meilleurs poids/planches. Les checkpoints restent
conservés ; une erreur ou un manque réel d'espace interrompt le suivi.
