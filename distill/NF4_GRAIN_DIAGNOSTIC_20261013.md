# Diagnostic du grain, sans modification de la paire A/B

## Haute fréquence en racine d'altitude

32 captures TRAIN indépendantes du diagnostic de niveau, nouveaux bruits,
trois coarse. `q = 38,6*C4 − 31,4`, passe-haut `q − G_sigma(q)`, sigma1,
réflexion aux bords et statistiques sur les32x32pixels centraux, tous pixels.
SS est l'élève aux deux étapes ; TT le teacher aux deux étapes ; ST le
teacher en deuxième étape sur les **mêmes entrées** que l'élève.

| Catégorie | RMS SS/TT, médiane | Corrélation SS/TT | RMS SS/ST | Corrélation SS/ST |
|---|---:|---:|---:|---:|
| Plaine désertique |1,895|0,359|1,919|0,391|
| Plaine tempérée |1,933|0,391|1,965|0,429|
| Côte humide complexe |1,646|0,421|1,666|0,475|
| Plaine basse/mer |1,552|0,421|1,586|0,476|
| Montagne humide |1,298|0,538|1,320|0,586|

Le grain excédentaire existe donc **en q**, indépendamment de sa conversion en
mètres. La cible même entrée ST augmente modérément la corrélation. Cela ne
démontre pas qu'une perte appariée suffira : une bande mal corrélée peut être
lissée par une régression quadratique. Les incisions et les amplitudes restent
des critères à contrôler séparément.

Pour chaque paire, un déplacement constant de q de ±0,001 a aussi été testé.
La variation maximale de la perte passe-haut en q est5,83e−16 : elle est
numériquement indépendante de ce déplacement. La perte en `h=q|q|` a au
contraire une dérivée mesurable selon le niveau. La preuve est descriptive
sur TRAIN ; aucune nouvelle loss ni aucun coefficient n'a été introduit dans
la paire en cours.

Données : `eval/native-nf4-root-highpass-train-diagnostic.json`, sous
`/home/delete/data/distill/`. Le rapport porte les SHA256 des sources
`native-nf4-level-train-attribution-5090/{report.json,arrays.npz}` et conserve
les64 lignes par catégorie, coarse, seed et type de cible.

## Essai sélectif INT8 rejeté

Hypothèse testée sur4090 : remplacer uniquement le kernelNF4
`dec.512x512_block3.conv_res1` par son encodageINT8, à partir des maîtres
du même70k.331776poids, surcoût165888octets : le paquet resterait proche
de149,821Mo. Aucun fitting ou optimizer ; sept capturesTRAIN, quatre cas
critiques et trois montagnes, teacher/decoder/entrée identiques.

Le remplacement **aggrave le RMS du grain dans les quatre plaines/côtes** :
1,662→1,743 ;1,922→2,005 ;1,782→1,962 ;1,603→1,726.
En montagne, il ne réduit presque pas le grain et déplace le niveauLOD0
d'environ+23,3 /+39,1 /+29,2m. Les poids ont été restaurés après le probe ;
aucun fichier de modèle servi ni poids de la paire n'a changé.

**Ce changement après entraînement n'est pas retenu.** Il ne prouve pas qu'un
QAT mixte correctement entraîné échouerait, ni que les autres couches sont
innocentes ; il rejette cette substitution comme correctif immédiat.

Code : `native_last_layer_probe.py`. Preuve :
`eval/native-nf4-last-int8-train-probe-4090.json`, avec métriques de niveau,
côte, haute fréquence en q et morphologie décodée. Job terminé avec code0.

La paire reste inchangée : B corrige d'abord le niveau. Son premier bloc doit
être comparé à A avant de choisir l'expérience suivante sur le grain.
