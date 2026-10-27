# Audit des vrais sols ouverts et des vues MID

[English](render-v2-audit-real-open-ground.md)

État de lecture du 23 octobre 2026. Aucun changement de production dans cet audit. Les anciennes matrices synthétiques sont exclues des preuves visuelles et ne valident pas le rendu utilisateur.

## Provenance et résolution

Les captures de `output/render-v2-40km/real-matrix/` viennent du renderer WebGPU de production, du même monde et de DEM réellement générés. Le runner `tools/verification/review_render_v2_real_matrix.cjs` impose le decoder LOD0, une résolution source de **30 m**, zéro tuile visible en attente, le même centre et le même éclairage entre styles. Chaque ligne complète a douze PNG : quatre zooms, Render v2 juillet/janvier et Render3D juillet. Les dates sont calendaires ; juillet n'est pas l'été dans l'hémisphère sud.

| Emprise | Image | Pas écran | Source DEM |
| --- | --- | --- | --- |
| 40 km | 1024² | 39,0625 m/px | decoder 30 m |
| 10 km | 1024² | 9,765625 m/px | même decoder 30 m |
| 2,56 km | 1024² | 2,5 m/px | même decoder 30 m agrandi |
| 640 m | 1024² | 0,625 m/px | même decoder 30 m agrandi |

La dernière image ne prouve donc pas un relief généré à 0,625 m : sa fenêtre ne contient qu'environ 21 intervalles du DEM. Elle permet d'examiner l'apparence procédurale sur ce relief. Les classifications et morphologies sont reconstruites sur le DEM natif ; le repérage initial à 19,53 km ne suffit pas à accepter une case.

## Douze croisements demandés

Les douze candidats ont leurs douze PNG et ont été inspectés, sans remplir de case par un terrain artificiel. **Cinq étiquettes de relief ne sont pas confirmées au decoder 30 m** : ce tableau ne valide donc pas douze croisements distincts. Le rapport natif reste l'autorité pour les éventuels remplacements.

| Case | Climat/relief natifs | 40 km | 10 km | 2,56 km | 640 m |
| --- | --- | --- | --- | --- | --- |
| BSK plain | BSH 100 % ; plaine ; 508–700 m | Relief faible lisible, pas de pommelage minéral distinct | Taches de couvert arrondies, peu visibles en Render3D | Sol plus sombre/terne et masses floues | Arbres épars ponctuels sombres, déjà présents mais plus clairs en Render3D |
| BSK rolling | BSH 65,95 %, BSK 34,05 % ; ondulé ; 556–909 m | Relief lisible, proche de Render3D | Plaques olive arrondies ajoutées, beaucoup moins visibles en Render3D | Couvert en grandes masses floues | Apparition de couronnes sombres ponctuelles ; contraste net avec les masses floues précédentes |
| BSK plateau | **Étiquette non confirmée : montagne native** ; BSK 82,55 %, BWK 17,45 % ; 1784–2793 m | Crêtes et incisions lisibles, deux styles proches | Même réseau de vallons, v2 plus sombre | Versants en masses très sombres/floues | Couronnes beaucoup plus pâles en juillet qu'en janvier ; sol très lisse |
| BSK mountains | BSK 95,54 %, DSC 4,46 % ; montagne ; 1228–4032 m | Relief fort ; neige janvier localisée sur l'altitude supérieure | Incisions conservées ; différences de couvert plus visibles | Grandes plages de couvert/sol clair plus tranchées et floues que Render3D | Couronnes sombres et sol uniforme, transition visible depuis les masses précédentes |
| ARID plain | BWH 100 % ; plaine ; 315–526 m | Relief/grain proches du comparateur | Vue douce sans dalmatien minéral distinct | Matériau plus gris, moins ocre | Sol presque uniforme, grain proche perdu ; DEM agrandi |
| ARID rolling | BWH 100 % ; ondulé ; 322–573 m | Relief lisible, proche du comparateur | Même incision centrale, pas de nouvelles taches minérales fortes | Virage ocre vers brun-gris, texture atténuée | Brun-gris très uni ; différence de teinte manifeste avec Render3D |
| ARID plateau | **Étiquette non confirmée : montagne native** ; BWK 99,93 % ; 1956–3230 m | Relief sombre et rocheux proche du comparateur | Incisions lisibles, pas de dalmatien minéral spécifique évident | Sol plus brun et très lissé | Brun uniforme ; Render3D conserve du grain |
| ARID mountains | **Étiquette non confirmée : plateau natif** ; BWK 100 % ; 818–2499 m | Surfaces élevées et bords incisés, plateau visuellement plausible | Réseau de canyons conservé | Incisions deviennent brun rouge et s'aplatissent visuellement ; Render3D les garde grises | Faible variation de brun et relief agrandi flou, sans couronnes |
| POLAR plain | **Étiquette non confirmée : montagne native** ; EF 100 % ; 5338–7649 m | Contraste neige blanche/affleurements sombres ; Render3D bien plus brun | Même rétention par pente, vallée sombre | Neige de pente avec centre très lissé ; texture du DEM agrandie | Neige gris clair presque uniforme ; juillet/janvier identiques, sans couronnes visibles |
| POLAR rolling | ET 100 % ; ondulé ; 145–593 m | Juillet enneigé, relief conservé ; janvier sol sombre | Même contraste saisonnier ; relief agrandi | Juillet plus gris que Render3D, sans fort pommelage isolé au centre | Petits points de couvert visibles, surtout en janvier ; ne pas les appeler arbres EF : la case est ET |
| POLAR plateau | **Étiquette non confirmée : montagne native** ; EF 100 % ; 3071–5694 m | Versants gris/neige blanche très contrastés, Render3D plus uniformément blanc | Même géométrie, larges vallées grises | Relief agrandi très lissé, peu de grain neige | Gris moyen presque uniforme ; juillet/janvier identiques, sans couronnes visibles |
| POLAR mountains | EF 100 % ; montagne ; 4581–6489 m | Neige/crêtes blanches et versants gris ; Render3D beaucoup plus brun | Vallées sombres et neige de pente bien distinctes | Grandes surfaces grises, relief flou agrandi | Centre gris très sombre, presque uniforme ; juillet/janvier visuellement identiques |

BSK-rolling confirme un problème de transition du couvert, sans démontrer une erreur du bruit minéral. Les deux BWH ne montrent pas de « dalmatien » rocheux spécifique évident à 40/10 km ; leur problème distinct est le changement de matière/teinte au zoom proche. La neige EF de montagne n'est pas uniformément blanche : la rétention dépend de la pente, donc EF seul ne garantit pas une couverture de neige égale à un.

Les planches de lecture appariées sont dans `output/render-v2-40km/real-matrix/audit-sheets/`. Ce sont des réductions des vrais PNG, pas des rendus de terrain reconstruits.

## Sources différentes à emprise constante, positions utilisateur

Les quatre dossiers `transitions-lod0`, `transitions-lod3`, `transitions-lod4` et `transitions-lod6` conservent **exactement 40 km, le même centre et 39,0625 m/px** pour chacune des trois positions utilisateur. Ils permettent de séparer substitution de données et changement de zoom. Les sources effectives sont decoder30m, latent240m, puis coarse7680m. La demande LOD6 restitue ici la même qualité4 et la même image que LOD4 ; ce n'est pas un quatrième niveau de détail indépendant.

`output/render-v2-40km/audit-real-transitions.json` contient les 36 mesures, SHA256 et snapshots. Le proxy vert G/R>1,05 et G/B>1,15 désigne uniquement une couleur, pas un couvert végétal sémantique. La bande Y est mesurée aux mêmes sigmas physiques 100 et 400 m pour toutes les sources.

| Position `user-green`, juillet | decoder30m | latent240m | coarse7680m |
| --- | --- | --- | --- |
| Y moyen v2 | 61,16 | 62,82 | 52,70 |
| G/R moyen v2 | 1,125 | 1,121 | 1,347 |
| Fraction du proxy vert v2 | 71,5 % | 70,0 % | 94,4 % |
| RMS bande Y v2, 100–400 m | 3,413 | 1,146 | 0,128 |
| RMS bande Y Render3D, 100–400 m | 3,436 | 0,862 | 0,123 |

Le coarse reste donc vert : il devient **plus uniformément vert et plus sombre**. Il ne faut pas écrire que la verdure disparaît. La perte de structure à cette échelle existe dans les deux styles et suit fortement la résolution du DEM ; les masses colorées v2 apportent ensuite une différence de couvert supplémentaire. Les nombres ne constituent pas une preuve que le couvert réel serait plus élevé en coarse.

À la position `user-pack`, la même bande géographique vaut respectivement v2/Render3D **5,241/2,686** (decoder), **4,264/0,442** (latent) et **2,245/0,253** (coarse). Le contraste de couvert supplémentaire v2 persiste donc à centre et emprise constants ; la résolution seule ne suffit pas à expliquer cette différence entre styles. À `user-desert`, les paires sont au contraire très proches : 6,173/5,986, puis 0,845/0,829, puis 0,112/0,112. Son coarse v2 est plus sombre (Y87,38 contre Y110,16), mais ne présente pas un excès comparable de contraste à 100–400 m.

## Mesure de teinte à géographie identique

`output/render-v2-40km/real-matrix/audit-real-open-metrics.json` conserve la mesure, le centre, la résolution, les fichiers et leurs SHA256. La comparaison utilise **le même carré central de 640 m**, extrait avec des bornes fractionnaires de chaque zoom ; elle ne compare pas les moyennes de fenêtres géographiques différentes. À 40 km, ce carré ne fait que 16,384 pixels de large : sa moyenne renseigne sur la teinte, pas sur le détail. Les valeurs Y sont des luminances d'affichage Rec.709 sur RGB 8 bits, pas une radiance physique.

| Cas, v2 juillet | RGB moyen à 40 km | RGB moyen à 640 m | Y 40 km → 640 m |
| --- | --- | --- | --- |
| BSK rolling | 127,3 / 113,3 / 75,0 | 93,8 / 88,2 / 57,4 | 113,5 → 87,2 |
| ARID plain | 147,1 / 122,2 / 85,1 | 151,0 / 134,5 / 101,0 | 124,8 → 135,6 |
| ARID rolling | 143,7 / 119,6 / 82,9 | 111,0 / 98,5 / 73,3 | 122,0 → 99,3 |

La direction de variation n'est pas universelle : la plaine aride s'éclaircit, l'ondulé aride s'assombrit. Le changement chromatique et la disparition du détail sont des constats ; ces moyennes ne séparent pas le matériau et sa réponse à l'éclairage.

## Vraies vues MID : autres données et autre voie shader

| Dossier | Emprise | Pas écran | Source déclarée |
| --- | --- | --- | --- |
| `mid-lod3` | 153,6 km | 300 m/px | latent, 240 m |
| `mid-lod4` | 307,2 km | 600 m/px | coarse, **7680 m** |
| `mid-lod6` | 1228,8 km | 2400 m/px | coarse, **7680 m** |

Ces vues ne sont pas des plaques de 40 km. Un plein cadre MID ne se compare pas statistiquement à un plein cadre decoder de 40 km. Les résolutions proviennent des snapshots des frames, pas du nom du dossier.

Lecture appariée MID3 : les déserts chaud/froid et la glace polaire présentent peu de différence visuelle entre v2 et Render3D ; la toundra alpine présente beaucoup plus de neige blanche morcelée en v2. À 40 km decoder, la même toundra alpine est réellement ET à 78,12 % et EF à 21,88 %, à 3171–4391 m : Render3D presque sans neige n'est donc pas une référence absolue de justesse climatique. MID4 montre des amas de neige beaucoup plus gros et un relief diffus. Le désert chaud MID6 conserve un fin grain diagonal très similaire entre styles, ce qui ne permet pas de l'imputer spécifiquement à v2.

`output/render-v2-40km/audit-real-mid-metrics.json` mesure les paires de styles sur leur fenêtre identique, avec les SHA256 et résolutions sources. La bande est la différence de deux flous gaussiens de sigma 2 et 8 pixels ; ses sigmas physiques changent donc entre les LOD. Elle mesure du contraste spatial, pas « du bruit erroné », et contient aussi les variations de neige et d'ombrage.

| Vue réelle, juillet | MAE Y v2/Render3D | RMS de bande v2 / Render3D | Sigmas de bande |
| --- | --- | --- | --- |
| MID3 désert chaud | 3,22 | 2,00 / 1,93 | 600 / 2400 m |
| MID3 désert froid haut | 2,48 | 3,35 / 3,33 | 600 / 2400 m |
| MID3 toundra alpine | 46,81 | 10,30 / 5,43 | 600 / 2400 m |
| MID3 côte utilisateur | 21,94 | 4,29 / 2,79 | 600 / 2400 m |
| MID4 toundra alpine | 84,86 | 2,87 / 1,46 | 1200 / 4800 m |
| MID4 côte utilisateur | 34,94 | 2,71 / 0,61 | 1200 / 4800 m |
| MID6 désert chaud | 9,33 | 0,45 / 0,46 | 4800 / 19200 m |

Ces écarts localisent le contraste supplémentaire dans les cas de couvert/neige plutôt que dans le contrôle désertique. Ils ne disent pas si le bon résultat serait d'ajouter la neige à Render3D ou d'adoucir celle de v2.

## Causes vérifiables et hypothèses à distinguer

1. **Voie coarse distincte.** Dans `web/terrain_renderer.js:457`, puis dans la construction de `coarseColor` vers la ligne 654, le rendu coarse appelle directement `v2Evaluate` avec le climat `r32float` et l'espacement source comme footprint. Il ne passe ni par `aux3 rgba8unorm`, ni par `v2Fragment`. Les amas MID4 ne peuvent donc pas être expliqués par la seule quantification de température aux3.
2. **Pas de microtexture terrestre à 7680 m.** `web/terrain_render_v2.js:29` filtre les mosaïques 2400/670 m ; à footprint 7680 m, elles sont éteintes. `v2Appearance` neutralise aussi tone et mineral affichés. En revanche, la neige dépend toujours de la température, du potentiel, de la pente et du TPI (`:30–54`). Hypothèse compatible avec les PNG : des champs climatiques/rétention multi-km sur un DEM de 7,68 km agrandi produisent les gros amas MID4. Une ablation ciblée est nécessaire pour attribuer chaque contour à un terme précis.
3. **Contraste de neige partielle.** La neige mélange un sol coloré à RGB(.88,.91,.91), puis garde l'éclairage macro. Même avec une fonction continue, les plages de couverture intermédiaire peuvent dominer les faibles variations d'un relief doux. Continuité de formule ne signifie pas faible contraste visuel. Les images seules ne prouvent pas un contour thermique quantifié.
4. **Changement de matière proche explicite.** `v2Fragment` (`:271–273`) remplace le sol hérité par `v2Open` entre 6 et 1,5 m/px. Les vues 2,56 km et 640 m traversent cette transition ; c'est une cause concrète du changement de teinte observé, sans preuve que tout son contraste vient de ce seul terme.
5. **Contrat climatique des couronnes vérifié dans l'assemblage.** La lecture isolée de `forestClimate` legacy suggérerait un décalage avec aux3 v2, mais `web/terrain_render_v2.js:310` remplace explicitement son préambule lors de `drawShader` : annual=(Ts+Tw)/2, humid=c.g×5 et coldest=min(Ts,Tw). Le shader assemblé possède le décodage approprié. Cette piste est écartée ; elle n'explique pas les couronnes sombres.
6. **Précision du transport climatique fine/latent.** Aux3 est une texture `rgba8unorm` (`web/terrain_renderer.js:940`) et encode 100 °C dans 255 intervalles : le pas des températures stockées est 0,39216 °C. Le filtrage bilinéaire interpole ces valeurs mais ne récupère pas la précision perdue au bake. Cette propriété peut contribuer aux paliers de couverture fine/latent ; aucune attribution quantitative d'un contour des vrais PNG à ce terme n'est revendiquée ici. Elle ne concerne pas la voie coarse décrite plus haut.
7. **Normalisation de pluie distincte au bake et au draw.** Aux3.g (`web/terrain_renderer.js:562`) divise la pluie par un seuil calculé avec le refroidissement fixe 0,0065 °C/m, tandis que `v2EnvironmentAt` (`web/terrain_render_v2.js:240`) reconstruit la pluie avec le seuil des Ts/Tw effectivement transportées, dont les gradients altitudinaux peuvent différer. Avant quantification/saturation, le rapport pluie décodée/pluie source vaut seuil(Ts,Tw effectives)/seuil(température annuelle avec lapse fixe). C'est une incohérence de transport lisible dans la voie fine/latent, pas une preuve que chaque tache des captures en découle, et pas une explication de MID4 coarse.

## Canopée réelle et limite du comptage de plage

L'analyse complémentaire `output/render-v2-40km/real-detail/audit-real-canopy.json` compare les mêmes centres, fenêtres 1,28 km, éclairages et DEM decoder 30 m. Les 400 premières lignes des images 512² excluent HUD et minimap. Un masque de texture de couronnes est défini une seule fois sur Render3D et appliqué aux deux images ; c'est un proxy visuel, pas une segmentation sémantique du shader.

| Site réel, juillet | Y moyen v2 / Render3D dans le proxy | Fraction Y<40 v2 / Render3D | RMS haute fréquence v2 / Render3D, sigma 1 px |
| --- | --- | --- | --- |
| Forêt tropicale basse | 31,56 / 57,89 | 83,86 % / 12,18 % | 4,04 / 6,97 |
| Tempéré humide bas | 74,41 / 90,68 | 0 % / 0,17 % | 1,90 / 9,22 |

Le site tempéré est à latitude −50,01° : juillet correspond à son hiver. Ces données corroborent l'assombrissement AF et la perte du contraste des couronnes tempérées ; elles n'établissent pas une luminance physique correcte du comparateur.

Aucun nombre de pixels « plage » fiable n'est revendiqué sur AF/warm-coast à 39,0625 m/px. Les liserés sont généralement subpixel ou d'un pixel, mêlant eau/terre, plage et éclairage des crêtes. Un seuil de couleur inventerait une précision sémantique ; il faudrait un diagnostic plage de production ou une capture côtière réellement plus proche.
