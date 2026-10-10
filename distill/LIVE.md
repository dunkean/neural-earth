# Comparaison des NN en direct

Le serveur de revue utilise le viewer existant, avec un menu **NN** dans la barre
du haut. Au premier lancement, les choix sont **coarse distillé 4 étapes,
base original et decoder distillé**. Coarse et decoder peuvent être comparés
à leurs modèles d'origine ; coarse8 reste disponible si son export existe.
Les bases distillés rejetés ne figurent plus dans le menu ni dans l'API.
Une ancienne configuration qui les sélectionnait est migrée vers le base original,
sans modifier coarse, decoder ou le reset. Les autres choix sauvegardés sont
conservés au redémarrage. Les choix sont communs aux onglets de ce serveur.

Les poids restent hors Git. Pour un nouveau checkout, placer les deux exports
vérifiés dans les chemins suivants (relatifs à `DISTILL_ROOT`, par défaut
`~/data/distill`) avant le premier lancement :

- `live/weights/coarse4/coarse.pt` : coarse à 4 étapes, checkpoint 50 000.
- `final/weights/candidate-1/decoder.pt` : decoder distillé, checkpoint 200 000.

Le démarrage échoue avec un message explicite si un export par défaut manque.
Le coarse4 a été jugé bon visuellement en direct ; les seuils physiques stricts
ne sont pas tous validés. Les anciens checkpoints base restent archivés pour
la recherche et ne sont jamais chargés par ce serveur.

**Appliquer les NN** recharge le moteur et rouvre la même vue. **Reset après
Orogen** change le namespace des caches neuronaux : coarse, latent, decoder et
leurs tuiles sont recalculés. Le cache Orogen reste intact, ainsi que les réglages
de génération, le seed et la caméra. Les anciens caches neuronaux ne sont pas
effacés du disque : ils deviennent inactifs. Une combinaison réutilise son cache
tant qu'aucun reset n'est demandé.

```bash
source distill/env.sh
python -m distill.jobs start live-model-review --gpu 1 -- python -u -m distill.live_server --port 8765
python -m distill.jobs status live-model-review --compact
python -m distill.jobs stop live-model-review
```

Adresse depuis Windows ou WSL : <http://localhost:8765/>. Le serveur est local,
sur la 5090 seule ; le changement de modèles prend quelques secondes. Il est
supervisé dans tmux et conserve les choix dans `~/data/distill/live/config.json`.
Les exports sont lus aux chemins indiqués ci-dessus.

Les deux hooks du serveur de production sont actifs uniquement sous
`TERRAIN_DISTILL_LIVE=1`. Le lancement habituel reste sur les modèles d'origine.
L'identité des poids et du code participe à chaque cache neuronal ; les poids
sont revérifiés avant chargement. Les candidats restent expérimentaux : ouvrir
ce serveur ne constitue pas une acceptation des seuils de qualité.
