# Comparaison des NN en direct

Le serveur de revue utilise le viewer existant, avec un menu **NN** dans la barre
du haut. Coarse, base et decoder se sélectionnent séparément. Les quatre bases
du dossier final sont disponibles, ainsi que les modèles d'origine, coarse8
et decoder200k. Les choix sont communs aux onglets de ce serveur.

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
Les exports restent sous `~/data/distill/final/weights/`.

Les deux hooks du serveur de production sont actifs uniquement sous
`TERRAIN_DISTILL_LIVE=1`. Le lancement habituel reste sur les modèles d'origine.
L'identité des poids et du code participe à chaque cache neuronal ; les poids
sont revérifiés avant chargement. Les candidats restent expérimentaux : ouvrir
ce serveur ne constitue pas une acceptation des seuils de qualité.
