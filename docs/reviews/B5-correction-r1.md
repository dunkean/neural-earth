# B5 — décisions après la double review finale

Astra high et Opus 5.5 high concluent **ACCEPT WITH LIMITATIONS**. Les modèles et prompts sont enregistrés dans les reviews. Les correctifs précédents sont conservés par bloc ; aucun passage de gate échoué n’a promu un profil expérimental.

| Finding Opus B5 r1 | Décision du coordinateur |
|---|---|
| P2-1 benchmark par sauts, rAF pris pour coût du rendu | Accepté et corrigé. Reçu final harness `75689d9fb02e` : ±6 km/s, 120 intervalles / 2 s, environ 12 km par pan. Segment dessin CPU p95 0,6 / 0,4 ms, caméra 0,1 ms ; cadence headless ~60 Hz, 0 intervalle >1,5× période. Baselines/fins natives confirmées, samples sans pending, transports hits. Gate GPU/présentation et travail main-thread complet non mesurés. |
| P2-2 attribution des snapshots navigateur | Accepté et corrigé. Bilan avec hashes client, anciens runs historiques, échec 180,948 s conservé, promotion corrigée 25,443 s avec cache partiellement réchauffé, client final et revisite 224 ms explicitement chaude. Le scénario du harness est renommé. |
| P2-3 probe mémoire/lock et restart avec prepare=0 | Limite déclarée. Préparation automatique active par défaut ; la sérialisation des revalidations supprime la rafale prouvée mais ne supprime pas toute attente sur le GPU. Probe metadata indépendante souhaitable ensuite. |
| P2-4 flags numériques et VRAM | Manque de preuve accepté comme limite. Identité actuelle inclut runtime/dtype/batch/backend, mais ne certifie ni les flags cuDNN/TF32 ni la stabilité sous pression VRAM. Les claims d’exactitude sont limités aux essais enregistrés et aux bytes persistés. |
| P2-5 double vérification et scan des fenêtres | Goulot mesuré conservé : 112 s de persistance sur 439,6 s. Optimisation future, aucun gain annoncé aujourd’hui. |
| P2-6 coarse LOD 4–6 au bord | Limite déclarée : ces halos peuvent générer hors du plan fini et différer du clamp LOD ≥7. Les bords ne sont pas déclarés identiques entre tous les LOD. |
| P2-7 API read_mip | L’API relative au grid refuse les blocs incomplets ou hors grid ; elle n’est pas utilisée par les tuiles globales. Ne pas confondre son niveau local avec LOD 12 du viewer. Aucun mip mondial préfabriqué n’est revendiqué. |
| P2-8 concurrence PNG Windows | Hypothèse non reproduite, conservée comme risque du fallback PNG. Le chemin actif transporte les champs FP32 et ne dépend pas d’un PNG pendant sa promotion. |

Le défaut de rafale a été découvert par le dernier navigateur réel et corrigé dans `index.html`, avec régression ciblée et test rendu appris sur 24 patches sans NN supplémentaire. Les contrôles CUDA ont été rejoués après la dernière extension de manifest ; les reçus finaux lèvent la réserve temporelle citée par Opus. Le générateur CUDA et ses 107 sources n’ont pas changé pendant les corrections client/mesure.

La review Opus B5 r2 acceptait le design mais lisait le harness avant sa correction d’attente et avant publication du reçu. Les P2 A–E ont conduit aux corrections finales : recalcul forcé et snapshot re-vérifié, configuration préchargement explicitée (désactivée), samples LOD/fallback, backend WebGPU exigé et adapter enregistré, CPU décrit comme borne inférieure, second pan renommé selon son vrai point de départ, captures liées par SHA, réponses attribuées avant l’attente des headers. Astra a trouvé le second défaut de settlement sur le premier reçu continu ; ce reçu reste conservé. Le bilan attribue chaque mesure à son rapport/client et mentionne les autres sessions serveur. Aucune métrique chaude n’est présentée comme débit NN froid.

Les portes restent explicites : référence naturelle préservée sur le paquet témoin, macro non approuvée visuellement, WebGPU E2 NO-GO, profil latent scalaire NO-GO, cold NN ultra rapide et débit soutenu non acquis. Le raccord procédural attend la phase suivante.
