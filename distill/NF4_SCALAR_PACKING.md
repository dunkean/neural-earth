# Paquet NF4 avec gain de sortie FP32

Le format `native-weight-quant-scalar-packed` conserve les codes NF4 et leurs
échelles FP32, ainsi que les tenseurs sensibles BF16. Le seul changement de
payload par rapport au paquet précédent est `net.out_gain`, de 2 à 4 octets.
Le manifeste conserve aussi sa référence BF16 `gain_anchor`.

Pour la convolution de sortie, le facteur est :

```text
fan_in = sqrt(nombre d'éléments d'un filtre)
scale = float32(round_bf16(gain_anchor / fan_in))
        + (gain_fp32 - float32(gain_anchor)) / fan_in
kernel = round_bf16(normalize_fp32(raw_bf16_kernel) * scale)
```

Cette expression reproduit exactement le facteur initial BF16 et conserve les
petits incréments du gain. Les convolutions internes NF4 sont déjà normalisées
et incluent leur fan-in : **ne pas les normaliser à nouveau**. Les activations
et convolutions restent BF16 dans le runtime CUDA ; aucun calcul quatre bits
ou gain de vitesse n'est revendiqué.

## Validation du prototype

Audit terminé sur RTX 4090 avec un gain délibérément déplacé de 1/16 de cran
BF16, pour que l'erreur de serialization ne puisse pas rester invisible :
−0,19427490234375 en FP32, contre −0,1943359375 une fois arrondi en BF16.
Ce prototype est un test de format sur TRAIN, pas un modèle calibré ou accepté.

- 174 tenseurs vérifiés, dont 93 kernels NF4 : égalité exacte des valeurs utiles.
- Trois batchs réels de huit fenêtres : sorties exactes avant/après export.
- Un champ complet de montagne humide : les deux étapes, leurs entrées et le
  cœur fusionné sont identiques ; les hauteurs du decoder original aussi.
- Même contrôle pour le runtime compact aux poids résidents BF16, gain FP32.
- Trois tests CPU vérifient les offsets, les payloads inchangés, la précision
  du gain et le rejet de données tronquées.

Poids binaires : **149 604 748 octets**. Manifeste : **49 994 octets**.
Total : **149,654742 Mo** (Mo décimaux). Les deux octets supplémentaires
n'affectent donc pas le budget de téléchargement de 150 Mo.

Preuves sous `/home/delete/data/distill/eval/` :

- `native-nf4-scalar-pack-prototype-4090/audit.json` et `packed/` ;
- `native-nf4-scalar-pack-runtime-prototype-4090/audit.json` : contrôle incluant
  le runtime compact, avec référence au même paquet immuable.

Un premier audit avait échoué avant ses forwards à cause d'un argument manquant
du harness. Le paquet a été conservé, puis réaudité explicitement : aucun
export précédent n'a été écrasé. Les preuves de succès sont les rapports ci-dessus.

Les codecs sont dans `native_scalar_pack.py`. Pour un checkpoint entraîné :

```python
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
manifest = export_scalar_packed(saved, output, device='cuda')
net = runtime_scalar_packed(output, device='cuda')
```

Le checkpoint doit déclarer `native-weight-quant-scalar` et
`anchored-fp32-v1`. Les exports refusent d'écraser un paquet existant. Le GPU
d'export est à choisir explicitement : le prototype4090 valide le format, pas
l'identité de la quantification aux frontières de codes sur5090. Le checkpoint
retenu devra être exporté et audité sur5090 après entraînement, puis évalué sur
les bancs physiques et les raccords. Aucun poids expérimental n'est placé dans Git.
