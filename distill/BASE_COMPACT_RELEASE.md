# Base compact B75 — première version pour intégration

Release : [`base-nf4-b75-v1`](https://github.com/dunkean/neural-earth/releases/tag/base-nf4-b75-v1).
Source : EMA du bras B à 75 000, teacher natif, poids NF4 et gain de sortie FP32.
Cette version fige un candidat utilisable pendant le développement des suivants.

## Téléchargement

- [Paquet ZIP](https://github.com/dunkean/neural-earth/releases/download/base-nf4-b75-v1/base-nf4-b75-v1.zip) : **132 287 323 octets**, soit 132,3 Mo / 126,2 Mio.
- [weights.bin](https://github.com/dunkean/neural-earth/releases/download/base-nf4-b75-v1/weights.bin) : **149 604 748 octets**, soit 149,6 Mo.
- [manifest.json](https://github.com/dunkean/neural-earth/releases/download/base-nf4-b75-v1/manifest.json) : configuration, formes, offsets, codebook et empreinte du binaire.
- [SHA256SUMS.txt](https://github.com/dunkean/neural-earth/releases/download/base-nf4-b75-v1/SHA256SUMS.txt) et audits accompagnent la release.

SHA256 de `weights.bin` :
`4f84df2853aecb830b7cf4bee60572e08e2503232a25d7ebaaee0231930d6dd2`.
Le paquet n'est donc pas exactement un modèle de 120 Mo : cette version fait
132,3 Mo à télécharger en ZIP et 149,6 Mo décompressée.

## Interface et calcul

Conserver la géométrie et le solver du base original : fenêtres 64×64 au pas
de 32, **deux étapes**, entrées et sorties à cinq canaux, vecteur conditionnel
de 58 valeurs par fenêtre, mêmes bruits et fusion. `config` dans le manifeste
décrit l'U-Net complet. Il conserve 253 688 037 paramètres.

Référence Python CUDA, depuis la branche/tag contenant ce document :

```python
from distill.native_scalar_pack import runtime_scalar_packed

net = runtime_scalar_packed("/chemin/base-nf4-b75-v1", device="cuda")
velocity = net(inputs.bfloat16(), noise_labels=labels.bfloat16(),
               conditional_inputs=[condition.bfloat16()])
```

Cette fonction charge un **réseau**, pas le solver ni toute la chaîne terrain.
Le coarse et le decoder sont les modèles existants. Le paquet publié fournit
des poids et une configuration ; il ne contient pas un graphe ONNX.

## Lecture dans le navigateur

[`packed_base.mjs`](packed_base.mjs) décode chaque tenseur en `Float32Array`.
Les valeurs BF16 effectives sont conservées et le gain de sortie reste FP32.
Le lecteur est indépendant du backend WASM/WebGPU de l'application :

```javascript
import {decodePackedTensor} from './packed_base.mjs';

const manifest = await fetch('/models/base/manifest.json').then(r => r.json());
const bytes = await fetch('/models/base/weights.bin').then(r => r.arrayBuffer());
const tensor = decodePackedTensor(manifest, bytes, 'net.enc.512x512_conv.weight');
```

Choisir les noms depuis `manifest.tensors`. Le binaire suit l'ordre C, little
endian ; les offsets doivent être lus avec `DataView`, car certains ne sont
pas alignés sur quatre octets. NF4 : premier code dans le nibble bas, deuxième
dans le haut ; un facteur FP32 par bloc de 512 coefficients.

Les noyaux NF4 contiennent déjà la normalisation MP globale et le fan-in :
**ne pas les renormaliser**. Les tenseurs BF16 des autres MPConv restent des
poids bruts, soumis à leur expression habituelle de normalisation. Pour la
convolution de sortie, appliquer le gain ancré décrit dans
[`NF4_SCALAR_PACKING.md`](NF4_SCALAR_PACKING.md), avec `gain_anchor` du manifeste.

Le stockage compact ne réduit pas le nombre de calculs. La référence CUDA
déquantifie au chargement et calcule en BF16. La représentation temporaire
Float32 du lecteur JavaScript peut occuper environ 1 Go si tous les tenseurs
sont gardés simultanément ; lire puis transférer les tenseurs vers le format
résident choisi par l'intégrateur. La qualité et la vitesse du backend
navigateur restent à mesurer après portage.

## Qualité et validation de cette version

Audit RTX 5090 : **174 tenseurs exacts**, trois lots de fenêtres exacts et
cinq champs autonomes à deux étapes exacts avant/après export, dont plaines,
côtes et montagne humide. Le runtime compact est identique à la source QAT.
Le décodeur JavaScript a aussi été comparé à la référence Python : les octets
Float32 reconstruits des **174 tenseurs sont exactement identiques**.

Sur les sept lieux historiques, coarse 4 et decoder distillé communs :

| MAE moyenne face au base BF16 original | LOD 3 | LOD 0 |
|---|---:|---:|
| NF4 EMA 70k précédent | 37,26 m | 36,06 m |
| B75 publié | 24,06 m | 22,15 m |
| Écart naturel base FP32/BF16 | 22,43 m | 21,10 m |

Le grain LOD 3 en plaine et les pixels côtiers restent à améliorer. Sur les
sept captures de progression, l'amplitude HF moyenne vaut environ 1,51× celle
du teacher. Les raccords physiques froids sur sept scènes et deux LOD sont
du même ordre que ceux de l'original ; certains maxima augmentent. Le mode
canonique de ce nouveau checkpoint n'a pas encore été vérifié.

Cette publication fixe une référence d'intégration ; elle ne déclare pas
l'achèvement de la recherche ni une amélioration de vitesse. Les temps publiés
pour la même architecture sur les checkpoints précédents sont légèrement
plus lents que l'original. B75 n'a pas encore son benchmark propre.

Provenance : teacher `xandergos/terrain-diffusion-30m`, révision
`9ef8030cb805b433b98ec25c5dddefbac07a9e26`, licence MIT ; notice et licence
inclues dans la release. Le codebook NF4 suit QLoRA/bitsandbytes ; le codec est
une implémentation indépendante.
