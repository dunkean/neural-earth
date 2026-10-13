//! Magnitude-preserving EDM2 UNet of Terrain Diffusion (coarse and base networks)
//! on the direct WebGPU runtime. Same graph as the shared Rust port: blocks fuse
//! `mp_silu`, the embedding modulation, `mp_sum` and the clamp into their convolutions.
use crate::gpu::{Act, Aux, Buf, Conv, ConvArgs, Gpu, Modulation, Second, Tensor, align};
use crate::weights::Bundle;
use std::rc::Rc;

pub use crate::gpu::up;

const SILU_GAIN: f32 = 1.0 / 0.596;

pub struct EdmConfig {
    pub image_size: usize,
    pub channels: Vec<usize>,
    pub layers: usize,
    pub midblock_attention: bool,
    pub concat_balance: f32,
    /// Conditional inputs: `(fourier channels, weight)`; zero channels means a tensor input.
    pub conditional: Vec<(usize, f32)>,
}

impl EdmConfig {
    pub fn coarse() -> Self {
        Self {
            image_size: 16,
            channels: vec![128],
            layers: 2,
            midblock_attention: false,
            concat_balance: 0.5,
            conditional: vec![(64, 0.2); 5],
        }
    }
    pub fn base() -> Self {
        Self {
            image_size: 512,
            channels: vec![192, 384, 576, 768],
            layers: 3,
            midblock_attention: true,
            concat_balance: 0.5,
            conditional: vec![(0, 1.0)],
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Resample {
    Keep,
    Down,
    Up,
}

struct Block {
    encoder: bool,
    resample: Resample,
    skip: Option<Conv>,
    res0: Conv,
    res1: Conv,
    modulation: usize,
    heads: usize,
    qkv: Option<Conv>,
    proj: Option<Conv>,
}

#[allow(clippy::large_enum_variant)]
enum Layer {
    Conv(Conv),
    Block(Block),
}

struct Conditional {
    fourier: Option<(Vec<f32>, Vec<f32>)>,
    /// `[dim, inputs]` row-major.
    linear: Vec<f32>,
    inputs: usize,
}

pub struct Edm {
    gpu: Gpu,
    dim: usize,
    noise_freqs: Vec<f32>,
    noise_linear: Vec<f32>,
    conditional: Vec<Conditional>,
    weights: Vec<f32>,
    encoder: Vec<Layer>,
    decoder: Vec<(bool, Block)>,
    out_conv: Conv,
    concat_balance: f32,
    modulation: Modulation,
}

/// Per-sample block modulations of one forward pass.
pub struct Embedding {
    buf: Rc<Buf>,
}

fn mp_weights(t: f32) -> (f32, f32) {
    let norm = ((1.0 - t) * (1.0 - t) + t * t).sqrt();
    ((1.0 - t) / norm, t / norm)
}

/// `y = x · Wᵀ` for `x: [n, inputs]`, `W: [out, inputs]`.
fn linear(x: &[f32], w: &[f32], inputs: usize) -> Vec<f32> {
    let n = x.len() / inputs;
    let out = w.len() / inputs;
    let mut y = vec![0.0; n * out];
    for r in 0..n {
        let row = &x[r * inputs..(r + 1) * inputs];
        for o in 0..out {
            let wr = &w[o * inputs..(o + 1) * inputs];
            y[r * out + o] = row.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    y
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

impl Edm {
    pub fn load(gpu: &Gpu, bundle: &Bundle, config: &EdmConfig) -> Result<Self, String> {
        let conv = |name: &str| -> Result<Conv, String> {
            let entry = bundle.get(name)?;
            gpu.conv(entry, None, align(entry.shape[1]))
        };
        let mut matrices: Vec<(Vec<f32>, usize)> = Vec::new();
        let noise_linear_entry = bundle.get("noise_linear.weight")?;
        let dim = noise_linear_entry.shape[0];
        let mut block = |name: &str,
                         encoder: bool,
                         resample: Resample,
                         attention: bool|
         -> Result<Block, String> {
            let res0 = conv(&format!("{name}.conv_res0.weight"))?;
            let heads = if attention { res0.cout / 64 } else { 0 };
            let skip_name = format!("{name}.conv_skip.weight");
            let emb = bundle.get(&format!("{name}.emb_linear.weight"))?;
            if emb.shape != [res0.cout, dim] {
                return Err(format!("Unexpected modulation shape in {name}"));
            }
            matrices.push((emb.to_f32(), res0.cout));
            Ok(Block {
                encoder,
                resample,
                skip: if bundle.has(&skip_name) {
                    Some(conv(&skip_name)?)
                } else {
                    None
                },
                res0,
                res1: conv(&format!("{name}.conv_res1.weight"))?,
                modulation: matrices.len() - 1,
                heads,
                qkv: if attention {
                    Some(conv(&format!("{name}.attn_qkv.weight"))?)
                } else {
                    None
                },
                proj: if attention {
                    Some(conv(&format!("{name}.attn_proj.weight"))?)
                } else {
                    None
                },
            })
        };
        let res = |level: usize| config.image_size >> level;
        let mut encoder = Vec::new();
        for level in 0..config.channels.len() {
            let r = res(level);
            if level == 0 {
                encoder.push(Layer::Conv(conv(&format!("enc.{r}x{r}_conv.weight"))?));
            } else {
                encoder.push(Layer::Block(block(
                    &format!("enc.{r}x{r}_down"),
                    true,
                    Resample::Down,
                    false,
                )?));
            }
            for i in 0..config.layers {
                encoder.push(Layer::Block(block(
                    &format!("enc.{r}x{r}_block{i}"),
                    true,
                    Resample::Keep,
                    false,
                )?));
            }
        }
        let mut decoder = Vec::new();
        for level in (0..config.channels.len()).rev() {
            let r = res(level);
            if level == config.channels.len() - 1 {
                decoder.push((
                    false,
                    block(
                        &format!("dec.{r}x{r}_in0"),
                        false,
                        Resample::Keep,
                        config.midblock_attention,
                    )?,
                ));
                decoder.push((
                    false,
                    block(&format!("dec.{r}x{r}_in1"), false, Resample::Keep, false)?,
                ));
            } else {
                decoder.push((
                    false,
                    block(&format!("dec.{r}x{r}_up"), false, Resample::Up, false)?,
                ));
            }
            for i in 0..=config.layers {
                decoder.push((
                    true,
                    block(
                        &format!("dec.{r}x{r}_block{i}"),
                        false,
                        Resample::Keep,
                        false,
                    )?,
                ));
            }
        }
        let mut conditional = Vec::new();
        let mut weights = vec![1.0];
        for (i, &(fourier, weight)) in config.conditional.iter().enumerate() {
            let (fourier, name) = if fourier > 0 {
                (
                    Some((
                        bundle
                            .get(&format!("conditional_layers.{i}.0.freqs"))?
                            .to_f32(),
                        bundle
                            .get(&format!("conditional_layers.{i}.0.phases"))?
                            .to_f32(),
                    )),
                    format!("conditional_layers.{i}.1.weight"),
                )
            } else {
                (None, format!("conditional_layers.{i}.weight"))
            };
            let entry = bundle.get(&name)?;
            conditional.push(Conditional {
                fourier,
                linear: entry.to_f32(),
                inputs: entry.shape[1],
            });
            weights.push(weight);
        }
        let (modulation, offsets) = Modulation::new(gpu, &matrices, dim);
        let assign = |b: &mut Block| b.modulation = offsets[b.modulation];
        for layer in &mut encoder {
            if let Layer::Block(b) = layer {
                assign(b);
            }
        }
        for (_, b) in &mut decoder {
            assign(b);
        }
        Ok(Self {
            gpu: gpu.clone(),
            dim,
            noise_freqs: bundle.get("noise_fourier.freqs")?.to_f32(),
            noise_linear: noise_linear_entry.to_f32(),
            conditional,
            weights,
            encoder,
            decoder,
            out_conv: conv("out_conv.weight")?,
            concat_balance: config.concat_balance,
            modulation,
        })
    }

    /// Conditioning embedding (FP32, on the CPU: a few hundred thousand MACs),
    /// then every block's normalized modulation on the GPU.
    pub fn embedding(
        &self,
        noise_labels: &[f32],
        scalars: &[Vec<f32>],
        tensor: Option<(&[f32], usize)>,
    ) -> Embedding {
        let n = noise_labels.len();
        let mut positional = Vec::with_capacity(n * self.noise_freqs.len() * 2);
        for &t in noise_labels {
            let row: Vec<f32> = self.noise_freqs.iter().map(|&f| t * f).collect();
            positional.extend(row.iter().map(|y| y.sin() * std::f32::consts::SQRT_2));
            positional.extend(row.iter().map(|y| y.cos() * std::f32::consts::SQRT_2));
        }
        let mut embeds = vec![linear(
            &positional,
            &self.noise_linear,
            self.noise_freqs.len() * 2,
        )];
        let mut scalar_index = 0;
        for layer in &self.conditional {
            embeds.push(match &layer.fourier {
                Some((freqs, phases)) => {
                    let values = &scalars[scalar_index];
                    scalar_index += 1;
                    let mut features = Vec::with_capacity(n * freqs.len());
                    for &x in values {
                        features.extend(
                            freqs
                                .iter()
                                .zip(phases)
                                .map(|(f, p)| (x * f + p).cos() * std::f32::consts::SQRT_2),
                        );
                    }
                    linear(&features, &layer.linear, layer.inputs)
                }
                None => {
                    let (values, _) = tensor.expect("tensor conditioning");
                    linear(values, &layer.linear, layer.inputs)
                        .into_iter()
                        .map(|v| silu(v) / 0.596)
                        .collect()
                }
            });
        }
        let norm = self.weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        let mut emb = vec![0.0f32; n * self.dim];
        for (e, w) in embeds.iter().zip(&self.weights) {
            for (a, b) in emb.iter_mut().zip(e) {
                *a += b * (w / norm);
            }
        }
        for v in &mut emb {
            *v = silu(*v) / 0.596;
        }
        let emb = self.gpu.floats(&emb);
        Embedding {
            buf: self.gpu.modulation(&emb, n, &self.modulation),
        }
    }

    fn block(
        &self,
        block: &Block,
        x: Act,
        concat: Option<(&Act, f32, f32)>,
        emb: &Embedding,
    ) -> Act {
        let gpu = &self.gpu;
        let x = match block.resample {
            Resample::Keep => x,
            Resample::Down => gpu.resample(&x, 2, 1),
            Resample::Up => gpu.resample(&x, 1, 2),
        };
        let modulation = || Aux::Modulation {
            buf: &emb.buf,
            stride: self.modulation.total,
            offset: block.modulation,
        };
        let (scale, second) = match concat {
            Some((skip, a, b)) => (a, Some((skip, b))),
            None => (1.0, None),
        };
        let input = || ConvArgs {
            scale,
            second: match second {
                Some((act, scale)) => Second::Concat { act, scale, div: 1 },
                None => Second::None,
            },
            ..Default::default()
        };
        let (x, y) = if block.encoder {
            let x = match &block.skip {
                Some(skip) => gpu.conv2d(&x, skip, ConvArgs::default()),
                None => x,
            };
            let x = gpu.scale_pixels(&x, &gpu.pixel_scale(&x, 1, None, true, 0.0));
            let y = gpu.conv2d(
                &x,
                &block.res0,
                ConvArgs {
                    padding: 1,
                    silu_in: Some(SILU_GAIN),
                    aux: modulation(),
                    silu_out: Some(SILU_GAIN),
                    ..Default::default()
                },
            );
            (x, y)
        } else {
            let y = gpu.conv2d(
                &x,
                &block.res0,
                ConvArgs {
                    padding: 1,
                    silu_in: Some(SILU_GAIN),
                    aux: modulation(),
                    silu_out: Some(SILU_GAIN),
                    ..input()
                },
            );
            let x = match &block.skip {
                Some(skip) => gpu.conv2d(&x, skip, input()),
                None => {
                    assert!(
                        second.is_none() && scale == 1.0,
                        "concatenation without skip"
                    );
                    x
                }
            };
            (x, y)
        };
        let (wa, wb) = mp_weights(0.3);
        let out = gpu.conv2d(
            &y,
            &block.res1,
            ConvArgs {
                padding: 1,
                second: Second::Residual { act: &x, wa, wb },
                clamp: (block.heads == 0).then_some(256.0),
                ..Default::default()
            },
        );
        if block.heads == 0 {
            return out;
        }
        let qkv = gpu.conv2d(&out, block.qkv.as_ref().unwrap(), ConvArgs::default());
        let attended = gpu.attention(&qkv, block.heads, 64);
        gpu.conv2d(
            &attended,
            block.proj.as_ref().unwrap(),
            ConvArgs {
                second: Second::Residual { act: &out, wa, wb },
                clamp: Some(256.0),
                ..Default::default()
            },
        )
    }

    /// Network output in FP32. `x` is `[n, in, h, w]`.
    pub fn forward(&self, x: Tensor<4>, emb: &Embedding) -> Tensor<4> {
        let gpu = &self.gpu;
        let mut x = gpu.to_act(&x, 1);
        let mut skips: Vec<(Act, usize)> = Vec::new();
        let mut channels = 0;
        for layer in &self.encoder {
            x = match layer {
                Layer::Conv(conv) => {
                    channels = conv.cout;
                    gpu.conv2d(
                        &x,
                        conv,
                        ConvArgs {
                            padding: 1,
                            ..Default::default()
                        },
                    )
                }
                Layer::Block(block) => {
                    channels = block.res1.cout;
                    self.block(block, x, None, emb)
                }
            };
            skips.push((x.clone(), channels));
        }
        let t = self.concat_balance;
        for (concat, block) in &self.decoder {
            if *concat {
                let (skip, b) = skips.pop().unwrap();
                let (a, b) = (channels as f32, b as f32);
                let scale = ((a + b) / ((1.0 - t) * (1.0 - t) + t * t)).sqrt();
                x = self.block(
                    block,
                    x,
                    Some((&skip, scale / a.sqrt() * (1.0 - t), scale / b.sqrt() * t)),
                    emb,
                );
            } else {
                x = self.block(block, x, None, emb);
            }
            channels = block.res1.cout;
        }
        let out = gpu.conv2d(
            &x,
            &self.out_conv,
            ConvArgs {
                padding: 1,
                ..Default::default()
            },
        );
        gpu.from_act(&out, self.out_conv.cout)
    }
}
