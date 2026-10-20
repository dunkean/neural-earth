//! Distilled one-pass decoder ("student200k") on the direct WebGPU runtime.
//! Pixel norms are folded into the following convolution's input, and the
//! decoder's `cat(up(x), skip)` is read in place by its convolutions.
use crate::gpu::{Act, Aux, Conv, ConvArgs, Gpu, Second, Tensor, align};
use crate::weights::Bundle;
use std::f32::consts::FRAC_1_SQRT_2;

struct Block {
    skip: Option<Conv>,
    first: Conv,
    second: Conv,
    dilation: usize,
}

pub struct Student {
    gpu: Gpu,
    stem: Conv,
    encoders: Vec<Block>,
    down: Vec<Conv>,
    middle: Vec<Block>,
    decoders: Vec<Block>,
    /// FP32 `[head (c), head bias, direct (in)]`.
    head: wgpu::Buffer,
    eps: f32,
}

impl Student {
    pub fn load(
        gpu: &Gpu,
        bundle: &Bundle,
        depth: usize,
        dilations: &[usize],
    ) -> Result<Self, String> {
        let conv = |name: &str| -> Result<Conv, String> {
            let weight = bundle.get(&format!("{name}.weight"))?;
            let bias = format!("{name}.bias");
            gpu.conv(
                weight,
                if bundle.has(&bias) {
                    Some(bundle.get(&bias)?)
                } else {
                    None
                },
                align(weight.shape[1]),
            )
        };
        let block = |name: &str, dilation: usize| -> Result<Block, String> {
            Ok(Block {
                skip: if bundle.has(&format!("{name}.skip.weight")) {
                    Some(conv(&format!("{name}.skip"))?)
                } else {
                    None
                },
                first: conv(&format!("{name}.layers.1"))?,
                second: conv(&format!("{name}.layers.4"))?,
                dilation,
            })
        };
        let head = bundle.get("head.weight")?;
        let direct = bundle.get("direct.weight")?;
        if head.shape[0] != 1
            || direct.shape[0] != 1
            || head.shape[2..] != [1, 1]
            || direct.shape[2..] != [1, 1]
        {
            return Err("Unexpected student head".into());
        }
        let mut weights = head.to_f32();
        weights.resize(align(head.shape[1]), 0.0);
        weights.push(if bundle.has("head.bias") {
            bundle.scalar("head.bias")?
        } else {
            0.0
        });
        weights.extend(direct.to_f32());
        Ok(Self {
            gpu: gpu.clone(),
            stem: conv("stem")?,
            encoders: (0..depth)
                .map(|i| block(&format!("encoders.{i}"), 1))
                .collect::<Result<_, _>>()?,
            down: (0..depth - 1)
                .map(|i| conv(&format!("down.{i}")))
                .collect::<Result<_, _>>()?,
            middle: dilations
                .iter()
                .enumerate()
                .map(|(i, &d)| block(&format!("middle.{i}"), d))
                .collect::<Result<_, _>>()?,
            decoders: (0..depth - 1)
                .map(|i| block(&format!("decoders.{i}"), 1))
                .collect::<Result<_, _>>()?,
            head: gpu.constant(&weights),
            eps: 1e-6,
        })
    }

    /// `x` (or `cat(up(x), skip)`) → `(skip + silu(conv(pn(silu(conv(pn(x)))))) / √2`.
    fn block(&self, block: &Block, x: &Act, skip: Option<&Act>) -> Act {
        let gpu = &self.gpu;
        let d = block.dilation;
        let div = if skip.is_some() { 2 } else { 1 };
        let input = || ConvArgs {
            div,
            second: match skip {
                Some(act) => Second::Concat {
                    act,
                    scale: 1.0,
                    div: 1,
                },
                None => Second::None,
            },
            ..Default::default()
        };
        let residual = match &block.skip {
            Some(c) => gpu.conv2d(x, c, input()),
            None => {
                assert!(skip.is_none(), "concatenation without skip");
                x.clone()
            }
        };
        let scale = gpu.pixel_scale(x, div, skip, false, self.eps);
        let y = gpu.conv2d(
            x,
            &block.first,
            ConvArgs {
                padding: d,
                dilation: d,
                aux: Aux::Pixel(&scale),
                silu_out: Some(1.0),
                ..input()
            },
        );
        let scale = gpu.pixel_scale(&y, 1, None, false, self.eps);
        gpu.conv2d(
            &y,
            &block.second,
            ConvArgs {
                padding: d,
                dilation: d,
                aux: Aux::Pixel(&scale),
                silu_out: Some(1.0),
                second: Second::Residual {
                    act: &residual,
                    wa: FRAC_1_SQRT_2,
                    wb: FRAC_1_SQRT_2,
                },
                ..Default::default()
            },
        )
    }

    /// `[n, 16, h, w]` features to the `[n, 1, h, w]` FP32 normalized residual.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let gpu = &self.gpu;
        let mut x = gpu.conv2d(
            &gpu.to_act(&input, 0),
            &self.stem,
            ConvArgs {
                padding: 1,
                ..Default::default()
            },
        );
        let mut skips = Vec::new();
        for (level, block) in self.encoders.iter().enumerate() {
            x = self.block(block, &x, None);
            if level < self.down.len() {
                let down = gpu.conv2d(
                    &x,
                    &self.down[level],
                    ConvArgs {
                        stride: 2,
                        padding: 1,
                        ..Default::default()
                    },
                );
                skips.push(x);
                x = down;
            }
        }
        for block in &self.middle {
            x = self.block(block, &x, None);
        }
        for block in &self.decoders {
            let skip = skips.pop().unwrap();
            x = self.block(block, &x, Some(&skip));
        }
        gpu.student_head(&x, &input, &self.head)
    }
}
