//! Loader for the same NF4 B75 artifact used by Python's `native_scalar_pack`.
//!
//! Internal convolutions are already normalized effective kernels. Decode their
//! nibbles once at load time; never apply MP normalization a second time. The
//! output convolution retains the release's anchored FP32 gain arithmetic.
use crate::weights::{Bundle, Entry, Values};
use half::{bf16, f16};
use serde_json::Value;

fn integer(row: &Value, key: &str) -> Result<usize, String> {
    row[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("Invalid packed-base {key}"))
}

fn payload(data: &[u8], offset: usize, count: usize) -> Result<&[u8], String> {
    data.get(
        offset
            ..offset
                .checked_add(count)
                .ok_or("Packed-base size overflow")?,
    )
    .ok_or_else(|| "Truncated packed-base weights".into())
}

fn scalar(data: &[u8], tensors: &Value, name: &str) -> Result<f32, String> {
    let row = &tensors[name];
    let bytes = payload(data, integer(row, "offset")?, integer(row, "bytes")?)?;
    let value = match (row["dtype"].as_str(), bytes.len()) {
        (Some("bfloat16"), 2) => bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        (Some("float32"), 4) => f32::from_le_bytes(bytes.try_into().unwrap()),
        _ => return Err(format!("Invalid packed-base scalar {name}")),
    };
    if !value.is_finite() {
        return Err(format!("Nonfinite packed-base scalar {name}"));
    }
    Ok(value)
}

pub fn parse(manifest: &Value, data: &[u8]) -> Result<Bundle, String> {
    if manifest["architecture"] != "native-weight-quant-scalar-packed"
        || manifest["scalar_precision"] != "anchored-fp32-v1"
        || manifest["scheme"] != "nf4"
        || manifest["endian"] != "little"
        || integer(manifest, "bytes")? != data.len()
    {
        return Err("Unsupported packed-base contract".into());
    }
    // The existing Rust EDM forward is shared with this native-interface student.
    let config = &manifest["config"];
    if config["image_size"] != 512
        || config["in_channels"] != 5
        || config["out_channels"] != 5
        || config["model_channels"] != 192
        || config["model_channel_mults"] != serde_json::json!([1, 2, 3, 4])
        || config["layers_per_block"] != 3
        || config["midblock_attention"] != true
        || config["concat_balance"] != 0.5
        || config["conditional_inputs"].as_array().map(Vec::len) != Some(1)
        || config["conditional_inputs"][0].as_array().map(Vec::len) != Some(3)
        || config["conditional_inputs"][0][0] != "tensor"
        || config["conditional_inputs"][0][1].as_u64() != Some(58)
        || config["conditional_inputs"][0][2].as_f64() != Some(1.0)
    {
        return Err("Packed-base architecture does not match the shared Rust EDM".into());
    }
    let block = integer(manifest, "block")?;
    if block < 16 || !block.is_multiple_of(2) {
        return Err("Invalid NF4 block length".into());
    }
    let codebook: Vec<f32> = manifest["codebook"]
        .as_array()
        .ok_or("Missing NF4 codebook")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
        .collect();
    if codebook.len() != 16 || codebook.iter().any(|v| !v.is_finite()) {
        return Err("Invalid NF4 codebook".into());
    }
    let tensors = &manifest["tensors"];
    let gain = scalar(data, tensors, "net.out_gain")?;
    let anchor = manifest["gain_anchor"]
        .as_f64()
        .ok_or("Missing gain anchor")? as f32;
    if !anchor.is_finite() || bf16::from_f32(anchor).to_f32() != anchor {
        return Err("Invalid BF16 gain anchor".into());
    }
    let mut bundle = Bundle::default();
    bundle
        .metadata
        .insert("config".into(), "base-nf4-b75".into());
    for (original, row) in tensors.as_object().ok_or("Missing packed tensors")? {
        let name = original
            .strip_prefix("net.")
            .ok_or("Invalid base tensor prefix")?;
        if name.starts_with("logvar_")
            || !(name.ends_with(".weight") || name.ends_with("freqs") || name.ends_with("phases"))
        {
            continue;
        }
        let shape: Vec<usize> = row["shape"]
            .as_array()
            .ok_or("Missing packed tensor shape")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or("Invalid tensor shape")
            })
            .collect::<Result<_, _>>()?;
        let count = shape
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or("Tensor size overflow")?;
        let bytes = payload(data, integer(row, "offset")?, integer(row, "bytes")?)?;
        let values = if row["dtype"] == "nf4" {
            if shape.len() != 4 || row["effective_gain"] != 1 {
                return Err(format!("Invalid effective NF4 convolution {name}"));
            }
            let scales_count = integer(row, "scale_count")?;
            let scale_bytes = payload(
                data,
                integer(row, "scale_offset")?,
                scales_count.checked_mul(4).ok_or("Scale size overflow")?,
            )?;
            if count > bytes.len() * 2 || scales_count.checked_mul(block) != Some(bytes.len() * 2) {
                return Err(format!("Invalid NF4 block lengths for {name}"));
            }
            let scales: Vec<f32> = scale_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            if scales.iter().any(|v| !v.is_finite()) {
                return Err("Nonfinite NF4 scales".into());
            }
            // A block shares one scale: decode its 16 possible values once.
            let mut decoded = Vec::with_capacity(count);
            for (b, &scale) in scales.iter().enumerate() {
                let table: [f16; 16] = std::array::from_fn(|code| {
                    f16::from_f32(bf16::from_f32(codebook[code] * scale).to_f32())
                });
                let end = ((b + 1) * block).min(count);
                for i in b * block..end {
                    decoded.push(table[((bytes[i / 2] >> ((i % 2) * 4)) & 15) as usize]);
                }
            }
            Values::F16(decoded)
        } else {
            if row["dtype"] != "bfloat16" || bytes.len() != count * 2 {
                return Err(format!("Invalid sensitive tensor {name}"));
            }
            let mut values: Vec<f32> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16::from_le_bytes(b).to_f32())
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(format!("Nonfinite sensitive tensor {name}"));
            }
            if let Some(module) = name.strip_suffix(".weight") {
                if !matches!(shape.len(), 2 | 4) || count == 0 {
                    return Err(format!("Invalid MP weight shape {name}"));
                }
                let fan = (shape[1..].iter().product::<usize>() as f32).sqrt();
                let scale = if module == "out_conv" {
                    bf16::from_f32(anchor / fan).to_f32() + (gain - anchor) / fan
                } else if let Some(block_name) = module.strip_suffix(".emb_linear") {
                    bf16::from_f32(
                        scalar(data, tensors, &format!("net.{block_name}.emb_gain"))? / fan,
                    )
                    .to_f32()
                } else {
                    1.0 / fan
                };
                let rms = (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / count as f64)
                    .sqrt() as f32;
                for value in &mut values {
                    *value = bf16::from_f32((*value / (rms + 1e-4)) * scale).to_f32();
                }
            }
            if shape.len() == 4 {
                Values::F16(values.into_iter().map(f16::from_f32).collect())
            } else {
                Values::F32(values)
            }
        };
        bundle
            .tensors
            .insert(name.to_owned(), Entry { shape, values });
    }
    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> (Value, Vec<u8>) {
        let mut data = (-0.194_658_6_f32).to_le_bytes().to_vec();
        data.extend([0x10, 0x32, 0, 0, 0, 0, 0, 0]);
        data.extend(0.25f32.to_le_bytes());
        data.extend(bf16::from_f32(0.75).to_le_bytes());
        data.extend(bf16::from_f32(-0.25).to_le_bytes());
        let manifest = json!({
            "architecture":"native-weight-quant-scalar-packed", "scalar_precision":"anchored-fp32-v1",
            "scheme":"nf4", "endian":"little", "bytes":data.len(), "block":16,
            "codebook":[-1.,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,
                -0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.,
                0.07958029955625534,0.16093020141124725,0.24611230194568634,
                0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.],
            "gain_anchor":-0.1943359375,
            "config":{"image_size":512,"in_channels":5,"out_channels":5,"model_channels":192,
                "model_channel_mults":[1,2,3,4],"layers_per_block":3,"midblock_attention":true,
                "concat_balance":0.5,"conditional_inputs":[["tensor",58,1.0]]},
            "tensors":{
                "net.out_gain":{"dtype":"float32","shape":[],"offset":0,"bytes":4},
                "net.enc.test.weight":{"dtype":"nf4","shape":[1,4,1,1],"offset":4,"bytes":8,
                    "scale_offset":12,"scale_count":1,"effective_gain":1},
                "net.out_conv.weight":{"dtype":"bfloat16","shape":[1,2,1,1],"offset":16,"bytes":4}
            }
        });
        (manifest, data)
    }

    #[test]
    fn packed_codes_keep_nibble_order_and_baked_scale() {
        let (manifest, data) = fixture();
        let bundle = parse(&manifest, &data).unwrap();
        // Values from Python read_tensor: codebook[c] * scale -> BF16.
        assert_eq!(
            bundle.get("enc.test.weight").unwrap().to_f32(),
            vec![-0.25, -0.173_828_13, -0.130_859_38, -0.098_632_81]
        );
    }

    #[test]
    fn anchored_gain_matches_python_scalar_runtime() {
        let (manifest, data) = fixture();
        let bundle = parse(&manifest, &data).unwrap();
        // Python anchored_output_conv with these two BF16 raw weights.
        assert_eq!(
            bundle.get("out_conv.weight").unwrap().to_f32(),
            vec![-0.184_570_31, 0.061_767_578]
        );
    }

    #[test]
    fn truncated_or_incompatible_artifacts_are_rejected() {
        let (mut manifest, mut data) = fixture();
        data.pop();
        assert!(parse(&manifest, &data).is_err());
        let (_, data) = fixture();
        manifest["config"]["layers_per_block"] = json!(2);
        assert!(parse(&manifest, &data).is_err());
    }

    #[test]
    fn javascript_integer_serialization_preserves_conditioning_contract() {
        let (mut manifest, data) = fixture();
        manifest["config"]["conditional_inputs"][0][2] = json!(1);
        assert!(parse(&manifest, &data).is_ok());
    }
}
