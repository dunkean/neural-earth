//! Raster shading in WASM. Canvas only presents these finished RGBA pixels.
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t)
}
pub fn raster(
    h: &[f32],
    fields: &[f32],
    w: usize,
    n: usize,
    step: f32,
    mode: &str,
    settings: &str,
) -> Result<Vec<u8>, String> {
    if w == 0
        || n == 0
        || w > 2048
        || n > 2048
        || h.len() != w * n
        || fields.len() != 5 * w * n
        || !step.is_finite()
        || step <= 0.0
        || !h.iter().chain(fields).all(|v| v.is_finite())
    {
        return Err("Invalid render raster".into());
    }
    let config: serde_json::Value = serde_json::from_str(settings).map_err(|e| e.to_string())?;
    let lighting = &config["lighting"];
    let materials = &config["materials"];
    let material =
        |key: &str, default: f32| materials[key].as_f64().unwrap_or(default as f64) as f32;
    let tint = |key: &str, default: [f32; 3]| -> [f32; 3] {
        std::array::from_fn(|i| materials[key][i].as_f64().unwrap_or(default[i] as f64) as f32)
    };
    let number = |key: &str, default: f32| lighting[key].as_f64().unwrap_or(default as f64) as f32;
    let strength = number("strength", 1.0).clamp(0.0, 1.0);
    let ambient = number("ambient", 0.35).clamp(0.0, 1.0);
    let exaggeration = number("exaggeration", 6.0).clamp(0.0, 20.0);
    let az = number("azimuth", 315.0).to_radians();
    let alt = number("altitude", 45.0).to_radians();
    let light = [az.sin() * alt.cos(), -az.cos() * alt.cos(), alt.sin()];
    let mut out = vec![0; w * n * 4];
    for y in 0..n {
        for x in 0..w {
            let p = y * w + x;
            let z = h[p];
            let mut temp = fields[w * n + p] - 0.006 * (z - fields[p]);
            let mut rain = fields[3 * w * n + p].max(0.0);
            let season = fields[2 * w * n + p].max(0.0) / 100.0;
            if mode == "render" {
                temp += season
                    * std::f32::consts::SQRT_2
                    * ((material("season", 0.5) - 0.5) * std::f32::consts::TAU).cos();
                rain *= 2.0f32.powf(material("moisture", 0.0) * 2.0);
            }
            let dx =
                (h[y * w + (x + 1).min(w - 1)] - h[y * w + x.saturating_sub(1)]) / (2.0 * step);
            let dy =
                (h[(y + 1).min(n - 1) * w + x] - h[y.saturating_sub(1) * w + x]) / (2.0 * step);
            let mut rgb = if mode == "temperature" {
                mix([0.18, 0.38, 0.88], [0.95, 0.22, 0.08], (temp + 35.0) / 70.0)
            } else if mode == "precipitation" {
                mix(
                    [0.78, 0.61, 0.32],
                    [0.08, 0.36, 0.72],
                    rain.ln_1p() / 4001f32.ln(),
                )
            } else if z < 0.0 {
                mix([0.02, 0.12, 0.25], [0.28, 0.54, 0.70], 1.0 + z / 6000.0)
            } else if mode == "relief" || mode == "orogen-height" {
                mix(
                    mix([0.38, 0.54, 0.32], [0.68, 0.60, 0.44], z / 2600.0),
                    [0.92, 0.95, 0.97],
                    (z - 2700.0) / 2600.0,
                )
            } else {
                let effective = rain / (1.0 + fields[4 * w * n + p].max(0.0) / 200.0);
                let dry = (450.0 + temp.max(0.0) * 25.0 + season * 8.0 - effective) / 650.0;
                let forest = if mode == "render" {
                    material("forest", 1.0)
                } else {
                    1.0
                };
                let mut land = mix(
                    [0.42, 0.57, 0.28],
                    [0.08, 0.35, 0.20],
                    effective / 2500.0 * forest,
                );
                land = mix(land, [0.82, 0.69, 0.42], dry);
                let rock = if mode == "render" {
                    ((dx * dx + dy * dy).sqrt().atan().to_degrees() - material("rock_slope", 40.0))
                        / 20.0
                } else {
                    (z - 2300.0) / 2200.0
                };
                let mut rock_color = [0.52, 0.48, 0.43];
                let mut snow_color = [0.93, 0.97, 0.99];
                let mut snow = 1.0;
                if mode == "render" {
                    let vegetation_tint = tint("vegetation_tint", [1.0; 3]);
                    let rock_tint = tint("rock_tint", [1.0; 3]);
                    for c in 0..3 {
                        land[c] *= vegetation_tint[c];
                        rock_color[c] *= rock_tint[c];
                    }
                    snow_color = tint("snow_color", [0.94, 0.96, 0.97]);
                    snow = material("snow", 1.0);
                    let curvature = h[y * w + (x + 1).min(w - 1)]
                        + h[y * w + x.saturating_sub(1)]
                        + h[(y + 1).min(n - 1) * w + x]
                        + h[y.saturating_sub(1) * w + x]
                        - 4.0 * z;
                    let variation = 1.0
                        + (curvature / step).clamp(-1.0, 1.0) * material("variation", 1.0) * 0.12;
                    for c in &mut land {
                        *c *= variation;
                    }
                }
                land = mix(land, rock_color, rock);
                mix(land, snow_color, (1.5 - temp) / 7.0 * snow)
            };
            if !matches!(mode, "temperature" | "precipitation") && z > 0.0 {
                let normal = [-dx * exaggeration, -dy * exaggeration, 1.0];
                let length = (normal.iter().map(|v| v * v).sum::<f32>()).sqrt();
                let dot = (0..3)
                    .map(|i| normal[i] * light[i] / length)
                    .sum::<f32>()
                    .max(0.0);
                let shade = 1.0 - strength + strength * (ambient + (1.0 - ambient) * dot);
                for c in &mut rgb {
                    *c *= shade;
                }
                let contours = &config["contours"];
                if contours["enabled"].as_bool().unwrap_or(false) {
                    let interval = contours["interval"].as_f64().unwrap_or(200.0).max(1.0) as f32;
                    let f = (z / interval).rem_euclid(1.0);
                    let distance = f.min(1.0 - f) * interval;
                    let thickness = ((dx * dx + dy * dy).sqrt() * step * 0.7).max(0.5);
                    if distance < thickness {
                        rgb = mix(rgb, [0.08, 0.10, 0.08], 0.55);
                    }
                }
            }
            for c in 0..3 {
                out[p * 4 + c] = (rgb[c].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            out[p * 4 + 3] = 255;
        }
    }
    Ok(out)
}
