//! Software globe projection in WASM; the browser only presents RGBA bytes.
//! Projection and lighting match web/terrain_globe.js's perspective shader.
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;
struct Texture {
    rgba: Vec<u8>,
    width: usize,
    height: usize,
}
impl Texture {
    fn sample(&self, u: f64, v: f64) -> [f64; 4] {
        if self.width == 0 {
            return [0.12, 0.29, 0.43, 1.0];
        }
        let x = (u * self.width as f64 - 0.5).clamp(0.0, (self.width - 1) as f64);
        let y = (v * self.height as f64 - 0.5).clamp(0.0, (self.height - 1) as f64);
        let (xa, ya) = (x.floor() as usize, y.floor() as usize);
        let (xb, yb) = ((xa + 1).min(self.width - 1), (ya + 1).min(self.height - 1));
        let (tx, ty) = (x - xa as f64, y - ya as f64);
        std::array::from_fn(|c| {
            let get = |y, x| self.rgba[(y * self.width + x) * 4 + c] as f64 / 255.0;
            (get(ya, xa) * (1.0 - tx) + get(ya, xb) * tx) * (1.0 - ty)
                + (get(yb, xa) * (1.0 - tx) + get(yb, xb) * tx) * ty
        })
    }
}
#[wasm_bindgen]
pub struct GlobeRaster {
    atlas: Texture,
    detail: Texture,
}
#[wasm_bindgen]
impl GlobeRaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            atlas: Texture {
                rgba: Vec::new(),
                width: 0,
                height: 0,
            },
            detail: Texture {
                rgba: Vec::new(),
                width: 0,
                height: 0,
            },
        }
    }
    pub fn set_atlas(&mut self, rgba: Vec<u8>, width: usize, height: usize) -> Result<(), JsValue> {
        self.atlas = texture(rgba, width, height)?;
        Ok(())
    }
    pub fn set_detail(
        &mut self,
        rgba: Vec<u8>,
        width: usize,
        height: usize,
    ) -> Result<(), JsValue> {
        self.detail = texture(rgba, width, height)?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &self,
        width: usize,
        height: usize,
        aspect: f64,
        yaw: f64,
        pitch: f64,
        altitude: f64,
        bounds: Vec<f64>,
        light: Vec<f64>,
    ) -> Result<Vec<u8>, JsValue> {
        if width == 0
            || height == 0
            || width > 2048
            || height > 2048
            || bounds.len() != 4
            || light.len() != 8
            || !aspect.is_finite()
            || aspect <= 0.0
            || !yaw.is_finite()
            || !pitch.is_finite()
            || !altitude.is_finite()
            || altitude <= 0.0
        {
            return Err(JsValue::from_str("Invalid globe frame"));
        }
        let mut out = vec![0; width * height * 4];
        let (cp, sp, cy, sy) = (pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin());
        let center = (0.75 - yaw / (2.0 * PI)).rem_euclid(1.0);
        for y in 0..height {
            for x in 0..width {
                let rx = (2.0 * (x as f64 + 0.5) / width as f64 - 1.0) * aspect;
                let ry = 1.0 - 2.0 * (y as f64 + 0.5) / height as f64;
                let rz = -2.41421356;
                let norm = (rx * rx + ry * ry + rz * rz).sqrt();
                let z = rz / norm;
                let b = (1.0 + altitude) * z;
                let c = altitude * (2.0 + altitude);
                let disc = b * b - c;
                let mut rgb = [0.035, 0.065, 0.10];
                if disc >= 0.0 {
                    let t = c / (-b + disc.sqrt());
                    let px = -rx / norm * t;
                    let py = ry / norm * t;
                    let pz = 1.0 + altitude + z * t;
                    let qy = py * cp + pz * sp;
                    let qz = -py * sp + pz * cp;
                    let nx = px * cy + qz * sy;
                    let nz = -px * sy + qz * cy;
                    let u = (nz.atan2(nx) / (2.0 * PI) + 0.5).rem_euclid(1.0);
                    let polar = nx.hypot(nz).atan2(qy.abs()) / PI;
                    let v = if qy >= 0.0 { polar } else { 1.0 - polar };
                    let color = self.atlas.sample(u, v);
                    rgb.copy_from_slice(&color[..3]);
                    let du = u + (center - u).round();
                    if self.detail.width > 0
                        && du >= bounds[0]
                        && du <= bounds[2]
                        && v >= bounds[1]
                        && v <= bounds[3]
                    {
                        let color = self.detail.sample(
                            (du - bounds[0]) / (bounds[2] - bounds[0]),
                            (v - bounds[1]) / (bounds[3] - bounds[1]),
                        );
                        for c in 0..3 {
                            rgb[c] = rgb[c] * (1.0 - color[3]) + color[c] * color[3];
                        }
                    }
                    let shade = (nx * light[4] + qy * light[5] + nz * light[6])
                        .max(0.0)
                        .powf(light[2]);
                    let intensity =
                        1.0 - light[0] + light[0] * (light[1] + (1.0 - light[1]) * shade);
                    for c in &mut rgb {
                        *c *= intensity;
                    }
                }
                for c in 0..3 {
                    out[(y * width + x) * 4 + c] = (rgb[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                out[(y * width + x) * 4 + 3] = 255;
            }
        }
        Ok(out)
    }
}
impl Default for GlobeRaster {
    fn default() -> Self {
        Self::new()
    }
}
fn texture(rgba: Vec<u8>, width: usize, height: usize) -> Result<Texture, JsValue> {
    if width > 2048 || height > 2048 || rgba.len() != width * height * 4 {
        return Err(JsValue::from_str("Invalid globe texture"));
    }
    Ok(Texture {
        rgba,
        width,
        height,
    })
}
