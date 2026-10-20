include!(concat!(env!("OUT_DIR"), "/engine.rs"));
pub mod engine {
    pub mod kernels;
}
pub mod atlas;
#[path = "engine/edm.rs"]
pub mod edm;
#[path = "engine/gpu.rs"]
pub mod gpu;
pub mod packed_base;
pub mod render;
#[path = "engine/student.rs"]
pub mod student;
pub use pipeline::{Clock, Networks, Request};
pub use sketch::{Climate, Family, Regional, Sketch, Validation};
use std::{rc::Rc, sync::Arc};
use wasm_bindgen::prelude::*;

pub fn world_seed(seed: &str) -> u64 {
    seed.trim().parse().unwrap_or_else(|_| {
        seed.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        })
    })
}
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace=performance,js_name=now)]
    pub(crate) fn now() -> f64;
}
struct Performance;
impl Clock for Performance {
    fn now_ms(&self) -> f64 {
        now()
    }
}
fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Acquires the GPU and uploads the three networks.
pub async fn load_networks(
    coarse: Vec<u8>,
    base: Vec<u8>,
    decoder: Vec<u8>,
) -> Result<Networks, String> {
    let t0 = gpu::clock_ms();
    let gpu = gpu::Gpu::new().await?;
    let t1 = gpu::clock_ms();
    let bundles = (
        weights::Bundle::parse(&coarse)?,
        weights::Bundle::parse(&base)?,
        weights::Bundle::parse(&decoder)?,
    );
    let t2 = gpu::clock_ms();
    let networks = Networks::load(gpu, &bundles.0, &bundles.1, &bundles.2);
    let t3 = gpu::clock_ms();
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("NEURAL_EARTH_VERBOSE").is_ok() {
        eprintln!(
            "device {:.0} ms, parse {:.0} ms, upload {:.0} ms",
            t1 - t0,
            t2 - t1,
            t3 - t2
        );
    }
    let _ = (t0, t1, t2, t3);
    networks
}

#[wasm_bindgen]
pub struct NeuralEarth {
    networks: Rc<Networks>,
    precision: String,
    atlas: Arc<atlas::Atlas>,
    seed: u64,
}
#[wasm_bindgen]
impl NeuralEarth {
    pub async fn create(coarse: Vec<u8>, base: Vec<u8>, decoder: Vec<u8>) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        let networks = load_networks(coarse, base, decoder).await.map_err(err)?;
        let precision = networks.precision().to_owned();
        Ok(Self {
            networks: Rc::new(networks),
            precision,
            atlas: Arc::new(atlas::Atlas {
                width: 4,
                height: 2,
                extent: [40000000.0, 20000000.0],
                values: vec![0.0; 40],
            }),
            seed: 0,
        })
    }
    pub fn precision(&self) -> String {
        self.precision.clone()
    }
    pub fn set_world(
        &mut self,
        seed: String,
        xyz: Vec<f32>,
        fields: Vec<f32>,
        circumference: f64,
    ) -> Result<(), JsValue> {
        if !circumference.is_finite() || !(10000.0..=100000000.0).contains(&circumference) {
            return Err(err("Invalid planet size"));
        }
        self.atlas = Arc::new(
            atlas::rasterize(
                &xyz,
                &fields,
                1024,
                512,
                [circumference, circumference / 2.0],
            )
            .map_err(err)?,
        );
        self.seed = world_seed(&seed);
        self.networks.clear_cache();
        Ok(())
    }
    pub fn source(
        &self,
        x: f64,
        y: f64,
        step: f64,
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if width == 0
            || height == 0
            || width > 2048
            || height > 2048
            || !x.is_finite()
            || !y.is_finite()
            || !step.is_finite()
            || step <= 0.0
        {
            return Err(err("Invalid sampling rectangle"));
        }
        let mut fields = vec![0.0; width * height * 5];
        for iy in 0..height {
            for ix in 0..width {
                let v = self
                    .atlas
                    .sample(x + (ix as f64 + 0.5) * step, y + (iy as f64 + 0.5) * step);
                for c in 0..5 {
                    fields[c * width * height + iy * width + ix] = v[c];
                }
            }
        }
        Ok(fields)
    }

    /// Development knob: `bm,bn,bk,tm,tn` GEMM tile, or empty for the default.
    pub fn set_conv_tile(&self, spec: String) {
        let v: Vec<usize> = spec.split(',').filter_map(|x| x.parse().ok()).collect();
        self.networks
            .gpu()
            .set_tile((v.len() == 5).then(|| engine::kernels::Tile {
                bm: v[0],
                bn: v[1],
                bk: v[2],
                tm: v[3],
                tn: v[4],
            }));
    }

    /// Development statistics: `[GPU wait ms, dispatches, convolution GFLOP]`.
    pub fn stats(&self) -> Vec<f64> {
        use std::sync::atomic::Ordering::Relaxed;
        vec![
            gpu::WAIT_US.load(Relaxed) as f64 / 1000.0,
            gpu::DISPATCHES.load(Relaxed) as f64,
            gpu::CONV_FLOPS.load(Relaxed) as f64 * 1e-9,
        ]
    }

    /// Compiles the network pipelines (run while Orogen builds the world).
    pub fn warm(&self) -> js_sys::Promise {
        let networks = self.networks.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            networks.warm().await.map_err(err)?;
            Ok(JsValue::UNDEFINED)
        })
    }

    pub fn clear_neural_cache(&self) {
        self.networks.clear_cache();
    }
    pub fn tile(
        &self,
        lod: u8,
        tx: i32,
        ty: i32,
        height_noise: f32,
        climate_noise: Vec<f32>,
    ) -> js_sys::Promise {
        let networks = self.networks.clone();
        let atlas = self.atlas.clone();
        let seed = self.seed;
        wasm_bindgen_futures::future_to_promise(async move {
            if lod > 7
                || climate_noise.len() != 4
                || !height_noise.is_finite()
                || !(0.001..=2.0).contains(&height_noise)
                || climate_noise
                    .iter()
                    .any(|r| !r.is_finite() || !(0.001..=2.0).contains(r))
            {
                return Err(err("Invalid neural tile request"));
            }
            let request = tile_request(seed, atlas, lod, tx, ty, height_noise, &climate_noise);
            let output = networks
                .generate(&request, &Performance)
                .await
                .map_err(err)?;
            Ok(js_sys::Float32Array::from(output.heights.as_slice()).into())
        })
    }
}

/// Neural request of one 256² viewer tile with its 24-cell halo (304² samples).
pub fn tile_request(
    seed: u64,
    atlas: Arc<atlas::Atlas>,
    lod: u8,
    tx: i32,
    ty: i32,
    height_noise: f32,
    climate_noise: &[f32],
) -> Request {
    let stride = 256i64 * (1i64 << lod);
    let mut request = Request::new(seed, Sketch::Planet(atlas));
    let halo = 24i64 * (1i64 << lod);
    request.origin = Some(ty as i64 * stride - halo);
    request.origin_x = Some(tx as i64 * stride - halo);
    request.extent_m = (stride + 2 * halo) as f64 * 30.0;
    request.resolution = 304;
    request.exact = true;
    // Batched base windows fill the GPU at the 8×8 and 16×16 UNet levels.
    request.base_batch = 16;
    request.decoder_batch = 1;
    request.output_stage = lod;
    request.height_noise = height_noise;
    request.tuning.climate_noise.copy_from_slice(climate_noise);
    request
}

#[wasm_bindgen]
pub fn render_tile(
    heights: Vec<f32>,
    fields: Vec<f32>,
    width: usize,
    height: usize,
    step: f32,
    mode: String,
    settings: String,
) -> Result<Vec<u8>, JsValue> {
    render::raster(&heights, &fields, width, height, step, &mode, &settings).map_err(err)
}
