//! Native timing of browser tiles on the same engine (wgpu native backend).
//! cargo run --release --example bench -- [lod tx ty]... [--dump out.bin]
use neural_earth_wasm::{Clock, atlas::Atlas, tile_request};
use std::{sync::Arc, time::Instant};

struct Wall(Instant);
impl Clock for Wall {
    fn now_ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}

fn synthetic_atlas(circumference: f64) -> Atlas {
    let (width, height) = (1024, 512);
    let plane = width * height;
    let mut values = vec![0.0f32; plane * 5];
    for y in 0..height {
        for x in 0..width {
            let (u, v) = (x as f32 / width as f32, y as f32 / height as f32);
            let ridge = (u * 37.0).sin() * (v * 23.0).cos() + 0.5 * (u * 91.0 + v * 53.0).sin();
            let at = y * width + x;
            values[at] = 1200.0 + 1800.0 * ridge;
            values[plane + at] = 12.0 - 25.0 * (v - 0.5).abs();
            values[2 * plane + at] = 700.0;
            values[3 * plane + at] = 900.0 + 300.0 * ridge;
            values[4 * plane + at] = 55.0;
        }
    }
    Atlas {
        width,
        height,
        extent: [circumference, circumference / 2.0],
        values,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dump = args
        .iter()
        .position(|a| a == "--dump")
        .map(|i| args[i + 1].clone());
    let numbers: Vec<i32> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let tiles: Vec<(u8, i32, i32)> = if numbers.is_empty() {
        vec![(4, 100, -40), (0, 1600, -640)]
    } else {
        numbers
            .chunks(3)
            .map(|c| (c[0] as u8, c[1], c[2]))
            .collect()
    };
    let models =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../wasm/dist/runtime/models");
    let read = |name: &str| std::fs::read(models.join(name)).expect(name);
    let clock = Wall(Instant::now());
    let networks = pollster::block_on(neural_earth_wasm::load_networks(
        read("coarse.safetensors"),
        read("base.nf4"),
        read("decoder.safetensors"),
    ))
    .unwrap();
    println!("load {:.0} ms", clock.now_ms());
    if std::env::var("NEURAL_EARTH_WARM").is_ok() {
        let start = clock.now_ms();
        pollster::block_on(networks.warm()).unwrap();
        println!("warm {:.0} ms", clock.now_ms() - start);
    }
    if std::env::var("NEURAL_EARTH_PROFILE").is_ok() {
        *neural_earth_wasm::gpu::PROFILE.lock().unwrap() = Some(Default::default());
    }
    let atlas = Arc::new(synthetic_atlas(40_030_000.0));
    let mut dumped = Vec::new();
    for &(lod, tx, ty) in &tiles {
        for pass in 0..2 {
            networks.clear_cache();
            let mut request = tile_request(42, atlas.clone(), lod, tx, ty, 0.05, &[0.5; 4]);
            if let Ok(b) = std::env::var("BASE_BATCH") {
                request.base_batch = b.parse().unwrap();
            }
            let wait = neural_earth_wasm::gpu::WAIT_US.load(std::sync::atomic::Ordering::Relaxed);
            let flops =
                neural_earth_wasm::gpu::CONV_FLOPS.load(std::sync::atomic::Ordering::Relaxed);
            let start = clock.now_ms();
            let output = pollster::block_on(networks.generate(&request, &clock)).unwrap();
            let t = &output.timings;
            println!(
                "lod {lod} ({tx},{ty}) pass {pass}: {:.0} ms  [coarse {:.0} base {:.0} decoder {:.0} recon {:.0}]",
                clock.now_ms() - start,
                t.coarse_ms,
                t.base_ms,
                t.decoder_ms,
                t.reconstruction_ms
            );
            let flops = neural_earth_wasm::gpu::CONV_FLOPS
                .load(std::sync::atomic::Ordering::Relaxed)
                - flops;
            let wait =
                neural_earth_wasm::gpu::WAIT_US.load(std::sync::atomic::Ordering::Relaxed) - wait;
            println!("  GPU wait {:.0} ms", wait as f64 / 1000.0);
            println!(
                "  conv {:.2} TFLOP, {:.2} TFLOPS",
                flops as f64 * 1e-12,
                flops as f64 * 1e-9 / (clock.now_ms() - start)
            );
            if pass == 1 {
                let (lo, hi) = output
                    .heights
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &h| (a.min(h), b.max(h)));
                println!("  heights {lo:.1} .. {hi:.1}");
                dumped.extend(output.heights.iter().flat_map(|h| h.to_le_bytes()));
            }
        }
    }
    if let Some(profile) = neural_earth_wasm::gpu::PROFILE.lock().unwrap().take() {
        let mut rows: Vec<_> = profile.into_iter().collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let total: f64 = rows.iter().map(|r| r.1.1).sum();
        println!("profile total {total:.0} ms");
        for (key, (count, ms, flops)) in rows.iter().take(45) {
            println!(
                "{ms:8.1} ms {count:5}x {:6.2} TFLOPS  {}",
                *flops as f64 * 1e-9 / ms,
                &key[..key.len().min(120)]
            );
        }
    }
    if let Some(path) = dump {
        std::fs::write(path, dumped).unwrap();
    }
}
