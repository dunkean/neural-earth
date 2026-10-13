// Exercise the exact generated Rust reconstruction without a GPU. Build's
// prepare-engine step supplies the same modules used by the WASM inference.
#[allow(dead_code)]
#[path = "../../neural-earth/generated-engine/plan.rs"]
mod plan;
#[allow(dead_code)]
#[path = "../../neural-earth/generated-engine/reconstruct.rs"]
mod reconstruct;

fn tile(x: i64, y: i64) -> Vec<f32> {
    let py = plan::Plan::new(y, 304, true);
    let px = plan::Plan::new(x, 304, true);
    let n = py.residual.len();
    assert_eq!(n, px.residual.len());
    let field = |x: i64, y: i64| ((x as f64 * 0.051).sin() * (y as f64 * 0.073).cos()) as f32;
    let residual = reconstruct::Grid {
        n,
        v: (0..n * n)
            .map(|i| {
                field(
                    px.residual.start + (i % n) as i64,
                    py.residual.start + (i / n) as i64,
                )
            })
            .collect(),
    };
    let m = n / 8;
    let lowfreq = reconstruct::Grid {
        n: m,
        v: (0..m * m)
            .map(|i| {
                2.0 + field(
                    px.residual.start / 8 + (i % m) as i64,
                    py.residual.start / 8 + (i / m) as i64,
                )
            })
            .collect(),
    };
    reconstruct::elevation(
        &residual,
        &lowfreq,
        (
            (y - py.residual.start) as usize,
            (x - px.residual.start) as usize,
        ),
        304,
    )
}

#[test]
fn adjacent_crop_halos_include_the_entire_laplacian_filter_support() {
    let (x, y) = (1188 * 256 - 24, -1076 * 256 - 24);
    let first = tile(x, y);
    let east = tile(x + 256, y);
    let south = tile(x, y + 256);
    for row in 0..304 {
        for overlap in 0..48 {
            assert_eq!(first[row * 304 + 256 + overlap], east[row * 304 + overlap]);
            assert_eq!(
                first[(256 + overlap) * 304 + row],
                south[overlap * 304 + row]
            );
        }
    }
}
