#[path = "../../neural-earth/src/atlas.rs"]
#[allow(dead_code)]
mod atlas;
#[path = "../../neural-earth/src/render.rs"]
mod render;

#[test]
fn atlas_wraps_longitude_and_reflects_at_poles() {
    let atlas = atlas::Atlas {
        width: 4,
        height: 2,
        extent: [400.0, 200.0],
        values: (0..40).map(|i| i as f32).collect(),
    };
    assert_eq!(atlas.sample(-190.0, 0.0), atlas.sample(210.0, 0.0));
    assert_eq!(atlas.sample(0.0, -110.0), atlas.sample(200.0, -90.0));
    assert_eq!(atlas.sample(0.0, 110.0), atlas.sample(200.0, 90.0));
}

#[test]
fn coarse_signed_root_and_physical_climate_keep_their_units() {
    let mut values = vec![0.0; 40];
    values[..8].fill(-100.0);
    values[8..16].fill(25.0);
    values[16..24].fill(600.0);
    values[24..32].fill(1800.0);
    values[32..40].fill(35.0);
    let atlas = atlas::Atlas {
        width: 4,
        height: 2,
        extent: [40000000.0, 20000000.0],
        values,
    };
    assert_eq!(
        atlas.conditioning(-7, 9, 1),
        vec![-10.0, 25.0, 600.0, 1800.0, 35.0]
    );
}

#[test]
fn raster_is_finite_repeatable_and_recoloring_does_not_change_dem() {
    let n = 16;
    let mut heights = vec![500.0; n * n];
    heights[..n].fill(-100.0);
    let original = heights.clone();
    let mut fields = vec![0.0; 5 * n * n];
    fields[n * n..2 * n * n].fill(18.0);
    fields[2 * n * n..3 * n * n].fill(600.0);
    fields[3 * n * n..4 * n * n].fill(1000.0);
    let a = render::raster(&heights, &fields, n, n, 30.0, "render", "{}").unwrap();
    let b = render::raster(&heights, &fields, n, n, 30.0, "render", "{}").unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), n * n * 4);
    assert!(a.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
    assert_ne!(
        a,
        render::raster(&heights, &fields, n, n, 30.0, "temperature", "{}").unwrap()
    );
    assert_eq!(heights, original);
}
