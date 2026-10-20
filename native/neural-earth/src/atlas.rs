//! Shared physical conventions of terrain_orogen._metres and terrain_conditioning.
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Atlas {
    pub width: usize,
    pub height: usize,
    pub extent: [f64; 2],
    // channel-major: elevation m, BIO1 °C, BIO4 °C×100, BIO12 mm, BIO15 %.
    pub values: Vec<f32>,
}

impl Atlas {
    pub fn sample(&self, x: f64, y: f64) -> [f32; 5] {
        let mut u = x / self.extent[0] + 0.5;
        let mut v = y / self.extent[1] + 0.5;
        // Reflect across either pole and rotate longitude by half a turn.
        v = v.rem_euclid(2.0);
        if v > 1.0 {
            v = 2.0 - v;
            u += 0.5;
        }
        let gx = u.rem_euclid(1.0) * self.width as f64 - 0.5;
        let gy = (v * self.height as f64 - 0.5).clamp(0.0, (self.height - 1) as f64);
        let xa = (gx.floor() as i64).rem_euclid(self.width as i64) as usize;
        let xb = (xa + 1) % self.width;
        let ya = gy.floor() as usize;
        let yb = (ya + 1).min(self.height - 1);
        let (tx, ty) = ((gx - gx.floor()) as f32, (gy - gy.floor()) as f32);
        let plane = self.width * self.height;
        std::array::from_fn(|c| {
            let get = |yy, xx| self.values[c * plane + yy * self.width + xx];
            let a = get(ya, xa) * (1.0 - tx) + get(ya, xb) * tx;
            let b = get(yb, xa) * (1.0 - tx) + get(yb, xb) * tx;
            a * (1.0 - ty) + b * ty
        })
    }

    pub fn conditioning(&self, y0: i64, x0: i64, n: usize) -> Vec<f32> {
        let mut out = vec![0.0; 5 * n * n];
        for y in 0..n {
            for x in 0..n {
                let fields = self.sample(
                    (x0 as f64 + x as f64 + 0.5) * 7680.0,
                    (y0 as f64 + y as f64 + 0.5) * 7680.0,
                );
                for c in 0..5 {
                    out[c * n * n + y * n + x] = if c == 0 {
                        fields[0].signum() * fields[0].abs().sqrt()
                    } else {
                        fields[c]
                    };
                }
            }
        }
        out
    }
}

struct Node {
    index: usize,
    axis: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}
impl Node {
    fn build(ids: &mut [usize], xyz: &[f32], depth: usize) -> Option<Box<Self>> {
        if ids.is_empty() {
            return None;
        }
        let axis = depth % 3;
        let mid = ids.len() / 2;
        ids.select_nth_unstable_by(mid, |&a, &b| {
            xyz[a * 3 + axis].total_cmp(&xyz[b * 3 + axis])
        });
        let (left, rest) = ids.split_at_mut(mid);
        Some(Box::new(Self {
            index: rest[0],
            axis,
            left: Self::build(left, xyz, depth + 1),
            right: Self::build(&mut rest[1..], xyz, depth + 1),
        }))
    }
    fn nearest(&self, point: &[f32; 3], xyz: &[f32], best: &mut [(f32, usize); 4]) {
        let delta = point[self.axis] - xyz[self.index * 3 + self.axis];
        let distance = (0..3)
            .map(|a| (point[a] - xyz[self.index * 3 + a]).powi(2))
            .sum::<f32>();
        if distance < best[3].0 {
            best[3] = (distance, self.index);
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        let (near, far) = if delta < 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };
        if let Some(node) = near {
            node.nearest(point, xyz, best);
        }
        if delta * delta < best[3].0
            && let Some(node) = far
        {
            node.nearest(point, xyz, best);
        }
    }
}

pub fn rasterize(
    xyz: &[f32],
    fields: &[f32],
    width: usize,
    height: usize,
    extent: [f64; 2],
) -> Result<Atlas, String> {
    let count = xyz.len() / 3;
    if count < 4
        || xyz.len() != count * 3
        || fields.len() != count * 5
        || !xyz.iter().chain(fields).all(|v| v.is_finite())
    {
        return Err("Invalid Orogen fields".into());
    }
    let tree = Node::build(&mut (0..count).collect::<Vec<_>>(), xyz, 0).unwrap();
    let mut values = vec![0.0; width * height * 5];
    for y in 0..height {
        for x in 0..width {
            let lat = PI / 2.0 - (y as f64 + 0.5) * PI / height as f64;
            let lon = (x as f64 + 0.5) * 2.0 * PI / width as f64 - PI;
            let point = [
                (lat.cos() * lon.cos()) as f32,
                lat.sin() as f32,
                (lat.cos() * lon.sin()) as f32,
            ];
            let mut best = [(f32::INFINITY, 0); 4];
            tree.nearest(&point, xyz, &mut best);
            let weights = best.map(|(d, _)| 1.0 / d.max(1e-12));
            let total: f32 = weights.iter().sum();
            for c in 0..5 {
                values[c * width * height + y * width + x] = (0..4)
                    .map(|i| weights[i] * fields[c * count + best[i].1])
                    .sum::<f32>()
                    / total;
            }
        }
    }
    Ok(Atlas {
        width,
        height,
        extent,
        values,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sample_field(
    values: &[f32],
    channels: usize,
    ystart: i64,
    xstart: i64,
    n: usize,
    channel: usize,
    stride: i64,
    oy: i64,
    ox: i64,
    extent: f64,
    resolution: usize,
    scale: f32,
    bias: f32,
) -> Vec<f32> {
    let get = |y: usize, x: usize| {
        values[channel * n * n + y * n + x] / values[channels * n * n + y * n + x].max(1e-12)
    };
    let mut out = Vec::with_capacity(resolution * resolution);
    for y in 0..resolution {
        for x in 0..resolution {
            let coord = |o: i64, s: i64, i: usize| {
                ((o as f64 + (i as f64 + 0.5) * extent / resolution as f64 / 30.0) / stride as f64
                    - 0.5
                    - s as f64)
                    .clamp(0.0, (n - 1) as f64)
            };
            let (gy, gx) = (coord(oy, ystart, y), coord(ox, xstart, x));
            let (ya, xa) = (gy.floor() as usize, gx.floor() as usize);
            let (yb, xb) = ((ya + 1).min(n - 1), (xa + 1).min(n - 1));
            let (ty, tx) = ((gy - ya as f64) as f32, (gx - xa as f64) as f32);
            let root = ((get(ya, xa) * (1.0 - tx) + get(ya, xb) * tx) * (1.0 - ty)
                + (get(yb, xa) * (1.0 - tx) + get(yb, xb) * tx) * ty)
                * scale
                + bias;
            out.push(root.signum() * root * root);
        }
    }
    out
}
