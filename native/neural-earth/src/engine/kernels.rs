//! WGSL kernels. Storage is `T` (f16 with `shader-f16`, f32 otherwise);
//! arithmetic and accumulation are FP32.

pub fn header(half: bool) -> String {
    if half {
        "enable f16;\nalias T = f16;\n".into()
    } else {
        "alias T = f32;\n".into()
    }
}

/// Flat index of a 2D-dispatched 1D kernel.
const INDEX: &str = "fn flat(wg: vec3<u32>, n: vec3<u32>, lid: u32, size: u32) -> u32 {
    return (wg.y * n.x + wg.x) * size + lid;
}
";

fn elementwise(params: &str, bindings: &str, size: u32, body: &str) -> String {
    format!(
        "struct P {{ count: u32, {params} }}
@group(0) @binding(0) var<uniform> p: P;
{bindings}
{INDEX}
@compute @workgroup_size({size})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {{
    let i = flat(wg, nwg, lid, {size}u);
    if (i >= p.count) {{ return; }}
{body}
}}
"
    )
}

pub fn fill() -> String {
    elementwise(
        "value: f32",
        "@group(0) @binding(1) var<storage, read_write> o: array<f32>;",
        256,
        "    o[i] = p.value;",
    )
}

pub fn axpby() -> String {
    elementwise(
        "two: u32, a: f32, b: f32",
        "@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;",
        256,
        "    var v = x[i] * p.a;
    if (p.two != 0u) { v = v + y[i] * p.b; }
    o[i] = v;",
    )
}

pub fn upsample() -> String {
    elementwise(
        "h: u32, w: u32, f: u32",
        "@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;",
        256,
        "    let ow = p.w * p.f;
    let plane = p.h * p.f * ow;
    let nc = i / plane;
    let r = i % plane;
    o[i] = x[(nc * p.h + (r / ow) / p.f) * p.w + (r % ow) / p.f];",
    )
}

pub fn to_act() -> String {
    elementwise(
        "plane: u32, c: u32, ones: u32, c4: u32",
        "@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<vec4<T>>;",
        256,
        "    let q = i % p.c4;
    let pix = i / p.c4;
    let n = pix / p.plane;
    let r = pix % p.plane;
    var v = vec4<f32>(0.0);
    for (var j = 0u; j < 4u; j++) {
        let c = q * 4u + j;
        if (c < p.c) { v[j] = x[(n * p.c + c) * p.plane + r]; }
        else if (c < p.ones) { v[j] = 1.0; }
    }
    o[i] = vec4<T>(v);",
    )
}

pub fn from_act() -> String {
    elementwise(
        "plane: u32, c: u32, stride: u32",
        "@group(0) @binding(1) var<storage, read> x: array<T>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;",
        256,
        "    let r = i % p.plane;
    let nc = i / p.plane;
    let n = nc / p.c;
    let c = nc % p.c;
    o[i] = f32(x[(n * p.plane + r) * p.stride + c]);",
    )
}

pub fn resample() -> String {
    elementwise(
        "h: u32, w: u32, ih: u32, iw: u32, c4: u32, mul: u32, div: u32",
        "@group(0) @binding(1) var<storage, read> x: array<vec4<T>>;
@group(0) @binding(2) var<storage, read_write> o: array<vec4<T>>;",
        256,
        "    let q = i % p.c4;
    let pix = i / p.c4;
    let plane = p.h * p.w;
    let n = pix / plane;
    let r = pix % plane;
    let y = (r / p.w) * p.mul / p.div;
    let xx = (r % p.w) * p.mul / p.div;
    o[i] = x[((n * p.ih + y) * p.iw + xx) * p.c4 + q];",
    )
}

pub fn pixel_scale(edm: bool) -> String {
    let finish = if edm {
        "1.0 / (1e-4 + sqrt(mean))"
    } else {
        "inverseSqrt(mean + p.eps)"
    };
    elementwise(
        "h: u32, w: u32, a4: u32, b4: u32, adiv: u32, eps: f32",
        "@group(0) @binding(1) var<storage, read> a: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<T>>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;",
        64,
        &format!(
            "    let plane = p.h * p.w;
    let n = i / plane;
    let r = i % plane;
    let ah = p.h / p.adiv;
    let aw = p.w / p.adiv;
    let ai = ((n * ah + (r / p.w) / p.adiv) * aw + (r % p.w) / p.adiv) * p.a4;
    var s = 0.0;
    for (var c = 0u; c < p.a4; c++) {{ let v = vec4<f32>(a[ai + c]); s += dot(v, v); }}
    for (var c = 0u; c < p.b4; c++) {{ let v = vec4<f32>(b[i * p.b4 + c]); s += dot(v, v); }}
    let mean = s / f32(4u * (p.a4 + p.b4));
    o[i] = {finish};"
        ),
    )
}

pub fn scale_pixels() -> String {
    elementwise(
        "c4: u32",
        "@group(0) @binding(1) var<storage, read> x: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4<T>>;",
        256,
        "    o[i] = vec4<T>(vec4<f32>(x[i]) * s[i / p.c4]);",
    )
}

pub fn student_head() -> String {
    elementwise(
        "plane: u32, c: u32, cin: u32",
        "@group(0) @binding(1) var<storage, read> x: array<T>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> wt: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;",
        256,
        // Weights: head [c], head bias, direct [cin].
        "    let n = i / p.plane;
    let r = i % p.plane;
    var s = wt[p.c];
    for (var c = 0u; c < p.c; c++) { s += f32(x[i * p.c + c]) * wt[c]; }
    var d = 0.0;
    for (var c = 0u; c < p.cin; c++) { d += input[(n * p.cin + c) * p.plane + r] * wt[p.c + 1u + c]; }
    o[i] = s + d;",
    )
}

/// One workgroup per (block, sample): `c = W·emb + 1`, then `c / sqrt(mean(c²) + 1e-8)`.
pub fn modulation() -> String {
    "struct P { dim: u32, total: u32, blocks: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> table: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let block = wg.x;
    let n = wg.y;
    let offset = table[block * 4u];
    let cout = table[block * 4u + 1u];
    let at = n * p.total + table[block * 4u + 2u];
    var s = 0.0;
    for (var c = lid; c < cout; c += 256u) {
        var v = 1.0;
        for (var e = 0u; e < p.dim; e++) { v += emb[n * p.dim + e] * wt[offset + e * cout + c]; }
        o[at + c] = v;
        s += v * v;
    }
    partial[lid] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { partial[lid] += partial[lid + k]; }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(partial[0] / f32(cout) + 1e-8);
    for (var c = lid; c < cout; c += 256u) { o[at + c] *= scale; }
}
"
    .into()
}

/// Normalizes each (sample, head, q/k/v, token) vector: `y / (|y| / sqrt(ch) + 1e-4)`.
/// QKV channel `(head · ch + c) · 3 + i`; output `[n, head, i, token, c]` FP32.
pub fn attention_norm() -> String {
    elementwise(
        "tokens: u32, heads: u32, ch: u32, stride: u32",
        "@group(0) @binding(1) var<storage, read> x: array<T>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;",
        64,
        "    let token = i % p.tokens;
    let qi = (i / p.tokens) % 3u;
    let head = (i / (3u * p.tokens)) % p.heads;
    let n = i / (3u * p.tokens * p.heads);
    let base = (n * p.tokens + token) * p.stride;
    var s = 0.0;
    for (var c = 0u; c < p.ch; c++) { let v = f32(x[base + (head * p.ch + c) * 3u + qi]); s += v * v; }
    let scale = 1.0 / (sqrt(s) / sqrt(f32(p.ch)) + 1e-4);
    for (var c = 0u; c < p.ch; c++) { o[i * p.ch + c] = f32(x[base + (head * p.ch + c) * 3u + qi]) * scale; }",
    )
}

/// One thread per (sample, head, query): softmax(q·k / sqrt(ch)) · v.
pub fn attention() -> String {
    elementwise(
        "tokens: u32, heads: u32, ch: u32, c4: u32",
        "@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<T>;",
        64,
        "    let tq = i % p.tokens;
    let head = (i / p.tokens) % p.heads;
    let n = i / (p.tokens * p.heads);
    let base = (n * p.heads + head) * 3u * p.tokens * p.ch;
    let q = base + tq * p.ch;
    let k = base + p.tokens * p.ch;
    let v = base + 2u * p.tokens * p.ch;
    let scale = 1.0 / sqrt(f32(p.ch));
    var w: array<f32, 64>;
    var top = -3.0e38;
    for (var t = 0u; t < p.tokens; t++) {
        var s = 0.0;
        for (var c = 0u; c < p.ch; c++) { s += y[q + c] * y[k + t * p.ch + c]; }
        w[t] = s * scale;
        top = max(top, w[t]);
    }
    var total = 0.0;
    for (var t = 0u; t < p.tokens; t++) { w[t] = exp(w[t] - top); total += w[t]; }
    let out = (n * p.tokens + tq) * p.c4 * 4u + head * p.ch;
    for (var c = 0u; c < p.ch; c++) {
        var s = 0.0;
        for (var t = 0u; t < p.tokens; t++) { s += w[t] * y[v + t * p.ch + c]; }
        o[out + c] = T(s / total);
    }",
    )
}

/// Workgroup tile of the convolution GEMM: `bm` output pixels × `bn` output
/// channels, K in chunks of `bk` input channels of one tap, `tm × tn` per thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub tm: usize,
    pub tn: usize,
}

impl Tile {
    pub fn threads(&self) -> usize {
        (self.bm / self.tm) * (self.bn / self.tn)
    }
    pub fn shared_bytes(&self) -> usize {
        self.bk * (self.bm + self.bn) * 4
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConvFlags {
    pub concat: bool,
    pub residual: bool,
    pub modulation: bool,
    pub pixel: bool,
    pub silu_in: bool,
    pub silu_out: bool,
    pub bias: bool,
    pub clamp: bool,
}

/// Implicit-GEMM convolution. The next K-chunk is fetched into registers while
/// the current one is multiplied from workgroup memory. Per-thread arrays are
/// emitted as scalars with constant indices so that they stay in registers.
pub fn conv(f: &ConvFlags, t: &Tile) -> String {
    let b = |v: bool| if v { "true" } else { "false" };
    let threads = t.threads();
    let la = t.bm * t.bk / 4 / threads;
    let lb = t.bk * t.bn / 4 / threads;
    assert!(
        la >= 1 && lb >= 1 && la * threads * 4 == t.bm * t.bk && lb * threads * 4 == t.bk * t.bn
    );
    assert!(
        t.tm.is_multiple_of(4)
            && t.tn.is_multiple_of(4)
            && t.bk.is_multiple_of(4)
            && 16usize.is_multiple_of(t.bk)
    );
    let (tm4, tn4) = (t.tm / 4, t.tn / 4);
    let each = |n: usize, body: &str| -> String {
        (0..n).map(|j| body.replace("$J", &j.to_string())).collect()
    };
    let mut fma = String::new();
    for i in 0..tm4 {
        fma += &format!("            let a{i} = As[k * $BM4 + ty * $TM4 + {i}u];\n");
    }
    for j in 0..tn4 {
        fma += &format!("            let b{j} = Bs[k * $BN4 + tx * $TN4 + {j}u];\n");
    }
    for i in 0..t.tm {
        for j in 0..tn4 {
            let lane = ["x", "y", "z", "w"][i % 4];
            fma += &format!("            acc{i}_{j} += a{}.{lane} * b{j};\n", i / 4);
        }
    }
    let mut epilogue = String::new();
    for i in 0..t.tm {
        epilogue +=
            &format!("    {{\n        let m = m0 + ty * $TM + {i}u;\n        if (m < p.m) {{\n");
        for j in 0..tn4 {
            epilogue +=
                &format!("            store(m, wg.y * $BN4 + tx * $TN4 + {j}u, acc{i}_{j});\n");
        }
        epilogue += "        }\n    }\n";
    }
    let declare: String = (0..t.tm)
        .flat_map(|i| (0..tn4).map(move |j| format!("    var acc{i}_{j} = vec4<f32>(0.0);\n")))
        .collect();
    CONV_BODY
        .replace("$ROWS", &each(la, CONV_ROW))
        .replace("$LOAD", &(each(la, CONV_LOAD_A) + &each(lb, CONV_LOAD_B)))
        .replace(
            "$STORE",
            &(each(la, CONV_STORE_A) + &each(lb, "        Bs[tid + $Ju * $THREADS] = rb$J;\n")),
        )
        .replace(
            "$REGS",
            &(each(la, "    var ra$J = vec4<f32>(0.0);\n")
                + &each(lb, "    var rb$J = vec4<f32>(0.0);\n")),
        )
        .replace("$ACC", &declare)
        .replace("$FMA", &fma)
        .replace("$EPILOGUE", &epilogue)
        .replace("$CONCAT", b(f.concat))
        .replace("$RESIDUAL", b(f.residual))
        .replace("$MODULATION", b(f.modulation))
        .replace("$PIXEL", b(f.pixel))
        .replace("$SILU_IN", b(f.silu_in))
        .replace("$SILU_OUT", b(f.silu_out))
        .replace("$BIAS", b(f.bias))
        .replace("$CLAMP", b(f.clamp))
        .replace("$THREADS", &format!("{threads}u"))
        .replace("$BM4", &format!("{}u", t.bm / 4))
        .replace("$BN4", &format!("{}u", t.bn / 4))
        .replace("$BK4", &format!("{}u", t.bk / 4))
        .replace("$TM4", &format!("{tm4}u"))
        .replace("$TN4", &format!("{tn4}u"))
        .replace("$TM", &format!("{}u", t.tm))
        .replace("$ASHARED", &(t.bk * t.bm / 4).to_string())
        .replace("$BSHARED", &(t.bk * t.bn / 4).to_string())
        .replace("$BM", &format!("{}u", t.bm))
        .replace("$BK", &format!("{}u", t.bk))
        .replace("$WG", &threads.to_string())
}

/// Output-pixel coordinates of A row `$J` of this thread.
const CONV_ROW: &str = "
    let row$J = m0 + (tid + $Ju * $THREADS) / $BK4;
    let ok$J = row$J < p.m;
    let mrow$J = min(row$J, p.m - 1u);
    let n$J = mrow$J / hw;
    let y$J = i32(((mrow$J % hw) / p.ow) * p.stride) - i32(p.pad);
    let x$J = i32(((mrow$J % hw) % p.ow) * p.stride) - i32(p.pad);
    let q$J = ((tid + $Ju * $THREADS) % $BK4) * 4u;
";

/// Fetches A row `$J` of K-chunk `ls` (tap `ky, kx`, channel base `cc`).
const CONV_LOAD_A: &str = "
            {
                let ci = cc * $BK + q$J;
                let yy = y$J + ky;
                let xx = x$J + kx;
                var v = vec4<f32>(0.0);
                if (ok$J && yy >= 0 && yy < i32(p.ih) && xx >= 0 && xx < i32(p.iw)) {
                    let y = u32(yy);
                    let x = u32(xx);
                    if (CONCAT && ci >= p.ca) {
                        v = vec4<f32>(xb[((n$J * bh + (y >> p.bshift)) * bw + (x >> p.bshift)) * cb4 + (ci - p.ca) / 4u]) * p.bscale;
                    } else {
                        v = vec4<f32>(xa[((n$J * ah + (y >> p.ashift)) * aw + (x >> p.ashift)) * ca4 + ci / 4u]) * p.ascale;
                    }
                    if (PIXEL) { v = v * aux[(n$J * p.ih + y) * p.iw + x]; }
                    if (SILU_IN) { v = silu(v) * p.pgain; }
                }
                ra$J = v;
            }
";

const CONV_LOAD_B: &str = "
            {
                let idx = tid + $Ju * $THREADS;
                let c4 = wg.y * $BN4 + idx % $BN4;
                let krow = tap * cin + cc * $BK + idx / $BN4;
                var w = vec4<f32>(0.0);
                if (c4 < p.cout4) { w = vec4<f32>(wt[krow * p.cout4 + c4]); }
                rb$J = w;
            }
";

/// Transposes A row `$J` into `As[k][m / 4]`.
const CONV_STORE_A: &str = "
        {
            let idx = tid + $Ju * $THREADS;
            let r = idx / $BK4;
            let at = ((idx % $BK4) * 4u) * $BM4 + (r >> 2u);
            let lane = r & 3u;
            As[at][lane] = ra$J.x;
            As[at + $BM4][lane] = ra$J.y;
            As[at + 2u * $BM4][lane] = ra$J.z;
            As[at + 3u * $BM4][lane] = ra$J.w;
        }
";

const CONV_BODY: &str = "
const CONCAT: bool = $CONCAT;
const RESIDUAL: bool = $RESIDUAL;
const MODULATION: bool = $MODULATION;
const PIXEL: bool = $PIXEL;
const SILU_IN: bool = $SILU_IN;
const SILU_OUT: bool = $SILU_OUT;
const BIAS: bool = $BIAS;
const CLAMP: bool = $CLAMP;
struct P {
    m: u32, cout4: u32, ca: u32, cb: u32,
    ih: u32, iw: u32, oh: u32, ow: u32,
    kw: u32, taps: u32, stride: u32, pad: u32,
    dil: u32, ashift: u32, bshift: u32, cmod: u32,
    moff: u32, krows: u32, r0: u32, r1: u32,
    ascale: f32, bscale: f32, pgain: f32, egain: f32,
    wa: f32, wb: f32, lim: f32, r2: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> xa: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> xb: array<vec4<T>>;
@group(0) @binding(3) var<storage, read> wt: array<vec4<T>>;
@group(0) @binding(4) var<storage, read> aux: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<vec4<T>>;

var<workgroup> As: array<vec4<f32>, $ASHARED>;
var<workgroup> Bs: array<vec4<f32>, $BSHARED>;

fn silu(v: vec4<f32>) -> vec4<f32> { return v / (vec4<f32>(1.0) + exp(-v)); }

fn store(m: u32, c4: u32, acc: vec4<f32>) {
    if (c4 >= p.cout4) { return; }
    var v = acc;
    if (BIAS) { v += vec4<f32>(wt[p.krows * p.cout4 + c4]); }
    if (MODULATION) {
        let o = (m / (p.oh * p.ow)) * p.cmod + p.moff + c4 * 4u;
        v = v * vec4<f32>(aux[o], aux[o + 1u], aux[o + 2u], aux[o + 3u]);
    }
    if (SILU_OUT) { v = silu(v) * p.egain; }
    if (RESIDUAL) { v = vec4<f32>(xb[m * p.cout4 + c4]) * p.wa + v * p.wb; }
    if (CLAMP) { v = clamp(v, vec4<f32>(-p.lim), vec4<f32>(p.lim)); }
    outp[m * p.cout4 + c4] = vec4<T>(v);
}

@compute @workgroup_size($WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let m0 = wg.x * $BM;
    let hw = p.oh * p.ow;
    let cin = p.ca + p.cb;
    let chunks = cin / $BK;
    let total = p.taps * chunks;
    let ca4 = p.ca / 4u;
    let cb4 = p.cb / 4u;
    let ah = p.ih >> p.ashift;
    let aw = p.iw >> p.ashift;
    let bh = p.ih >> p.bshift;
    let bw = p.iw >> p.bshift;
$ROWS
    let tx = tid % ($BN4 / $TN4);
    let ty = tid / ($BN4 / $TN4);
$ACC
$REGS
    var tap = 0u;
    var cc = 0u;
    var ky = 0;
    var kx = 0;
$LOAD
    for (var s = 0u; s < total; s++) {
$STORE
        workgroupBarrier();
        cc++;
        if (cc == chunks) {
            cc = 0u;
            tap++;
            ky = i32((tap / p.kw) * p.dil);
            kx = i32((tap % p.kw) * p.dil);
        }
        if (s + 1u < total) {
$LOAD
        }
        for (var k = 0u; k < $BK; k++) {
$FMA        }
        workgroupBarrier();
    }
$EPILOGUE}
";
