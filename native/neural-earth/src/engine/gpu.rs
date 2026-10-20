//! Direct WebGPU runtime of the three terrain networks.
//!
//! Network activations are NHWC (channels innermost, padded to 16) in FP16 when
//! the device has `shader-f16`, FP32 otherwise. Every kernel accumulates in FP32.
//! Pipeline-level tensors (solver states, network inputs and outputs) are FP32
//! NCHW. Work is recorded and submitted once per readback, so a whole forward
//! pass is a single queue submission.
use crate::engine::kernels;
use crate::weights::{Entry, Values};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Channel padding of every activation: one GEMM K-chunk.
pub const CHANNEL_ALIGN: usize = 16;
const UNIFORM_SLOT: u64 = 256;
const UNIFORM_SLOTS: u64 = 4096;

/// Convolution multiply-adds × 2 recorded so far (benchmarks).
pub static CONV_FLOPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Wall time spent waiting for readbacks, in µs (benchmarks).
pub static WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-kernel GPU time with `NEURAL_EARTH_PROFILE` (native benchmarks; serializes work).
#[cfg(not(target_arch = "wasm32"))]
pub static PROFILE: std::sync::Mutex<Option<HashMap<String, ProfileEntry>>> =
    std::sync::Mutex::new(None);
/// Dispatches, milliseconds and convolution FLOPs of one profiled kernel.
pub type ProfileEntry = (usize, f64, u64);
#[cfg(not(target_arch = "wasm32"))]
static PROFILE_TAG: std::sync::Mutex<Option<(String, u64)>> = std::sync::Mutex::new(None);

/// Recorded dispatches (benchmarks).
pub static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Milliseconds on a monotonic clock.
pub fn clock_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    return crate::now();
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.0
    }
}

pub fn align(c: usize) -> usize {
    c.div_ceil(CHANNEL_ALIGN) * CHANNEL_ALIGN
}

pub struct Kernel {
    label: String,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

enum Command {
    Dispatch {
        kernel: Rc<Kernel>,
        bind: wgpu::BindGroup,
        groups: [u32; 3],
    },
    Copy {
        src: wgpu::Buffer,
        src_offset: u64,
        dst: wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    },
}

#[derive(Default)]
struct Recorder {
    commands: Vec<Command>,
    params: Vec<u8>,
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    half: bool,
    kernels: RefCell<HashMap<String, Rc<Kernel>>>,
    recorder: RefCell<Recorder>,
    pool: RefCell<HashMap<u64, Vec<wgpu::Buffer>>>,
    uniform: wgpu::Buffer,
    dummy: wgpu::Buffer,
    pub tile_override: RefCell<Option<kernels::Tile>>,
}

#[derive(Clone)]
pub struct Gpu(Rc<Inner>);

/// A pooled storage buffer, returned to the pool when the last handle drops.
/// Pending commands keep their own handles; queue order makes reuse safe.
pub struct Buf {
    pub buffer: wgpu::Buffer,
    gpu: Gpu,
}

impl Drop for Buf {
    fn drop(&mut self) {
        let size = self.buffer.size();
        self.gpu
            .0
            .pool
            .borrow_mut()
            .entry(size)
            .or_default()
            .push(self.buffer.clone());
    }
}

/// FP32 NCHW tensor of the denoising solvers.
#[derive(Clone)]
pub struct Tensor<const D: usize = 4> {
    gpu: Gpu,
    buf: Rc<Buf>,
    shape: [usize; 4],
}

/// Network activation: NHWC with `c` (aligned) channels in the compute precision.
#[derive(Clone)]
pub struct Act {
    pub buf: Rc<Buf>,
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Act {
    pub fn pixels(&self) -> usize {
        self.n * self.h * self.w
    }
}

/// Per-pixel FP32 values (pixel-norm scales).
pub struct Pixels {
    pub buf: Rc<Buf>,
}

/// Convolution weights as a `[taps · cin, cout]` GEMM matrix (+ an optional bias row).
pub struct Conv {
    pub buffer: wgpu::Buffer,
    pub cin: usize,
    pub cout: usize,
    pub cout_pad: usize,
    pub kh: usize,
    pub kw: usize,
    pub bias: bool,
}

struct Layout {
    cout: usize,
    cin: usize,
    taps: usize,
    cin_pad: usize,
    cout_pad: usize,
    rows: usize,
}

/// Optional second operand of a convolution: an implicit channel concatenation,
/// or the residual of the epilogue.
pub enum Second<'a> {
    None,
    Concat {
        act: &'a Act,
        scale: f32,
        div: usize,
    },
    Residual {
        act: &'a Act,
        wa: f32,
        wb: f32,
    },
}

pub enum Aux<'a> {
    None,
    /// Per-(sample, channel) multiplier at an offset of a modulation buffer.
    Modulation {
        buf: &'a Buf,
        stride: usize,
        offset: usize,
    },
    /// Per-input-pixel multiplier (pixel normalization of the input).
    Pixel(&'a Pixels),
}

pub struct ConvArgs<'a> {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    /// First source: scale and spatial divisor (2 reads it upsampled).
    pub scale: f32,
    pub div: usize,
    pub second: Second<'a>,
    pub aux: Aux<'a>,
    /// Prologue `silu(x) · gain`.
    pub silu_in: Option<f32>,
    /// Epilogue `silu(y) · gain`.
    pub silu_out: Option<f32>,
    pub clamp: Option<f32>,
}

impl Default for ConvArgs<'_> {
    fn default() -> Self {
        Self {
            stride: 1,
            padding: 0,
            dilation: 1,
            scale: 1.0,
            div: 1,
            second: Second::None,
            aux: Aux::None,
            silu_in: None,
            silu_out: None,
            clamp: None,
        }
    }
}

/// Uniform parameters: 4-byte words.
#[derive(Default)]
pub struct Params(Vec<u32>);
impl Params {
    pub fn u(mut self, v: usize) -> Self {
        self.0
            .push(u32::try_from(v).expect("GPU parameter overflow"));
        self
    }
    pub fn f(mut self, v: f32) -> Self {
        self.0.push(v.to_bits());
        self
    }
}

fn groups_1d(count: usize, size: usize) -> [u32; 3] {
    let groups = count.div_ceil(size).max(1);
    let x = groups.min(65535);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

impl Gpu {
    pub async fn new() -> Result<Self, String> {
        #[cfg(target_arch = "wasm32")]
        let backends = wgpu::Backends::BROWSER_WEBGPU;
        #[cfg(not(target_arch = "wasm32"))]
        let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let request = |power_preference| {
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
        };
        // Some browsers expose no adapter for an explicit high-performance request.
        let adapter = match request(wgpu::PowerPreference::HighPerformance).await {
            Ok(adapter) => adapter,
            Err(_) => request(wgpu::PowerPreference::None)
                .await
                .map_err(|e| format!("No WebGPU adapter: {e}"))?,
        };
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("NEURAL_EARTH_VERBOSE").is_ok() {
            eprintln!("{:?}", adapter.get_info());
        }
        let half = adapter.features().contains(wgpu::Features::SHADER_F16)
            && std::env::var("NEURAL_EARTH_FP32").is_err();
        let limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("neural-earth"),
                required_features: if half {
                    wgpu::Features::SHADER_F16
                } else {
                    wgpu::Features::empty()
                },
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("WebGPU device: {e}"))?;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: UNIFORM_SLOT * UNIFORM_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unused"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(Self(Rc::new(Inner {
            device,
            queue,
            half,
            kernels: Default::default(),
            recorder: Default::default(),
            pool: Default::default(),
            uniform,
            dummy,
            tile_override: Default::default(),
        })))
    }

    pub fn set_tile(&self, tile: Option<kernels::Tile>) {
        *self.0.tile_override.borrow_mut() = tile;
    }

    pub fn half(&self) -> bool {
        self.0.half
    }

    pub fn precision(&self) -> &'static str {
        if self.0.half { "fp16" } else { "fp32" }
    }

    /// Bytes of one activation element.
    fn element(&self) -> usize {
        if self.0.half { 2 } else { 4 }
    }

    /// Pipeline tensors are FP32 whatever the network precision.
    pub fn f16_options(&self) -> Gpu {
        self.clone()
    }

    pub fn f32_options(&self) -> Gpu {
        self.clone()
    }

    pub fn alloc(&self, bytes: usize) -> Rc<Buf> {
        let size = (bytes.max(16) as u64).next_multiple_of(256);
        let reused = self.0.pool.borrow_mut().get_mut(&size).and_then(Vec::pop);
        let buffer = reused.unwrap_or_else(|| {
            self.0.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Rc::new(Buf {
            buffer,
            gpu: self.clone(),
        })
    }

    fn upload_bytes(&self, bytes: &[u8]) -> wgpu::Buffer {
        let size = bytes.len().max(16).next_multiple_of(4);
        let buffer = self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer
                .slice(..)
                .get_mapped_range_mut()
                .expect("mapped at creation");
            if bytes.len() == size {
                view.copy_from_slice(bytes);
            } else {
                let mut padded = bytes.to_vec();
                padded.resize(size, 0);
                view.copy_from_slice(&padded);
            }
        }
        buffer.unmap();
        buffer
    }

    /// Constant FP32 storage.
    pub fn constant(&self, values: &[f32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_bytes(&bytes)
    }

    fn write(&self, buf: &Buf, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Ordered before the next submission, after every already submitted one.
        self.flush();
        self.0.queue.write_buffer(&buf.buffer, 0, &bytes);
    }

    /// Convolution `[cout, cin, kh, kw]` (+ bias) as a padded GEMM matrix.
    /// `cin_pad` lays out input channels of an aligned (possibly concatenated) activation.
    pub fn conv(
        &self,
        weight: &Entry,
        bias: Option<&Entry>,
        cin_pad: usize,
    ) -> Result<Conv, String> {
        let [cout, cin, kh, kw]: [usize; 4] = weight
            .shape
            .clone()
            .try_into()
            .map_err(|_| "4D convolution weight expected")?;
        if cin > cin_pad {
            return Err("Convolution input channels exceed their layout".into());
        }
        let cout_pad = align(cout);
        let rows = kh * kw * cin_pad + usize::from(bias.is_some());
        let bias = match bias {
            Some(b) if b.shape != [cout] => return Err("Bias length mismatch".into()),
            Some(b) => Some(b.to_f32()),
            None => None,
        };
        let shape = Layout {
            cout,
            cin,
            taps: kh * kw,
            cin_pad,
            cout_pad,
            rows,
        };
        let buffer = match (&weight.values, self.0.half) {
            (Values::F16(v), true) => self.layout(
                v,
                |x| x.to_le_bytes(),
                &shape,
                bias,
                |x| half::f16::from_f32(x).to_le_bytes(),
            ),
            (Values::F16(v), false) => self.layout(
                v,
                |x| x.to_f32().to_le_bytes(),
                &shape,
                bias,
                f32::to_le_bytes,
            ),
            (Values::F32(v), true) => self.layout(
                v,
                |x| half::f16::from_f32(*x).to_le_bytes(),
                &shape,
                bias,
                |x| half::f16::from_f32(x).to_le_bytes(),
            ),
            (Values::F32(v), false) => {
                self.layout(v, |x| x.to_le_bytes(), &shape, bias, f32::to_le_bytes)
            }
        };
        Ok(Conv {
            buffer,
            cin: cin_pad,
            cout,
            cout_pad,
            kh,
            kw,
            bias: shape.rows > shape.taps * shape.cin_pad,
        })
    }

    /// `[cout, cin, taps]` → `[taps · cin_pad (+ bias row), cout_pad]`, in blocks of
    /// 16 output channels so that both sides stay in cache.
    fn layout<S, const N: usize>(
        &self,
        src: &[S],
        element: impl Fn(&S) -> [u8; N],
        l: &Layout,
        bias: Option<Vec<f32>>,
        scalar: impl Fn(f32) -> [u8; N],
    ) -> wgpu::Buffer {
        let mut matrix = vec![[0u8; N]; (l.rows * l.cout_pad).max(16).next_multiple_of(2)];
        for ob in (0..l.cout).step_by(16) {
            let oe = (ob + 16).min(l.cout);
            for i in 0..l.cin {
                for t in 0..l.taps {
                    let row = (t * l.cin_pad + i) * l.cout_pad;
                    for o in ob..oe {
                        matrix[row + o] = element(&src[(o * l.cin + i) * l.taps + t]);
                    }
                }
            }
        }
        if let Some(bias) = bias {
            let row = l.taps * l.cin_pad * l.cout_pad;
            for (o, b) in bias.into_iter().enumerate() {
                matrix[row + o] = scalar(b);
            }
        }
        self.upload_bytes(matrix.as_flattened())
    }

    pub fn kernel(&self, key: &str, source: impl FnOnce() -> String) -> Rc<Kernel> {
        if let Some(kernel) = self.0.kernels.borrow().get(key) {
            return kernel.clone();
        }
        let source = kernels::header(self.0.half) + &source();
        let device = &self.0.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let kernel = Rc::new(Kernel {
            label: key.to_owned(),
            pipeline,
            layout,
        });
        self.0
            .kernels
            .borrow_mut()
            .insert(key.to_owned(), kernel.clone());
        kernel
    }

    /// Records one dispatch. `inputs` bind 1..; the output binds last.
    pub fn dispatch(
        &self,
        kernel: &Rc<Kernel>,
        params: Params,
        inputs: &[Option<&wgpu::Buffer>],
        output: &wgpu::Buffer,
        groups: [u32; 3],
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        if PROFILE.lock().unwrap().is_some() {
            self.flush();
            let _ = self.0.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let slot = {
            let recorder = self.0.recorder.borrow();
            recorder.params.len() as u64
        };
        if slot + UNIFORM_SLOT > UNIFORM_SLOT * UNIFORM_SLOTS {
            self.flush();
        }
        let mut recorder = self.0.recorder.borrow_mut();
        let offset = recorder.params.len() as u64;
        let mut bytes: Vec<u8> = params.0.iter().flat_map(|w| w.to_le_bytes()).collect();
        bytes.resize(UNIFORM_SLOT as usize, 0);
        recorder.params.extend_from_slice(&bytes);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.0.uniform,
                offset,
                size: wgpu::BufferSize::new(UNIFORM_SLOT),
            }),
        }];
        for (i, input) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: input.unwrap_or(&self.0.dummy).as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32 + 1,
            resource: output.as_entire_binding(),
        });
        let bind = self.0.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.layout,
            entries: &entries,
        });
        recorder.commands.push(Command::Dispatch {
            kernel: kernel.clone(),
            bind,
            groups,
        });
        drop(recorder);
        DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(target_arch = "wasm32"))]
        if PROFILE.lock().unwrap().is_some() {
            let started = std::time::Instant::now();
            self.flush();
            let _ = self.0.device.poll(wgpu::PollType::wait_indefinitely());
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let (key, flops) = PROFILE_TAG
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| (format!("{} {:?}", kernel.label, groups), 0));
            let mut profile = PROFILE.lock().unwrap();
            let entry = profile.as_mut().unwrap().entry(key).or_default();
            entry.0 += 1;
            entry.1 += ms;
            entry.2 += flops;
        }
    }

    fn copy(
        &self,
        src: &wgpu::Buffer,
        src_offset: u64,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        self.0.recorder.borrow_mut().commands.push(Command::Copy {
            src: src.clone(),
            src_offset,
            dst: dst.clone(),
            dst_offset,
            size,
        });
    }

    fn encode(&self, extra: Option<(&wgpu::Buffer, &wgpu::Buffer, u64)>) {
        let recorder = std::mem::take(&mut *self.0.recorder.borrow_mut());
        if recorder.commands.is_empty() && extra.is_none() {
            return;
        }
        let device = &self.0.device;
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut pass: Option<wgpu::ComputePass<'static>> = None;
        for command in &recorder.commands {
            match command {
                Command::Dispatch {
                    kernel,
                    bind,
                    groups,
                } => {
                    let pass = pass.get_or_insert_with(|| {
                        encoder
                            .begin_compute_pass(&Default::default())
                            .forget_lifetime()
                    });
                    pass.set_pipeline(&kernel.pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
                Command::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    pass = None;
                    encoder.copy_buffer_to_buffer(src, *src_offset, dst, *dst_offset, *size);
                }
            }
        }
        drop(pass);
        if let Some((src, dst, size)) = extra {
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        }
        if !recorder.params.is_empty() {
            self.0
                .queue
                .write_buffer(&self.0.uniform, 0, &recorder.params);
        }
        self.0.queue.submit([encoder.finish()]);
    }

    /// Submits all recorded work.
    pub fn flush(&self) {
        self.encode(None);
    }

    pub async fn read_bytes(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
        let size = size.next_multiple_of(4);
        let staging = self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let started = clock_ms();
        self.encode(Some((buffer, &staging, size)));
        let (sender, receiver) = futures_channel::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        #[cfg(not(target_arch = "wasm32"))]
        self.0
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU wait: {e}"))?;
        receiver
            .await
            .map_err(|_| "GPU readback cancelled".to_string())?
            .map_err(|e| format!("GPU readback: {e}"))?;
        WAIT_US.fetch_add(
            ((clock_ms() - started) * 1000.0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let bytes = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("GPU readback: {e:?}"))?
            .to_vec();
        staging.unmap();
        Ok(bytes)
    }

    // ---- pipeline tensors (FP32 NCHW) -------------------------------------------------

    pub fn image(&self, values: Vec<f32>, shape: [usize; 4]) -> Tensor<4> {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        let buf = self.alloc(values.len() * 4);
        self.write(&buf, &values);
        Tensor {
            gpu: self.clone(),
            buf,
            shape,
        }
    }

    fn fill(&self, shape: [usize; 4], value: f32) -> Tensor<4> {
        let count = shape.iter().product::<usize>();
        let buf = self.alloc(count * 4);
        let kernel = self.kernel("fill", kernels::fill);
        self.dispatch(
            &kernel,
            Params::default().u(count).f(value),
            &[],
            &buf.buffer,
            groups_1d(count, 256),
        );
        Tensor {
            gpu: self.clone(),
            buf,
            shape,
        }
    }

    // ---- network activations ----------------------------------------------------------

    pub fn act(&self, n: usize, h: usize, w: usize, c: usize) -> Act {
        Act {
            buf: self.alloc(n * h * w * c * self.element()),
            n,
            h,
            w,
            c,
        }
    }

    /// FP32 NCHW → aligned NHWC, appending `ones` constant channels (EDM inputs).
    pub fn to_act(&self, x: &Tensor<4>, ones: usize) -> Act {
        let [n, c, h, w] = x.shape;
        let out = self.act(n, h, w, align(c + ones));
        let kernel = self.kernel("to_act", kernels::to_act);
        let count = out.pixels() * out.c / 4;
        self.dispatch(
            &kernel,
            Params::default()
                .u(count)
                .u(h * w)
                .u(c)
                .u(c + ones)
                .u(out.c / 4),
            &[Some(&x.buf.buffer)],
            &out.buf.buffer,
            groups_1d(count, 256),
        );
        out
    }

    /// First `channels` of an activation as FP32 NCHW.
    pub fn from_act(&self, x: &Act, channels: usize) -> Tensor<4> {
        let shape = [x.n, channels, x.h, x.w];
        let count = shape.iter().product::<usize>();
        let buf = self.alloc(count * 4);
        let kernel = self.kernel("from_act", kernels::from_act);
        self.dispatch(
            &kernel,
            Params::default().u(count).u(x.h * x.w).u(channels).u(x.c),
            &[Some(&x.buf.buffer)],
            &buf.buffer,
            groups_1d(count, 256),
        );
        Tensor {
            gpu: self.clone(),
            buf,
            shape,
        }
    }

    /// Nearest resampling by `mul / div` (2/1 subsamples, 1/2 doubles), times `scale`.
    pub fn resample(&self, x: &Act, mul: usize, div: usize) -> Act {
        let (h, w) = (x.h * div / mul, x.w * div / mul);
        let out = self.act(x.n, h, w, x.c);
        let kernel = self.kernel("resample", kernels::resample);
        let count = out.pixels() * x.c / 4;
        self.dispatch(
            &kernel,
            Params::default()
                .u(count)
                .u(h)
                .u(w)
                .u(x.h)
                .u(x.w)
                .u(x.c / 4)
                .u(mul)
                .u(div),
            &[Some(&x.buf.buffer)],
            &out.buf.buffer,
            groups_1d(count, 256),
        );
        out
    }

    /// Per-pixel inverse RMS over the channels of `a` (and of `b`, an implicit
    /// concatenation, at `a` read through `adiv`). EDM: `1 / (1e-4 + rms)`;
    /// student: `1 / sqrt(mean + eps)`.
    pub fn pixel_scale(
        &self,
        a: &Act,
        adiv: usize,
        b: Option<&Act>,
        edm: bool,
        eps: f32,
    ) -> Pixels {
        let (n, h, w) = match b {
            Some(b) => (b.n, b.h, b.w),
            None => (a.n, a.h * adiv, a.w * adiv),
        };
        let count = n * h * w;
        let buf = self.alloc(count * 4);
        let kernel = self.kernel(&format!("pixel_scale_{edm}"), || kernels::pixel_scale(edm));
        self.dispatch(
            &kernel,
            Params::default()
                .u(count)
                .u(h)
                .u(w)
                .u(a.c / 4)
                .u(b.map_or(0, |b| b.c / 4))
                .u(adiv)
                .f(eps),
            &[Some(&a.buf.buffer), b.map(|b| &b.buf.buffer)],
            &buf.buffer,
            groups_1d(count, 64),
        );
        Pixels { buf }
    }

    /// `x · scale[pixel]`.
    pub fn scale_pixels(&self, x: &Act, scale: &Pixels) -> Act {
        let out = self.act(x.n, x.h, x.w, x.c);
        let kernel = self.kernel("scale_pixels", kernels::scale_pixels);
        let count = out.pixels() * x.c / 4;
        self.dispatch(
            &kernel,
            Params::default().u(count).u(x.c / 4),
            &[Some(&x.buf.buffer), Some(&scale.buf.buffer)],
            &out.buf.buffer,
            groups_1d(count, 256),
        );
        out
    }

    /// GEMM tile of a convolution with `m` output pixels.
    fn tile(&self, m: usize) -> kernels::Tile {
        if let Some(tile) = *self.0.tile_override.borrow() {
            return tile;
        }
        if let Ok(spec) = std::env::var("NEURAL_EARTH_TILE") {
            let v: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect();
            return kernels::Tile {
                bm: v[0],
                bn: v[1],
                bk: v[2],
                tm: v[3],
                tn: v[4],
            };
        }
        let _ = m;
        // Fastest in Chrome (D3D12) on an RTX 3090, and the smallest
        // workgroup memory (8 KiB) for portability.
        kernels::Tile {
            bm: 64,
            bn: 64,
            bk: 16,
            tm: 4,
            tn: 4,
        }
    }

    /// Implicit-GEMM convolution with fused prologue and epilogue.
    pub fn conv2d(&self, x: &Act, conv: &Conv, args: ConvArgs) -> Act {
        let (cb, second) = match &args.second {
            Second::Concat { act, .. } => (act.c, Some(&act.buf.buffer)),
            Second::Residual { act, .. } => (0, Some(&act.buf.buffer)),
            Second::None => (0, None),
        };
        assert_eq!(x.c + cb, conv.cin, "convolution input layout");
        assert!(args.div.is_power_of_two(), "resampling divisor");
        let (ih, iw) = (x.h * args.div, x.w * args.div);
        let span = |size: usize, k: usize| {
            (size + 2 * args.padding - args.dilation * (k - 1) - 1) / args.stride + 1
        };
        let (oh, ow) = (span(ih, conv.kh), span(iw, conv.kw));
        let out = self.act(x.n, oh, ow, conv.cout_pad);
        let m = x.n * oh * ow;
        let flags = kernels::ConvFlags {
            concat: matches!(args.second, Second::Concat { .. }),
            residual: matches!(args.second, Second::Residual { .. }),
            modulation: matches!(args.aux, Aux::Modulation { .. }),
            pixel: matches!(args.aux, Aux::Pixel(_)),
            silu_in: args.silu_in.is_some(),
            silu_out: args.silu_out.is_some(),
            bias: conv.bias,
            clamp: args.clamp.is_some(),
        };
        let tile = self.tile(m);
        let kernel = self.kernel(&format!("conv{flags:?}{tile:?}"), || {
            kernels::conv(&flags, &tile)
        });
        let (bscale, bdiv) = match &args.second {
            Second::Concat { scale, div, .. } => (*scale, *div),
            _ => (1.0, 1),
        };
        let (wa, wb) = match &args.second {
            Second::Residual { act, wa, wb } => {
                assert_eq!((act.n, act.h, act.w, act.c), (x.n, oh, ow, conv.cout_pad));
                (*wa, *wb)
            }
            _ => (0.0, 1.0),
        };
        let (aux, cmod, moff) = match &args.aux {
            Aux::Modulation {
                buf,
                stride,
                offset,
            } => (Some(&buf.buffer), *stride, *offset),
            Aux::Pixel(p) => (Some(&p.buf.buffer), 0, 0),
            Aux::None => (None, 0, 0),
        };
        let flops = 2 * (m * conv.cout) as u64 * (conv.kh * conv.kw * conv.cin) as u64;
        CONV_FLOPS.fetch_add(flops, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(target_arch = "wasm32"))]
        if PROFILE.lock().unwrap().is_some() {
            *PROFILE_TAG.lock().unwrap() = Some((
                format!(
                    "conv {}->{} {}x{} s{} @{}x{} n{}{}{}{}",
                    conv.cin,
                    conv.cout,
                    conv.kh,
                    conv.kw,
                    args.stride,
                    oh,
                    ow,
                    x.n,
                    if args.silu_in.is_some() {
                        " silu_in"
                    } else {
                        ""
                    },
                    if matches!(args.second, Second::Concat { .. }) {
                        " cat"
                    } else {
                        ""
                    },
                    if matches!(args.aux, Aux::Pixel(_)) {
                        " pix"
                    } else {
                        ""
                    },
                ),
                flops,
            ));
        }
        let params = Params::default()
            .u(m)
            .u(conv.cout_pad / 4)
            .u(x.c)
            .u(cb)
            .u(ih)
            .u(iw)
            .u(oh)
            .u(ow)
            .u(conv.kw)
            .u(conv.kh * conv.kw)
            .u(args.stride)
            .u(args.padding)
            .u(args.dilation)
            .u(args.div.trailing_zeros() as usize)
            .u(bdiv.trailing_zeros() as usize)
            .u(cmod)
            .u(moff)
            .u(conv.kh * conv.kw * conv.cin)
            .u(0)
            .u(0)
            .f(args.scale)
            .f(bscale)
            .f(args.silu_in.unwrap_or(1.0))
            .f(args.silu_out.unwrap_or(1.0))
            .f(wa)
            .f(wb)
            .f(args.clamp.unwrap_or(0.0))
            .f(0.0);
        let groups = [
            m.div_ceil(tile.bm) as u32,
            conv.cout_pad.div_ceil(tile.bn) as u32,
            1,
        ];
        assert!(groups[0] <= 65535, "convolution too large for one dispatch");
        self.dispatch(
            &kernel,
            params,
            &[Some(&x.buf.buffer), second, Some(&conv.buffer), aux],
            &out.buf.buffer,
            groups,
        );
        out
    }

    /// Normalized modulation vectors of every EDM block for `n` samples.
    pub fn modulation(&self, emb: &Buf, n: usize, table: &Modulation) -> Rc<Buf> {
        let out = self.alloc(n * table.total * 4);
        let kernel = self.kernel("modulation", kernels::modulation);
        self.dispatch(
            &kernel,
            Params::default()
                .u(table.dim)
                .u(table.total)
                .u(table.blocks),
            &[Some(&emb.buffer), Some(&table.weights), Some(&table.table)],
            &out.buffer,
            [table.blocks as u32, n as u32, 1],
        );
        out
    }

    pub fn floats(&self, values: &[f32]) -> Rc<Buf> {
        let buf = self.alloc(values.len() * 4);
        self.write(&buf, values);
        buf
    }

    /// Cosine attention of a `[n, h·w, 3·heads·64]` QKV projection.
    pub fn attention(&self, qkv: &Act, heads: usize, channels: usize) -> Act {
        let tokens = qkv.h * qkv.w;
        assert!(tokens <= 64 && channels == 64, "attention size");
        let normalized = self.alloc(qkv.n * heads * 3 * tokens * channels * 4);
        let kernel = self.kernel("attention_norm", kernels::attention_norm);
        let count = qkv.n * heads * 3 * tokens;
        self.dispatch(
            &kernel,
            Params::default()
                .u(count)
                .u(tokens)
                .u(heads)
                .u(channels)
                .u(qkv.c),
            &[Some(&qkv.buf.buffer)],
            &normalized.buffer,
            groups_1d(count, 64),
        );
        let out = self.act(qkv.n, qkv.h, qkv.w, align(heads * channels));
        let kernel = self.kernel("attention", kernels::attention);
        let count = qkv.n * heads * tokens;
        self.dispatch(
            &kernel,
            Params::default()
                .u(count)
                .u(tokens)
                .u(heads)
                .u(channels)
                .u(out.c / 4),
            &[Some(&normalized.buffer)],
            &out.buf.buffer,
            groups_1d(count, 64),
        );
        out
    }

    /// Student output: `head(x) + direct(input)` with FP32 weights.
    pub fn student_head(&self, x: &Act, input: &Tensor<4>, weights: &wgpu::Buffer) -> Tensor<4> {
        let [n, cin, h, w] = input.shape;
        let count = n * h * w;
        let buf = self.alloc(count * 4);
        let kernel = self.kernel("student_head", kernels::student_head);
        self.dispatch(
            &kernel,
            Params::default().u(count).u(h * w).u(x.c).u(cin),
            &[Some(&x.buf.buffer), Some(&input.buf.buffer), Some(weights)],
            &buf.buffer,
            groups_1d(count, 256),
        );
        Tensor {
            gpu: self.clone(),
            buf,
            shape: [n, 1, h, w],
        }
    }
}

/// Transposed `[dim, cout]` FP32 embedding projections of all blocks.
pub struct Modulation {
    pub weights: wgpu::Buffer,
    pub table: wgpu::Buffer,
    pub dim: usize,
    pub total: usize,
    pub blocks: usize,
}

impl Modulation {
    /// `matrices[i]` are `[cout, dim]`; returns the offset of each block's vector.
    pub fn new(gpu: &Gpu, matrices: &[(Vec<f32>, usize)], dim: usize) -> (Self, Vec<usize>) {
        let mut weights = Vec::new();
        let mut table = Vec::new();
        let mut offsets = Vec::new();
        let mut total = 0;
        for (matrix, cout) in matrices {
            table.extend([(weights.len()) as u32, *cout as u32, total as u32, 0]);
            for e in 0..dim {
                for o in 0..*cout {
                    weights.push(matrix[o * dim + e]);
                }
            }
            offsets.push(total);
            total += align(*cout);
        }
        let table: Vec<f32> = table.into_iter().map(f32::from_bits).collect();
        (
            Self {
                weights: gpu.constant(&weights),
                table: gpu.constant(&table),
                dim,
                total,
                blocks: matrices.len(),
            },
            offsets,
        )
    }
}

impl<const D: usize> Tensor<D> {
    pub fn dims(&self) -> [usize; 4] {
        self.shape
    }

    fn axpby(&self, a: f32, other: Option<&Self>, b: f32) -> Self {
        let count = self.shape.iter().product::<usize>();
        if let Some(o) = other {
            assert_eq!(o.shape, self.shape, "elementwise shapes");
        }
        let gpu = &self.gpu;
        let buf = gpu.alloc(count * 4);
        let kernel = gpu.kernel("axpby", kernels::axpby);
        gpu.dispatch(
            &kernel,
            Params::default()
                .u(count)
                .u(usize::from(other.is_some()))
                .f(a)
                .f(b),
            &[Some(&self.buf.buffer), other.map(|o| &o.buf.buffer)],
            &buf.buffer,
            groups_1d(count, 256),
        );
        Self {
            gpu: gpu.clone(),
            buf,
            shape: self.shape,
        }
    }

    pub fn mul_scalar(self, s: f32) -> Self {
        self.axpby(s, None, 0.0)
    }

    // Method names of the tensor API used by the shared pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        self.axpby(1.0, Some(&other), 1.0)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        self.axpby(1.0, Some(&other), -1.0)
    }

    pub fn zeros(shape: [usize; 4], gpu: Gpu) -> Self {
        let t = gpu.fill(shape, 0.0);
        Self {
            gpu: t.gpu,
            buf: t.buf,
            shape,
        }
    }

    /// Concatenation along channels.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        assert_eq!(dim, 1, "only channel concatenation is supported");
        let first = &tensors[0];
        let gpu = first.gpu.clone();
        let [n, _, h, w] = first.shape;
        let c: usize = tensors.iter().map(|t| t.shape[1]).sum();
        let buf = gpu.alloc(n * c * h * w * 4);
        let plane = (h * w * 4) as u64;
        for b in 0..n {
            let mut at = 0u64;
            for t in &tensors {
                assert_eq!((t.shape[0], t.shape[2], t.shape[3]), (n, h, w));
                let size = t.shape[1] as u64 * plane;
                gpu.copy(
                    &t.buf.buffer,
                    b as u64 * size,
                    &buf.buffer,
                    (b * c) as u64 * plane + at,
                    size,
                );
                at += size;
            }
        }
        Self {
            gpu,
            buf,
            shape: [n, c, h, w],
        }
    }
}

/// Nearest upsampling of a pipeline tensor.
pub fn up(x: Tensor<4>, factor: usize) -> Tensor<4> {
    let [n, c, h, w] = x.shape;
    let gpu = x.gpu.clone();
    let shape = [n, c, h * factor, w * factor];
    let count = shape.iter().product::<usize>();
    let buf = gpu.alloc(count * 4);
    let kernel = gpu.kernel("upsample", kernels::upsample);
    gpu.dispatch(
        &kernel,
        Params::default().u(count).u(h).u(w).u(factor),
        &[Some(&x.buf.buffer)],
        &buf.buffer,
        groups_1d(count, 256),
    );
    Tensor { gpu, buf, shape }
}

/// Pipeline tensors are already FP32.
pub fn f16<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x
}

pub async fn read(x: Tensor<4>) -> Result<Vec<f32>, String> {
    let count = x.shape.iter().product::<usize>();
    let bytes = x.gpu.read_bytes(&x.buf.buffer, (count * 4) as u64).await?;
    Ok(bytes[..count * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
