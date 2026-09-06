//! Time-conditioned residual U-Net using only the public tensor API.
use super::{Conv2D, GroupNorm, Layer, Linear};
use crate::trainer::{TrainableModel, UnsupervisedModel};
use crate::{ContextId, ExecutionContext, MlResult, Parameter, Tensor, TensorError, Variable};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn invalid(reason: &str) -> crate::MlError {
    TensorError::InvalidOperation {
        op: "diffusion",
        reason: reason.into(),
    }
    .into()
}
fn conv(
    ctx: &ExecutionContext,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    label: &str,
) -> MlResult<Conv2D> {
    Conv2D::new(
        ctx,
        input,
        output,
        (kernel, kernel),
        (stride, stride),
        (kernel / 2, kernel / 2),
        label,
    )
}

#[derive(Debug)]
struct Residual {
    norm1: GroupNorm,
    conv1: Conv2D,
    norm2: GroupNorm,
    conv2: Conv2D,
    skip: Option<Conv2D>,
    time: Linear,
    channels: usize,
}
impl Residual {
    fn new(
        ctx: &ExecutionContext,
        input: usize,
        output: usize,
        groups: usize,
        time: usize,
    ) -> MlResult<Self> {
        Ok(Self {
            norm1: GroupNorm::new(ctx, groups, input, 1e-5, "norm1")?,
            conv1: conv(ctx, input, output, 3, 1, "conv1")?,
            norm2: GroupNorm::new(ctx, groups, output, 1e-5, "norm2")?,
            conv2: conv(ctx, output, output, 3, 1, "conv2")?,
            skip: if input != output {
                Some(conv(ctx, input, output, 1, 1, "skip")?)
            } else {
                None
            },
            time: Linear::new(ctx, time, output, "time")?,
            channels: output,
        })
    }
    fn forward(&self, input: &Variable, time: &Variable) -> MlResult<Variable> {
        let batch = input.tensor().shape()?[0];
        let h = self.conv1.forward(&self.norm1.forward(input)?.silu()?)?;
        let t = self
            .time
            .forward(time)?
            .reshape(&[batch, self.channels, 1, 1])?;
        let h = h.add(t.tensor())?;
        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;
        let skip = if let Some(layer) = &self.skip {
            layer.forward(input)?
        } else {
            input.clone()
        };
        h.add(skip.tensor())
    }
    fn parameters(&self) -> Vec<&Parameter> {
        let mut p = self.norm1.parameters();
        p.extend(self.conv1.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.time.parameters());
        if let Some(skip) = &self.skip {
            p.extend(skip.parameters());
        }
        p
    }
}

#[derive(Debug)]
struct Attention {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    channels: usize,
}
impl Attention {
    fn new(ctx: &ExecutionContext, channels: usize, groups: usize) -> MlResult<Self> {
        Ok(Self {
            norm: GroupNorm::new(ctx, groups, channels, 1e-5, "attention_norm")?,
            q: Linear::new(ctx, channels, channels, "query")?,
            k: Linear::new(ctx, channels, channels, "key")?,
            v: Linear::new(ctx, channels, channels, "value")?,
            out: Linear::new(ctx, channels, channels, "projection")?,
            channels,
        })
    }
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        let shape = input.tensor().shape()?;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let x = self
            .norm
            .forward(input)?
            .reshape(&[n, c, h * w])?
            .transpose(&[0, 2, 1])?
            .reshape(&[n * h * w, c])?;
        let q = self.q.forward(&x)?.reshape(&[n, h * w, c])?;
        let k = self
            .k
            .forward(&x)?
            .reshape(&[n, h * w, c])?
            .transpose(&[0, 2, 1])?;
        let v = self.v.forward(&x)?.reshape(&[n, h * w, c])?;
        let scale = input
            .tensor()
            .execution_context()?
            .scalar(1.0 / (self.channels as f32).sqrt())?;
        let scores = q.matmul(k.tensor())?.mul(&scale)?.softmax(2)?;
        let weighted = scores.matmul(v.tensor())?.reshape(&[n * h * w, c])?;
        let output = self
            .out
            .forward(&weighted)?
            .reshape(&[n, h * w, c])?
            .transpose(&[0, 2, 1])?
            .reshape(&shape)?;
        output.add(input.tensor())
    }
    fn parameters(&self) -> Vec<&Parameter> {
        let mut p = self.norm.parameters();
        for layer in [&self.q, &self.k, &self.v, &self.out] {
            p.extend(layer.parameters());
        }
        p
    }
}

#[derive(Debug)]
struct Stage {
    first: Residual,
    second: Residual,
    attention: Option<Attention>,
    resize: Conv2D,
}
impl Stage {
    fn body(&self, input: &Variable, time: &Variable) -> MlResult<Variable> {
        let h = self
            .second
            .forward(&self.first.forward(input, time)?, time)?;
        if let Some(a) = &self.attention {
            a.forward(&h)
        } else {
            Ok(h)
        }
    }
    fn parameters(&self) -> Vec<&Parameter> {
        let mut p = self.first.parameters();
        p.extend(self.second.parameters());
        if let Some(a) = &self.attention {
            p.extend(a.parameters());
        }
        p.extend(self.resize.parameters());
        p
    }
}

#[derive(Debug)]
pub struct Unet {
    config: serde_json::Value,
    context: ExecutionContext,
    channels: usize,
    dim: usize,
    initial: Conv2D,
    time1: Linear,
    time2: Linear,
    down: Vec<Stage>,
    middle1: Residual,
    middle_attention: Attention,
    middle2: Residual,
    up: Vec<Stage>,
    final_residual: Residual,
    final_conv: Conv2D,
}
impl Unet {
    pub fn new(
        ctx: &ExecutionContext,
        channels: usize,
        dim: usize,
        multipliers: &[usize],
        groups: usize,
        attention_at: &[usize],
    ) -> MlResult<Self> {
        if channels == 0
            || dim == 0
            || dim % 2 != 0
            || multipliers.is_empty()
            || multipliers.contains(&0)
        {
            return Err(invalid(
                "channels and dimensions must be positive, embedding dimension even, and stages nonempty",
            ));
        }
        let time = dim
            .checked_mul(4)
            .ok_or_else(|| invalid("dimension overflow"))?;
        let mut dims = vec![dim];
        for &m in multipliers {
            dims.push(
                dim.checked_mul(m)
                    .ok_or_else(|| invalid("dimension overflow"))?,
            );
        }
        let mut down = Vec::new();
        let mut up = Vec::new();
        for (i, pair) in dims.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            down.push(Stage {
                first: Residual::new(ctx, a, b, groups, time)?,
                second: Residual::new(ctx, b, b, groups, time)?,
                attention: if attention_at.contains(&i) {
                    Some(Attention::new(ctx, b, groups)?)
                } else {
                    None
                },
                resize: conv(ctx, b, b, 3, 2, "downsample")?,
            });
        }
        let mid = dims[dims.len() - 1];
        for i in (0..multipliers.len()).rev() {
            let previous = if i + 1 == multipliers.len() {
                mid
            } else {
                dims[i + 1]
            };
            up.push(Stage {
                first: Residual::new(ctx, previous + dims[i + 1], dims[i], groups, time)?,
                second: Residual::new(ctx, dims[i], dims[i], groups, time)?,
                attention: if attention_at.contains(&i) {
                    Some(Attention::new(ctx, dims[i], groups)?)
                } else {
                    None
                },
                resize: conv(ctx, previous, previous, 3, 1, "upsample")?,
            });
        }
        Ok(Self {
            config: serde_json::json!({"channels":channels,"dim":dim,"multipliers":multipliers,"groups":groups,"attention_at":attention_at}),
            context: ctx.clone(),
            channels,
            dim,
            initial: conv(ctx, channels, dim, 3, 1, "initial")?,
            time1: Linear::new(ctx, dim, time, "time1")?,
            time2: Linear::new(ctx, time, time, "time2")?,
            down,
            middle1: Residual::new(ctx, mid, mid, groups, time)?,
            middle_attention: Attention::new(ctx, mid, groups)?,
            middle2: Residual::new(ctx, mid, mid, groups, time)?,
            up,
            final_residual: Residual::new(ctx, 2 * dim, dim, groups, time)?,
            final_conv: conv(ctx, dim, channels, 1, 1, "final")?,
        })
    }
    /// Timesteps are normalized to [0, 1] with shape [batch, 1].
    pub fn forward(&self, input: &Variable, timesteps: &Tensor) -> MlResult<Variable> {
        self.context.validate(input.tensor())?;
        self.context.validate(timesteps)?;
        let shape = input.tensor().shape()?;
        let divisor = 1usize
            .checked_shl(self.down.len() as u32)
            .ok_or_else(|| invalid("too many stages"))?;
        if shape.len() != 4
            || shape[0] == 0
            || shape[1] != self.channels
            || shape[2] == 0
            || shape[3] == 0
            || shape[2] % divisor != 0
            || shape[3] % divisor != 0
        {
            return Err(invalid(
                "expected NCHW images with spatial dimensions divisible by 2^stages",
            ));
        }
        if timesteps.shape()? != vec![shape[0], 1] {
            return Err(invalid("timesteps must have shape [batch, 1]"));
        }
        let frequencies = self.context.tensor(
            (0..self.dim / 2)
                .map(|i| 10000f32.powf(-2.0 * i as f32 / self.dim as f32))
                .collect(),
            &[1, self.dim / 2],
        )?;
        let angles = timesteps.mul(&frequencies)?;
        let embedding = self
            .context
            .concat(&[&angles.sin()?, &angles.cos()?], 1)?
            .as_variable()?;
        let time = self
            .time2
            .forward(&self.time1.forward(&embedding)?.silu()?)?;
        let initial = self.initial.forward(input)?;
        let mut h = initial.clone();
        let mut skips = Vec::new();
        for stage in &self.down {
            h = stage.body(&h, &time)?;
            skips.push(h.clone());
            h = stage.resize.forward(&h)?;
        }
        h = self.middle2.forward(
            &self
                .middle_attention
                .forward(&self.middle1.forward(&h, &time)?)?,
            &time,
        )?;
        for (stage, skip) in self.up.iter().zip(skips.iter().rev()) {
            h = stage.resize.forward(&h.nearest_upsample2d((2, 2))?)?;
            h = self
                .context
                .concat(&[h.tensor(), skip.tensor()], 1)?
                .as_variable()?;
            h = stage.body(&h, &time)?;
        }
        h = self
            .context
            .concat(&[h.tensor(), initial.tensor()], 1)?
            .as_variable()?;
        self.final_conv
            .forward(&self.final_residual.forward(&h, &time)?)
    }
    pub fn predict(&self, input: &Tensor, timesteps: &Tensor) -> MlResult<Tensor> {
        self.context.no_grad(|| {
            Ok(self
                .forward(&input.as_variable()?, timesteps)?
                .tensor()
                .clone())
        })
    }
}
impl TrainableModel for Unet {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        let mut p = self.initial.parameters();
        p.extend(self.time1.parameters());
        p.extend(self.time2.parameters());
        for stage in &self.down {
            p.extend(stage.parameters());
        }
        p.extend(self.middle1.parameters());
        p.extend(self.middle_attention.parameters());
        p.extend(self.middle2.parameters());
        for stage in &self.up {
            p.extend(stage.parameters());
        }
        p.extend(self.final_residual.parameters());
        p.extend(self.final_conv.parameters());
        p
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionScheduler {
    betas: Vec<f32>,
    alpha_bars: Vec<f32>,
}
impl DiffusionScheduler {
    pub fn cosine(steps: usize, offset: f32) -> MlResult<Self> {
        if steps == 0 || !offset.is_finite() || offset < 0.0 || offset > 1.0 {
            return Err(invalid("invalid cosine schedule"));
        }
        let f = |t: f32| {
            ((t / steps as f32 + offset) / (1.0 + offset) * std::f32::consts::FRAC_PI_2)
                .cos()
                .powi(2)
        };
        let f0 = f(0.0);
        let mut previous = 1.0;
        let mut betas = Vec::with_capacity(steps);
        for t in 0..steps {
            let current = (f((t + 1) as f32) / f0).clamp(1e-4, 1.0);
            betas.push((1.0 - current / previous).clamp(0.0, 0.999));
            previous = current;
        }
        Self::from_betas(betas)
    }
    pub fn linear(steps: usize, start: f32, end: f32) -> MlResult<Self> {
        if steps == 0
            || !start.is_finite()
            || !end.is_finite()
            || start <= 0.0
            || start > end
            || end >= 1.0
        {
            return Err(invalid("invalid beta schedule"));
        }
        Self::from_betas(
            (0..steps)
                .map(|i| start + (end - start) * i as f32 / steps.saturating_sub(1).max(1) as f32)
                .collect(),
        )
    }
    pub fn from_betas(betas: Vec<f32>) -> MlResult<Self> {
        if betas.is_empty()
            || betas
                .iter()
                .any(|b| !b.is_finite() || *b < 0.0 || *b >= 1.0)
        {
            return Err(invalid("betas must be finite and in [0,1)"));
        }
        let mut product = 1.0;
        let alpha_bars = betas
            .iter()
            .map(|b| {
                product *= 1.0 - b;
                product
            })
            .collect();
        Ok(Self { betas, alpha_bars })
    }
    pub fn timesteps(&self) -> usize {
        self.betas.len()
    }
    pub fn q_sample(&self, image: &Tensor, noise: &Tensor, t: usize) -> MlResult<Tensor> {
        let &alpha = self
            .alpha_bars
            .get(t)
            .ok_or_else(|| invalid("timestep out of range"))?;
        if image.shape()? != noise.shape()? {
            return Err(invalid("noise shape differs from image"));
        }
        let ctx = image.execution_context()?;
        image
            .mul(&ctx.scalar(alpha.sqrt())?)?
            .add(&noise.mul(&ctx.scalar((1.0 - alpha).sqrt())?)?)
    }
    pub fn reverse_step(
        &self,
        image: &Tensor,
        prediction: &Tensor,
        noise: &Tensor,
        t: usize,
    ) -> MlResult<Tensor> {
        let &beta = self
            .betas
            .get(t)
            .ok_or_else(|| invalid("timestep out of range"))?;
        if image.shape()? != prediction.shape()? || image.shape()? != noise.shape()? {
            return Err(invalid("reverse-step shapes differ"));
        }
        let ctx = image.execution_context()?;
        ctx.validate(noise)?;
        let alpha_bar = self.alpha_bars[t];
        let coefficient = beta / (1.0 - alpha_bar).sqrt().max(1e-8);
        let mean = image
            .sub(&prediction.mul(&ctx.scalar(coefficient)?)?)?
            .mul(&ctx.scalar(1.0 / (1.0 - beta).sqrt())?)?;
        if t == 0 {
            return Ok(mean);
        }
        let variance = beta * (1.0 - self.alpha_bars[t - 1]) / (1.0 - alpha_bar).max(1e-8);
        mean.add(&noise.mul(&ctx.scalar(variance.sqrt())?)?)
    }
}

#[derive(Debug)]
pub struct Diffusion {
    context: ExecutionContext,
    pub unet: Unet,
    pub scheduler: DiffusionScheduler,
    noise: StdRng,
}
impl Diffusion {
    pub fn new(
        ctx: &ExecutionContext,
        unet: Unet,
        scheduler: DiffusionScheduler,
        seed: u64,
    ) -> MlResult<Self> {
        if unet.context_id() != ctx.id() {
            return Err(crate::ContextError::Mismatch.into());
        }
        Ok(Self {
            context: ctx.clone(),
            unet,
            scheduler,
            noise: StdRng::seed_from_u64(seed),
        })
    }
    fn noise(&mut self, shape: &[usize]) -> MlResult<Tensor> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("shape overflow"))?;
        let data = (0..count)
            .map(|_| {
                (-2.0 * self.noise.random::<f32>().max(f32::MIN_POSITIVE).ln()).sqrt()
                    * (std::f32::consts::TAU * self.noise.random::<f32>()).cos()
            })
            .collect();
        self.context.tensor(data, shape)
    }
    pub fn forward_loss_with_noise(
        &self,
        image: &Variable,
        noise: &Tensor,
        t: usize,
    ) -> MlResult<(Variable, Variable)> {
        let noisy = self
            .scheduler
            .q_sample(image.tensor(), noise, t)?
            .as_variable()?;
        let shape = image.tensor().shape()?;
        if shape.len() != 4 {
            return Err(invalid("expected NCHW image"));
        }
        let times = self.context.tensor(
            vec![t as f32 / self.scheduler.timesteps() as f32; shape[0]],
            &[shape[0], 1],
        )?;
        let prediction = self.unet.forward(&noisy, &times)?;
        let loss = prediction.mse_loss(noise, crate::Reduction::Mean)?;
        Ok((prediction, loss))
    }
    pub fn sample_with_noise(&self, initial: &Tensor, step_noise: &[Tensor]) -> MlResult<Tensor> {
        self.context.validate(initial)?;
        if step_noise.len() != self.scheduler.timesteps() {
            return Err(invalid("one noise tensor per timestep is required"));
        }
        self.context.no_grad(|| {
            let mut image = initial.clone();
            let shape = image.shape()?;
            if shape.len() != 4 {
                return Err(invalid("expected NCHW image"));
            }
            for t in (0..self.scheduler.timesteps()).rev() {
                let times = self.context.tensor(
                    vec![t as f32 / self.scheduler.timesteps() as f32; shape[0]],
                    &[shape[0], 1],
                )?;
                let prediction = self.unet.predict(&image, &times)?;
                image = self
                    .scheduler
                    .reverse_step(&image, &prediction, &step_noise[t], t)?;
            }
            Ok(image)
        })
    }
    pub fn sample(&mut self, shape: &[usize]) -> MlResult<Tensor> {
        let initial = self.noise(shape)?;
        let mut noises = Vec::new();
        for _ in 0..self.scheduler.timesteps() {
            noises.push(self.noise(shape)?);
        }
        self.sample_with_noise(&initial, &noises)
    }
}
impl TrainableModel for Diffusion {
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.unet.parameters()
    }
}
impl UnsupervisedModel for Diffusion {
    fn forward_loss(&mut self, image: &Variable) -> MlResult<(Variable, Variable)> {
        let noise = self.noise(&image.tensor().shape()?)?;
        let t = self.noise.random_range(0..self.scheduler.timesteps());
        self.forward_loss_with_noise(image, &noise, t)
    }
}

impl crate::trainer::CheckpointableModel for Diffusion {
    fn save_checkpoint(&self, path: &std::path::Path) -> MlResult<()> {
        let path = path
            .to_str()
            .ok_or_else(|| invalid("checkpoint path is not valid UTF-8"))?;
        let mut params = Vec::new();
        for (i, p) in self.parameters().iter().enumerate() {
            params.push(super::ParamState {
                name: format!("parameter_{i}"),
                shape: p.tensor().shape()?,
                data: p.tensor().to_vec()?,
                blob_offset: None,
                blob_length: None,
            });
        }
        super::ModelState::new(vec![super::LayerState {
            layer_type: "Diffusion".into(),
            label: "ddpm".into(),
            config: serde_json::json!({"unet":self.unet.config,"betas":self.scheduler.betas}),
            params,
        }])
        .save(path)
    }
    fn load_checkpoint(&mut self, path: &std::path::Path) -> MlResult<()> {
        let path = path
            .to_str()
            .ok_or_else(|| invalid("checkpoint path is not valid UTF-8"))?;
        let state = super::ModelState::load(path)?;
        if state.layers.len() != 1 {
            return Err(invalid("invalid diffusion checkpoint layer count"));
        }
        let layer = &state.layers[0];
        let betas: Vec<f32> = serde_json::from_value(layer.config["betas"].clone())
            .map_err(|_| invalid("invalid checkpoint scheduler"))?;
        if layer.layer_type != "Diffusion"
            || layer.config["unet"] != self.unet.config
            || betas != self.scheduler.betas
        {
            return Err(invalid("checkpoint architecture or scheduler mismatch"));
        }
        let parameters = self.parameters();
        if parameters.len() != layer.params.len() {
            return Err(invalid("checkpoint parameter count mismatch"));
        }
        let mut buffers = Vec::new();
        for (i, (p, saved)) in parameters.iter().zip(&layer.params).enumerate() {
            if saved.name != format!("parameter_{i}") || saved.shape != p.tensor().shape()? {
                return Err(invalid("checkpoint parameter identity or shape mismatch"));
            }
            buffers.push(crate::TensorBuffer::from_vec(
                saved.data.clone(),
                &saved.shape,
            )?);
        }
        for (p, buffer) in parameters.iter().zip(buffers) {
            self.context.replace_parameter(p.variable(), buffer)?;
        }
        Ok(())
    }
}
