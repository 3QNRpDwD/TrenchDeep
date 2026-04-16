use super::*;

/// DDPM noise schedule 및 forward/reverse diffusion process.
///
/// Ho et al., "Denoising Diffusion Probabilistic Models" (2020) 기반.
///
/// ## 핵심 사전계산 값
/// - `betas`:      β_1 … β_T  (linear schedule)
/// - `alphas`:     α_t = 1 - β_t
/// - `alpha_bars`: ᾱ_t = ∏_{s=1}^{t} α_s
/// - `sqrt_alpha_bars`:         √ᾱ_t
/// - `sqrt_one_minus_alpha_bars`: √(1 - ᾱ_t)
/// - `posterior_variance`:      β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
pub struct DDPMScheduler {
    pub timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alpha_bars: Vec<f32>,
    pub sqrt_alpha_bars: Vec<f32>,
    pub sqrt_one_minus_alpha_bars: Vec<f32>,
    pub sqrt_recip_alphas: Vec<f32>,
    pub posterior_variance: Vec<f32>,
}

impl DDPMScheduler {
    /// Linear β-schedule: β 가 `beta_start` 에서 `beta_end` 까지 선형 증가.
    ///
    /// DDPM 논문 기본값: beta_start=1e-4, beta_end=0.02, timesteps=1000.
    pub fn linear_schedule(timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let betas: Vec<f32> = (0..timesteps)
            .map(|i| beta_start + (beta_end - beta_start) * i as f32 / (timesteps - 1).max(1) as f32)
            .collect();

        Self::from_betas(timesteps, betas)
    }

    /// Cosine β-schedule (Nichol & Dhariwal, 2021).
    ///
    /// ᾱ_t = f(t) / f(0), f(t) = cos²((t/T + s) / (1+s) · π/2)
    pub fn cosine_schedule(timesteps: usize, s: f32) -> Self {
        let f = |t: f32| -> f32 {
            ((t / timesteps as f32 + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2)
        };
        let f0 = f(0.0);

        let mut alpha_bars = Vec::with_capacity(timesteps);
        for t in 0..timesteps {
            alpha_bars.push((f((t + 1) as f32) / f0).clamp(1e-4, 1.0));
        }

        // α_bar → β 역산: β_t = 1 - ᾱ_t / ᾱ_{t-1}
        let mut betas = Vec::with_capacity(timesteps);
        for t in 0..timesteps {
            let alpha_bar_prev = if t == 0 { 1.0 } else { alpha_bars[t - 1] };
            betas.push((1.0 - alpha_bars[t] / alpha_bar_prev).clamp(0.0, 0.999));
        }

        Self::from_betas(timesteps, betas)
    }

    /// β 벡터로부터 모든 사전계산 값을 도출한다.
    fn from_betas(timesteps: usize, betas: Vec<f32>) -> Self {
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();

        // ᾱ_t = cumulative product of α
        let mut alpha_bars = Vec::with_capacity(timesteps);
        let mut cumprod = 1.0f32;
        for &a in &alphas {
            cumprod *= a;
            alpha_bars.push(cumprod);
        }

        let sqrt_alpha_bars: Vec<f32> = alpha_bars.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alpha_bars: Vec<f32> = alpha_bars.iter().map(|&a| (1.0 - a).sqrt()).collect();
        let sqrt_recip_alphas: Vec<f32> = alphas.iter().map(|&a| a.sqrt().recip()).collect();

        // posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        let mut posterior_variance = Vec::with_capacity(timesteps);
        for t in 0..timesteps {
            let alpha_bar_prev = if t == 0 { 1.0 } else { alpha_bars[t - 1] };
            let pv = betas[t] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bars[t]).max(1e-8);
            posterior_variance.push(pv);
        }

        Self {
            timesteps,
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
            sqrt_recip_alphas,
            posterior_variance,
        }
    }

    /// Forward diffusion process: q(x_t | x_0).
    ///
    /// ```text
    /// x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε,   ε ~ N(0, I)
    /// ```
    ///
    /// # Arguments
    /// * `x_0`   - 원본 데이터 `[N, C, H, W]`
    /// * `t`     - 타임스텝 인덱스 (0-based, 배치 내 모든 샘플에 동일 적용)
    /// * `noise` - 미리 생성된 가우시안 노이즈 (x_0 과 동일 shape)
    ///
    /// # Returns
    /// `(x_t, noise)` — noisy 이미지와 사용된 노이즈
    pub fn q_sample(&self, x_0: &dyn TensorBase, t: usize, noise: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let sqrt_ab = self.sqrt_alpha_bars[t];
        let sqrt_omab = self.sqrt_one_minus_alpha_bars[t];

        // x_t = sqrt_ab * x_0 + sqrt_omab * noise
        let x_t_data: Vec<f32> = x_0.data().iter()
            .zip(noise.data().iter())
            .map(|(&x, &n)| sqrt_ab * x + sqrt_omab * n)
            .collect();

        GlobalTensor::from_vec(x_t_data, x_0.shape())
    }

    /// Forward diffusion (학습 경로, Variable 반환).
    #[cfg(feature = "enableBackward")]
    pub fn q_sample_variable(&self, x_0: &Variable, t: usize, noise: &Variable) -> MlResult<Variable> {
        let sqrt_ab = self.sqrt_alpha_bars[t];
        let sqrt_omab = self.sqrt_one_minus_alpha_bars[t];

        let scale_signal = Variable::new(Tensor::from_vec(vec![sqrt_ab], &[1, 1])?);
        let scale_noise = Variable::new(Tensor::from_vec(vec![sqrt_omab], &[1, 1])?);

        // x_t = scale_signal * x_0  +  scale_noise * noise
        let term1 = Mul::new()?.apply(&[x_0, &scale_signal])?;
        let term2 = Mul::new()?.apply(&[noise, &scale_noise])?;
        Ok(&term1 + &term2)
    }

    /// Reverse diffusion 한 스텝: p(x_{t-1} | x_t).
    ///
    /// DDPM 논문 Algorithm 2:
    /// ```text
    /// x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))  +  √β̃_t · z
    /// ```
    /// 여기서 z ~ N(0,I) (t > 0), z = 0 (t = 0).
    pub fn p_sample(
        &self,
        x_t: &dyn TensorBase,
        predicted_noise: &dyn TensorBase,
        t: usize,
    ) -> MlResult<GlobalTensor<f32>> {
        let sqrt_recip_alpha = self.sqrt_recip_alphas[t];
        let beta = self.betas[t];
        let sqrt_omab = self.sqrt_one_minus_alpha_bars[t];
        let noise_coef = beta / sqrt_omab.max(1e-8);

        // mean = 1/√α_t · (x_t - noise_coef · ε_θ)
        let mean_data: Vec<f32> = x_t.data().iter()
            .zip(predicted_noise.data().iter())
            .map(|(&x, &eps)| sqrt_recip_alpha * (x - noise_coef * eps))
            .collect();

        if t == 0 {
            // t=0 이면 노이즈 추가 없이 반환
            return GlobalTensor::from_vec(mean_data, x_t.shape());
        }

        // z ~ N(0, I), variance = β̃_t
        let z = Tensor::randn(x_t.shape());
        let sigma = self.posterior_variance[t].sqrt();

        let result: Vec<f32> = mean_data.iter()
            .zip(z.data().iter())
            .map(|(&m, &zi)| m + sigma * zi)
            .collect();

        GlobalTensor::from_vec(result, x_t.shape())
    }

    /// 전체 reverse process: x_T → x_0.
    ///
    /// DDPM 논문 Algorithm 2 전체 루프.
    /// `denoise_fn` 은 (x_t, t) → predicted_noise 를 반환하는 클로저.
    pub fn p_sample_loop<F>(
        &self,
        shape: &[usize],
        denoise_fn: F,
    ) -> MlResult<GlobalTensor<f32>>
    where
        F: Fn(&GlobalTensor<f32>, usize) -> MlResult<GlobalTensor<f32>>,
    {
        // x_T ~ N(0, I)
        let mut x = Tensor::randn(shape);
        let mut x_global = GlobalTensor::from_vec(x.data().to_vec(), x.shape())?;

        // t = T-1, T-2, ..., 0
        for t in (0..self.timesteps).rev() {
            let predicted_noise = denoise_fn(&x_global, t)?;
            x_global = self.p_sample(&x_global, &predicted_noise, t)?;
        }

        Ok(x_global)
    }

    /// 랜덤 타임스텝 생성 (학습용).
    /// [0, timesteps) 범위의 uniform random 정수.
    pub fn sample_timestep(&self) -> usize {
        (rand::random::<f32>() * self.timesteps as f32) as usize % self.timesteps
    }
}

// Scheduler 구조체를 DDPMScheduler로 대체
pub struct Scheduler {
    pub inner: DDPMScheduler,
}

impl Scheduler {
    pub fn ddpm(timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        Self {
            inner: DDPMScheduler::linear_schedule(timesteps, beta_start, beta_end),
        }
    }

    pub fn ddpm_cosine(timesteps: usize) -> Self {
        Self {
            inner: DDPMScheduler::cosine_schedule(timesteps, 0.008),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_schedule_values() {
        let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
        assert_eq!(s.betas.len(), 1000);
        // β_0 ≈ 1e-4, β_999 ≈ 0.02
        assert!((s.betas[0] - 1e-4).abs() < 1e-6);
        assert!((s.betas[999] - 0.02).abs() < 1e-4);
        // ᾱ 는 단조감소
        for i in 1..1000 {
            assert!(s.alpha_bars[i] < s.alpha_bars[i - 1],
                "alpha_bars must decrease: t={} ({}) >= t={} ({})",
                i, s.alpha_bars[i], i - 1, s.alpha_bars[i - 1]);
        }
        // ᾱ_0 ≈ 1, ᾱ_999 ≈ 0
        assert!(s.alpha_bars[0] > 0.99);
        assert!(s.alpha_bars[999] < 0.1);
    }

    #[test]
    fn cosine_schedule_monotonic() {
        let s = DDPMScheduler::cosine_schedule(1000, 0.008);
        assert_eq!(s.betas.len(), 1000);
        for i in 1..1000 {
            assert!(s.alpha_bars[i] <= s.alpha_bars[i - 1] + 1e-6,
                "cosine alpha_bars must decrease: t={}", i);
        }
    }

    #[test]
    fn q_sample_shape_preserved() -> MlResult<()> {
        let s = DDPMScheduler::linear_schedule(100, 1e-4, 0.02);
        let x_0 = Tensor::from_vec(vec![1.0; 2 * 3 * 4 * 4], &[2, 3, 4, 4])?;
        let noise = Tensor::randn(&[2, 3, 4, 4]);
        let x_t = s.q_sample(&x_0, 50, &noise)?;
        assert_eq!(x_t.shape(), &[2, 3, 4, 4]);
        Ok(())
    }

    #[test]
    fn q_sample_t0_close_to_original() -> MlResult<()> {
        let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
        let x_0 = Tensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4])?;
        let noise = Tensor::from_vec(vec![0.5; 16], &[1, 1, 4, 4])?;

        // t=0: √ᾱ_0 ≈ 1, √(1-ᾱ_0) ≈ 0 → x_t ≈ x_0
        let x_t = s.q_sample(&x_0, 0, &noise)?;
        for (&xt, &x0) in x_t.data().iter().zip(x_0.data().iter()) {
            assert!((xt - x0).abs() < 0.05, "t=0: x_t should be close to x_0");
        }
        Ok(())
    }

    #[test]
    fn q_sample_large_t_dominated_by_noise() -> MlResult<()> {
        let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
        let x_0 = Tensor::from_vec(vec![10.0; 16], &[1, 1, 4, 4])?;
        let noise = Tensor::from_vec(vec![0.0; 16], &[1, 1, 4, 4])?;

        // t=999: ᾱ_999 ≈ 0 → x_t ≈ noise (≈0)
        let x_t = s.q_sample(&x_0, 999, &noise)?;
        let mean: f32 = x_t.data().iter().sum::<f32>() / x_t.data().len() as f32;
        assert!(mean.abs() < 2.0,
            "t=999: signal should be nearly gone, mean={}", mean);
        Ok(())
    }

    #[test]
    fn p_sample_shape_preserved() -> MlResult<()> {
        let s = DDPMScheduler::linear_schedule(100, 1e-4, 0.02);
        let x_t = Tensor::from_vec(vec![0.5; 2 * 1 * 4 * 4], &[2, 1, 4, 4])?;
        let eps = Tensor::from_vec(vec![0.1; 2 * 1 * 4 * 4], &[2, 1, 4, 4])?;
        let x_prev = s.p_sample(&x_t, &eps, 50)?;
        assert_eq!(x_prev.shape(), &[2, 1, 4, 4]);
        Ok(())
    }

    #[test]
    fn p_sample_loop_returns_correct_shape() -> MlResult<()> {
        let s = DDPMScheduler::linear_schedule(10, 1e-4, 0.02);
        let shape = [1, 1, 4, 4];

        let result = s.p_sample_loop(&shape, |x_t, _t| {
            // 더미 denoise: 그냥 0 반환
            GlobalTensor::from_vec(vec![0.0; x_t.data().len()], x_t.shape())
        })?;

        assert_eq!(result.shape(), &[1, 1, 4, 4]);
        Ok(())
    }

    #[test]
    fn posterior_variance_t0_is_zero() {
        let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
        // t=0: ᾱ_{t-1} = ᾱ_{-1} = 1.0 → posterior_var = β_0 * 0 / (1-ᾱ_0) ≈ 0
        assert!(s.posterior_variance[0].abs() < 1e-6,
            "posterior_variance[0] = {}", s.posterior_variance[0]);
    }
}
