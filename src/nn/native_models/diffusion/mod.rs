#[path = "encoder.rs"]
pub mod encoder;
#[path = "decoder.rs"]
mod decoder;
#[path = "unet.rs"]
pub mod unet;
#[path = "scheduler.rs"]
pub mod scheduler;
#[path = "embedding.rs"]
pub mod embedding;

use super::*; // info, MlResult, Layer, Linear, Sequential, ... (from model/mod.rs)

// diffusion 하위 모듈 전용 import
use std::fmt::Debug;
use crate::legacy::{
    nn::{Conv2D, GroupNorm, activation::{SiLU, SoftmaxOp}},
    tensor::operators::{Concat, Cos, Mul, NearestUpsample2d, ReshapeOp, Sin, Transpose},
};
use crate::legacy::loss::MeanSquaredError;
use crate::legacy::nn::models::diffusion::unet::Unet;
use self::embedding::TimeEmbeddingMLP;
use self::encoder::SinusoidalPE;
use self::scheduler::DDPMScheduler;
// NOTE: `Decoder`, `Encoder`, `Scheduler` (wrapper) 스텁은 LatentDiffusion 구현 시
//        사용될 예정이므로 제거하지 않는다. 현재는 dead import 방지를 위해 제외.

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                     DDPM — Diffusion 모델                               ║
// ║                                                                         ║
// ║  "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)           ║
// ║                                                                         ║
// ║  ── 전체 알고리즘 개요 ──                                                ║
// ║                                                                         ║
// ║  【학습 (Algorithm 1)】                                                  ║
// ║    repeat:                                                              ║
// ║      1. x₀ ~ q(x₀)              데이터에서 이미지 샘플링                ║
// ║      2. t ~ Uniform({1,...,T})   랜덤 타임스텝                          ║
// ║      3. ε ~ N(0, I)             가우시안 노이즈 생성                    ║
// ║      4. x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε    forward process               ║
// ║      5. L = ‖ε - ε_θ(x_t, t)‖²  U-Net 의 노이즈 예측과 실제 비교     ║
// ║      6. ∇_θ L → optimizer step   gradient descent                      ║
// ║    until converged                                                      ║
// ║                                                                         ║
// ║  【샘플링 (Algorithm 2)】                                                ║
// ║    1. x_T ~ N(0, I)             순수 노이즈에서 시작                    ║
// ║    2. for t = T, T-1, ..., 1:                                           ║
// ║         z ~ N(0, I) if t > 1, else z = 0                               ║
// ║         x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t))       ║
// ║                   + √β̃_t · z                                           ║
// ║    3. return x₀                  생성된 이미지!                         ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// DDPM 디퓨전 모델.
///
/// 세 가지 컴포넌트로 구성
/// - `unet`:      노이즈 예측 네트워크 ε_θ(x_t, t)
/// - `scheduler`: noise schedule (β, ᾱ) 및 forward/reverse process
/// - `loss`:      MSE loss ‖ε - ε_θ‖²
pub struct Diffusion {
    pub unet: Unet,
    pub scheduler: DDPMScheduler,
    /// MSE loss 연산자 (Arc<dyn Function>)
    pub loss: GlobalFunction,
    /// 입력 이미지 shape [C, H, W] (batch 제외)
    pub image_shape: Vec<usize>,
}

impl Diffusion {
    /// DDPM 모델 생성.
    ///
    /// ## 파라미터
    ///
    /// * `image_channels` - 이미지 채널 수 (1=grayscale, 3=RGB)
    /// * `image_size`     - 이미지 한 변의 크기 (정사각형 가정, 예: 32)
    /// * `dim`            - U-Net 기본 채널 수 (예: 64)
    /// * `dim_mults`      - 각 해상도 단계의 채널 배수 (예: [1, 2, 4])
    /// * `resnet_groups`  - GroupNorm 그룹 수
    /// * `use_attn_at`    - 각 해상도에서 attention 사용 여부
    /// * `timesteps`      - 디퓨전 타임스텝 수 T (기본: 1000)
    /// * `beta_start`     - β 스케줄 시작값 (기본: 1e-4)
    /// * `beta_end`       - β 스케줄 끝값 (기본: 0.02)
    pub fn new(
        image_channels: usize,
        image_size: usize,
        dim: usize,
        dim_mults: &[usize],
        resnet_groups: usize,
        use_attn_at: &[bool],
        timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> MlResult<Self> {
        let unet = Unet::new(
            dim,
            None,               // init_dim = dim
            None,               // out_dim = channels
            dim_mults,
            image_channels,
            resnet_groups,
            use_attn_at,
        )?;

        let scheduler = DDPMScheduler::linear_schedule(timesteps, beta_start, beta_end);

        Ok(Self {
            unet,
            scheduler,
            loss: MeanSquaredError::new()?,
            image_shape: vec![image_channels, image_size, image_size],
        })
    }

    /// 학습 한 스텝 수행 (추론 경로, gradient 추적 없음).
    ///
    /// ## 반환값
    ///
    /// MSE loss 스칼라 값 (모니터링용).
    ///
    /// ## Algorithm 1 (DDPM 학습) 의 한 반복
    ///
    /// ```text
    /// 입력: x₀ (원본 이미지 배치)
    ///
    /// 1. t ~ Uniform({0,...,T-1})     랜덤 타임스텝 선택
    /// 2. ε ~ N(0, I)                  노이즈 생성
    /// 3. x_t = q_sample(x₀, t, ε)    forward diffusion
    /// 4. ε_θ = unet(x_t, t)          노이즈 예측
    /// 5. loss = MSE(ε, ε_θ)          예측 오차
    /// ```
    pub fn compute_loss(&self, x_0: &dyn TensorBase) -> MlResult<f32> {
        let batch_size = x_0.shape()[0];

        // Step 1: 랜덤 타임스텝 — 배치 내 모든 샘플에 동일 t 적용 (간단화)
        let t = self.scheduler.sample_timestep();

        // Step 2: 가우시안 노이즈 생성 — ε ~ N(0, I)
        let noise = Tensor::randn(x_0.shape());

        // Step 3: Forward diffusion — x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
        let x_t = self.scheduler.q_sample(x_0, t, &noise)?;

        // Step 4: U-Net 노이즈 예측 — ε_θ(x_t, t)
        //
        // timestep t 를 [N, 1] 텐서로 변환 (SinusoidalPE 입력 형식)
        // 정규화: t / T 로 [0, 1) 범위에 매핑
        let t_normalized = t as f32 / self.scheduler.timesteps as f32;
        let t_tensor = GlobalTensor::from_vec(
            vec![t_normalized; batch_size],
            &[batch_size, 1],
        )?;
        let predicted_noise = self.unet.predict_with_t(&x_t, &t_tensor)?;

        // Step 5: MSE Loss — ‖ε - ε_θ‖²
        //
        // 왜 단순 MSE 가 작동하는가?
        // Ho et al. 은 변분 하한(ELBO)을 단순화하면
        // L_simple = E_t,x₀,ε[‖ε - ε_θ(x_t, t)‖²] 가 됨을 보임.
        // 이 단순한 목표가 실제로 더 좋은 샘플 품질을 냄.
        let loss_output = self.loss.forward(&[&noise, &predicted_noise])?;
        Ok(loss_output[0].data[0])
    }

    /// 이미지 생성 (샘플링).
    ///
    /// ## Algorithm 2 (DDPM 샘플링)
    ///
    /// ```text
    /// x_T ~ N(0, I)                        순수 노이즈에서 시작
    /// for t = T-1, T-2, ..., 0:
    ///     ε_θ = unet(x_t, t)              노이즈 예측
    ///     x_{t-1} = p_sample(x_t, ε_θ, t) 한 스텝 역방향 이동
    /// return x₀                            생성된 이미지
    /// ```
    ///
    /// * `batch_size` - 한 번에 생성할 이미지 수
    pub fn sample(&self, batch_size: usize) -> MlResult<GlobalTensor<f32>> {
        let mut shape = vec![batch_size];
        shape.extend_from_slice(&self.image_shape);

        self.scheduler.p_sample_loop(&shape, |x_t, t| {
            // timestep 정규화
            let t_normalized = t as f32 / self.scheduler.timesteps as f32;
            let t_tensor = GlobalTensor::from_vec(
                vec![t_normalized; batch_size],
                &[batch_size, 1],
            )?;
            self.unet.predict_with_t(x_t, &t_tensor)
        })
    }
}

// TODO(LatentDiffusion): DDPM 안정화 후 latent space 학습 구현 예정.
// Encoder/Decoder (VAE) + DDPMScheduler 를 조합한 Stable Diffusion 계열 아키텍처.
// 삭제 금지.
#[allow(dead_code)]
struct LatentDiffusion;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  UnsupervisedModel 구현 (학습 경로)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  Diffusion × UnsupervisedModel 어댑터                                      ║
// ║                                                                           ║
// ║  DDPM 은 자기지도(self-supervised) 학습이므로 타깃 `t` 를 외부에서          ║
// ║  받지 않는다. `UnsupervisedTrainer` 는 입력 `x` 만 전달하며,                 ║
// ║  모델이 내부에서 랜덤 노이즈 ε 를 생성한다:                                  ║
// ║                                                                           ║
// ║    - 입력: x₀ (원본 이미지)                                                 ║
// ║    - 타겟: ε (내부에서 랜덤 생성한 노이즈)                                    ║
// ║    - 예측: ε_θ(x_t, t) (U-Net 이 예측한 노이즈)                              ║
// ║    - 손실: ‖ε - ε_θ‖²                                                      ║
// ║                                                                           ║
// ║  이전에는 `TrainableModel` (지도학습용) 에 dummy target 을 넘기는 방식으로   ║
// ║  구현되어 있었으나, P3 에서 `UnsupervisedModel` 로 정식 이관.                ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

#[cfg(feature = "enableBackward")]
impl Diffusion {
    /// 학습 경로 forward pass — Algorithm 1 의 한 스텝.
    ///
    /// ```text
    /// 1. t ~ Uniform({0,...,T-1})              랜덤 타임스텝
    /// 2. ε ~ N(0, I)                           노이즈 생성
    /// 3. x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε     forward diffusion (Variable)
    /// 4. ε_θ = unet.forward_with_t(x_t, t)    노이즈 예측 (gradient 추적)
    /// 5. loss = MSE(ε, ε_θ)                    예측 오차
    /// ```
    ///
    /// ## 왜 Variable 경로가 필요한가?
    ///
    /// `compute_loss()` 는 `GlobalTensor` (no-grad) 경로로 모니터링용이지만,
    /// 실제 학습에서는 gradient 가 U-Net 파라미터까지 역전파되어야 함.
    /// `Variable` 은 연산 그래프에 기록되므로 `.backward()` 호출 시
    /// 자동으로 ∂L/∂θ 를 계산할 수 있음.
    pub fn forward_loss_diffusion(&mut self, x_0: &Variable) -> MlResult<(Variable, Variable)> {
        let batch_size = x_0.tensor().shape()[0];

        // Step 1: 랜덤 타임스텝 — t ~ Uniform({0, ..., T-1})
        let t = self.scheduler.sample_timestep();

        // Step 2: 가우시안 노이즈 — ε ~ N(0, I)
        //
        // U-Net 이 예측할 노이즈.
        // Variable 로 감싸서 MSE loss 의 backward 가 작동하도록 함.
        let noise = Variable::new(Tensor::randn(x_0.tensor().shape()));

        // Step 3: Forward diffusion (Variable 경로)
        //
        //   x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
        //
        // q_sample_variable 은 Mul + Add 연산자를 사용하여
        // gradient 가 x₀ → ... → loss 까지 흐를 수 있게 함.
        // (실제로는 x₀ 의 gradient 는 필요 없지만, noise 와 unet 파라미터의 gradient 가  요지)
        let x_t = self.scheduler.q_sample_variable(x_0, t, &noise)?;

        // Step 4: Timestep 정규화 + U-Net 순전파
        //
        // t 를 [0, 1) 범위로 정규화하여 SinusoidalPE 에 입력.
        // 왜 정규화? → timestep 의 절대값(0~999)보다 상대적 위치가
        // sinusoidal encoding 에 더 안정적인 입력을 제공.
        let t_normalized = t as f32 / self.scheduler.timesteps as f32;
        let t_var = Variable::new(
            Tensor::from_vec(vec![t_normalized; batch_size], &[batch_size, 1])?
        );

        // ε_θ(x_t, t) — gradient 추적되는 노이즈 예측
        let predicted_noise = self.unet.forward_with_t(&x_t, &t_var)?;

        // Step 5: MSE Loss — ‖ε - ε_θ‖²
        //
        //   L = (1/n) Σᵢ (εᵢ - ε_θᵢ)²
        //
        // .backward() 호출 시 ∂L/∂ε_θ = 2(ε_θ - ε)/n 이 계산되고,
        // chain rule 을 통해 U-Net 의 모든 파라미터 θ 까지 역전파됨.
        let loss = self.loss.apply_with_label(&[&noise, &predicted_noise], "mse_loss")?;

        Ok((predicted_noise, loss))
    }
}

/// UnsupervisedModel 구현 — `UnsupervisedTrainer::fit()` 과 통합.
///
/// ## 인터페이스 매핑
///
/// | UnsupervisedModel | DDPM 에서의 의미                          |
/// |-------------------|-------------------------------------------|
/// | `x` (입력)        | 원본 이미지 배치 x₀                       |
/// | `forward_loss()`  | Algorithm 1 전체 (q_sample → unet → MSE) |
/// | `predict_raw()`   | U-Net 추론 (dummy timestep)               |
/// | `params()`        | U-Net 의 모든 학습 파라미터               |
#[cfg(feature = "enableBackward")]
impl crate::legacy::trainer::UnsupervisedModel for Diffusion {
    fn forward_loss(
        &mut self,
        x: &Variable,
    ) -> MlResult<(Variable, Variable)> {
        self.forward_loss_diffusion(x)
    }

    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<GlobalTensor<f32>> {
        self.unet.predict(x)
    }
}

impl crate::legacy::trainer::TrainableModel for Diffusion {
    fn params(&self) -> Vec<&dyn crate::legacy::nn::Parameter> { self.unet.params() }
}
impl crate::legacy::trainer::CheckpointableModel for Diffusion {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  테스트
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
#[path = "../../../tests/nn/native_models/diffusion/mod_tests.rs"]
mod tests;
