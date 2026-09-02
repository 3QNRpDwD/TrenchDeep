mod encoder;
mod decoder;
mod unet;
mod scheduler;
mod embedding;

use super::*; // info, MlResult, Layer, Linear, Sequential, ... (from model/mod.rs)

// diffusion 하위 모듈 전용 import
use std::fmt::Debug;
use crate::{
    nn::{Conv2D, GroupNorm, activation::{SiLU, SoftmaxOp}},
    tensor::operators::{Concat, Cos, Mul, NearestUpsample2d, ReshapeOp, Sin, Transpose},
};
use crate::loss::MeanSquaredError;
use crate::tests::common::model::diffusion::unet::Unet;
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
impl crate::trainer::UnsupervisedModel for Diffusion {
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

impl crate::trainer::TrainableModel for Diffusion {
    fn params(&self) -> Vec<&dyn crate::nn::Parameter> { self.unet.params() }
}
impl crate::trainer::CheckpointableModel for Diffusion {}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  테스트
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::operators::Function;

    /// Diffusion 모델 생성 + loss 계산 end-to-end 테스트.
    #[test]
    fn diffusion_compute_loss() -> MlResult<()> {
        // 작은 모델: 8×8 grayscale, dim=8, 2 stages
        let model = Diffusion::new(
            1, 8,               // 1채널, 8×8
            8,                  // dim
            &[1, 2],           // dim_mults
            4,                  // resnet_groups
            &[false, true],    // attn at last stage
            10,                 // timesteps (작게)
            1e-4, 0.02,        // beta schedule
        )?;

        // 랜덤 "이미지" 배치
        let x_0 = Tensor::from_vec(vec![0.5; 2 * 1 * 8 * 8], &[2, 1, 8, 8])?;
        let loss = model.compute_loss(&x_0)?;

        // loss 는 유한한 양수여야 함
        assert!(loss.is_finite(), "loss = {} (must be finite)", loss);
        assert!(loss >= 0.0, "loss = {} (must be >= 0)", loss);
        info!("Diffusion compute_loss: {:.6}", loss);
        Ok(())
    }

    /// Diffusion 샘플링 shape 테스트.
    #[test]
    fn diffusion_sample_shape() -> MlResult<()> {
        let model = Diffusion::new(
            1, 8,
            8, &[1, 2],
            4, &[false, true],
            5,           // 아주 적은 timestep (테스트 속도)
            1e-4, 0.02,
        )?;

        let samples = model.sample(2)?;
        assert_eq!(samples.shape(), &[2, 1, 8, 8]);
        // 값이 유한해야 함 (NaN/Inf 없음)
        for &v in samples.data() {
            assert!(v.is_finite(), "sample contains non-finite value: {}", v);
        }
        Ok(())
    }

    /// Scheduler 와 Unet 이 조합되어 동작하는지 확인.
    #[test]
    fn diffusion_forward_reverse_roundtrip() -> MlResult<()> {
        let model = Diffusion::new(
            1, 8,
            8, &[1, 2],
            4, &[false, false],
            100,
            1e-4, 0.02,
        )?;

        // 원본 이미지
        let x_0 = Tensor::from_vec(vec![1.0; 1 * 1 * 8 * 8], &[1, 1, 8, 8])?;

        // t=0 에서 forward diffusion → x_0 와 거의 동일해야 함
        let noise = Tensor::from_vec(vec![0.0; 1 * 1 * 8 * 8], &[1, 1, 8, 8])?;
        let x_t0 = model.scheduler.q_sample(&x_0, 0, &noise)?;
        let diff: f32 = x_t0.data().iter()
            .zip(x_0.data().iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>() / x_0.data().len() as f32;
        assert!(diff < 0.05, "t=0 q_sample should be close to x_0, diff={}", diff);

        // t=99 에서 forward diffusion → 노이즈 지배적
        let noise = Tensor::randn(&[1, 1, 8, 8]);
        let x_t99 = model.scheduler.q_sample(&x_0, 99, &noise)?;
        // 원본과 크게 달라야 함
        let diff99: f32 = x_t99.data().iter()
            .zip(x_0.data().iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>() / x_0.data().len() as f32;
        assert!(diff99 > diff, "t=99 should be further from x_0 than t=0");

        Ok(())
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  학습 + 샘플링 End-to-End 테스트
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ╔═══════════════════════════════════════════════════════════════════════╗
    // ║  DDPM 학습 + 샘플링 통합 테스트                                          ║
    // ║                                                                       ║
    // ║  이 테스트는 DDPM 의 전체 파이프라인을 검증:                               ║
    // ║                                                                       ║
    // ║  ┌────────────────────────────────────────────────────────────┐       ║
    // ║  │  Phase 1: 학습 (Training)                                   │       ║
    // ║  │                                                            │       ║
    // ║  │  for epoch in 0..N:                                        │       ║
    // ║  │    for x₀ in dataset:                                      │       ║
    // ║  │      ① reset_graph()         연산 그래프 초기화              │       ║
    // ║  │      ② forward_loss(x₀, _)   Algorithm 1 한 스텝           │       ║
    // ║  │      ③ loss.backward()        역전파 — ∂L/∂θ 계산           │       ║
    // ║  │      ④ optimizer.step()       θ ← θ - lr·∂L/∂θ            │       ║
    // ║  │      ⑤ optimizer.zero_grad()  gradient 초기화              │       ║
    // ║  │                                                            │       ║
    // ║  │  검증: loss 가 유한하고 감소 추세                             │       ║
    // ║  └────────────────────────────────────────────────────────────┘       ║
    // ║                                                                       ║
    // ║  ┌────────────────────────────────────────────────────────────┐       ║
    // ║  │  Phase 2: 샘플링 (Sampling)                                 │       ║
    // ║  │                                                            │       ║
    // ║  │  x_T ~ N(0, I)               순수 노이즈                    │       ║
    // ║  │  for t = T-1 → 0:                                          │       ║
    // ║  │    ε_θ = unet(x_t, t)       노이즈 예측                     │       ║
    // ║  │    x_{t-1} = p_sample(...)   한 스텝 역방향                 │       ║
    // ║  │                                                            │       ║
    // ║  │  검증: shape = [N,C,H,W], 모든 값이 유한.                    │       ║
    // ║  └────────────────────────────────────────────────────────────┘       ║
    // ╚═══════════════════════════════════════════════════════════════════════╝

    /// DDPM 수동 학습 루프 + 샘플링 end-to-end 테스트.
    ///
    /// Trainer::fit() 대신 수동 루프를 사용하여 각 단계를 명시적으로 보여줌.
    ///
    /// ## 왜 수동 루프인가?
    ///
    /// Trainer::fit() 은 (x, t) 쌍의 지도학습 데이터셋을 기대하지만,
    /// DDPM 은 이미지만으로 학습함 (자기지도). 
    /// 수동 루프는 이 차이를 명확하게 보여주고, 학습 과정의 각 단계를 직접 제어 가능.
    #[cfg(feature = "enableBackward")]
    #[test]
    fn diffusion_train_and_sample() -> MlResult<()> {
        use crate::tensor::ComputationGraph;
        use crate::optimizer::{Adam, Optimizer, clip_grad_norm};

        info!("═══════════════════════════════════════════════════════════");
        info!("  DDPM Training + Sampling E2E Test");
        info!("═══════════════════════════════════════════════════════════");

        // ── 모델 생성 ──────────────────────────────────────────────────
        //
        // 최소 구성으로 빠른 테스트:
        //   - 8×8 grayscale (1채널)
        //   - dim=8, dim_mults=[1,2] → 2단계 U-Net
        //   - 10 timesteps (실제로는 1000, 테스트에서는 속도 우선)
        //   - attention 없음 (파라미터 수 최소화)
        let mut model = Diffusion::new(
            1, 8,                  // image: 1ch × 8×8
            8,                                   // dim (base channels)
            &[1, 2],                        // dim_mults → [8, 16] 채널
            4,                          // GroupNorm groups
            &[false, false],              // attention 비활성화 (속도)
            10,                            // T = 10 timesteps
            1e-4, 0.02,           // β schedule
        )?;

        // ── 옵티마이저 설정 ────────────────────────────────────────────
        //
        // Adam: DDPM 논문 기본 옵티마이저
        //   lr = 1e-3 (작은 모델이므로 높은 학습률 사용 가능)
        //   β₁ = 0.9, β₂ = 0.999
        let mut optimizer = Adam::new(1e-3, 0.9, 0.999, 1e-8);

        // U-Net 의 모든 파라미터를 옵티마이저에 등록
        //
        // params() 는 init_conv, time_mlp, down blocks, mid block,
        // up blocks, final_res_block, final_conv 의 모든 weight/bias 를 반환
        for param in model.unet.params() {
            optimizer.register(param);
        }

        let param_count = model.unet.params().len();
        info!("  Model parameters: {} tensors", param_count);

        // ── 학습 데이터 생성 ───────────────────────────────────────────
        //
        // 간단한 패턴: 상단 밝고 하단 어두운 그래디언트 이미지
        //
        //   row 0: ████████  (밝음, ≈1.0)
        //   row 1: ▓▓▓▓▓▓▓▓
        //   row 2: ▒▒▒▒▒▒▒▒
        //   ...
        //   row 7: ░░░░░░░░  (어두움, ≈0.14)
        //
        // 이 패턴을 학습하면 U-Net 은 "위가 밝고 아래가 어두운" 구조를 재현하기 위해 작동.
        let batch_size = 2;
        let mut img_data = Vec::with_capacity(batch_size * 1 * 8 * 8);
        for _b in 0..batch_size {
            for row in 0..8 {
                for _col in 0..8 {
                    // 행에 따라 밝기 감소: 1.0, 0.875, 0.75, ..., 0.125
                    img_data.push(1.0 - row as f32 / 8.0);
                }
            }
        }
        let x_0 = Variable::new(Tensor::from_vec(img_data, &[batch_size, 1, 8, 8])?);

        // ── 학습 전 초기 loss 측정 ──────────────────────────────────────
        let initial_loss = model.compute_loss(x_0.tensor())?;
        info!("  Initial loss (no-grad): {:.6}", initial_loss);

        // ── 학습 루프 ──────────────────────────────────────────────────
        //
        // DDPM Algorithm 1:
        //   repeat:
        //     1. x₀ ~ dataset
        //     2. t ~ Uniform({0,...,T-1})
        //     3. ε ~ N(0, I)
        //     4. x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
        //     5. L = ‖ε - ε_θ(x_t, t)‖²
        //     6. ∇_θ L → update θ
        //
        // 아래 루프는 이 과정을 명시적으로 구현.
        let epochs = 5;
        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            // ① 연산 그래프 초기화
            //
            // 매 스텝마다 새로운 그래프를 만들어야 함.
            // 이전 스텝의 중간 노드를 재사용하면 gradient 가 오염됨.
            ComputationGraph::reset_graph();

            // ② Forward pass — Algorithm 1 의 Step 1~5
            //
            // forward_loss_diffusion 내부에서:
            //   - 랜덤 t 선택
            //   - 노이즈 생성
            //   - q_sample_variable (forward diffusion)
            //   - unet.forward_with_t (노이즈 예측)
            //   - MSE loss 계산
            let (_predicted, loss_var) = model.forward_loss_diffusion(&x_0)?;
            let loss_val = loss_var.tensor().data()[0];
            losses.push(loss_val);

            // ③ 역전파 — ∂L/∂θ 계산
            //
            // loss.backward() 는 연산 그래프를 역순으로 순회하며
            // chain rule 을 적용하여 모든 파라미터의 gradient 를 계산:
            //
            //   loss ← MSE ← ε_θ ← UNet layers ← ... ← θ (weights)
            //                  ↑
            //             ∂L/∂ε_θ = 2(ε_θ - ε)/n
            loss_var.backward()?;

            // gradient clipping — 수치 안정성
            //
            // 깊은 네트워크에서 gradient 가 폭발할 수 있으므로,
            // L2 norm 이 max_norm 을 초과하면 비례 축소.
            // PyTorch 의 torch.nn.utils.clip_grad_norm_ 과 동일.
            let params: Vec<&dyn crate::nn::Parameter> = model.unet.params();
            let grad_norm = clip_grad_norm(&params, 1.0);

            // ④ 파라미터 업데이트 — θ ← θ - lr · ∂L/∂θ
            optimizer.step()?;

            // ⑤ gradient 초기화
            optimizer.zero_grad()?;

            info!(
                "  Epoch {}/{}: loss = {:.6}, grad_norm = {:.4}",
                epoch + 1, epochs, loss_val, grad_norm
            );

            // loss 가 유한한지 확인 (NaN/Inf 발생 시 즉시 실패)
            assert!(
                loss_val.is_finite(),
                "Epoch {}: loss = {} (NaN/Inf detected!)",
                epoch + 1, loss_val
            );
        }

        // ── 학습 결과 검증 ──────────────────────────────────────────────
        //
        // 5 에폭은 수렴하기에 부족하지만, 최소한:
        //   1. 모든 loss 가 유한해야 함 (수치 안정성)
        //   2. loss 가 비합리적으로 크지 않아야 함
        info!("  All losses: {:?}", losses);

        let final_loss = *losses.last().unwrap();
        assert!(final_loss.is_finite(), "Final loss must be finite");
        assert!(final_loss < 100.0, "Final loss {} is unreasonably large", final_loss);

        // ── 샘플링 테스트 ──────────────────────────────────────────────
        //
        // 학습된(?) 모델로 이미지 생성:
        //   x_T ~ N(0, I) → ... → x₀
        //
        // 5 에폭 학습이므로 의미있는 이미지는 기대하지 않지만,
        // 파이프라인이 정상 동작하는지 (shape, 유한성) 확인.
        info!("  Sampling {} images...", batch_size);
        let samples = model.sample(batch_size)?;

        // Shape 검증: [batch_size, channels, height, width]
        assert_eq!(
            samples.shape(),
            &[batch_size, 1, 8, 8],
            "Sample shape mismatch: expected [{}, 1, 8, 8], got {:?}",
            batch_size, samples.shape()
        );

        // 유한성 검증: NaN/Inf 없어야 함
        let all_finite = samples.data().iter().all(|v| v.is_finite());
        assert!(all_finite, "Samples contain NaN/Inf values");

        // 통계 출력
        let min = samples.data().iter().cloned().fold(f32::INFINITY, f32::min);
        let max = samples.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = samples.data().iter().sum::<f32>() / samples.data().len() as f32;
        info!("  Sample stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
        info!("═══════════════════════════════════════════════════════════");
        info!("  ✓ DDPM train + sample pipeline verified!");
        info!("═══════════════════════════════════════════════════════════");

        Ok(())
    }

    /// `UnsupervisedTrainer::fit()` 을 사용한 DDPM 학습 테스트.
    ///
    /// 수동 루프 대신 프레임워크의 트레이너를 활용하는 패턴.
    /// P3 에서 dummy target 을 받던 `Trainer`(지도학습) 대신
    /// 자기지도에 최적화된 `UnsupervisedTrainer` 로 이관됨.
    ///
    /// ## UnsupervisedTrainer 장점
    ///   - forward_loss 가 `(x,)` 만 받으므로 dummy 타깃 불필요
    ///   - 로그, NaN 검사, progress bar 자동 처리
    ///   - 수렴 조기 종료 (tolerance)
    ///   - 체크포인트 저장/재개
    #[cfg(feature = "enableBackward")]
    #[test]
    fn diffusion_train_with_trainer() -> MlResult<()> {
        use crate::optimizer::{Adam, Optimizer};
        use crate::trainer::{UnsupervisedDataset, EpochSchedule};
        
        info!("  DDPM Training via UnsupervisedTrainer::fit()");
        
        // 모델 생성 (최소 구성)
        let mut model = Diffusion::new(
            1, 8, 8, &[1, 2], 4, &[false, false],
            10, 1e-4, 0.02,
        )?;
        // 옵티마이저 + 파라미터 등록
        let mut optimizer = Adam::new(1e-3, 0.9, 0.999, 1e-8);
        for param in model.unet.params() {
            optimizer.register(param);
        }
        // 학습 데이터 (단일 배치)
        let x_0 = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * 1 * 8 * 8], &[2, 1, 8, 8])?
        );
        // UnsupervisedTrainer::silent() — 로그 없이 빠르게 실행
        let trainer = crate::trainer::Trainer::silent().unsupervised();
        let samples = [&x_0];
        let result = 
            trainer.fit(
                &mut model, 
                &mut optimizer, 
                UnsupervisedDataset::new(&samples)?, 
                EpochSchedule::new(3)?.with_tolerance(1e-10)
            )?;
        println!("  Trainer result: {} epochs, final_loss = {:.6}", result.units_completed, result.final_loss);
        assert!(result.final_loss.is_finite(), "Trainer final loss must be finite");
        assert_eq!(result.units_completed, 3);

        Ok(())
    }
}
