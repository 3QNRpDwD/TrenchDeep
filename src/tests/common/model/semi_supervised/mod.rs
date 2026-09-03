//! 반지도학습 파일럿: **Pi-model 스타일 이진 분류기**.
//!
//! 이 모듈은 P4 에서 도입된 `SemiSupervisedTrainer` + `SemiSupervisedModel`
//! 인터페이스가 end-to-end 로 동작함을 최소 예제로 증명한다.
//!
//! ## 구조
//!
//! - 단층 선형 모델: `y = Wx + b`, `(n_input=2) → (n_output=2)` 이진 분류
//! - 지도 손실: `SoftmaxCrossEntropyLoss(y_l, t_l)`
//! - 일관성 손실: `MSE(f(x_u + ε₁), f(x_u + ε₂))` — Pi-model (Laine & Aila, 2017)
//! - 총 손실: `sup + λ · con` (λ 는 트레이너의 `ConsistencyRamp` 로 결정)
//!
//! 실제 논문은 dropout/augmentation 기반 stochastic forward pass 를 쓰지만,
//! 여기서는 **입력 가우시안 노이즈** 로 대체해 의존성을 최소화했다.

use super::*;

use crate::{
    loss::{MeanSquaredError, SoftmaxCrossEntropyLoss},
    nn::Variable,
    tensor::{
        operators::{Mul, Function},
        GlobalFunction,
        GlobalTensor,
        Tensor,
        TensorBase,
    },
    var_with_label,
    MlResult,
};

// ────────────────────────────────────────────────────────────────────────────
// PiToyClassifier
// ────────────────────────────────────────────────────────────────────────────

/// 반지도학습 파일럿용 장난감 이진 분류기.
pub struct PiToyClassifier {
    pub w1: Variable,
    pub b1: Variable,
    sup_loss: GlobalFunction,
    con_loss: GlobalFunction,
    noise_scale: f32,
}

impl PiToyClassifier {
    /// `(n_input → n_output)` 선형층 + Pi-model 일관성 손실용 MSE 를 구성.
    pub fn new(n_input: usize, n_output: usize, noise_scale: f32) -> MlResult<Self> {
        let sup_loss = SoftmaxCrossEntropyLoss::new()?;
        let con_loss = MeanSquaredError::new()?;

        let w1_data: Vec<f32> = (0..n_input * n_output)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_input, n_output])?,
            "pi_weight"
        );
        let b1_data: Vec<f32> = vec![0.0; n_output];
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_output])?,
            "pi_bias"
        );

        Ok(Self { w1, b1, sup_loss, con_loss, noise_scale })
    }

    #[cfg(feature = "enableBackward")]
    fn forward_pass(&self, x: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;
        let pre = matmul.apply(&[x, &self.w1])?;
        Ok(&pre + &self.b1)
    }

    #[cfg(feature = "enableBackward")]
    fn add_noise(&self, x: &Variable) -> MlResult<Variable> {
        let data_len = x.tensor().data().len();
        let shape    = x.tensor().shape().to_vec();
        let noise: Vec<f32> = (0..data_len)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * self.noise_scale)
            .collect();
        let noise_var = Variable::new(Tensor::from_vec(noise, &shape)?);
        Ok(x + &noise_var)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SemiSupervisedModel impl
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "enableBackward")]
impl crate::trainer::SemiSupervisedModel for PiToyClassifier {
    fn forward_loss(
        &mut self,
        x_l: &Variable,
        t_l: &Variable,
        x_u: &Variable,
        lambda: f32,
    ) -> MlResult<(Variable, Variable)> {
        // ── 지도 손실 ───────────────────────────────────────────────────
        let y_l   = self.forward_pass(x_l)?;
        let l_sup = self.sup_loss.apply_with_label(&[&y_l, t_l], "pi_sup")?;

        // ── 일관성 손실: 동일 입력에 두 번 다른 노이즈로 forward ─────────
        let x_u1  = self.add_noise(x_u)?;
        let x_u2  = self.add_noise(x_u)?;
        let y_u1  = self.forward_pass(&x_u1)?;
        let y_u2  = self.forward_pass(&x_u2)?;
        let l_con = self.con_loss.apply_with_label(&[&y_u1, &y_u2], "pi_con")?;

        // ── 결합: total = sup + λ · con ────────────────────────────────
        let lambda_var = Variable::new(Tensor::from_vec(vec![lambda], &[1, 1])?);
        let scaled_con = Mul::new()?.apply(&[&l_con, &lambda_var])?;
        let total = &l_sup + &scaled_con;

        Ok((y_l, total))
    }

    fn predict_raw(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let add    = Add::new()?;
        let pre    = matmul.forward(&[x, self.w1.tensor()])?.remove(0);
        let y      = add.forward(&[&pre, self.b1.tensor()])?.remove(0);
        Ok(y)
    }
}

impl crate::trainer::TrainableModel for PiToyClassifier {
    fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w1, &self.b1] }
}
impl crate::trainer::CheckpointableModel for PiToyClassifier {}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "enableBackward")]
mod tests {
    use super::*;
    use crate::{
        optimizer::{Adam, Optimizer},
        trainer::{ConsistencyRamp, SemiSupervisedModel, Trainer, TrainableModel},
    };

    /// labeled 4 개 + unlabeled 8 개로 Pi-model 학습이 **발산 없이** 동작하는지 확인.
    /// 파일럿의 목적은 트레이너 인터페이스 검증이지 정확도 달성이 아니다.
    #[test]
    fn pi_model_pilot_runs() -> MlResult<()> {
        // ── 모델 + 옵티마이저 ────────────────────────────────────────────
        let mut model = PiToyClassifier::new(2, 2, 0.1)?;
        let mut opt   = Adam::new(1e-2, 0.9, 0.999, 1e-8);
        for p in model.params() {
            opt.register(p);
        }

        // ── 데이터 (2D 이진 분류: 사분면 기반) ──────────────────────────
        // class 0 : (+1, +1) 근처
        // class 1 : (-1, -1) 근처
        let labeled = vec![
            ([ 1.0,  1.0], [1.0, 0.0]),
            ([ 0.9,  1.1], [1.0, 0.0]),
            ([-1.0, -1.0], [0.0, 1.0]),
            ([-1.1, -0.9], [0.0, 1.0]),
        ];
        // unlabeled: 두 군집 주변에 흩뿌린 8 개 점
        let unlabeled = vec![
            [ 1.2,  0.8], [ 0.7,  1.3], [ 1.1,  1.0], [ 0.8,  0.9],
            [-1.2, -0.8], [-0.7, -1.3], [-1.1, -1.0], [-0.8, -0.9],
        ];
        let labeled_dataset = crate::trainer::DatasetBuilder::from_source(
            crate::trainer::MemorySource::new(labeled),
        )
        .map(|(input, target): ([f32; 2], [f32; 2])| Ok(crate::trainer::SupervisedSample::new(
            Tensor::from_vec(input.to_vec(), &[2])?,
            Tensor::from_vec(target.to_vec(), &[2])?,
        )))
        .build()?;
        let unlabeled_dataset = crate::trainer::DatasetBuilder::from_source(
            crate::trainer::MemorySource::new(unlabeled),
        )
        .map(|input: [f32; 2]| Ok(crate::trainer::UnsupervisedSample::new(
            Tensor::from_vec(input.to_vec(), &[2])?,
        )))
        .build()?;
        let mut loader = crate::trainer::SemiSupervisedDataLoader::builder(
            labeled_dataset,
            unlabeled_dataset,
        )
        .labeled_collator(crate::trainer::SupervisedStackCollator::new())
        .unlabeled_collator(crate::trainer::UnsupervisedStackCollator::new())
        .labeled_batch_size(2)
        .unlabeled_batch_size(4)
        .build()?;

        // ── 트레이너: silent + 짧은 램프 ────────────────────────────────
        let trainer = Trainer::silent().semi_supervised()
            .with_ramp(ConsistencyRamp::Sigmoid { max_weight: 1.0, ramp_epochs: 5 });

        let result = trainer.fit(&mut model, &mut opt,
            &mut loader,
            crate::trainer::EpochSchedule::new(8)?.with_tolerance(1e-10))?;

        // ── 검증 ────────────────────────────────────────────────────────
        assert!(result.units_completed > 0, "적어도 1 에폭은 학습되어야 함");
        assert!(
            result.final_loss.is_finite(),
            "최종 손실이 유한해야 함: got {}", result.final_loss
        );
        assert!(
            result.final_loss >= 0.0,
            "손실은 음이 아니어야 함: got {}", result.final_loss
        );

        Ok(())
    }

    /// `ConsistencyRamp::Constant(0.0)` 이면 일관성 손실 기여가 0 이어야 하고,
    /// 파일럿이 그 경우도 정상 동작하는지 확인.
    #[test]
    fn pi_model_pilot_zero_ramp_is_supervised_only() -> MlResult<()> {
        let mut model = PiToyClassifier::new(2, 2, 0.1)?;
        let mut opt   = Adam::new(1e-2, 0.9, 0.999, 1e-8);
        for p in model.params() {
            opt.register(p);
        }

        let x_l = Variable::new(Tensor::from_vec(vec![1.0, 1.0], &[1, 2])?);
        let t_l = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[1, 2])?);
        let x_u = Variable::new(Tensor::from_vec(vec![0.5, 0.5], &[1, 2])?);

        let x_l_slice = [&x_l];
        let t_l_slice = [&t_l];
        let x_u_slice = [&x_u];

        let trainer = Trainer::silent().semi_supervised()
            .with_ramp(ConsistencyRamp::Constant(0.0));

        let result = trainer.fit(&mut model, &mut opt,
            crate::trainer::SemiSupervisedDataset::new(&x_l_slice, &t_l_slice, &x_u_slice)?,
            crate::trainer::EpochSchedule::new(3)?.with_tolerance(1e-10))?;

        assert!(result.final_loss.is_finite());
        Ok(())
    }
}
