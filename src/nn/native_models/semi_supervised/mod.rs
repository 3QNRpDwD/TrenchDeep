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

use crate::legacy::{
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
impl crate::legacy::trainer::SemiSupervisedModel for PiToyClassifier {
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

impl crate::legacy::trainer::TrainableModel for PiToyClassifier {
    fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w1, &self.b1] }
}
impl crate::legacy::trainer::CheckpointableModel for PiToyClassifier {}

// ────────────────────────────────────────────────────────────────────────────
// 테스트
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "enableBackward")]
#[path = "../../../tests/nn/native_models/semi_supervised/mod_tests.rs"]
mod tests;
