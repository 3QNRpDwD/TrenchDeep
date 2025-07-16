use super::*;

pub mod mlp;
pub mod regression;

pub trait Model {
    #[cfg(feature = "enableBackpropagation")]
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()>;
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, test_data: &Tensor) -> MlResult<GlobalTensor<f32>>;
    #[cfg(feature = "enableBackpropagation")]
    fn update(&self, lr: &dyn TensorBase) -> MlResult<()>;
    #[cfg(feature = "enableBackpropagation")]
    fn zero_grad(&mut self) -> MlResult<()>;
    fn save(&self, path: &str) -> MlResult<()>;
    fn load(&mut self, path: &str) -> MlResult<()>;
    fn get_loss(&self) -> f32;
    fn compute_total_error(&mut self, X: &[&Variable], T: &[&Variable]) -> MlResult<f32>;
}

use std::sync::Arc;
use crate::tensor::operators::Function;

pub struct MLP {
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    layer: Sequential,
    loss_function: Arc<dyn Function + Send + Sync>,
}

pub struct SoftmaxRegression {
    layer: Sequential,
    loss_function: Arc<CrossEntropyLoss>,
}

// impl std::fmt::Debug for MLP {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         writeln!(f, "MLP {{")?;
//         writeln!(f, "  w1.shape = {:?}, w2.shape = {:?}",
//                  self.w1.tensor().shape(),
// 
//                  self.w2.tensor().shape())?;
//         // 활성화 함수 정보 추가
//         writeln!(f, "  layer = {:?}", self.layer)?;
//         writeln!(f, "  loss_function = {}", self.loss_function.name())?;
//         writeln!(f, "}}")
//     }
// }