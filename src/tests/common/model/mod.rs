use crate::nn::activation::{SigmoidLayer, SoftmaxLayer};
use crate::nn::Layer;
use super::*;
pub mod mlp;
pub mod regression;

pub trait Model {
    #[cfg(feature = "enableBackpropagation")]
    fn train(&mut self, x_set: &[Arc<Variable>], t_set: &[Arc<Variable>], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()>;
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, x: &Arc<Variable>) -> MlResult<Arc<Variable>>;
    fn predict(&mut self, test_data: &Tensor) -> MlResult<GlobalTensor<f32>>;
    #[cfg(feature = "enableBackpropagation")]
    fn update(&mut self, lr: &dyn TensorBase) -> MlResult<()>;
    #[cfg(feature = "enableBackpropagation")]
    fn zero_grad(&mut self) -> MlResult<()>;
    fn save(&self, path: &str) -> MlResult<()>;
    fn load(&mut self, path: &str) -> MlResult<()>;
    fn get_loss(&self) -> f32;
    fn compute_total_error(&mut self, X: &[Arc<Variable>], T: &[Arc<Variable>]) -> MlResult<f32>;
}

pub struct MLP {
    pub w1: Arc<Variable>, // shape = [hidden_node, input_node]
    pub w2: Arc<Variable>, // shape = [output_node, hidden_node]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    pub b2: Arc<Variable>, // shape = [output_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    layer: crate::nn::Sequential,
    loss_function: GlobalFunction,
}

pub struct SoftmaxRegression {
    pub w1: Arc<Variable>, // shape = [hidden_node, input_node]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    activation: GlobalFunction,
    loss_function: GlobalFunction,
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  w1.shape = {:?}, w2.shape = {:?}",
                 self.w1.tensor().shape(),

                 self.w2.tensor().shape())?;
        // 활성화 함수 정보 추가
        writeln!(f, "  layer = {:?}", self.layer)?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}