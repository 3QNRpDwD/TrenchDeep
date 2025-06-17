pub(crate) mod data;
pub(crate) mod evaluation;
pub(crate) mod utils;
pub(crate) mod config;
pub mod model;

use serde::{Deserialize, Serialize};
use mnist::{MnistBuilder};
use std::{sync::Arc};
use log::{info, warn};
use crate::{
    nn::{
        activation::Sigmoid,
        activation::Softmax
    },
    var_with_label,
    var_input,
    MlResult,
    scalar,
    MlError,
    tensor::{
        AutogradFunction,
        ComputationGraph,
        Variable,
        Tensor,
        operators::Function,
        TensorBase,
        GlobalFunction,
        operators::{Add, Matmul}
    }
};
use rand::{rng, seq::SliceRandom};
use tracing_subscriber::{
    prelude::*,
    EnvFilter,
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt
};
use std::time::Instant;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use crate::loss::SoftmaxWithCrossEntropyLoss;
use time::macros::format_description;

pub trait Model {
    fn new(layer_parms: &[usize], activations: &[GlobalFunction], loss: GlobalFunction) -> Self
    where
        Self: Sized;
    #[cfg(feature = "enableBackpropagation")]
    fn train(&mut self, x_set: &[Arc<Variable>], t_set: &[Arc<Variable>], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()>;
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&self, x: &Arc<Variable>) -> MlResult<Arc<Variable>>;
    fn predict(&self, test_data: &Tensor) -> MlResult<Tensor>;
    #[cfg(feature = "enableBackpropagation")]
    fn update(&mut self, lr: &Tensor) -> MlResult<()>;
    #[cfg(feature = "enableBackpropagation")]
    fn zero_grad(&mut self) -> MlResult<()>;
    fn save(&self, path: &str) -> MlResult<()>;
    fn load(&mut self, path: &str) -> MlResult<()>;
    fn get_loss(&self) -> f32;
}

pub trait Data {

}

#[derive(Serialize, Deserialize)]
struct ModelParameters {
    w1_data: Vec<f32>,
    w1_shape: Vec<usize>,
    b1_data: Vec<f32>,
    b1_shape: Vec<usize>,
    w2_data: Vec<f32>,
    w2_shape: Vec<usize>,
    b2_data: Vec<f32>,
    b2_shape: Vec<usize>,
}

pub struct MLP {
    pub w1: Arc<Variable>, // shape = [hidden_node, input_node]
    pub w2: Arc<Variable>, // shape = [output_node, hidden_node]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    pub b2: Arc<Variable>, // shape = [output_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    hidden_activation: GlobalFunction,
    output_activation: GlobalFunction,
    loss_function: GlobalFunction,
}

pub struct SoftmaxRegression {
    pub w1: Arc<Variable>, // shape = [hidden_node, input_node]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    output_activation: GlobalFunction,
    loss_function: GlobalFunction,
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  w1.shape = {:?}, w2.shape = {:?}",
                 self.w1.tensor().shape(),

                 self.w2.tensor().shape())?;
        // 활성화 함수 정보 추가
        writeln!(f, "  hidden_activation = {}", self.hidden_activation.name())?;
        writeln!(f, "  output_activation = {}", self.output_activation.name())?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}