pub(crate) mod data;
pub(crate) mod training;
pub(crate) mod evaluation;
pub(crate) mod utils;
pub(crate) mod config;


use serde::{Deserialize, Serialize};
use mnist::{MnistBuilder};
use std::{sync::Arc, fs::File, fmt};
use log::{debug, error, info, trace, warn};
use crate::{nn::{
    activation::Sigmoid,
    activation::Softmax
}, loss::CrossEntropyLoss, tensor::{
    Variable,
    Tensor,
    operators::Function,
    TensorBase,
    GlobalFunction
}, var_with_label, var_input, MlResult, scalar, MlError};
#[cfg(feature = "enableBackpropagation")]
use crate::tensor::{AutogradFunction, ComputationGraph};
use crate::tensor::operators::{Add, Matmul, Square, Sub, Sum};

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
    pub w3: Arc<Variable>, // shape = [output_node, hidden_node]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    pub b2: Arc<Variable>, // shape = [output_node, 1]
    pub b3: Arc<Variable>, // shape = [output_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    hidden_activation: GlobalFunction,
    output_activation: GlobalFunction,
    loss_function: GlobalFunction,
}

impl fmt::Debug for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl MLP {
    /// n_input : 입력 뉴런 개수
    /// n_hidden: 은닉 뉴런 개수
    /// n_output: 출력 뉴런 개수
    /// hidden_activation: 은닉층에 적용할 활성화 함수
    /// output_activation: 출력층에 적용할 활성화 함수
    pub fn new(
        n_input: usize,
        n_hidden_1: usize,
        n_hidden_2: usize,
        n_output: usize,
        hidden_activation: GlobalFunction,
        output_activation: GlobalFunction,
        loss_function: GlobalFunction,
    ) -> Self {
        // He 초기화 또는 Xavier 초기화와 같은 더 나은 가중치 초기화 방법을 고려할 수 있음
        let w1_data: Vec<f32> = (0..n_hidden_1 * n_input)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5) // 0을 중심으로 분포
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_hidden_1, n_input]).unwrap(),
            "weight_1"
        );

        let w2_data: Vec<f32> = (0..n_hidden_2 * n_hidden_1)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();
        let w2 = var_with_label!(
            Tensor::from_vec(w2_data, &[n_hidden_2, n_hidden_1]).unwrap(),
            "weight_2"
        );

        let w3_data: Vec<f32> = (0..n_output * n_hidden_2)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();
        let w3 = var_with_label!(
            Tensor::from_vec(w3_data, &[n_output, n_hidden_2]).unwrap(),
            "weight_3"
        );

        // bias 항들 초기화
        let b1_data: Vec<f32> = vec![0.0; n_hidden_1]; // 0으로 초기화하는 것이 일반적
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_hidden_1, 1]).unwrap(),
            "bias_1"
        );

        let b2_data: Vec<f32> = vec![0.0; n_hidden_2];
        let b2 = var_with_label!(
            Tensor::from_vec(b2_data, &[n_hidden_2, 1]).unwrap(),
            "bias_2"
        );

        let b3_data: Vec<f32> = vec![0.0; n_output];
        let b3 = var_with_label!(
            Tensor::from_vec(b3_data, &[n_output, 1]).unwrap(),
            "bias_3"
        );

        Self { w1, w2, w3, b1, b2, b3, hidden_activation, output_activation, loss_function }
    }
}