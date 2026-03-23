pub(crate) mod logging;
pub(crate) mod utils;
pub(crate) mod data;
pub mod model;

use crate::{
    nn::{
        activation::Softmax,
        Variable,
        Parameter,
        Layer,
        Sequential
    },
    var_with_label,
    MlResult,
    tensor::{
        AutogradFunction,
        Tensor,
        operators::Function,
        TensorBase,
        GlobalFunction,
        GlobalTensor,
        operators::{Add, Matmul}
    },
    loss::{CrossEntropyLoss, SoftmaxCrossEntropyLoss},
};
