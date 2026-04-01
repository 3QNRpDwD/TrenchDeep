use super::*; // info, MlResult (from common/mod.rs)

// ── model 하위 모듈 공통 import ──────────────────────────────────────────────
use crate::{
    loss::{CrossEntropyLoss, SoftmaxCrossEntropyLoss},
    nn::{
        activation::{Sigmoid, Softmax},
        Layer,
        Linear,
        Model,
        Parameter,
        Sequential,
        Variable,
    },
    tensor::{
        AutogradFunction,
        GlobalFunction,
        GlobalTensor,
        operators::{Add, Function, Matmul},
        Tensor,
        TensorBase,
    },
    var_with_label,
};

pub mod classification;
pub mod regression;
pub mod generation;
pub mod language;

pub use self::classification::{MLP, LogisticRegression, SoftmaxRegression};
pub use self::regression::LinearRegression;
