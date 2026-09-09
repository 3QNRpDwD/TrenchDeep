use super::*; // info, MlResult (from common/mod.rs)

// ── model 하위 모듈 공통 import ──────────────────────────────────────────────
use crate::legacy::{
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

#[path = "linear/mod.rs"]
pub mod linear;
#[path = "nonlinear/mod.rs"]
pub mod nonlinear;
#[path = "mlp/mod.rs"]
pub mod mlp;
#[path = "diffusion/mod.rs"]
pub mod diffusion;
#[path = "transformer/mod.rs"]
pub mod transformer;
#[path = "semi_supervised/mod.rs"]
pub mod semi_supervised;
#[path = "reinforcement/mod.rs"]
pub mod reinforcement;
#[path = "autoregressive/mod.rs"]
pub mod autoregressive;

pub use self::linear::LinearRegression;
pub use self::nonlinear::{LogisticRegression, SoftmaxRegression};
pub use self::mlp::MLP;
