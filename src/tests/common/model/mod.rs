pub(crate) use super::*; // info, MlResult (from common/mod.rs)

// ── model 하위 모듈 공통 import ──────────────────────────────────────────────
pub(crate) use crate::{
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

pub mod mlp;
pub mod regression;
mod diffusion;

pub struct MLP {
    pub layer: Sequential,
    loss_function: GlobalFunction,
}

pub struct LogisticRegression {
    pub w1: Variable,
    pub b1: Variable,
    activation: GlobalFunction,
    loss_function: GlobalFunction,
}

pub struct LinearRegression {
    pub w1: Variable,
    pub b1: Variable,
    activation: GlobalFunction,
    loss_function: GlobalFunction,
}

pub struct SoftmaxRegression {
    pub w1: Variable,
    pub b1: Variable,
    activation: GlobalFunction,
    loss_function: GlobalFunction,
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  layer = {:?}", self.layer)?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}

impl std::fmt::Debug for LinearRegression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LinearRegression {{")?;
        writeln!(f, "  w1 = {:?}", self.w1)?;
        writeln!(f, "  b1 = {:?}", self.b1)?;
        writeln!(f, "  activation = {}", self.activation.name())?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}

impl std::fmt::Debug for LogisticRegression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LogisticRegression {{")?;
        writeln!(f, "  w1 = {:?}", self.w1)?;
        writeln!(f, "  b1 = {:?}", self.b1)?;
        writeln!(f, "  activation = {}", self.activation.name())?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}

impl std::fmt::Debug for SoftmaxRegression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SoftmaxRegression {{")?;
        writeln!(f, "  w1 = {:?}", self.w1)?;
        writeln!(f, "  b1 = {:?}", self.b1)?;
        writeln!(f, "  activation = {}", self.activation.name())?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}
