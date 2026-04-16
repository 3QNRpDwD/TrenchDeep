use super::*;

use crate::nn::activation::{SoftmaxOp, SigmoidOp};

pub mod logistic;
pub mod softmax;

pub struct LogisticRegression {
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
