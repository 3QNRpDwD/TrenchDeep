use super::*; // info, MlResult, Layer, Parameter, Variable, Matmul, Add, ... (from model/mod.rs)

// regression 하위 모듈 전용 import
use crate::nn::activation::Identity; // linear.rs

pub mod linear;

pub struct LinearRegression {
    pub w1: Variable,
    pub b1: Variable,
    activation: GlobalFunction,
    loss_function: GlobalFunction,
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
