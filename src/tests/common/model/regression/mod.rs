pub(crate) use super::*; // info, MlResult, Layer, Parameter, Variable, Matmul, Add, ... (from model/mod.rs)

// regression 하위 모듈 전용 import
pub(crate) use crate::nn::activation::Identity; // linear.rs

pub mod softmax;
pub mod linear;
pub mod logistic;
