use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use crate::{
    define_op,
    MlResult,
    scalar,
    backend::{
        Backend,
        CpuBackend,
        Device
    },
    tensor::{
        HandleId,
        NODE_ID_GEN,
        PooledTensor,
        Tensor,
        TensorBase,
        operators::{Function, OPERATOR_STORAGE}
    },
    MlError,
    TensorError::InvalidInputCount
};

mod display;
mod function;

const EPSILON: f32 = 1e-15;

#[derive(Debug, Clone)]
pub enum LossError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidOperation {
        op: &'static str,
        reason: String,
    },
}

define_op!(MeanSquaredError);
define_op!(MeanAbsoluteError);
define_op!(HuberLoss, delta: f32);
define_op!(BinaryCrossEntropyLoss);
define_op!(CrossEntropyLoss);
define_op!(SoftmaxCrossEntropyLoss);

pub trait Loss: Function {
    fn loss(&self, predict: Tensor, target: Tensor) -> MlResult<f32> {
        Ok(*<Self as Function>::forward(&self, &[&predict, &target])?.remove(0).data().first().unwrap())
    }
}


impl Loss for MeanSquaredError {}

impl Loss for MeanAbsoluteError {}

impl Loss for HuberLoss {}

impl Loss for BinaryCrossEntropyLoss {}

impl Loss for CrossEntropyLoss {}