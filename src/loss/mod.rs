use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use crate::{
    register_operator,
    MlResult,
    scalar,
    backend::{
        Backend,
        CpuBackend,
        Device
    },
    tensor::{
        GlobalFunction,
        HandleId,
        NODE_ID_GEN,
        OPERATOR_STORAGE,
        PooledTensor,
        Tensor,
        TensorBase,
        operators::Function
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

pub struct MeanSquaredError {
    backend: Arc<dyn Backend>, node_id: HandleId,
}

pub struct MeanAbsoluteError {
    backend: Arc<dyn Backend>, node_id: HandleId,
}

pub struct HuberLoss {
    backend: Arc<dyn Backend>, node_id: HandleId, delta: f32,
}

pub struct BinaryCrossEntropyLoss {
    backend: Arc<dyn Backend>, node_id: HandleId,
}

pub struct CrossEntropyLoss {
    backend: Arc<dyn Backend>, node_id: HandleId,
}

pub struct SoftmaxCrossEntropyLoss {
    backend: Arc<dyn Backend>, node_id: HandleId,
}

pub trait Loss: Function {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function>::new()
    }
    fn loss(&self, predict: Tensor, target: Tensor) -> MlResult<f32> {
        Ok(*<Self as Function>::forward(&self, &[&predict, &target])?.remove(0).data().first().unwrap())
    }
}


impl Loss for MeanSquaredError {}

impl Loss for MeanAbsoluteError {}

impl Loss for HuberLoss {}

impl Loss for BinaryCrossEntropyLoss {}

impl Loss for CrossEntropyLoss {}