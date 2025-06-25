mod display;
mod function;

use crate::tensor::operators::{Function, Matmax, Sub};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use crate::backend::{Backend, CpuBackend, Device};
use crate::{register_operator, scalar, MlResult};
use crate::tensor::{AutogradFunction, GlobalFunction, HandleId, Tensor, TensorBase, NODE_ID_GEN, OPERATOR_STORAGE, GlobalTensor};

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