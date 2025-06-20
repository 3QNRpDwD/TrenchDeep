mod display;
mod function;

use crate::tensor::operators::{Function, Matmax, Sub};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use crate::backend::{Backend, CpuBackend, Device};
use crate::{register_operator, scalar, MlResult};
use crate::tensor::{AutogradFunction, GlobalFunction, NodeId, Tensor, TensorBase, Variable, NODE_ID_GEN, OPERATOR_STORAGE, GlobalTensor};

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
    backend: Arc<dyn Backend>, node_id: NodeId,
    #[cfg(feature = "enableBackpropagation")]
    inputs: Vec<Tensor>,
    #[cfg(feature = "enableBackpropagation")]
    outputs: Vec<Tensor>,
}

pub struct MeanAbsoluteError {
    backend: Arc<dyn Backend>, node_id: NodeId,
    #[cfg(feature = "enableBackpropagation")]
    inputs: Vec<Tensor>,
    #[cfg(feature = "enableBackpropagation")]
    outputs: Vec<Tensor>,
}

pub struct HuberLoss {
    backend: Arc<dyn Backend>, node_id: NodeId, delta: f32,
    #[cfg(feature = "enableBackpropagation")]
    inputs: Vec<Tensor>,
    #[cfg(feature = "enableBackpropagation")]
    outputs: Vec<Tensor>,
}

pub struct BinaryCrossEntropyLoss {
    backend: Arc<dyn Backend>, node_id: NodeId,
    #[cfg(feature = "enableBackpropagation")]
    inputs: Vec<Tensor>,
    #[cfg(feature = "enableBackpropagation")]
    outputs: Vec<Tensor>,
}

pub struct CrossEntropyLoss {
    backend: Arc<dyn Backend>, node_id: NodeId,
    #[cfg(feature = "enableBackpropagation")]
    inputs: Vec<Tensor>,
    #[cfg(feature = "enableBackpropagation")]
    outputs: Vec<Tensor>,
}

pub struct SoftmaxWithCrossEntropyLoss {
    backend: Arc<dyn Backend>, node_id: NodeId,
    #[cfg(feature = "enableBackpropagation")]
    inputs: Vec<Tensor>,
    #[cfg(feature = "enableBackpropagation")]
    outputs: Vec<Tensor>,
}

pub trait Loss: Function {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function>::new()
    }
    fn loss(predict: Tensor, target: Tensor) -> MlResult<f32>;
}


impl Loss for MeanSquaredError {
    fn loss(predict: Tensor, target: Tensor) -> MlResult<f32> {
        if predict.shape() != target.shape() {
            return Err(LossError::InvalidShape { expected: predict.shape().to_vec(), got: target.shape().to_vec() }.into());
        }
        let n = predict.data().len() as f32;
        let squared_error = predict.data().iter().zip(target.data().iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>();
        Ok(squared_error / n)
    }
}

impl Loss for MeanAbsoluteError {
    fn loss(predict: Tensor, target: Tensor) -> MlResult<f32> {
        if predict.shape() != target.shape() {
            return Err(LossError::InvalidShape { expected: predict.shape().to_vec(), got: target.shape().to_vec() }.into());
        }
        let n = predict.data().len() as f32;
        let abs_error = predict.data().iter().zip(target.data().iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum::<f32>();
        Ok(abs_error / n)
    }
}

impl Loss for HuberLoss {
    fn loss(predict: Tensor, target: Tensor) -> MlResult<f32> {
        if predict.shape() != target.shape() {
            return Err(LossError::InvalidShape { expected: predict.shape().to_vec(), got: target.shape().to_vec() }.into());
        }
        let n = predict.data().len() as f32;
        let delta = 1.0; // Default delta
        let huber_error = predict.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let diff = (p - t).abs();
            if diff <= delta { 0.5 * diff.powi(2) } else { delta * (diff - 0.5 * delta) }
        }).sum::<f32>();
        Ok(huber_error / n)
    }
}

impl Loss for BinaryCrossEntropyLoss {
    fn loss(predict: Tensor, target: Tensor) -> MlResult<f32> {
        if predict.shape() != target.shape() {
            return Err(LossError::InvalidShape { expected: predict.shape().to_vec(), got: target.shape().to_vec() }.into());
        }
        let n = predict.data().len() as f32;
        let bce_loss = predict.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            - (t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln())
        }).sum::<f32>();
        Ok(bce_loss / n)
    }
}

impl Loss for CrossEntropyLoss {
    fn loss(predict: Tensor, target: Tensor) -> MlResult<f32> {
        if predict.shape() != target.shape() {
            return Err(LossError::InvalidShape { expected: predict.shape().to_vec(), got: target.shape().to_vec() }.into());
        }
        let batch_size = if predict.shape().len() > 1 { predict.shape()[0] } else { 1 } as f32;
        let cce_loss = predict.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON);
            - t * p_clipped.ln()
        }).sum::<f32>();
        Ok(cce_loss / batch_size)
    }
}