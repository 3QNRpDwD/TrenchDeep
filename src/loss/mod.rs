mod diplay;
mod function;

use crate::tensor::operators::{Div, Function, Matmax, Sub};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use crate::backend::{Backend, CpuBackend, Device};
use crate::{register_operator, scalar, MlResult};
use crate::tensor::{AutogradFunction, GlobalFunction, NodeId, Tensor, TensorBase, Variable, NODE_ID_GEN, OPERATOR_STORAGE};

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

struct MeanSquaredError {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct MeanAbsoluteError {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct HuberLoss {
    backend: Arc<dyn Backend>, node_id: NodeId, delta: f32
}

struct BinaryCrossEntropy {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct CategoricalCrossEntropy {
    backend: Arc<dyn Backend>, node_id: NodeId
}

pub trait Loss<T: Debug + Clone>: Function<T> {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function<T>>::new()
    }
    fn loss(predict: Tensor<T>, target: Tensor<T>) -> MlResult<f32>;
}


impl Loss<f32> for MeanSquaredError {
    fn new() -> MlResult<GlobalFunction> { <Self as Function<f32>>::new() }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
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

impl Loss<f32> for MeanAbsoluteError {
    fn new() -> MlResult<GlobalFunction> { <Self as Function<f32>>::new() }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
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

impl Loss<f32> for HuberLoss {
    fn new() -> MlResult<GlobalFunction> { <Self as Function<f32>>::new() }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
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

impl Loss<f32> for BinaryCrossEntropy {
    fn new() -> MlResult<GlobalFunction> { <Self as Function<f32>>::new() }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
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

impl Loss<f32> for CategoricalCrossEntropy {
    fn new() -> MlResult<GlobalFunction> { <Self as Function<f32>>::new() }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
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