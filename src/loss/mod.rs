use crate::tensor::operators::Function;
use std::fmt::Debug;
use std::sync::Arc;
use crate::backend::Backend;
use crate::MlResult;
use crate::tensor::{NodeId, Tensor, Variable};

struct MeanSquaredError {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct MeanAbsoluteError {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct HuberLoss {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct BinaryCrossEntropy {
    backend: Arc<dyn Backend>, node_id: NodeId
}

struct CategoricalCrossEntropy {
    backend: Arc<dyn Backend>, node_id: NodeId
}

pub trait Loss<T: Debug + Clone>: Function<T> {
    fn new() -> Self;
    fn loss(predict: Tensor<T>, target: Tensor<T>) -> MlResult<f32>;
}

impl Function<f32> for MeanSquaredError {}

impl Function<f32> for MeanAbsoluteError {}

impl Function<f32> for HuberLoss {}

impl Function<f32> for BinaryCrossEntropy {}

impl Function<f32> for CategoricalCrossEntropy {}

impl Loss<f32> for MeanSquaredError {
    fn new() -> Self {
        todo!()
    }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
        todo!()
    }
}

impl Loss<f32> for MeanAbsoluteError {
    fn new() -> Self {
        todo!()
    }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
        todo!()
    }
}

impl Loss<f32> for HuberLoss {
    fn new() -> Self {
        todo!()
    }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
        todo!()
    }
}

impl Loss<f32> for BinaryCrossEntropy {
    fn new() -> Self {
        todo!()
    }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
        todo!()
    }
}

impl Loss<f32> for CategoricalCrossEntropy {
    fn new() -> Self {
        todo!()
    }

    fn loss(predict: Tensor<f32>, target: Tensor<f32>) -> MlResult<f32> {
        todo!()
    }
}