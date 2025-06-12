use crate::tensor::operators::Function;
use std::fmt::Debug;
use std::sync::Arc;
use crate::backend::Backend;
use crate::tensor::NodeId;

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

    fn forward(&self, input: &T, target: &T) -> T;
}