pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

use super::*;

pub trait Activation: Layer {}

impl<T: Layer> Activation for T {}

#[derive(Debug, Clone)]
pub struct Sigmoid { 
    backend: Arc<dyn Backend>, node_id: NodeId
}

#[derive(Debug, Clone)]
pub struct SigmoidLayer {
    label: String,
    operator: GlobalFunction
}

#[derive(Debug, Clone)]
pub struct Tanh {
    backend: Arc<dyn Backend>, node_id: NodeId
}

#[derive(Debug, Clone)]
pub struct TanhLayer {
    label: String,
    operator: GlobalFunction
}

#[derive(Debug, Clone)]
pub struct ReLU {
    backend: Arc<dyn Backend>, node_id: NodeId
}

#[derive(Debug, Clone)]
pub struct ReLULayer {
    label: String,
    operator: GlobalFunction
}

#[derive(Debug, Clone)]
pub struct Softmax {
    backend: Arc<dyn Backend>, node_id: NodeId
}

#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    label: String,
    operator: GlobalFunction
}
