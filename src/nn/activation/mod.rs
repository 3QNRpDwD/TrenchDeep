use super::*;

pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

pub trait Activation: Layer {}

impl Activation for SoftmaxLayer {}
impl Activation for SigmoidLayer {}
impl Activation for ReLULayer {}
impl Activation for TanhLayer {}

#[derive(Debug, Clone)]
pub struct Sigmoid { 
    backend: Arc<dyn Backend>, node_id: HandleId
}

#[derive(Debug, Clone)]
pub struct SigmoidLayer {
    label: String,
    operator: GlobalFunction
}

#[derive(Debug, Clone)]
pub struct Tanh    { 
    backend: Arc<dyn Backend>, node_id: HandleId
}

#[derive(Debug, Clone)]
pub struct TanhLayer {
    label: String,
    operator: GlobalFunction
}

#[derive(Debug, Clone)]
pub struct ReLU { 
    backend: Arc<dyn Backend>, node_id: HandleId
}

#[derive(Debug, Clone)]
pub struct ReLULayer {
    label: String,
    operator: GlobalFunction
}

#[derive(Debug, Clone)]
pub struct Softmax { 
    backend: Arc<dyn Backend>, node_id: HandleId
}

#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    label: String,
    operator: GlobalFunction
}
