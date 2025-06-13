pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

use super::*;

pub trait Activation: Function {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function>::new()
    }
}

impl<T: Function> Activation for T {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function>::new()
    }
}

#[derive(Debug, Clone)]
pub struct Sigmoid { backend: Arc<dyn Backend>, node_id: NodeId }

#[derive(Debug, Clone)]
pub struct Tanh    { backend: Arc<dyn Backend>, node_id: NodeId }

#[derive(Debug, Clone)]
pub struct ReLu    { backend: Arc<dyn Backend>, node_id: NodeId }

#[derive(Debug, Clone)]
pub struct Softmax { backend: Arc<dyn Backend>, node_id: NodeId }