pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

use super::*;

pub trait Activation<Type: Debug + Clone>: Function<Type> {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function<Type>>::new()
    }
}

impl<T: Function<f32> + Clone> Activation<f32> for T {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        <Self as Function<f32>>::new()
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