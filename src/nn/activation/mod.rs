pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

use super::*;

pub trait Activation<Type: Debug + Clone>: Function<Type> + AutogradFunction<Type> {
    fn new() -> MlResult<Self> where Self: Sized {
        <Self as Function<Type>>::new()
    }
    fn apply(&mut self, input: &Arc<Variable<Type>>) -> MlResult<Arc<Variable<Type>>> where Self: AutogradFunction<Type> {
        <Self as AutogradFunction<Type>>::apply(self, &[input])
    }
}

impl<T: Function<f32> + Clone + 'static> Activation<f32> for T {
    fn new() -> MlResult<Self> where Self: Sized {
        <Self as Function<f32>>::new()
    }
    fn apply(&mut self, input: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        <Self as AutogradFunction<f32>>::apply(self, &[input])
    }
}

pub struct Sigmoid { backend: Arc<dyn Backend> }
pub struct Tanh    { backend: Arc<dyn Backend> }
pub struct Relu    { backend: Arc<dyn Backend> }
pub struct Softmax { backend: Arc<dyn Backend> }