pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

use super::*;

pub trait Activation<Type: Debug + Clone>: Function + AutogradFunction<Type> {
    fn new() -> MlResult<Self> where Self: Sized {
        <Self as Function>::new()
    }
    fn apply(&mut self, input: &Arc<Variable>) -> MlResult<Variable> where Self: AutogradFunction<Type> {
        <Self as AutogradFunction<Type>>::apply(self, &[input])
    }
}

impl<T: Function + Clone + 'static> Activation<f32> for T {
    fn new() -> MlResult<Self> where Self: Sized {
        <Self as Function>::new()
    }
    fn apply(&mut self, input: &Arc<Variable>) -> MlResult<Variable> {
        <Self as AutogradFunction<f32>>::apply(self, &[input])
    }
}

pub struct Sigmoid { backend: Arc<dyn Backend> }
pub struct Tanh    { backend: Arc<dyn Backend> }
pub struct Relu    { backend: Arc<dyn Backend> }
pub struct Softmax { backend: Arc<dyn Backend> }