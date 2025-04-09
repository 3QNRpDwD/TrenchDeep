mod functions;

use std::sync::Arc;
use crate::tensor::operators::Function;
use crate::tensor::TensorBase;

pub struct Sigmoid {
    exp: Arc<dyn Function<f32>>,

    #[cfg(all(feature = "enableBackpropagation"))]
    mul: Arc<dyn Function<f32>>,
    #[cfg(all(feature = "enableBackpropagation"))]
    sub: Arc<dyn Function<f32>>,
}
pub struct Tanh {
    exp: Arc<dyn Function<f32>>,
    sub: Arc<dyn Function<f32>>,
    div: Arc<dyn Function<f32>>,
    add: Arc<dyn Function<f32>>,

    #[cfg(all(feature = "enableBackpropagation"))]
    mul: Arc<dyn Function<f32>>,
}
pub struct Relu {
    #[cfg(all(feature = "enableBackpropagation"))]
    mul: Arc<dyn Function<f32>>
}