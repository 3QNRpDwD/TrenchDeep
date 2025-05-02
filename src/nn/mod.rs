pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;

use crate::{
    backend::{
        Backend,
        CpuBackend,
        Device
    },
    tensor::{
        operators::Function,
        AutogradFunction,
        Tensor,
        TensorBase,
        Variable
    },
    MlResult
};
use std::{
    fmt::Debug,
    sync::Arc
};

pub trait Layer<Type> {
    fn new() -> MlResult<Self> where Self: Sized;
    fn apply(&self, input: &Arc<Variable<Type>>) -> MlResult<Arc<Variable<Type>>>;
    fn forward(&self, input: &Tensor<Type>) -> MlResult<Tensor<Type>>;
}

pub struct Linear<Type>    { operators: Arc<dyn Function<Type>> }
pub struct Conv<Type>      { operators: Arc<dyn Function<Type>> }
pub struct Pooling<Type>   { operators: Arc<dyn Function<Type>> }

