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
    sync::Arc,
    collections::HashSet
};

pub trait Layer {
    fn new() -> MlResult<Self> where Self: Sized;
    fn parms(&self) -> MlResult<&[String]>;
    fn set_parms(&self, name: String, parm: &Arc<dyn Parameter>) -> MlResult<&HashSet<Arc<dyn Parameter>>>;
    fn get_parms(&self, name: String) -> MlResult<Arc<dyn Parameter>>;
    fn apply(&self, input: &Arc<dyn Parameter>) -> MlResult<Arc<dyn Parameter>>;
    fn forward(&self, input: &Tensor) -> MlResult<Tensor>;
}

pub trait Parameter {}
impl<Type> Parameter for Variable<Type> {}

pub struct Linear    { operators: Arc<dyn Function> }
pub struct Conv      { operators: Arc<dyn Function> }
pub struct Pooling   { operators: Arc<dyn Function> }

