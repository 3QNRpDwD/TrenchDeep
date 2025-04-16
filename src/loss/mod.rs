use std::fmt::Debug;
use crate::tensor::operators::Function;

pub trait Loss<T: Debug + Clone>: Function<T> {}