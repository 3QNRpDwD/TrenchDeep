use crate::tensor::operators::Function;
use std::fmt::Debug;

pub trait Loss<T: Debug + Clone>: Function<T> {}