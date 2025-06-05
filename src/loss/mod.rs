use crate::tensor::operators::Function;
use std::fmt::Debug;

pub trait Loss<T: Debug + Clone>: Function<T> {
    fn new() -> Self;

    fn forward(&self, input: &T, target: &T) -> T;
}