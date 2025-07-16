use crate::define_op;
use crate::backend::Device;
use super::*;

pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

pub trait Activation: Layer {}

impl Activation for SoftmaxLayer {}
impl Activation for SigmoidLayer {}
impl Activation for ReLULayer {}
impl Activation for TanhLayer {}

define_op!(Sigmoid);
define_op!(Tanh);
define_op!(ReLU);
define_op!(Softmax);

#[derive(Debug, Clone)]
pub struct SigmoidLayer {
    label: String,
    operator: Arc<Sigmoid>
}

#[derive(Debug, Clone)]
pub struct TanhLayer {
    label: String,
    operator: Arc<Tanh>
}

#[derive(Debug, Clone)]
pub struct ReLULayer {
    label: String,
    operator: Arc<ReLU>
}
#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    label: String,
    operator: Arc<Softmax>
}
