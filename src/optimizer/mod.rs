pub trait Optimizer {
    fn step(&self);
    fn zero_grad(&self);
    fn get_params(&self);
    fn set_params(&self);
    fn get_grads(&self);
    fn set_grads(&self);
    fn get_lr(&self);
    fn set_lr(&self);
}

pub struct SGD {}