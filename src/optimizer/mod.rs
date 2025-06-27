use std::{
    fmt::{Debug, Display, Formatter},
    sync::Arc
};
use crate::{
    backend::Backend,
    nn::Parameter,
    tensor::{
        HandleId,
        Tensor,
        operators::Function
    }
};


#[derive(thiserror::Error, Debug)]
pub enum OptimError {
    #[error("Gradient Error: {0}")]
    GradientError(String),
}

pub struct BGD { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct SGD { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct  MiniBGD { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct Momentum { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct NAG { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct AdaGrad { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct AdaDelta { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct RMSProp { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct Adam { backend: Arc<dyn Backend>, node_id: HandleId }

pub struct AdamW { backend: Arc<dyn Backend>, node_id: HandleId }

pub trait Optimizer<T: Debug + Clone> {
    fn step(&self);
    fn zero_grad(&self);
    fn get_params(&self);
    fn add_params(&self, parm: &dyn Parameter, grad: Option<Tensor>);
    fn get_lr(&self);
    fn set_lr(&self, ln: f32);
}

impl Function for BGD {}
impl Function for SGD {}
impl Function for MiniBGD {}
impl Function for Momentum {}
impl Function for NAG {}
impl Function for AdaGrad {}
impl Function for AdaDelta {}
impl Function for RMSProp {}
impl Function for Adam {}
impl Function for AdamW {}

impl Optimizer<f32> for BGD {
    fn step(&self) {
        todo!()
    }

    fn zero_grad(&self) {
        todo!()
    }

    fn get_params(&self) {
        todo!()
    }

    fn add_params(&self, parm: &dyn Parameter, grad: Option<Tensor>) {
        todo!()
    }

    fn get_lr(&self) {
        todo!()
    }

    fn set_lr(&self, ln: f32) {
        todo!()
    }
}