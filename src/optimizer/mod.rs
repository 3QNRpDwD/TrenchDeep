use std::fmt::Debug;
use std::sync::Arc;
use crate::backend::Backend;
use crate::loss::Loss;
use crate::nn::Parameter;
use crate::tensor::{NodeId, Tensor};
use crate::tensor::operators::Function;

pub struct BGD { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct SGD { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct  MiniBGD { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct Momentum { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct NAG { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct AdaGrad { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct AdaDelta { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct RMSProp { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct Adam { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct AdamW { backend: Arc<dyn Backend>, node_id: NodeId }

pub trait Optimizer<T: Debug + Clone>: Function<T> {
    fn step(&self);
    fn zero_grad(&self);
    fn get_params(&self);
    fn add_params(&self, parm: &dyn Parameter, grad: Option<Tensor<T>>);
    fn get_lr(&self);
    fn set_lr(&self, ln: f32);
}

impl Function<f32> for BGD {}
impl Function<f32> for SGD {}
impl Function<f32> for MiniBGD {}
impl Function<f32> for Momentum {}
impl Function<f32> for NAG {}
impl Function<f32> for AdaGrad {}
impl Function<f32> for AdaDelta {}
impl Function<f32> for RMSProp {}
impl Function<f32> for Adam {}
impl Function<f32> for AdamW {}

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

    fn add_params(&self, parm: &dyn Parameter, grad: Option<Tensor<f32>>) {
        todo!()
    }

    fn get_lr(&self) {
        todo!()
    }

    fn set_lr(&self, ln: f32) {
        todo!()
    }
}