use std::sync::Arc;
use crate::backend::Backend;
use crate::tensor::NodeId;

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