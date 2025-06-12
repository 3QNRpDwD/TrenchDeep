use std::sync::Arc;
use crate::backend::Backend;
use crate::tensor::NodeId;

pub struct GradientDescent { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct Momentum { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct NesterovAcceleratedGradient { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct Adagrad { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct Adadelta { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct RMSprop { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct Adam { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct AdamW { backend: Arc<dyn Backend>, node_id: NodeId }

pub struct L_BFGS { backend: Arc<dyn Backend>, node_id: NodeId }

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