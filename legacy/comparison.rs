//! Visibility bridge to unchanged baseline model implementations.
use crate::{MlResult,var_with_label};
use crate::nn::{Layer,Linear,Model,Parameter,Sequential,Variable,Conv2D,GroupNorm,activation::{SiLU,SoftmaxOp}};
use crate::tensor::{AutogradFunction,GlobalFunction,GlobalTensor,Tensor,TensorBase,operators::{Add,Function,Matmul,Concat,Cos,Mul,NearestUpsample2d,ReshapeOp,Sin,Transpose}};
use std::fmt::Debug;
use tracing::info;
#[path="src/tests/common/model/diffusion/encoder.rs"]
mod encoder;
#[path="src/tests/common/model/diffusion/embedding.rs"]
mod embedding;
#[path="src/tests/common/model/diffusion/unet.rs"]
pub mod unet;
#[path="src/tests/common/model/diffusion/scheduler.rs"]
pub mod scheduler;
use encoder::SinusoidalPE;
use embedding::TimeEmbeddingMLP;
pub fn clear_graph() { crate::tensor::ComputationGraph::reset_graph(); }
