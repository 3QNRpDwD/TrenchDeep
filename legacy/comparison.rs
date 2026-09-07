//! Visibility bridge to unchanged baseline model implementations.
pub use crate::tests::common::model::diffusion::Diffusion as ReferenceDiffusion;
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
#[path="src/tests/common/model/linear/mod.rs"]
pub mod linear;
use crate::loss::CrossEntropyLoss;
#[path="src/tests/common/model/mlp/mod.rs"]
pub mod mlp;
#[path="src/tests/common/model/autoregressive/mod.rs"]
pub mod autoregressive;
#[path="src/tests/common/model/semi_supervised/mod.rs"]
pub mod semi_supervised;
#[path="src/tests/common/model/reinforcement/mod.rs"]
pub mod reinforcement;
pub fn clear_graph() { crate::tensor::ComputationGraph::reset_graph(); }

/// Read the original DDPM's random choices after its unmodified forward call.
/// The source image identifies q_sample's signal multiplier; the loss identifies
/// the target noise. This does not reseed or modify the original computation.
pub fn diffusion_draw(
    model: &ReferenceDiffusion,
    image: &Variable,
    loss: &Variable,
) -> MlResult<(usize, GlobalTensor<f32>)> {
    crate::tensor::COMPUTATION_GRAPH.with(|graph| {
        let graph = graph.lock().map_err(|_| crate::MlError::StringError("legacy graph lock poisoned".into()))?;
        let mut signal_ids = Vec::new();
        let mut noise_id = None;
        graph.visit_nodes(|node| {
            if node.operation == Some("Mul") && node.inputs.first() == Some(&image.node_id()) && node.inputs.len() == 2 {
                signal_ids.push(node.inputs[1]);
            }
            if node.id == loss.node_id() { noise_id = node.inputs.first().copied(); }
        });
        if signal_ids.len() != 1 {
            return Err(crate::MlError::StringError("ambiguous DDPM signal multiplier".into()));
        }
        let mut timestep = None;
        let mut noise = None;
        graph.visit_nodes(|node| {
            if node.id == signal_ids[0] && node.tensor.data().len() == 1 {
                let value = node.tensor.data()[0];
                timestep = model.scheduler.sqrt_alpha_bars.iter().position(|v| v.to_bits() == value.to_bits());
            }
            if Some(node.id) == noise_id {
                noise = Some(GlobalTensor::from_vec(node.tensor.data().to_vec(), node.tensor.shape()));
            }
        });
        Ok((
            timestep.ok_or_else(|| crate::MlError::StringError("DDPM timestep not found".into()))?,
            noise.ok_or_else(|| crate::MlError::StringError("DDPM noise not found".into()))??,
        ))
    })
}
/// Read-only counters; no baseline algorithm or lifetime policy is changed.
pub fn statistics()->MlResult<(usize,usize)> {
    let tensors=crate::tensor::TENSOR_STORAGE.with(|storage|storage.try_borrow().map(|storage|storage.len()).map_err(|_|crate::MlError::StringError("legacy storage borrow conflict".into())))?;
    let nodes=crate::tensor::COMPUTATION_GRAPH.with(|graph|graph.lock().map(|graph|graph.node_map.len()).map_err(|_|crate::MlError::StringError("legacy graph lock poisoned".into())))?;
    Ok((tensors,nodes))
}
/// Capture the original model's sampled augmentations without changing its RNG or code.
pub fn direct_add_outputs(input:&Variable)->MlResult<Vec<GlobalTensor<f32>>> {
    crate::tensor::COMPUTATION_GRAPH.with(|graph| {
        let graph=graph.lock().map_err(|_|crate::MlError::StringError("legacy graph lock poisoned".into()))?;
        let mut outputs=Vec::new();
        graph.visit_nodes(|node| {
            if node.operation==Some("Add") && node.inputs.first()==Some(&input.node_id()) {
                outputs.push(GlobalTensor::from_vec(node.tensor.data().to_vec(),node.tensor.shape()));
            }
        });
        outputs.into_iter().collect()
    })
}
