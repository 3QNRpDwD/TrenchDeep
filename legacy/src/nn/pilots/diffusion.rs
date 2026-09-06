//! Minimal explicit-context DDPM training pilot.
//!
//! This intentionally uses a single shape-preserving convolution instead of the
//! legacy test U-Net. It exercises the same `q_sample -> noise prediction -> MSE`
//! training boundary while the complete U-Net is migrated independently.

use std::f32::consts::TAU;

use crate::loss::Reduction;
use crate::nn::{ContextConv2D, ContextLayer, ContextParameter};
use crate::trainer::{ContextTrainableModel, ContextUnsupervisedModel};
use crate::{ContextId, ContextTensor, ContextVariable, ExecutionContext, MlResult, TensorError};

#[derive(Debug)]
pub struct ContextDiffusionPilot {
    context: ExecutionContext,
    denoiser: ContextConv2D,
    alpha_bars: Vec<f32>,
    channels: usize,
}

impl ContextDiffusionPilot {
    pub fn new(
        context: &ExecutionContext,
        channels: usize,
        timesteps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> MlResult<Self> {
        if channels == 0
            || timesteps == 0
            || !beta_start.is_finite()
            || !beta_end.is_finite()
            || beta_start <= 0.0
            || beta_end < beta_start
            || beta_end >= 1.0
        {
            return Err(TensorError::InvalidOperation {
                op: "diffusion_pilot",
                reason: "channels/timesteps must be non-zero and 0 < beta_start <= beta_end < 1"
                    .into(),
            }.into());
        }
        let mut cumulative = 1.0;
        let denominator = timesteps.saturating_sub(1).max(1) as f32;
        let alpha_bars = (0..timesteps)
            .map(|index| {
                let ratio = index as f32 / denominator;
                let beta = beta_start + (beta_end - beta_start) * ratio;
                cumulative *= 1.0 - beta;
                cumulative
            })
            .collect();
        Ok(Self {
            context: context.clone(),
            denoiser: ContextConv2D::new(
                context,
                channels,
                channels,
                (3, 3),
                (1, 1),
                (1, 1),
                "diffusion_denoiser",
            )?,
            alpha_bars,
            channels,
        })
    }

    pub fn predict_noise(&self, noisy: &ContextTensor) -> MlResult<ContextTensor> {
        self.denoiser.predict(noisy)
    }

    fn standard_normal(count: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let radius = (-2.0 * rand::random::<f32>().max(f32::MIN_POSITIVE).ln()).sqrt();
            let angle = TAU * rand::random::<f32>();
            values.push(radius * angle.cos());
            if values.len() < count {
                values.push(radius * angle.sin());
            }
        }
        values
    }

    fn validate_image(&self, image: &ContextVariable) -> MlResult<Vec<usize>> {
        let shape = image.tensor().shape()?;
        if shape.len() != 4 || shape[1] != self.channels {
            return Err(TensorError::InvalidOperation {
                op: "diffusion_pilot",
                reason: "input must have shape [batch, channels, height, width]".into(),
            }.into());
        }
        Ok(shape)
    }
}

impl ContextTrainableModel for ContextDiffusionPilot {
    fn context_id(&self) -> ContextId { self.context.id() }
    fn parameters(&self) -> Vec<&ContextParameter> { self.denoiser.parameters() }
}

impl ContextUnsupervisedModel for ContextDiffusionPilot {
    fn forward_loss(
        &mut self,
        image: &ContextVariable,
    ) -> MlResult<(ContextVariable, ContextVariable)> {
        let shape = self.validate_image(image)?;
        let count = image.tensor().to_vec()?.len();
        let timestep = rand::random::<u64>() as usize % self.alpha_bars.len();
        let alpha_bar = self.alpha_bars[timestep];
        let noise = self.context.input(Self::standard_normal(count), &shape)?;
        let image_scale = self.context.input(vec![alpha_bar.sqrt()], &[])?;
        let noise_scale = self.context.input(vec![(1.0 - alpha_bar).sqrt()], &[])?;
        let clean_component = self.context.mul_variable(image, &image_scale)?;
        let noise_component = self.context.mul_variable(&noise, &noise_scale)?;
        let noisy = self.context.add_variable(&clean_component, &noise_component)?;
        let prediction = self.denoiser.apply(&noisy)?;
        let loss = self.context.mse_loss_variable(
            &prediction,
            noise.tensor(),
            Reduction::Mean,
        )?;
        Ok((prediction, loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{ContextAdam, ContextOptimizer};
    use crate::trainer::{ContextUnsupervisedDataset, ContextUnsupervisedTrainer, EpochSchedule};

    #[test]
    fn diffusion_pilot_trains_end_to_end() -> MlResult<()> {
        let context = ExecutionContext::new();
        let mut model = ContextDiffusionPilot::new(&context, 1, 8, 1e-4, 0.02)?;
        let images = [
            context.input(vec![0.25; 16], &[1, 1, 4, 4])?,
            context.input(vec![0.75; 16], &[1, 1, 4, 4])?,
        ];
        let refs = images.iter().collect::<Vec<_>>();
        let dataset = ContextUnsupervisedDataset::new(&context, &refs)?;
        let mut optimizer = ContextAdam::new(&context, 0.01, 0.9, 0.999, 1e-8)?;
        optimizer.register_all(&model.parameters())?;
        let result = ContextUnsupervisedTrainer::silent(&context).fit(
            &mut model,
            &mut optimizer,
            &dataset,
            EpochSchedule::new(3)?.with_tolerance(0.0),
        )?;
        assert!(result.final_loss.is_finite());
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
        Ok(())
    }
}
