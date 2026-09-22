use super::*;
use crate::nn::Diffusion;
#[derive(Debug, Clone, Copy)]
pub struct DiffusionObjective<L> { loss: L }
fn objective_error(message: &str) -> MlError {
    crate::TensorError::InvalidOperation { op: "diffusion", reason: message.into() }.into()
}
impl<L> DiffusionObjective<L> { pub fn new(loss: L) -> Self { Self { loss } } }
impl<L: Loss> DiffusionObjective<L> {
    pub fn forward_loss_with_feeds(&self, model: &Diffusion, image: &Variable, feeds: &ExecutionInputs) -> MlResult<(Variable, Variable)> {

        if feeds.variant() != "training" {
            return Err(objective_error("expected training inputs"));
        }
        let noise = feeds.get("noise")?;
        if image.tensor().shape()? != noise.shape()? {
            return Err(objective_error("noise shape differs from image"));
        }
        let noisy = image
            .tensor()
            .mul(feeds.get("image_scale")?)?
            .add(&noise.mul(feeds.get("noise_scale")?)?)?
            .as_variable()?;
        let prediction = model.unet.forward(&noisy, feeds.get("timesteps")?)?;
        let loss = self.loss.compute(&image.tensor().execution_context()?, &prediction, noise)?;
        Ok((prediction, loss))
    
    }
    pub fn forward_loss_with_noise(&self, model: &Diffusion, image: &Variable, noise: &Tensor, t: usize) -> MlResult<(Variable, Variable)> {
        self.forward_loss_with_feeds(model, image, &model.training_feeds(noise, t)?)
    }
    pub fn forward_loss(&self, model: &mut Diffusion, image: &Variable) -> MlResult<(Variable, Variable)> {
        let feeds = model.draw_training_feeds(&image.tensor().shape()?)?;
        self.forward_loss_with_feeds(model, image, &feeds)
    }
}
impl<L: Loss> Objective<Diffusion> for DiffusionObjective<L> {
    type Batch = UnsupervisedBatch;
    const PARADIGM: &'static str = "unsupervised";
    fn forward_batch(&self, model: &mut Diffusion, batch: &Self::Batch, _: &TrainingStepContext) -> MlResult<TrainingOutput> {
        let weight = sample_count(batch.samples.tensor())?;
        let (prediction, loss) = self.forward_loss(model, &batch.samples)?;
        Ok(TrainingOutput { loss, prediction: Some(prediction), target: None, weight, tokens: None, lambda: None })
    }
}
impl<L: Loss> PreparedObjective<Diffusion> for DiffusionObjective<L> {
    fn execution_batch(&self, model: &mut Diffusion, batch: &Self::Batch, _: &TrainingStepContext) -> MlResult<PreparedBatch> {
        let shape = batch.samples.tensor().shape()?;
        let inputs = model.draw_training_feeds(&shape)?.with("image", batch.samples.tensor().clone())?;
        Ok(PreparedBatch::new(inputs, shape[0]))
    }
    fn forward_inputs(&self, model: &Diffusion, inputs: &ExecutionInputs) -> MlResult<ModelOutput> {
        let (prediction, loss) = self.forward_loss_with_feeds(model, &inputs.get("image")?.as_variable()?, inputs)?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
