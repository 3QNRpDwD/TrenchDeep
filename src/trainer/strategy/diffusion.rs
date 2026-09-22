use super::*;
use crate::nn::Diffusion;
#[derive(Debug, Clone, Copy)]
pub struct DiffusionTraining;
fn objective_error(message: &str) -> MlError {
    crate::TensorError::InvalidOperation {
        op: "diffusion",
        reason: message.into(),
    }
    .into()
}

impl DiffusionTraining {
    pub fn forward_loss_with_feeds<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &Diffusion,
        image: &Variable,
        feeds: &ExecutionInputs,
    ) -> MlResult<(Variable, Variable)> {
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
        let loss = loss.compute(&image.tensor().execution_context()?, &prediction, noise)?;
        Ok((prediction, loss))
    }
    pub fn forward_loss_with_noise<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &Diffusion,
        image: &Variable,
        noise: &Tensor,
        t: usize,
    ) -> MlResult<(Variable, Variable)> {
        self.forward_loss_with_feeds(loss, model, image, &model.training_feeds(noise, t)?)
    }
    pub fn forward_loss<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &mut Diffusion,
        image: &Variable,
    ) -> MlResult<(Variable, Variable)> {
        let feeds = model.draw_training_feeds(&image.tensor().shape()?)?;
        self.forward_loss_with_feeds(loss, model, image, &feeds)
    }
}
impl TrainingStrategy<Diffusion> for DiffusionTraining {
    type Batch = UnsupervisedBatch;
    const PARADIGM: ParadigmTag = ParadigmTag::Unsupervised;
    fn forward_batch<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &mut Diffusion,
        batch: &Self::Batch,
        _: &TrainingStepContext,
    ) -> MlResult<TrainingOutput> {
        let weight = sample_count(batch.samples.tensor())?;
        let (prediction, loss) = self.forward_loss(loss, model, &batch.samples)?;
        Ok(TrainingOutput {
            loss,
            prediction: Some(prediction),
            target: None,
            weight,
            tokens: None,
            lambda: None,
        })
    }
}
impl PreparedTrainingStrategy<Diffusion> for DiffusionTraining {
    fn execution_batch(
        &self,
        model: &mut Diffusion,
        batch: &Self::Batch,
        _: &TrainingStepContext,
    ) -> MlResult<PreparedBatch> {
        let shape = batch.samples.tensor().shape()?;
        let inputs = model
            .draw_training_feeds(&shape)?
            .with("image", batch.samples.tensor().clone())?;
        Ok(PreparedBatch::new(inputs, shape[0]))
    }
    fn forward_inputs<L: Loss + ?Sized>(
        &self,
        loss: &L,
        model: &Diffusion,
        inputs: &ExecutionInputs,
    ) -> MlResult<ModelOutput> {
        let (prediction, loss) = self.forward_loss_with_feeds(
            loss,
            model,
            &inputs.get("image")?.as_variable()?,
            inputs,
        )?;
        Ok(ModelOutput::new(loss, Some(prediction)))
    }
}
