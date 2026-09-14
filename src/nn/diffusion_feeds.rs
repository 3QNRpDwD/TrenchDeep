//! Model-specific scheduler arithmetic; preparation lives in runtime::prepared.
use super::*;
use crate::runtime::prepared::ExecutionInputs;
fn nchw(shape: &[usize]) -> MlResult<usize> {
    if shape.len() != 4 || shape.contains(&0) {
        return Err(invalid("expected nonempty NCHW image"));
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize / 4)
        .ok_or_else(|| invalid("image size overflow"))
}
impl Diffusion {
    pub fn training_feeds(&self, noise: &Tensor, t: usize) -> MlResult<ExecutionInputs> {
        self.context.deny_preparation("scheduler feed selection")?;
        self.context.validate(noise)?;
        let shape = noise.shape()?;
        nchw(&shape)?;
        let &alpha = self
            .scheduler
            .alpha_bars
            .get(t)
            .ok_or_else(|| invalid("timestep out of range"))?;
        ExecutionInputs::new("training")
            .with("noise", noise.clone())?
            .with(
                "timesteps",
                self.context.tensor(
                    vec![t as f32 / self.scheduler.timesteps() as f32; shape[0]],
                    &[shape[0], 1],
                )?,
            )?
            .with("image_scale", self.context.scalar(alpha.sqrt())?)?
            .with("noise_scale", self.context.scalar((1.0 - alpha).sqrt())?)
    }
    pub fn draw_training_feeds(&mut self, shape: &[usize]) -> MlResult<ExecutionInputs> {
        nchw(shape)?;
        let noise = self.noise(shape)?;
        let t = self.noise.random_range(0..self.scheduler.timesteps());
        self.training_feeds(&noise, t)
    }
    pub fn forward_loss_with_feeds(
        &self,
        image: &Variable,
        feeds: &ExecutionInputs,
    ) -> MlResult<(Variable, Variable)> {
        if feeds.variant() != "training" {
            return Err(invalid("expected training inputs"));
        }
        let noise = feeds.get("noise")?;
        if image.tensor().shape()? != noise.shape()? {
            return Err(invalid("noise shape differs from image"));
        }
        let noisy = image
            .tensor()
            .mul(feeds.get("image_scale")?)?
            .add(&noise.mul(feeds.get("noise_scale")?)?)?
            .as_variable()?;
        let prediction = self.unet.forward(&noisy, feeds.get("timesteps")?)?;
        let loss = prediction.mse_loss(noise, crate::Reduction::Mean)?;
        Ok((prediction, loss))
    }
    pub fn step_feeds(&self, noise: &Tensor, t: usize) -> MlResult<ExecutionInputs> {
        self.context.deny_preparation("scheduler feed selection")?;
        self.context.validate(noise)?;
        let shape = noise.shape()?;
        nchw(&shape)?;
        let &beta = self
            .scheduler
            .betas
            .get(t)
            .ok_or_else(|| invalid("timestep out of range"))?;
        let alpha = self.scheduler.alpha_bars[t];
        let variance = if t == 0 {
            0.0
        } else {
            beta * (1.0 - self.scheduler.alpha_bars[t - 1]) / (1.0 - alpha).max(1e-8)
        };
        ExecutionInputs::new(if t == 0 { "final" } else { "step" })
            .with("noise", noise.clone())?
            .with(
                "timesteps",
                self.context.tensor(
                    vec![t as f32 / self.scheduler.timesteps() as f32; shape[0]],
                    &[shape[0], 1],
                )?,
            )?
            .with(
                "prediction_scale",
                self.context.scalar(beta / (1.0 - alpha).sqrt().max(1e-8))?,
            )?
            .with(
                "mean_scale",
                self.context.scalar(1.0 / (1.0 - beta).sqrt())?,
            )?
            .with("noise_scale", self.context.scalar(variance.sqrt())?)
    }
    pub fn reverse_step_with_feeds(
        &self,
        image: &Tensor,
        feeds: &ExecutionInputs,
    ) -> MlResult<Tensor> {
        if !matches!(feeds.variant(), "step" | "final") {
            return Err(invalid("expected reverse-step inputs"));
        }
        self.context.validate(image)?;
        self.context.validate(feeds.get("noise")?)?;
        if image.shape()? != feeds.get("noise")?.shape()? {
            return Err(invalid("reverse-step shapes differ"));
        }
        self.context.no_grad(|| {
            let prediction = self.unet.predict(image, feeds.get("timesteps")?)?;
            let mean = image
                .sub(&prediction.mul(feeds.get("prediction_scale")?)?)?
                .mul(feeds.get("mean_scale")?)?;
            if feeds.variant() == "final" {
                Ok(mean)
            } else {
                mean.add(&feeds.get("noise")?.mul(feeds.get("noise_scale")?)?)
            }
        })
    }
}
