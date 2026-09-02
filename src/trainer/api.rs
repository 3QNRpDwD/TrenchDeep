//! Shared public API types for all training paradigms.

use std::{collections::BTreeMap, path::Path, time::Duration};
use crate::{MlError, MlResult, nn::{Parameter, Variable}};

pub trait TrainableModel { fn params(&self) -> Vec<&dyn Parameter>; }

pub trait CheckpointableModel {
    fn save_checkpoint(&self, path: &Path) -> MlResult<()> {
        Err(MlError::StringError(format!("checkpoint save is not implemented: {}", path.display())))
    }
    fn load_checkpoint(&mut self, path: &Path) -> MlResult<()> {
        Err(MlError::StringError(format!("checkpoint load is not implemented: {}", path.display())))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason { Completed, Converged, Interrupted }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepUnit { Epoch, Episode, Token }

pub type MetricValues = BTreeMap<String, f32>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointPaths {
    pub model: std::path::PathBuf,
    pub metadata: std::path::PathBuf,
}

#[derive(Debug)]
pub struct TrainResult {
    pub stop_reason: StopReason,
    pub units_completed: usize,
    pub unit: StepUnit,
    pub final_loss: f32,
    pub metrics: MetricValues,
    pub checkpoint: Option<CheckpointPaths>,
    pub total_duration: Duration,
}

impl TrainResult {
    pub(crate) fn epochs(reason: StopReason, completed: usize, loss: f32, duration: Duration) -> Self {
        Self { stop_reason: reason, units_completed: completed, unit: StepUnit::Epoch,
            final_loss: loss, metrics: MetricValues::new(), checkpoint: None, total_duration: duration }
    }
    pub(crate) fn episodes(completed: usize, loss: f32, duration: Duration) -> Self {
        Self { stop_reason: StopReason::Completed, units_completed: completed, unit: StepUnit::Episode,
            final_loss: loss, metrics: MetricValues::new(), checkpoint: None, total_duration: duration }
    }

    pub(crate) fn with_checkpoint(mut self, checkpoint: Option<CheckpointPaths>) -> Self {
        self.checkpoint = checkpoint;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EpochSchedule { pub epochs: usize, pub convergence: super::Convergence }
impl EpochSchedule {
    pub fn new(epochs: usize) -> MlResult<Self> {
        if epochs == 0 { return Err(MlError::StringError("epochs must be > 0".into())); }
        Ok(Self { epochs, convergence: super::Convergence::Off })
    }
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.convergence = super::Convergence::from_tolerance(tolerance); self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EpisodeSchedule { pub episodes: usize, pub max_steps_per_episode: usize }
impl EpisodeSchedule {
    pub fn new(episodes: usize, max_steps_per_episode: usize) -> MlResult<Self> {
        if episodes == 0 || max_steps_per_episode == 0 {
            return Err(MlError::StringError("episodes and max_steps_per_episode must be > 0".into()));
        }
        Ok(Self { episodes, max_steps_per_episode })
    }
}

pub struct SupervisedDataset<'a> { pub inputs: &'a [&'a Variable], pub targets: &'a [&'a Variable] }
impl<'a> SupervisedDataset<'a> {
    pub fn new(inputs: &'a [&'a Variable], targets: &'a [&'a Variable]) -> MlResult<Self> {
        if inputs.is_empty() { return Err(MlError::StringError("supervised dataset must not be empty".into())); }
        if inputs.len() != targets.len() { return Err(MlError::StringError("input/target length mismatch".into())); }
        Ok(Self { inputs, targets })
    }
}

pub struct UnsupervisedDataset<'a> { pub samples: &'a [&'a Variable] }
impl<'a> UnsupervisedDataset<'a> {
    pub fn new(samples: &'a [&'a Variable]) -> MlResult<Self> {
        if samples.is_empty() { return Err(MlError::StringError("unsupervised dataset must not be empty".into())); }
        Ok(Self { samples })
    }
}

pub struct SemiSupervisedDataset<'a> {
    pub labeled_inputs: &'a [&'a Variable], pub labeled_targets: &'a [&'a Variable],
    pub unlabeled_inputs: &'a [&'a Variable],
}
impl<'a> SemiSupervisedDataset<'a> {
    pub fn new(labeled_inputs: &'a [&'a Variable], labeled_targets: &'a [&'a Variable],
               unlabeled_inputs: &'a [&'a Variable]) -> MlResult<Self> {
        if labeled_inputs.is_empty() || unlabeled_inputs.is_empty() {
            return Err(MlError::StringError("semi-supervised datasets must not be empty".into()));
        }
        if labeled_inputs.len() != labeled_targets.len() {
            return Err(MlError::StringError("labeled input/target length mismatch".into()));
        }
        Ok(Self { labeled_inputs, labeled_targets, unlabeled_inputs })
    }
}

pub struct AutoregressiveDataset<'a> { pub sequences: &'a [&'a Variable], pub pad_token_id: Option<usize> }
impl<'a> AutoregressiveDataset<'a> {
    pub fn new(sequences: &'a [&'a Variable]) -> MlResult<Self> {
        if sequences.is_empty() { return Err(MlError::StringError("autoregressive dataset must not be empty".into())); }
        Ok(Self { sequences, pad_token_id: None })
    }
    pub fn with_pad_token_id(mut self, id: usize) -> Self { self.pad_token_id = Some(id); self }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SupervisedOptions;
#[derive(Debug, Clone, Copy)]
pub struct SemiSupervisedOptions { pub ramp: super::ConsistencyRamp }
#[derive(Debug, Default, Clone, Copy)]
pub struct AutoregressiveOptions { pub pad_token_id: Option<usize> }
#[derive(Debug, Clone, Copy)]
pub struct ReinforcementOptions { pub gamma: f32, pub use_baseline: bool }
impl Default for ReinforcementOptions {
    fn default() -> Self { Self { gamma: 0.99, use_baseline: true } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::Variable, tensor::{Tensor, TensorBase}};
    #[test] fn schedules_reject_zero() {
        assert!(EpochSchedule::new(0).is_err());
        assert!(EpisodeSchedule::new(1, 0).is_err());
    }

    #[test] fn datasets_validate_contracts() {
        let x = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]).unwrap());
        let t = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]).unwrap());
        let xs = [&x]; let ts = [&t];
        assert!(SupervisedDataset::new(&xs, &ts).is_ok());
        assert!(SupervisedDataset::new(&[], &[]).is_err());
        assert!(SupervisedDataset::new(&xs, &[]).is_err());
        assert!(UnsupervisedDataset::new(&[]).is_err());
        assert!(AutoregressiveDataset::new(&[]).is_err());
        assert!(SemiSupervisedDataset::new(&xs, &ts, &[]).is_err());
    }

    #[test] fn identical_seed_produces_identical_stream() {
        let a = super::super::TrainingRuntime::new(7);
        let b = super::super::TrainingRuntime::new(7);
        let mut left = vec![1, 2, 3, 4, 5];
        let mut right = left.clone();
        a.shuffle(&mut left); b.shuffle(&mut right);
        assert_eq!(left, right);
        assert_eq!(a.random_f32(), b.random_f32());
    }

    #[test] fn facade_selectors_are_statically_typed() {
        let _: super::super::SupervisedTrainer = super::super::Trainer::silent().supervised();
        let _: super::super::UnsupervisedTrainer = super::super::Trainer::silent().unsupervised();
        let _: super::super::SemiSupervisedTrainer = super::super::Trainer::silent().semi_supervised();
        let _: super::super::AutoregressiveTrainer = super::super::Trainer::silent().autoregressive();
        let _: super::super::RLTrainer = super::super::Trainer::silent().reinforcement();
    }
}
