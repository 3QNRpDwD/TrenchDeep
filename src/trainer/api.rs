//! Shared public API types for all training paradigms.

use std::{collections::BTreeMap, path::Path, time::Duration};
use crate::{MlError, MlResult};


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

    pub(crate) fn with_metrics(mut self, metrics: MetricValues) -> Self {
        self.metrics = metrics;
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

