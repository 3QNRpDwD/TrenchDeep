#[path = "common/mod.rs"]
pub mod common;
#[path = "repro.rs"]
mod repro;

#[cfg(test)]
#[path = "mnist_test.rs"]
mod mnist_test;

#[cfg(test)]
#[path = "checkpoint_test.rs"]
mod checkpoint_test;

use crate::legacy::{
    MlResult,
    nn::{
        GroupNorm,
        Layer,
        Linear,
        Parameter,
        Sequential,
        Model,
        Variable,
    },
    tensor::TensorBase,
    optimizer::{Optimizer, SGD},
    tests::common::{
        data::mnist::{MnistConfig, MnistDataset},
        logging::setup_logging,
        model::{MLP, SoftmaxRegression},
    }
};
use tracing::{info, warn};
