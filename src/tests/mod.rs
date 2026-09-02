pub mod common;
mod repro;

#[cfg(test)]
mod mnist_test;

#[cfg(test)]
mod checkpoint_test;

use crate::{
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
        utils::generate_visualization,
    }
};
use log::{info, warn};