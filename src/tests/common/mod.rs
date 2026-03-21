pub(crate) mod data;
pub(crate) mod utils;
pub(crate) mod config;
pub mod model;

use serde::{Deserialize, Serialize};
use mnist::{MnistBuilder};
use std::{
    sync::Arc,
    time::Instant
};
use log::{info, warn};
use rand::{rng, seq::SliceRandom};
use tracing_subscriber::{
    prelude::*,
    EnvFilter,
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use crate::{
    loss::SoftmaxCrossEntropyLoss,
    nn::{
        activation::Sigmoid,
        activation::Softmax,
        Variable,
        Parameter,
        Layer,
        Sequential
    },
    var_with_label,
    var_input,
    MlResult,
    scalar,
    MlError,
    tensor::{
        AutogradFunction,
        ComputationGraph,
        Tensor,
        operators::Function,
        TensorBase,
        GlobalFunction,
        operators::{Add, Matmul}
    },
    tensor::GlobalTensor,
    loss::{CrossEntropyLoss},
    tests::common::model::{Model, SoftmaxRegression, MLP}
};
use time::macros::format_description;
