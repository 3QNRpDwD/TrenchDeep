use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{info, warn, error};
use mnist::MnistBuilder;
use rand::{rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use time::macros::format_description;
use tracing_subscriber::{
    EnvFilter,
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt
};

use crate::{
    loss::CrossEntropyLoss,
    loss::SoftmaxCrossEntropyLoss,
    MlError,
    MlResult,
    nn::{
        activation::Sigmoid,
        activation::Softmax,
        Layer,
        Linear,
        Parameter,
        Sequential,
        Variable
    }
    ,
    tensor::{
        AutogradFunction,
        ComputationGraph,
        GlobalFunction,
        operators::{Add, Matmul},
        operators::Function,
        Tensor,
        TensorBase
    },
    tensor::GlobalTensor,
    tests::common::model::{MLP, Model, SoftmaxRegression},
    var_input,
    var_with_label
};

pub(crate) mod data;
pub(crate) mod evaluation;
pub(crate) mod utils;
pub(crate) mod config;
pub mod model;

