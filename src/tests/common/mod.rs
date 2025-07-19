use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{info, warn};
use mnist::MnistBuilder;
use rand::{rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use time::macros::format_description;
use tracing_subscriber::{
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter
};

use crate::{
    loss::CrossEntropyLoss
    ,
    nn::{
        Layer,
        Linear,
        Parameter,
        Sequential,
        Variable
    },
    tensor::GlobalTensor,
    tensor::{
        operators::Function

        ,
        ComputationGraph,
        Tensor,
        TensorBase
    }
    ,
    tests::common::model::{Model, SoftmaxRegression},
    var_input,
    var_with_label,
    MlError,
    MlResult
};

pub(crate) mod data;
pub(crate) mod evaluation;
pub(crate) mod utils;
pub(crate) mod config;
pub mod model;

