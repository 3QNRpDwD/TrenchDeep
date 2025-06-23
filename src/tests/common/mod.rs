pub(crate) mod data;
pub(crate) mod evaluation;
pub(crate) mod utils;
pub(crate) mod config;
pub mod model;

use serde::{Deserialize, Serialize};
use mnist::{MnistBuilder};
use std::{sync::Arc};
use log::{info, warn};
use crate::{
    nn::{
        activation::Sigmoid,
        activation::Softmax,
        Variable,
        Parameter
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
    }
};
use rand::{rng, seq::SliceRandom};
use tracing_subscriber::{
    prelude::*,
    EnvFilter,
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt
};
use std::time::Instant;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use crate::loss::SoftmaxCrossEntropyLoss;
use time::macros::format_description;
use crate::tensor::GlobalTensor;
use crate::loss::{CrossEntropyLoss};
use crate::tests::common::model::{Model, SoftmaxRegression, MLP};
