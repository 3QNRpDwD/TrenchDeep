use tracing::error;
use crate::{
    nn::{
        Layer,
        Sequential
    },
    tensor::GlobalTensor
};

use super::*;

pub mod softmax;
pub mod linear;
pub mod logistic;

