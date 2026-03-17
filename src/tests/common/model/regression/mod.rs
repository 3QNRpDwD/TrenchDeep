use crate::{
    nn::{
        Layer,
        Sequential
    },
    tensor::GlobalTensor
};
use tracing::error;

use super::*;

pub mod softmax;
pub mod linear;
pub mod logistic;

