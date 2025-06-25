use super::*;

pub mod softmax;
pub mod linear;
pub mod logistic;

use tracing::error;
use crate::tensor::{GlobalTensor, TENSOR_ALLOCATOR};