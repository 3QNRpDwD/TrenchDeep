use super::*; // info, MlResult (from common/mod.rs)

// data 하위 모듈에서 사용
use ::mnist::MnistBuilder;
use crate::{
    nn::{Parameter, Variable},
    tensor::{Tensor, TensorBase},
    var_input,
    var_with_label,
};

pub mod mnist;
