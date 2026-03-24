mod Unet;
mod encoder;
mod decoder;
mod scheduler;
mod embedding;

use std::fmt::{Debug, Formatter};
use log::info;
use crate::{variable, MlResult};
use crate::loss::CrossEntropyLoss;
use crate::nn::{Layer, Linear, Model, Sequential};
use crate::tensor::{GlobalFunction, GlobalTensor, Tensor, TensorBase};
use crate::tensor::operators::{Cos, Function, Matmul, Pow, Sin, Concat};
use crate::nn::Variable;
use crate::optimizer::Optimizer;
use crate::nn::Parameter;
use crate::tests::common::model::diffusion::encoder::SinusoidalPE;
use crate::nn::activation::SiLULayer;
use crate::tests::common::model::MLP;
use crate::tests::common::model::diffusion::{decoder::Decoder, encoder::Encoder};

struct DiffusionModel {
    encoder: Encoder,
    decoder: Decoder,
    scheduler: MLP,
    loss: CrossEntropyLoss,
}