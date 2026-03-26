mod unet;
mod encoder;
mod decoder;
mod scheduler;
mod embedding;

use std::fmt::{Debug, Formatter};
use log::info;

use crate::{
    loss::CrossEntropyLoss,
    nn::{
        Variable,
        Parameter,
        activation::SiLULayer,
        Layer,
        Linear,
        Model,
        Sequential
    },
    tests::{
        common::{
            model::{
                diffusion::{
                    decoder::Decoder,
                    encoder::{
                        Encoder,
                        SinusoidalPE
                    }
                },
                MLP
            }
        }
    },
    optimizer::Optimizer,
    variable,
    MlResult,
    tensor::{
        GlobalFunction,
        GlobalTensor,
        Tensor,
        TensorBase,
        operators::{Cos, Function, Matmul, Sin, Concat}
    }
};


struct DiffusionModel {
    encoder: Encoder,
    decoder: Decoder,
    scheduler: MLP,
    loss: CrossEntropyLoss,
}