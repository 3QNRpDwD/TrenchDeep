mod encoder;
mod decoder;
mod unet;
mod scheduler;
mod embedding;

use super::*; // info, MlResult, Layer, Linear, Sequential, ... (from model/mod.rs)

// diffusion 하위 모듈 전용 import
use std::fmt::Debug;
use crate::{
    nn::{Conv2D, GroupNorm, activation::{SiLU, SoftmaxOp}},
    tensor::operators::{Concat, Cos, Mul, NearestUpsample2d, ReshapeOp, Sin, Transpose},
};
use crate::loss::MeanSquaredError;
use crate::tests::common::model::diffusion::unet::Unet;
use self::decoder::Decoder;
use self::encoder::{Encoder, SinusoidalPE};
use self::scheduler::Scheduler;

struct Diffusion {
    encoder: Encoder,
    unet: Unet,
    decoder: Decoder,
    scheduler: Scheduler,
    loss: MeanSquaredError,
}

struct StableDiffusion; // todo(DDPM 구현 후 예정)
