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
    tensor::operators::{Concat, Cos, Mul, ReshapeOp, Sin, Transpose},
};
use crate::loss::MeanSquaredError;
use self::decoder::Decoder;
use self::encoder::{Encoder, SinusoidalPE};
use self::scheduler::Scheduler;

struct Diffusion {
    encoder: Encoder,
    unet: Box<dyn Layer>,
    decoder: Decoder,
    scheduler: Scheduler,
    loss: MeanSquaredError,
}

struct StableDiffusion; // todo(DDPM 구현 후 예정)
