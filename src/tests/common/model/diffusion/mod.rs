mod encoder;
mod decoder;
mod unet;
mod scheduler;
mod embedding;

use super::*; // info, MlResult, Layer, Linear, Sequential, ... (from model/mod.rs)

// diffusion 하위 모듈 전용 import
use std::fmt::Debug;
use crate::{
    nn::{Conv2DLayer, GroupNormLayer, activation::SiLULayer},
    tensor::operators::{Concat, Cos, Sin},
};

use self::decoder::Decoder;
use self::encoder::{Encoder, SinusoidalPE};

struct Diffusion {
    encoder: Encoder,
    decoder: Decoder,
    scheduler: MLP,
    loss: CrossEntropyLoss,
}

struct StableDiffusion {

}
