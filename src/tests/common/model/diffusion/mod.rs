pub(crate) mod encoder;
pub(crate) mod decoder;
mod unet;
mod scheduler;
mod embedding;

pub(crate) use super::*; // info, MlResult, Layer, Linear, Sequential, ... (from model/mod.rs)

// diffusion 하위 모듈 전용 import
pub(crate) use std::fmt::Debug;
pub(crate) use crate::{
    nn::{Conv2DLayer, GroupNormLayer, activation::SiLULayer},
    tensor::operators::{Concat, Cos, Sin},
};

// diffusion 내부 서브모듈 re-export (하위 파일에서 super:: 로 접근 가능)
pub(crate) use self::decoder::Decoder;
pub(crate) use self::encoder::{Encoder, SinusoidalPE};

struct DiffusionModel {
    encoder: Encoder,
    decoder: Decoder,
    scheduler: MLP,
    loss: CrossEntropyLoss,
}
