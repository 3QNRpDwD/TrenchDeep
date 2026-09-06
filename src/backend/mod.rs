use std::fmt::{Debug, Display, Formatter};

#[cfg(feature="builtinKernels")]
pub use cpu::CpuBackend;
#[cfg(all(test,feature="builtinKernels"))]
pub(crate) use cpu::operations::{approx_sin_value,approx_cos_value,conv2d_forward_data,max_pool2d_forward_data,nearest_upsample2d_forward_data,group_norm_forward_data};
#[cfg(feature="builtinStorage")]
#[path="cpu/storage.rs"]
mod storage;
#[cfg(feature="builtinStorage")]
pub use storage::CpuTensorStore;
pub use device::{Device, DeviceType};

use crate::MlResult;

mod device;
mod feature;
#[cfg(feature="builtinKernels")]
mod cpu;

pub trait Backend: Debug + Send + Sync {
    fn device(&self) -> DeviceType;
    fn calc_device_flops(&self) -> f64;
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn multiply(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>;
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
    fn exp(&self, a: &[f32]) -> Vec<f32>;
    fn log(&self, a: &[f32]) -> Vec<f32>;
    fn pow(&self, a: &[f32], power: f32) -> Vec<f32>;
    fn sqrt(&self, a: &[f32]) -> Vec<f32>;
    fn sum(&self, a: &[f32]) -> f32;
    fn mean(&self, a: &[f32]) -> f32;
    fn execute_compute(&self, _dimensions: [u32; 3]) -> MlResult<()>;
}

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("{0}")]
    CpuError(String),
    #[error("{0}")]
    Other(String),
}
