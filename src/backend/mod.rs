use std::fmt::{Debug, Display, Formatter};

pub use cpu::CpuBackend;
pub use device::{Device, DeviceType};

use crate::MlResult;

mod device;
mod feature;
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

#[derive(Debug)]
pub enum BackendError {
    Other(String),
}

impl Display for BackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Other(s) => write!(f, "{}", s),
        }
    }
}