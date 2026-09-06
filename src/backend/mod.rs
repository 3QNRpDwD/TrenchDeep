use std::fmt::Debug;

#[cfg(feature = "builtinKernels")]
pub use cpu::CpuBackend;
#[cfg(all(test, feature = "builtinKernels"))]
pub(crate) use cpu::operations::{
    approx_cos_value, approx_sin_value, conv2d_forward_data, group_norm_forward_data,
    max_pool2d_forward_data, nearest_upsample2d_forward_data,
};
#[cfg(feature = "builtinStorage")]
#[path = "cpu/storage.rs"]
mod storage;
pub use device::{Device, DeviceType};
#[cfg(feature = "builtinStorage")]
pub use storage::CpuTensorStore;

#[cfg(feature = "builtinKernels")]
mod cpu;
mod device;
mod feature;

/// Device metadata plus the same fallible operation contract used by the runtime.
pub trait Backend: crate::contracts::OperationProvider + Debug + Send + Sync {
    fn device(&self) -> DeviceType;
    fn calc_device_flops(&self) -> f64;
}

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("{0}")]
    CpuError(String),
    #[error("{0}")]
    Other(String),
}
