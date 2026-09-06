use crate::MlResult;
use crate::backend::feature::DeviceFeatures;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda,
}

impl Display for DeviceType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Device {
    fn new() -> MlResult<Self>
    where
        Self: Sized;
    fn device_type(&self) -> DeviceType;
    fn get_features(&self) -> DeviceFeatures;
}
