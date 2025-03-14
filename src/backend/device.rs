use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::sync::Mutex;
use std::sync::Once;

use crate::backend::feature::*;
use crate::MlResult;

static INIT: Once = Once::new();
static mut GLOBAL_DEVICE_MANAGER: Option<DeviceManager> = None;
static mut DEFAULT_DEVICE: Option<Mutex<DeviceType>> = None;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda
}

impl Display for DeviceType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct DeviceManager {
    available_devices: HashSet<DeviceType>,
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceManager {
    pub fn new() -> Self {
        let mut available_devices = HashSet::new();
        available_devices.insert(DeviceType::Cpu);
        println!("Available devices: {:?}", available_devices);
        Self { available_devices }
    }

    pub fn available_devices(&self) -> &HashSet<DeviceType> {
        &self.available_devices
    }

    pub fn global() -> &'static DeviceManager {
        unsafe {
            INIT.call_once(|| {
                GLOBAL_DEVICE_MANAGER = Some(DeviceManager::new());

                // Initialize default device
                let _manager = GLOBAL_DEVICE_MANAGER.as_ref().unwrap();

                // Select default device based on priority and availability
                let device_type = {
                    {
                        DeviceType::Cpu
                    }
                };

                DEFAULT_DEVICE = Some(Mutex::new(device_type));
                println!("Default device set to: {:?}", device_type);
            });
            GLOBAL_DEVICE_MANAGER.as_ref().unwrap()
        }
    }

    pub fn get_default_device() -> DeviceType {
        unsafe {
            if let Some(ref mutex) = DEFAULT_DEVICE {
                *mutex.lock().unwrap()
            } else {
                DeviceType::Cpu
            }
        }
    }

    pub fn get_features(&self) -> DeviceFeatures {
        let mut features = DeviceFeatures::new();

        // Add CPU features
        #[cfg(target_arch = "x86_64")]
        {
            features.add_feature(
                CPU_FEATURE_AVX,
                is_x86_feature_detected!("avx"),
                Some("Advanced Vector Extensions".to_string()),
            );

            features.add_feature(
                CPU_FEATURE_AVX2,
                is_x86_feature_detected!("avx2"),
                Some("Advanced Vector Extensions 2".to_string()),
            );

            features.add_feature(
                CPU_FEATURE_AVX512F,
                is_x86_feature_detected!("avx512f"),
                Some("AVX-512 Foundation".to_string()),
            );
        }
        features
    }
}

pub trait Device {
    fn new() -> MlResult<Self>
    where
        Self: Sized;
    fn device_type(&self) -> DeviceType;
    fn get_features(&self) -> DeviceFeatures;
}
