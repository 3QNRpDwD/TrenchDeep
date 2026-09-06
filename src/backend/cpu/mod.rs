use compute::CpuCompute;
use core::CpuCore;

use crate::MlResult;
use crate::backend::feature::{
    CPU_FEATURE_AVX, CPU_FEATURE_AVX2, CPU_FEATURE_AVX512F, DeviceFeatures,
};
use crate::backend::{Backend, Device, DeviceType};

mod compute;
mod core;
pub(crate) mod operations;
mod parallel;

#[derive(Debug)]
pub struct CpuBackend {
    core: CpuCore,
    compute: CpuCompute,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self {
            core: CpuCore::new(),
            compute: CpuCompute::new(),
        }
    }
}

impl Device for CpuBackend {
    fn new() -> MlResult<Self> {
        Ok(CpuBackend {
            core: CpuCore::new(),
            compute: CpuCompute::new(),
        })
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn get_features(&self) -> DeviceFeatures {
        let mut features = DeviceFeatures::new();

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

impl Backend for CpuBackend {
    fn device(&self) -> DeviceType {
        self.core.device_type()
    }

    fn calc_device_flops(&self) -> f64 {
        self.core.calc_device_flops()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() -> MlResult<()> {
        let backend = CpuBackend::new()?;

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let sum = backend.compute.add(&a, &b)?;
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        let product = backend.compute.multiply(&a, &b)?;
        assert_eq!(product, vec![4.0, 10.0, 18.0]);

        Ok(())
    }

    #[test]
    fn test_matmul() -> MlResult<()> {
        let backend = CpuBackend::new()?;

        // 2x2 matrices
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = backend.compute.matmul(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

        Ok(())
    }
}
