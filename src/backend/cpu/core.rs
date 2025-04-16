use crate::backend::DeviceType;
use crate::tensor::operators::Matmul;
use crate::tensor::operators::Function;
use crate::tensor::{Tensor, TensorBase};

#[derive(Debug)]
pub struct CpuCore;

impl CpuCore {
    pub fn new() -> Self {
        CpuCore
    }

    pub fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    pub fn calc_device_flops(&self) -> f64 {
        let matmul = Matmul::new().unwrap();
        // Create two large tensors for benchmarking
        let size = 1024;
        let elements = size * size;

        let a = Tensor::from_vec(vec![1.0; elements], &[size, size].to_vec()).unwrap();
        let b = Tensor::from_vec(vec![2.0; elements], &[size, size].to_vec()).unwrap();

        // Measure matrix multiplication time (more compute intensive than addition)
        let start = std::time::Instant::now();
        let _c = matmul.forward(&[&a, &b]).unwrap();
        let duration = start.elapsed();

        // Calculate FLOPS:
        // For matrix multiplication of (n x n) matrices:
        // Each element requires n multiplications and n-1 additions
        // Total operations = n * n * (2n - 1)
        let operations = size as u64 * size as u64 * (2 * size as u64 - 1);
        let flops = (operations as f64) / duration.as_secs_f64();

        flops
    }
}

#[cfg(test)]
mod tests {

fn pretty_flops(flops: f64) -> String {
        if flops >= 1_000_000_000_000.0 {
            format!("{:.2} Tflops/s", flops / 1_000_000_000_000.0)
        } else if flops >= 1_000_000_000.0 {
            format!("{:.2} Gflops/s", flops / 1_000_000_000.0)
        } else if flops >= 1_000_000.0 {
            format!("{:.2} Mflops/s", flops / 1_000_000.0)
        } else if flops >= 1_000.0 {
            format!("{:.2} Kflops/s", flops / 1_000.0)
        } else {
            format!("{:.2} flops/s", flops)
        }
    }
}
