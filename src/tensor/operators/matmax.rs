use super::*; // Assuming this brings in Tensor, MlResult, MlError, TensorError, Function, etc.
// and potentially Backend, CpuBackend if they were part of the original 'super::*'
// For the sake of this refactoring, I'll assume necessary types are in scope.
use std::sync::Arc; // Added because Arc is used for backend.

// Assuming Matmax struct definition from the context:
// pub struct Matmax {
//     backend: Arc<dyn Backend>, // Or specific backend like CpuBackend
//     matmax: Option<(Option<i32>, bool)>, // (dim_option, keepdim_flag)
// }
// And CpuBackend::new() is available.
// And Tensor::zeros exists and takes &[usize].

impl Function for Matmax {
    fn new() -> MlResult<Self> {
        // Assuming CpuBackend and Backend are defined and accessible
        // For example, if CpuBackend needs to be created:
        // struct CpuBackend; impl CpuBackend { fn new() -> Result<Self, ()> { Ok(Self) } }
        // trait Backend {} impl Backend for CpuBackend {}
        // This part depends on your actual backend implementation.
        // For now, let's use a placeholder if CpuBackend isn't fully defined in context.
        // Ok(Self { backend: Arc::new(CpuBackend::new()?), matmax: None })
        // Given the original file:
        Ok(Self { backend: Arc::new(CpuBackend::new()?), matmax: None })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        let target_0 = targets[0];
        let (dim_option, keepdim_flag) = self.matmax.ok_or_else(|| MlError::StringError("Matmax parameters not set".to_string()))?;

        target_0.with_shape(|target_0_shape_slice| {
            target_0.with_data(|target_0_data_slice| {
                match dim_option {
                    None => {
                        let max_val = target_0_data_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        Ok(vec![
                            Tensor::from_vec(vec![max_val], &[1])?,
                            Tensor::zeros(target_0_shape_slice)
                        ])
                    }
                    Some(d) => {
                        let dim = if d < 0 {
                            (target_0_shape_slice.len() as i32 + d) as usize
                        } else {
                            d as usize
                        };

                        if dim >= target_0_shape_slice.len() {
                            return Err(MlError::TensorError(TensorError::InvalidAxis {
                                axis: dim,
                                shape: target_0_shape_slice.to_vec(),
                            }));
                        }

                        let mut new_shape = target_0_shape_slice.to_vec();
                        if !keepdim_flag {
                            if !new_shape.is_empty() {
                                new_shape.remove(dim);
                            }
                            if new_shape.is_empty() {
                                new_shape.push(1);
                            }
                        } else {
                            if !new_shape.is_empty() {
                                new_shape[dim] = 1;
                            } else {
                                // This case (keepdim=true on an empty shape input) is ill-defined.
                                // For a scalar input (shape []), dim >= len is true, caught above.
                                // If it somehow reached here, new_shape would be [], new_shape[dim] would panic.
                                // However, dim >= target_0_shape_slice.len() handles scalar inputs.
                                // If input shape was e.g. [0] and dim=0, keepdim=true. new_shape was [0], becomes [1].
                                if target_0_shape_slice.get(dim).map_or(false, |&s| s == 0) {
                                    // If original dim was 0 and we keepdim, it becomes 1.
                                    // e.g. [N,0,M] reduced on dim 1, keepdim=T -> [N,1,M]
                                }
                            }
                        }
                        if new_shape.is_empty() && target_0_shape_slice.is_empty() && keepdim_flag {
                            // Special case: input scalar [], keepdim=true for a non-existent dim (e.g. dim=0)
                            // This is usually caught by dim >= target_0_shape_slice.len()
                            // If input is scalar and keepdim=true, output shape should be [1]
                            new_shape.push(1);
                        }


                        let stride: usize = target_0_shape_slice.get(dim + 1..).map_or(1, |s| s.iter().product());
                        let outer_dims: usize = target_0_shape_slice.get(..dim).map_or(1, |s| s.iter().product());
                        let dim_size = target_0_shape_slice.get(dim).copied().unwrap_or(0);
                        let capacity: usize = new_shape.iter().product();
                        let mut max_values = Vec::with_capacity(capacity);
                        let mut max_indices = Vec::with_capacity(capacity);
                        let outer_stride_val: usize = target_0_shape_slice.get(dim..).map_or(1, |s| s.iter().product());

                        if target_0_data_slice.is_empty() && capacity > 0 && dim_size == 0 {
                            for _ in 0..capacity {
                                max_values.push(f32::NEG_INFINITY);
                                max_indices.push(0.0);
                            }
                        } else if !target_0_data_slice.is_empty() || capacity == 0 {
                            for i in 0..outer_dims {
                                for j in 0..stride {
                                    let mut max_val_local = f32::NEG_INFINITY;
                                    let mut max_idx_local = 0;

                                    if dim_size > 0 {
                                        for k in 0..dim_size {
                                            let current_idx = i * outer_stride_val + k * stride + j;
                                            if current_idx < target_0_data_slice.len() {
                                                let val = target_0_data_slice[current_idx];
                                                if val > max_val_local {
                                                    max_val_local = val;
                                                    max_idx_local = k;
                                                }
                                            } else {
                                                return Err(MlError::StringError(format!(
                                                    "Index out of bounds in Matmax: idx {}, len {}",
                                                    current_idx, target_0_data_slice.len()
                                                )));
                                            }
                                        }
                                    }
                                    // If dim_size is 0, max_val_local remains NEG_INFINITY, max_idx_local remains 0.
                                    // This is the desired behavior (e.g., max over an empty set).
                                    max_values.push(max_val_local);
                                    max_indices.push(max_idx_local as f32);
                                }
                            }
                        }


                        Ok(vec![
                            Tensor::from_vec(max_values, &new_shape)?,
                            Tensor::from_vec(max_indices, &new_shape)?
                        ])
                    }
                }
            })
        }) 
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _targets: &[Tensor], _grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }
    
    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    // Assuming these are correctly pathed
    // TensorBase for .data() and .shape() in tests
    use crate::{
        tensor::operators::{Function, Matmax},
        tensor::{Tensor, TensorBase, tests::assert_tensor_eq},
        tensor_ops,
        MlResult
    };

    // Helper to compare tensor data and shape for tests.
    // This might need adjustment if tests rely on old .data() / .shape() methods
    // For the refactored code, tests should ideally use with_data/with_shape or have accessors.
    // The existing tests in mod.rs use assert_tensor_eq which uses with_data/with_shape.


    #[test]
    fn test_max() -> MlResult<()> {
        let mut matmax = Matmax::new()?; // Ensure Matmax can be created without error
        matmax.matmax = Some((None, false)); // Set matmax parameters for global max
        
        // Test global maximum
        let buffer = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // shape [2,3]
        // tensor_ops! macro directly calls forward after setting matmax
        let mut results = matmax.forward(&[buffer])?;
        let (max_all_val, _max_all_idx) = (results.remove(0), results.remove(0));

        // Expected global max value tensor. The shape of value tensor for global max is typically [1].
        let expected_max_val = Tensor::from_vec(vec![6.0], &[1])?;
        assert_tensor_eq(&max_all_val, &expected_max_val)?;
        // _max_all_idx would be Tensor::zeros(&[2,3]) per current refactored code for global max.

        // Test maximum along dimension 0
        matmax.matmax = Some((Some(0), true));
        let mut results = matmax.forward(&[buffer])?;
        let (max_dim0_val, max_dim0_idx) = (results.remove(0), results.remove(0));
        let expected_max_dim0_val = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[1, 3])?;
        let expected_max_dim0_idx = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3])?;
        
        assert_tensor_eq(&max_dim0_val, &expected_max_dim0_val)?;
        assert_tensor_eq(&max_dim0_idx, &expected_max_dim0_idx)?;

        // Test maximum along dimension 1
        matmax.matmax = Some((Some(1), true));
        let mut results = matmax.forward(&[buffer])?;
        let (max_dim1_val, max_dim1_idx) = (results.remove(0), results.remove(0));
        let expected_max_dim1_val = Tensor::from_vec(vec![3.0, 6.0], &[2, 1])?;
        let expected_max_dim1_idx = Tensor::from_vec(vec![2.0, 2.0], &[2, 1])?;
        
        assert_tensor_eq(&max_dim1_val, &expected_max_dim1_val)?;
        assert_tensor_eq(&max_dim1_idx, &expected_max_dim1_idx)?;

        // Test maximum with negative dimension (-1, i.e., last dimension, which is dim 1 for a [2,3] tensor)
        matmax.matmax = Some((Some(-1), true));
        let mut results = matmax.forward(&[buffer])?;
        let (max_neg_val, max_neg_idx) = (results.remove(0), results.remove(0));

        assert_tensor_eq(&max_neg_val, &expected_max_dim1_val)?; // Should be same as dim 1
        assert_tensor_eq(&max_neg_idx, &expected_max_dim1_idx)?; // Should be same as dim 1

        Ok(())
    }
}