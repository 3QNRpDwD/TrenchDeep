use std::sync::Arc; // Assuming this is needed for Arc<dyn Backend>

use super::*; // Assuming this brings in Tensor, MlResult, MlError, TensorError, Function, Backend, CpuBackend etc.

// Matmul struct definition from context:
// pub struct Matmul {
//     backend: Arc<dyn Backend>,
// }

impl Function for Matmul {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        if targets.len() < 2 {
            return Err(MlError::StringError(
                "Matmul forward expects at least two target tensors".to_string(),
            ));
        }
        let tensor_0 = targets[0];
        let tensor_1 = targets[1];

        tensor_0.with_shape(|target_0_shape_slice| {
            tensor_0.with_data(|target_0_data_slice| {
                tensor_1.with_shape(|target_1_shape_slice| {
                    tensor_1.with_data(|target_1_data_slice| {
                        if target_0_data_slice.is_empty() || target_1_data_slice.is_empty() {
                            // This check handles cases where data is genuinely empty.
                            // For shapes like [0, N] or [N, 0], data would be empty.
                            return Err(MlError::TensorError(TensorError::EmptyTensor));
                        }

                        let a_dims = target_0_shape_slice.len();
                        let b_dims = target_1_shape_slice.len();

                        let buffer = match (a_dims, b_dims) {
                            // Case 1: 1D * 1D (dot product)
                            (1, 1) => {
                                if target_0_shape_slice[0] != target_1_shape_slice[0] {
                                    return Err(MlError::TensorError(TensorError::InvalidShape {
                                        // Using MatrixMultiplicationError for semantic consistency
                                        // as this is a form of matmul incompatibility.
                                        // Or a more specific DotProductShapeError.
                                        // For now, sticking to existing error types as much as possible.
                                        expected: target_0_shape_slice.to_vec(), // Or specifically the common dimension
                                        got: target_1_shape_slice.to_vec(), // Or specifically the common dimension
                                        // custom_message: Some("Inner dimensions must match for dot product.".to_string())
                                    }));
                                }
                                let sum = target_0_data_slice
                                    .iter()
                                    .zip(target_1_data_slice.iter())
                                    .map(|(&a_val, &b_val)| a_val * b_val)
                                    .sum::<f32>();
                                // Scalar output shape is `&[]` as per existing tests.
                                Tensor::from_vec(vec![sum], &[])?
                            }

                            // Case 2: 2D * 1D (Matrix-Vector)
                            (2, 1) => {
                                if target_0_shape_slice[1] != target_1_shape_slice[0] {
                                    return Err(MlError::TensorError(
                                        TensorError::MatrixMultiplicationError {
                                            left_shape: target_0_shape_slice.to_vec(),
                                            right_shape: target_1_shape_slice.to_vec(),
                                        },
                                    ));
                                }
                                let m = target_0_shape_slice[0];
                                let k = target_0_shape_slice[1]; // Inner dimension
                                let mut data = vec![0.0; m];

                                for i in 0..m {
                                    let mut sum = 0.0;
                                    for j_inner in 0..k {
                                        sum += target_0_data_slice[i * k + j_inner] * target_1_data_slice[j_inner];
                                    }
                                    data[i] = sum;
                                }
                                // Output shape is 1D vector [m]
                                Tensor::from_vec(data, &[m])?
                            }

                            // Case 3: 1D * 2D (Vector-Matrix)
                            (1, 2) => {
                                if target_0_shape_slice[0] != target_1_shape_slice[0] {
                                    return Err(MlError::TensorError(
                                        TensorError::MatrixMultiplicationError {
                                            left_shape: target_0_shape_slice.to_vec(),
                                            right_shape: target_1_shape_slice.to_vec(),
                                        },
                                    ));
                                }
                                let k = target_0_shape_slice[0]; // Inner dimension
                                let n = target_1_shape_slice[1];
                                let mut data = vec![0.0; n];

                                for j_outer in 0..n {
                                    let mut sum = 0.0;
                                    for i_inner in 0..k {
                                        sum += target_0_data_slice[i_inner] * target_1_data_slice[i_inner * n + j_outer];
                                    }
                                    data[j_outer] = sum;
                                }
                                // Output shape is 1D vector [n]
                                Tensor::from_vec(data, &[n])?
                            }

                            // Case 4: Higher dimensional tensor multiplication (includes 2D * 2D)
                            // This arm handles ad >= 2 && bd >= 2 based on original structure and tests.
                            // Other cases like (1, >=3) or (>=3, 1) would require more sophisticated
                            // unsqueezing or a more general matmul definition.
                            // The original code structure implies this arm is for when both tensors are at least 2D.
                            (ad, bd) if ad >= 2 && bd >= 2 => {
                                let m = target_0_shape_slice[ad - 2];
                                let k_left = target_0_shape_slice[ad - 1];
                                let k_right = target_1_shape_slice[bd - 2];
                                let n = target_1_shape_slice[bd - 1];

                                if k_left != k_right {
                                    return Err(MlError::TensorError(
                                        TensorError::MatrixMultiplicationError {
                                            left_shape: target_0_shape_slice.to_vec(),
                                            right_shape: target_1_shape_slice.to_vec(),
                                        },
                                    ));
                                }
                                let k_common = k_left;

                                let batch_dims_left_slice = &target_0_shape_slice[..ad - 2];
                                let batch_dims_right_slice = &target_1_shape_slice[..bd - 2];

                                let batch_numel_left: usize = batch_dims_left_slice.iter().product();
                                let batch_numel_right: usize = batch_dims_right_slice.iter().product();

                                let output_batch_numel: usize;
                                let mut final_output_shape_vec: Vec<usize> = Vec::new();

                                // Broadcasting logic for batch dimensions (simplified, as per original):
                                if batch_numel_left == 1 && batch_numel_right > 1 {
                                    output_batch_numel = batch_numel_right;
                                    final_output_shape_vec.extend_from_slice(batch_dims_right_slice);
                                } else if batch_numel_left > 1 && batch_numel_right == 1 {
                                    output_batch_numel = batch_numel_left;
                                    final_output_shape_vec.extend_from_slice(batch_dims_left_slice);
                                } else if batch_numel_left == batch_numel_right { // Also covers both numels being 1
                                    output_batch_numel = batch_numel_left;
                                    // If batch shapes are not identical, this simple broadcasting rule might be too naive.
                                    // Original code used left's batch shape if its numel > 1, else right's.
                                    // For numels equal and > 1, if shapes differ, it's an issue.
                                    // test_matmul_3d_3d [2,2,2]@[2,2,2] -> batch_left=[2], batch_right=[2].
                                    // Here batch_dims_left_slice == batch_dims_right_slice.
                                    if batch_dims_left_slice == batch_dims_right_slice {
                                        final_output_shape_vec.extend_from_slice(batch_dims_left_slice);
                                    } else if batch_numel_left == 1 { // Both scalar batches
                                        // final_output_shape_vec remains empty
                                    } else {
                                        // Numels are equal, >1, but batch shapes differ. This is an error.
                                        return Err(MlError::TensorError(TensorError::BroadcastError {
                                            from_shape: batch_dims_right_slice.to_vec(),
                                            to_shape: batch_dims_left_slice.to_vec(),
                                        }));
                                    }
                                } else { // batch_numel_left != batch_numel_right AND neither is 1. This is an error.
                                    return Err(MlError::TensorError(TensorError::BroadcastError {
                                        from_shape: batch_dims_right_slice.to_vec(),
                                        to_shape: batch_dims_left_slice.to_vec(),
                                    }));
                                }

                                let mut data = vec![0.0; output_batch_numel * m * n];

                                for batch_idx in 0..output_batch_numel {
                                    let left_matrix_batch_offset = if batch_numel_left == 1 { 0 } else { batch_idx };
                                    let right_matrix_batch_offset = if batch_numel_right == 1 { 0 } else { batch_idx };

                                    let start_left = left_matrix_batch_offset * m * k_common;
                                    let start_right = right_matrix_batch_offset * k_common * n;
                                    let result_start_offset = batch_idx * m * n;

                                    for i_row in 0..m {
                                        for j_col in 0..n {
                                            let mut sum = 0.0;
                                            for l_common in 0..k_common {
                                                sum += target_0_data_slice[start_left + i_row * k_common + l_common]
                                                    * target_1_data_slice[start_right + l_common * n + j_col];
                                            }
                                            data[result_start_offset + i_row * n + j_col] = sum;
                                        }
                                    }
                                }
                                final_output_shape_vec.push(m);
                                final_output_shape_vec.push(n);
                                Tensor::from_vec(data, &final_output_shape_vec)?
                            }
                            // Fallthrough for unhandled dimension combinations (e.g., (0,x), (x,0), (1, >=3), (>=3, 1))
                            _ => {
                                return Err(MlError::TensorError(
                                    TensorError::UnsupportedShapeForMatmul {
                                        left_shape: target_0_shape_slice.to_vec(),
                                        right_shape: target_1_shape_slice.to_vec(),
                                    },
                                ));
                            }
                        };
                        Ok(vec![buffer])
                    }) // End of with_data for tensor_1
                }) // End of with_shape for tensor_1
            }) // End of with_data for tensor_0
        }) // End of with_shape for tensor_0
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _targets: &[Tensor], _grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}


// Assume TensorError has these variants (from mod.rs and matmul.rs context):
// enum TensorError {
//     EmptyTensor,
//     InvalidShape { expected: Vec<usize>, got: Vec<usize> },
//     MatrixMultiplicationError { left_shape: Vec<usize>, right_shape: Vec<usize> },
//     UnsupportedShapeForMatmul { left_shape: Vec<usize>, right_shape: Vec<usize> },
//     BroadcastError { from_shape: Vec<usize>, to_shape: Vec<usize> },
//     // ... other errors
// }


#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Matmul};
    use crate::tensor::{Tensor, TensorBase}; // For .shape() and .data() in tests
    use crate::{tensor_ops, MlResult}; // For tensor_ops! macro

    // Helper for comparing tensors, assuming it uses with_data/with_shape if TensorBase methods are removed
    // For these tests, the original .shape() and .data() calls might be from TensorBase in tests.
    // If TensorBase is removed, these tests would need adjustment or direct use of with_data/with_shape.
    // Given mod.rs, TensorBase is still there and implemented for Tensor.

    #[test]
    fn test_matmul_2d_2d() -> MlResult<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[2, 2]);
        assert_eq!(c.with_data(|d| d.to_vec()), &[58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_2d() -> MlResult<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[2]);
        assert_eq!(c.with_data(|d| d.to_vec()), &[40.0, 46.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_2d_1d() -> MlResult<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0], &[3])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[2]);
        assert_eq!(c.with_data(|d| d.to_vec()), &[50.0, 122.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_3d() -> MlResult<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[2, 2, 2])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[2, 2, 2]);
        assert_eq!(
            c.with_data(|d| d.to_vec()),
            &[31.0, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0]
        );
        Ok(())
    }

    #[test]
    fn test_matmul_invalid_shapes() -> MlResult<()> {
        let matmul_op = Matmul::new()?; // Renamed to avoid conflict if Matmul is also a type/module
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;    // 1D
        let b = Tensor::from_vec(vec![4.0, 5.0], &[2])?;        // 1D, incompatible
        assert!(matmul_op.forward(&[a, b]).is_err()); // (1,1) case, inner dim mismatch

        let a2 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?; // 2D
        let b2 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?; // 2D, incompatible inner dim
        assert!(matmul_op.forward(&[a2, b2]).is_err()); // (2,2) case, k_left != k_right

        // Test incompatible batch dimensions - specific broadcasting rule
        // Example: left batch [2], right batch [3], numels differ and neither is 1.
        let a3 = Tensor::from_vec(vec![0.0; 2 * 2 * 2], &[2, 2, 2])?; // batch [2]
        let b3 = Tensor::from_vec(vec![0.0; 3 * 2 * 2], &[3, 2, 2])?; // batch [3]
        assert!(matmul_op.forward(&[a3,b3]).is_err());

        Ok(())
    }

    #[test]
    fn test_matmul_1x1_matrix() -> MlResult<()> { // Renamed from test_matmul_1x1 to be more descriptive
        let a = Tensor::from_vec(vec![2.0], &[1, 1])?;
        let b = Tensor::from_vec(vec![3.0], &[1, 1])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[1, 1]);
        assert_eq!(c.with_data(|d| d.to_vec()), &[6.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_1d_dot_product() -> MlResult<()> { // Renamed for clarity
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[]); // scalar output
        assert_eq!(c.with_data(|d| d.to_vec()), &[32.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d_broadcasting() -> MlResult<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?; // batch [2], M=2, K=2
        let b = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;                   // batch [], K=2, N=2 (scalar batch)
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[2, 2, 2]); // Output batch from 'a'
        assert_eq!(
            c.with_data(|d| d.to_vec()),
            // Batch 0 of a: [[1,2],[3,4]] matmul b [[9,10],[11,12]]
            // [1*9+2*11, 1*10+2*12] = [9+22, 10+24] = [31, 34]
            // [3*9+4*11, 3*10+4*12] = [27+44, 30+48] = [71, 78]
            // Batch 1 of a: [[5,6],[7,8]] matmul b
            // [5*9+6*11, 5*10+6*12] = [45+66, 50+72] = [111, 122]
            // [7*9+8*11, 7*10+8*12] = [63+88, 70+96] = [151, 166]
            &[31.0, 34.0, 71.0, 78.0, 111.0, 122.0, 151.0, 166.0]
        );
        Ok(())
    }

    #[test]
    fn test_matmul_4d_4d() -> MlResult<()> {
        // Data for a: 2x2 batches of 2x2 matrices
        let a_data = (0..16).map(|x| (x + 1) as f32).collect::<Vec<f32>>(); // Values 1 to 16
        // Data for b: 2x2 batches of 2x2 matrices, make them different
        let b_data = (0..16).map(|x| (16 - x) as f32).collect::<Vec<f32>>(); // Values 16 down to 1

        let a = Tensor::from_vec(a_data, &[2, 2, 2, 2])?; // batch [2,2]
        let b = Tensor::from_vec(b_data, &[2, 2, 2, 2])?; // batch [2,2]
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[2, 2, 2, 2]);

        // Calculate expected result for one batch item for verification (e.g. batch [0,0])
        // a_batch00 = [[1,2],[3,4]]
        // b_batch00 = [[16,15],[14,13]]
        // res_batch00_00 = 1*16 + 2*14 = 16 + 28 = 44
        // res_batch00_01 = 1*15 + 2*13 = 15 + 26 = 41
        // res_batch00_10 = 3*16 + 4*14 = 48 + 56 = 104
        // res_batch00_11 = 3*15 + 4*13 = 45 + 52 = 97
        // Expected for first 2x2 matrix: [44, 41, 104, 97]
        // The original test data seems to use same matrix for all batches in 'a' and 'b'
        // Let's re-use original test's values if they were specific.
        // Original test data:
        let a_orig_test = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,],
            &[2, 2, 2, 2]
        )?;
        let b_orig_test = Tensor::from_vec(
            vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,],
            &[2, 2, 2, 2]
        )?;
        let c_orig_test = tensor_ops!(a_orig_test, Matmul, b_orig_test);

        let expected_orig = vec![
            19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0, 19.0, 22.0,
            43.0, 50.0,
        ];
        assert_eq!(c_orig_test.with_data(|d| d.to_vec()), expected_orig);
        Ok(())
    }

    #[test]
    fn test_matmul_empty_tensor_error() -> MlResult<()> { // Renamed from test_matmul_empty
        let matmul_op = Matmul::new()?;
        let a = Tensor::from_vec(vec![], &[0, 2])?; // Data is empty
        let b = Tensor::from_vec(vec![1.0,2.0,3.0,4.0], &[2, 2])?;
        assert!(matmul_op.forward(&[a, b]).is_err());

        let a2 = Tensor::from_vec(vec![1.0,2.0,3.0,4.0], &[2,2])?;
        let b2 = Tensor::from_vec(vec![], &[2,0])?; // Data is empty
        assert!(matmul_op.forward(&[a2,b2]).is_err());
        Ok(())
    }

    #[test]
    fn test_matmul_broadcast_batch_dims() -> MlResult<()> {
        // Case: Left batch [1], Right batch [3,1]. Output batch [3,1]
        let matmul = Matmul::new()?;
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?; // Left: batch_shape [1], M=2, K=2
        let b_data_len = 3 * 1 * 2 * 2;
        let b_data = (0..b_data_len).map(|x| (x+5) as f32).collect::<Vec<f32>>();
        // b example data if needed: vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        let b = Tensor::from_vec(b_data, &[3, 1, 2, 2])?; // Right: batch_shape [3,1], K=2, N=2

        let c = matmul.forward(&[a, b])?.remove(0);

        assert_eq!(c.with_shape(|s| s.to_vec()), &[3, 1, 2, 2]); // Output batch from 'b'

        // Expected data: Matrix from 'a' is [[1,2],[3,4]]. This is broadcasted.
        // For each of the 3x1 batches of 'b', we matmul [[1,2],[3,4]] with that batch's 2x2 matrix.
        // b's matrices:
        // b[0,0]: [[5,6],[7,8]] -> res: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19,22; 43,50]
        // b[1,0]: [[9,10],[11,12]] -> res: [1*9+2*11, 1*10+2*12; 3*9+4*11, 3*10+4*12] = [31,34; 71,78]
        // b[2,0]: [[13,14],[15,16]] -> res: [1*13+2*15, 1*14+2*16; 3*13+4*15, 3*14+4*16] = [43,46; 99,106]
        let expected = vec![
            19.0, 22.0, 43.0, 50.0, // result for b[0,0]
            31.0, 34.0, 71.0, 78.0, // result for b[1,0]
            43.0, 46.0, 99.0, 106.0, // result for b[2,0]
        ];
        assert_eq!(c.with_data(|d| d.to_vec()), expected);
        Ok(())
    }
}