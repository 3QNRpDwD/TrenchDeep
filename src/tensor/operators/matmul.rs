use super::*;


impl Function<f32> for Matmul {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Performs matrix multiplication on two tensors
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the matrix multiplication
    // Handle empty tensors
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        if targets[0].data().is_empty() || targets[1].data().is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }
        let target_0 = targets[0];
        let target_0_shape = target_0.shape();
        let target_0_data = target_0.data();
        let target_1 = targets[1];
        let target_1_shape = target_1.shape();
        let target_1_data = target_1.data();

        let a = target_0_shape.len();
        let b = target_1_shape.len();

        let buffer = match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                match target_0.chk_shape(target_1) {
                    Err(e) => return Err(e),
                    _ => Tensor::<f32>::from_vec(vec![target_0_data.iter().zip(target_1_data.iter()).map(|(&a, &b)| a * b).sum::<f32>()], &vec![])?
                }
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if target_0_shape[1] != target_1_shape[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: target_0_shape.to_vec(),
                            right_shape: target_1_shape.to_vec(),
                        },
                    ));
                }
                let m = target_0_shape[0];
                let k = target_0_shape[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += target_0_data[i * k + j] * target_1_data[j];
                    }
                    data[i] = sum;
                }
                Tensor::<f32>::from_vec(data, &[m].to_vec())?
            }

            (1, 2) => {
                if target_0_shape[0] != target_1_shape[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: target_0_shape.to_vec(),
                            right_shape: target_1_shape.to_vec(),
                        },
                    ));
                }
                let k = target_0_shape[0];
                let n = target_1_shape[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += target_0_data[i] * target_1_data[i * n + j];
                    }
                    data[j] = sum;
                }
                Tensor::<f32>::from_vec(data, &[n].to_vec())?
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    target_0_shape[..a - 2].iter().product()
                } else {
                    1
                };
                let m = target_0_shape[a - 2];
                let k = target_0_shape[a - 1];
                let n = target_1_shape[b - 1];

                if k != target_1_shape[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: target_0_shape.to_vec(),
                            right_shape: target_1_shape.to_vec(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    target_1_shape[..b - 2].iter().product()
                } else {
                    1
                };

                let output_batch_size = if batch_size == 1 {
                    other_batch_size
                } else if other_batch_size == 1 {
                    batch_size
                } else if batch_size == other_batch_size {
                    batch_size
                } else {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: target_0_shape.to_vec(),
                            right_shape: target_1_shape.to_vec()
                        },
                    ));
                };

                let mut data = vec![0.0; output_batch_size * m * n];

                for batch in 0..output_batch_size {
                    let batch1 = if batch_size == 1 { 0 } else { batch };
                    let batch2 = if other_batch_size == 1 { 0 } else { batch };

                    let start1 = batch1 * m * k;
                    let start2 = batch2 * k * n;
                    let result_start = batch * m * n;

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum +=
                                    target_0_data[start1 + i * k + l] * target_1_data[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&target_0_shape[..a - 2]);
                    } else {
                        shape.extend_from_slice(&target_1_shape[..b - 2]);
                    }
                }
                shape.push(m);
                shape.push(n);
                Tensor::<f32>::from_vec(data, &shape)?
            }
        };
        Ok(vec![buffer])
    }


    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}


#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Matmul};
    use crate::tensor::{Tensor, TensorBase};
    use crate::{tensor_ops, MlResult};

    #[test]
    fn test_matmul_2d_2d() -> MlResult<()> {
        // Case 1: 2D * 2D Matrix Multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;
        let c = tensor_ops!(a, Matmul, b);


        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_2d() -> MlResult<()> {
        // Case 2: 1D * 2D (Vector-Matrix Multiplication)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[40.0, 46.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_2d_1d() -> MlResult<()> {
        // Case 3: 2D * 1D (Matrix-Vector Multiplication)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0], &[3])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[50.0, 122.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_3d() -> MlResult<()> {
        // Case 4: 3D * 3D (Batch Matrix Multiplication)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[2, 2, 2])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(
            c.data(),
            &[31.0, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0]
        );
        Ok(())
    }

    #[test]
    fn test_matmul_invalid_shapes() -> MlResult<()> {
        // Test incompatible shapes
        let matmul = Matmul::new()?;
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0], &[2])?;

        // This should return an error since the shapes are incompatible
        assert!(matmul.forward(&[&a, &b]).is_err());

        // Test incompatible batch dimensions
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

        // This should return an error since the batch dimensions don't match
        assert!(matmul.forward(&[&a, &b]).is_err());

        Ok(())
    }

    #[test]
    fn test_matmul_1x1() -> MlResult<()> {
        // Case 5: 1x1 Matrix Multiplication
        let a = Tensor::from_vec(vec![2.0], &[1, 1])?;
        let b = Tensor::from_vec(vec![3.0], &[1, 1])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c.data(), &[6.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_1d() -> MlResult<()> {
        // Case 6: 1D * 1D (Dot Product)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[]); // scalar output
        assert_eq!(c.data(), &[32.0]); // 1*4 + 2*5 + 3*6 = 32
        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d_broadcasting() -> MlResult<()> {
        // Case 7: 3D * 2D Broadcasting
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(
            c.data(),
            &[31.0, 34.0, 71.0, 78.0, 111.0, 122.0, 151.0, 166.0]
        );
        Ok(())
    }

    #[test]
    fn test_matmul_4d_4d() -> MlResult<()> {
        // Case 8: 4D * 4D Batch Matrix Multiplication
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,],
            &[2, 2, 2, 2]
        )?;
        let b = Tensor::from_vec(
            vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,],
            &[2, 2, 2, 2]
        )?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[2, 2, 2, 2]);
        let expected = vec![
            19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0, 19.0, 22.0,
            43.0, 50.0,
        ];
        assert_eq!(c.data(), &expected);
        Ok(())
    }

    #[test]
    fn test_matmul_empty() -> MlResult<()> {
        let matmul = Matmul::new()?;
        // Case 9: Empty Matrix Multiplication
        let a = Tensor::from_vec(vec![], &[0, 2])?;
        let b = Tensor::from_vec(vec![], &[2, 0])?;

        // This should return an error for empty tensors
        assert!(matmul.forward(&[&a, &b]).is_err());
        Ok(())
    }

    #[test]
    fn test_matmul_broadcast_batch_dims() -> MlResult<()> {
        // Case 10: Broadcasting with Different Batch Dimensions
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
        let b = Tensor::from_vec(
            vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0],
            &[3, 1, 2, 2]
        )?;
        let c = tensor_ops!(a, Matmul, b);

        assert_eq!(c.shape(), &[3, 1, 2, 2]);
        let expected = vec![
            19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0,
        ];
        assert_eq!(c.data(), &expected);
        Ok(())
    }
}