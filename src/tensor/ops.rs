use crate::{MlError, MlResult};
use crate::tensor::{Abs, TensorBase, Exp, Function, Log, Matmax, Matmul, Neg, OpsLayer, Pow, Sqrt, Square, TensorError, Topk};

impl<T: TensorBase> OpsLayer<T> for T {
    type Output = MlResult<Self>;

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn chk_shape(&self, other: &T) -> MlResult<()> {
        if self.shape() != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }

    /// Adds a scalar to each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to add
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element + scalar
    fn add_scalar(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| x + scalar).collect(), self.shape())
    }

    /// Subtracts a scalar from each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element - scalar
    fn sub_scalar(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| x - scalar).collect(), self.shape())
    }

    /// Multiplies a scalar by each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element * scalar
    fn mul_scalar(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| x * scalar).collect(), self.shape())
    }

    /// Divides each element in the tensor by a scalar
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element / scalar
    fn div_scalar(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| x / scalar).collect(), self.shape())
    }

    /// Subtracts a scalar from each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract
    ///
    /// # Returns
    /// A new tensor with each element being scalar - tensor_element
    fn scalar_sub(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| scalar - x).collect(), self.shape())
    }

    /// Divides a scalar by each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide
    ///
    /// # Returns
    /// A new tensor with each element being scalar / tensor_element
    fn scalar_div(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| scalar / x).collect(), self.shape())
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `exponent` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ exponent
    fn pow_scalar(&self, exponent: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| x.powf(exponent)).collect(), self.shape())
    }

    /// Raises a scalar to the power of each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to raise
    ///
    /// # Returns
    /// A new tensor with each element being scalar ^ tensor_element
    fn scalar_pow(&self, scalar: f32) -> Self::Output{
        TensorBase::from(self.data().iter().map(|&x| scalar.powf(x)).collect(), self.shape())
    }

    /// Compares each element in the tensor to a scalar and returns a new tensor with the result
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to compare each element to
    ///
    /// # Returns
    /// A new tensor with each element being 1.0 if tensor_element == scalar, otherwise 0.0
    fn eq_scalar(&self, scalar: f32) -> Self::Output {
        TensorBase::from(self.data().iter().map(|&x| (x == scalar) as i32 as f32).collect(), self.shape())
    }
}

impl<T: TensorBase> Function<T> for Abs<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.backend().exp(self.first.data()), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Exp<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.data().iter().map(|&x| x.abs()).collect(), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Log<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.data().iter().map(|&x| x.ln()).collect(), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Neg<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.data().iter().map(|&x| -x).collect(), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Sqrt<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.backend().sqrt(self.first.data()), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Square<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.data().iter().map(|&x| x * x).collect(), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for crate::tensor::Add<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&self) -> Self::Output {
        if self.first.shape().len() == 2 && self.second.shape().len() == 1 && self.first.shape()[1] == self.second.shape()[0] {
            let (_, features) = (self.first.shape()[0], self.first.shape()[1]);
            let mut data = vec![0.0; self.first.data().len()];

            for (i, chunk) in data.chunks_mut(features).enumerate() {
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = self.first.data()[i * features + j] + self.second.data()[j];
                }
            }
            return TensorBase::from(data, self.first.shape())
        }
        match self.first.chk_shape(&self.second) {
            Err(e) => Err(e),
            _ => TensorBase::from(
                self.first.backend()
                    .add(
                    self.first.data(),
                    self.second.data()
                    ),
                self.first.shape()
            )
        }
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for crate::tensor::Sub<T> {
    type Output = MlResult<T>;
    type Gradient = f64;
    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&self) -> Self::Output {
        if self.first.shape().len() == 2 && self.second.shape().len() == 1 && self.first.shape()[1] == self.second.shape()[0] {
            let (batch_size, features) = (self.first.shape()[0], self.first.shape()[1]);
            let mut data = vec![0.0; self.first.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = self.first.data()[i * features + j] - self.second.data()[j];
                }
            }
            return TensorBase::from(data, &self.first.shape());
        }

        match self.first.chk_shape(&self.second) {
            Err(e) => Err(e),
            _ => TensorBase::from(self.first.backend().sub(self.first.data(), self.second.data()), self.first.shape())
        }
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for crate::tensor::Mul<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn forward(&self) -> Self::Output {
        match self.first.chk_shape(&self.second) {
            Err(e) => Err(e),
            _ => TensorBase::from(self.first.backend().multiply(self.first.data(), self.second.data()), self.first.shape())
        }
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for crate::tensor::Div<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&self) -> Self::Output {

        match self.first.chk_shape(&self.second) {
            Err(e) => Err(e),
            _ => TensorBase::from(self.first.backend().div(self.first.data(), self.second.data()), self.first.shape())
        }
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Pow<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self) -> Self::Output {
        TensorBase::from(self.first.backend().pow(self.first.data(), self.power), self.first.shape())
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Matmul<T> {
    type Output = MlResult<T>;
    type Gradient = f64;

    /// Performs matrix multiplication on two tensors
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the matrix multiplication
    // Handle empty tensors
    fn forward(&self) -> Self::Output {
        if self.first.data().is_empty() || self.second.data().is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let a = self.first.shape().len();
        let b = self.second.shape().len();

        match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                match self.first.chk_shape(&self.second) {
                    Err(e) => Err(e),
                    _ => TensorBase::from(
                        vec![self.first.data().iter().zip(self.second.data().iter()).map(|(&a, &b)| a * b).sum::<f32>()],
                        &vec![]
                    )
                }
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if self.first.shape()[1] != self.second.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.first.shape().to_vec(),
                            right_shape: self.second.shape().to_vec(),
                        },
                    ));
                }
                let m = self.first.shape()[0];
                let k = self.first.shape()[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += self.first.data()[i * k + j] * self.second.data()[j];
                    }
                    data[i] = sum;
                }
                TensorBase::from(data, &[m].to_vec())
            }

            (1, 2) => {
                if self.first.shape()[0] != self.second.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.first.shape().to_vec(),
                            right_shape: self.second.shape().to_vec(),
                        },
                    ));
                }
                let k = self.first.shape()[0];
                let n = self.second.shape()[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += self.first.data()[i] * self.second.data()[i * n + j];
                    }
                    data[j] = sum;
                }
                TensorBase::from(data, &[n].to_vec())
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    self.first.shape()[..a - 2].iter().product()
                } else {
                    1
                };
                let m = self.first.shape()[a - 2];
                let k = self.first.shape()[a - 1];
                let n = self.second.shape()[b - 1];

                if k != self.second.shape()[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.first.shape().to_vec(),
                            right_shape: self.second.shape().to_vec(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    self.second.shape()[..b - 2].iter().product()
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
                            left_shape: self.first.shape().to_vec(),
                            right_shape: self.second.shape().to_vec()
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
                                    self.first.data()[start1 + i * k + l] * self.second.data()[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&self.first.shape()[..a - 2]);
                    } else {
                        shape.extend_from_slice(&self.second.shape()[..b - 2]);
                    }
                }
                shape.push(m);
                shape.push(n);

                TensorBase::from(data, &shape)
            }
        }
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Topk<T> {
    type Output = MlResult<(T, T)>;
    type Gradient = f64;

    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn forward(&self) -> Self::Output {
        if self.k == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = self.first.shape().len() - 1;
        let last_dim_size = self.first.shape()[last_dim];

        if self.k > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    self.k, last_dim_size
                ),
            }));
        }


        let slice_size = last_dim_size;
        let num_slices: usize = self.first.shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * self.k);
        let mut indices = Vec::with_capacity(num_slices * self.k);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &self.first.data()[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..self.k];
            let mut selected = top_k.to_vec();
            if !self.sorted {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = self.first.shape().to_vec();
        new_shape[last_dim] = self.k;

        Ok((
            TensorBase::from(values, &new_shape)?,
            TensorBase::from(indices, &new_shape)?,
        ))
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

impl<T: TensorBase> Function<T> for Matmax<T> {
    type Output = MlResult<(T, Option<T>)>;
    type Gradient = f64;

    /// Returns the maximum value of all elements in the input tensor.
    /// If dim is specified, returns the maximum values along the given dimension.
    ///
    /// # Arguments
    /// * `dim` - Optional dimension along which to find the maximum values
    /// * `keepdim` - Whether the output tensor has dim retained or not
    ///
    /// # Returns
    /// If dim is None, returns a tensor with a single element containing the maximum value.
    /// If dim is specified, returns a tuple of two tensors (values, indices) containing the
    /// maximum values and their indices along the specified dimension.
    fn forward(&self) -> Self::Output {
        match self.dim {
            None => {
                // Find global maximum
                let max_val = self.first.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                Ok((TensorBase::from(vec![max_val], &vec![1])?, None))

            }
            Some(d) => {
                let dim = if d < 0 {
                    (self.first.shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= self.first.shape().len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: self.first.shape().to_vec(),
                    }));
                }

                let mut new_shape = self.first.shape().to_vec();
                if !self.keepdim {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = self.first.shape()[dim + 1..].iter().product();
                let outer_stride: usize = self.first.shape()[dim..].iter().product();
                let outer_dims: usize = self.first.shape()[..dim].iter().product();
                let dim_size = self.first.shape()[dim];

                let mut max_values = Vec::with_capacity(self.first.data().len() / dim_size);
                let mut max_indices = Vec::with_capacity(self.first.data().len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = self.first.data()[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = k;
                            }
                        }

                        max_values.push(max_val);
                        max_indices.push(max_idx as f32);
                    }
                }

                Ok((
                    TensorBase::from(max_values, &new_shape)?,
                    Some(TensorBase::from(max_indices, &new_shape)?),
                ))
            }
        }
    }

    fn backward(&self, grad: Self::Gradient) -> Self::Output {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops;
    use crate::tensor::Tensor;

    use super::*;

    #[test]
    fn test_topk() -> MlResult<()> {
        // Test 1: Basic 1D tensor
        let tensor: Tensor = TensorBase::from(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = ops!(tensor, Topk, 3, true)?;
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let tensor: Tensor = TensorBase::from(
            vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0],
            &[2, 5],
        )?;
        let (values, indices) = ops!(tensor, Topk, 2, true)?;
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let tensor: Tensor = TensorBase::from(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = ops!(tensor, Topk, 3, false)?;
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
    #[test]
    fn test_max() -> MlResult<()> {
        // Test global maximum
        let a = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let (max_all, _) = ops!(a.clone(), Matmax, None, false)?;

        assert_eq!(max_all.data(), &[6.0]);

        // Test maximum along dimension 0
        let (max_dim0, indices0) = ops!(a.clone(), Matmax, Some(0), true)?;
        assert_eq!(max_dim0.shape(), &[1, 3]);
        assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(indices0.unwrap().data(), &[1.0, 1.0, 1.0]);

        // Test maximum along dimension 1
        let (max_dim1, indices1) = ops!(a.clone(), Matmax, Some(1), true)?;
        assert_eq!(max_dim1.shape(), &[2, 1]);
        assert_eq!(max_dim1.data(), &[3.0, 6.0]);
        assert_eq!(indices1.unwrap().data(), &[2.0, 2.0]);

        // Test maximum with negative dimension
        let (max_neg, indices_neg) = ops!(a.clone(), Matmax, Some(-1), true)?;
        assert_eq!(max_neg.data(), &[3.0, 6.0]);
        assert_eq!(indices_neg.unwrap().data(), &[2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_matmul_2d_2d() -> MlResult<()> {
        // Case 1: 2D * 2D Matrix Multiplication
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b: Tensor = TensorBase::from(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;
        let c = ops!(a, Matmul, b)?;


        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_2d() -> MlResult<()> {
        // Case 2: 1D * 2D (Vector-Matrix Multiplication)
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0], &[3])?;
        let b: Tensor = TensorBase::from(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[40.0, 46.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_2d_1d() -> MlResult<()> {
        // Case 3: 2D * 1D (Matrix-Vector Multiplication)
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b: Tensor = TensorBase::from(vec![7.0, 8.0, 9.0], &[3])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[50.0, 122.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_3d() -> MlResult<()> {
        // Case 4: 3D * 3D (Batch Matrix Multiplication)
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b: Tensor = TensorBase::from(
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[2, 2, 2],
        )?;
        let c = ops!(a, Matmul, b)?;

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
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0], &[3])?;
        let b: Tensor = TensorBase::from(vec![4.0, 5.0], &[2])?;

        // This should return an error since the shapes are incompatible
        assert!(ops!(a, Matmul, b).is_err());

        // Test incompatible batch dimensions
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

        // This should return an error since the batch dimensions don't match
        assert!(ops!(a, Matmul, b).is_err());

        Ok(())
    }

    #[test]
    fn test_matmul_1x1() -> MlResult<()> {
        // Case 5: 1x1 Matrix Multiplication
        let a: Tensor = TensorBase::from(vec![2.0], &[1, 1])?;
        let b: Tensor = TensorBase::from(vec![3.0], &[1, 1])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c.data(), &[6.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_1d() -> MlResult<()> {
        // Case 6: 1D * 1D (Dot Product)
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0], &[3])?;
        let b: Tensor = TensorBase::from(vec![4.0, 5.0, 6.0], &[3])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[]); // scalar output
        assert_eq!(c.data(), &[32.0]); // 1*4 + 2*5 + 3*6 = 32
        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d_broadcasting() -> MlResult<()> {
        // Case 7: 3D * 2D Broadcasting
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b: Tensor = TensorBase::from(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;
        let c = ops!(a, Matmul, b)?;

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
        let a: Tensor = TensorBase::from(
            vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            ],
            &[2, 2, 2, 2],
        )?;
        let b: Tensor = TensorBase::from(
            vec![
                5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
            ],
            &[2, 2, 2, 2],
        )?;
        let c: Tensor = ops!(a, Matmul, b)?;

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
        // Case 9: Empty Matrix Multiplication
        let a: Tensor = TensorBase::from(vec![], &[0, 2])?;
        let b: Tensor = TensorBase::from(vec![], &[2, 0])?;

        // This should return an error for empty tensors
        assert!(ops!(a, Matmul, b).is_err());
        Ok(())
    }

    #[test]
    fn test_matmul_broadcast_batch_dims() -> MlResult<()> {
        // Case 10: Broadcasting with Different Batch Dimensions
        let a: Tensor = TensorBase::from(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
        let b: Tensor = TensorBase::from(
            vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0],
            &[3, 1, 2, 2],
        )?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[3, 1, 2, 2]);
        let expected = vec![
            19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0, 19.0, 22.0, 43.0, 50.0,
        ];
        assert_eq!(c.data(), &expected);
        Ok(())
    }
}