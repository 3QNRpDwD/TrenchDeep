use std::ops::{Add, Div, Mul, Sub};
use crate::{MlError, MlResult};
use crate::tensor::{Tensor, OpsLayer, TensorError, DefaultLayer};

impl<T: DefaultLayer> OpsLayer<T> for T {
    type Output = MlResult<Self>;

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn can_op(&self, other: &T) -> MlResult<()> {
        if self.shape() != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }

    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn add(&self, other: &T) -> Self::Output {
        if self.shape().len() == 2 && other.shape().len() == 1 && self.shape()[1] == other.shape()[0] {
            let (_batch_size, features) = (self.shape()[0], self.shape()[1]);
            let mut data = vec![0.0; self.data().len()];

            for (i, chunk) in data.chunks_mut(features).enumerate() {
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = self.data()[i * features + j] + other.data()[j];
                }
            }
            return DefaultLayer::from(data, self.shape())
        }
        match self.can_op(other) {
            Err(e) => Err(e),
            _ => DefaultLayer::from(self.backend().add(self.data(), other.data()), self.shape())
        }
    }

    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn sub(&self, other: &T) -> Self::Output {
        if self.shape().len() == 2 && other.shape().len() == 1 && self.shape()[1] == other.shape()[0] {
            let mut data = vec![0.0; self.data().len()];
            let (batch_size, features) = (self.shape()[0], self.shape()[1]);

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = self.data()[i * features + j] - other.data()[j];
                }
            }
            return DefaultLayer::from(data, &self.shape());
        }

        match self.can_op(other) {
            Err(e) => Err(e),
            _ => DefaultLayer::from(self.backend().sub(self.data(), other.data()), self.shape())
        }

    }

    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn mul(&self, other: &T) -> Self::Output {
        match self.can_op(other) {
            Err(e) => Err(e),
            _ => DefaultLayer::from(self.backend().multiply(self.data(), other.data()), self.shape())
        }
    }

    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn div(&self, other: &T) -> Self::Output {
        match self.can_op(other) {
            Err(e) => Err(e),
            _ => DefaultLayer::from(self.backend().div(self.data(), other.data()), self.shape())
        }
    }

    /// Adds a scalar to each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to add
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element + scalar
    fn add_scalar(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x + scalar).collect(), self.shape())
    }

    /// Subtracts a scalar from each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element - scalar
    fn sub_scalar(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x - scalar).collect(), self.shape())
    }

    /// Multiplies a scalar by each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element * scalar
    fn mul_scalar(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x * scalar).collect(), self.shape())
    }

    /// Divides each element in the tensor by a scalar
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element / scalar
    fn div_scalar(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x / scalar).collect(), self.shape())
    }

    /// Subtracts a scalar from each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract
    ///
    /// # Returns
    /// A new tensor with each element being scalar - tensor_element
    fn scalar_sub(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| scalar - x).collect(), self.shape())
    }

    /// Divides a scalar by each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide
    ///
    /// # Returns
    /// A new tensor with each element being scalar / tensor_element
    fn scalar_div(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| scalar / x).collect(), self.shape())
    }

    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn neg(&self) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| -x).collect(), self.shape())
    }

    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn exp(&self) -> Self::Output {
        DefaultLayer::from(self.backend().exp(self.data()), self.shape())
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn pow(&self, _power: f32) -> Self::Output {
        DefaultLayer::from(self.backend().pow(self.data(), _power), self.shape())
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `exponent` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ exponent
    fn pow_scalar(&self, exponent: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x.powf(exponent)).collect(), self.shape())
    }

    /// Raises a scalar to the power of each element in the tensor
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to raise
    ///
    /// # Returns
    /// A new tensor with each element being scalar ^ tensor_element
    fn scalar_pow(&self, scalar: f32) -> Self::Output{
        DefaultLayer::from(self.data().iter().map(|&x| scalar.powf(x)).collect(), self.shape())
    }

    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn sqrt(&self) -> Self::Output {
        DefaultLayer::from(self.backend().sqrt(self.data()), self.shape())
    }

    /// Returns a new tensor with the square of the elements of input
    ///
    /// Args:
    ///     None
    ///
    /// Returns:
    ///     A new tensor with each element being the square of the corresponding element in the input tensor
    fn square(&self) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x * x).collect(), self.shape())
    }

    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn log(&self) ->    Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| x.ln()).collect(), self.shape())
    }

    /// Performs matrix multiplication on two tensors
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the matrix multiplication
    fn matmul(&self, other: &T) -> Self::Output {
        // Handle empty tensors
        if self.data().is_empty() || other.data().is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let a = self.shape().len();
        let b = other.shape().len();

        match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                match self.can_op(other) {
                    Err(e) => Err(e),
                    _ => DefaultLayer::from(
                        vec![self.data().iter().zip(other.data().iter()).map(|(&a, &b)| a * b).sum::<f32>()],
                        &vec![]
                    )
                }
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if self.shape()[1] != other.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape().to_vec(),
                            right_shape: other.shape().to_vec(),
                        },
                    ));
                }
                let m = self.shape()[0];
                let k = self.shape()[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += self.data()[i * k + j] * other.data()[j];
                    }
                    data[i] = sum;
                }
                DefaultLayer::from(data, &[m].to_vec())
            }

            (1, 2) => {
                if self.shape()[0] != other.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape().to_vec(),
                            right_shape: other.shape().to_vec(),
                        },
                    ));
                }
                let k = self.shape()[0];
                let n = other.shape()[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += self.data()[i] * other.data()[i * n + j];
                    }
                    data[j] = sum;
                }
                DefaultLayer::from(data, &[n].to_vec())
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    self.shape()[..a - 2].iter().product()
                } else {
                    1
                };
                let m = self.shape()[a - 2];
                let k = self.shape()[a - 1];
                let n = other.shape()[b - 1];

                if k != other.shape()[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape().to_vec(),
                            right_shape: other.shape().to_vec(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    other.shape()[..b - 2].iter().product()
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
                            left_shape: self.shape().to_vec(),
                            right_shape: other.shape().to_vec()
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
                                    self.data()[start1 + i * k + l] * other.data()[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&self.shape()[..a - 2]);
                    } else {
                        shape.extend_from_slice(&other.shape()[..b - 2]);
                    }
                }
                shape.push(m);
                shape.push(n);

                DefaultLayer::from(data, &shape)
            }
        }
    }

    /// Compares each element in the tensor to a scalar and returns a new tensor with the result
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to compare each element to
    ///
    /// # Returns
    /// A new tensor with each element being 1.0 if tensor_element == scalar, otherwise 0.0
    fn eq_scalar(&self, scalar: f32) -> Self::Output {
        DefaultLayer::from(self.data().iter().map(|&x| (x == scalar) as i32 as f32).collect(), self.shape())
    }

    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn abs(&self) -> MlResult<Self> {
        DefaultLayer::from(self.data().iter().map(|&x| x.abs()).collect(), self.shape())
    }

    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn topk(&self, k: usize, sorted: bool) -> MlResult<(T, T)> {
        if k == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = self.shape().len() - 1;
        let last_dim_size = self.shape()[last_dim];

        if k > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    k, last_dim_size
                ),
            }));
        }


        let slice_size = last_dim_size;
        let num_slices: usize = self.shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * k);
        let mut indices = Vec::with_capacity(num_slices * k);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &self.data()[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..k];
            let mut selected = top_k.to_vec();
            if !sorted {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[last_dim] = k;

        Ok((
            DefaultLayer::from(values, &new_shape)?,
            DefaultLayer::from(indices, &new_shape)?,
        ))
    }

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
    fn matmax(&self, dim: Option<i32>, keepdim: bool) -> MlResult<(Tensor, Option<Tensor>)> {
        match dim {
            None => {
                // Find global maximum
                let max_val = self.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                Ok((DefaultLayer::from(vec![max_val], &vec![1])?, None))

            }
            Some(d) => {
                let dim = if d < 0 {
                    (self.shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= self.shape().len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: self.shape().to_vec(),
                    }));
                }

                let mut new_shape = self.shape().to_vec();
                if !keepdim {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = self.shape()[dim + 1..].iter().product();
                let outer_stride: usize = self.shape()[dim..].iter().product();
                let outer_dims: usize = self.shape()[..dim].iter().product();
                let dim_size = self.shape()[dim];

                let mut max_values = Vec::with_capacity(self.data().len() / dim_size);
                let mut max_indices = Vec::with_capacity(self.data().len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = self.data()[idx];
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
                    DefaultLayer::from(max_values, &new_shape)?,
                    Some(DefaultLayer::from(max_indices, &new_shape)?),
                ))
            }
        }
    }
}

/// Add trait implementation for owned tensors
///
/// # Arguments
/// * `_other` - The tensor to add to self
///
/// # Returns
/// A new tensor containing the element-wise sum
///
/// # Broadcasting
/// * Supports broadcasting when adding a 1D tensor to each row of a 2D tensor
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, _other: Self) -> Self::Output {
        OpsLayer::add(&self, &_other).unwrap()
    }
}

/// Subtract trait implementation for owned tensors
///
/// # Arguments
/// * `_other` - The tensor to subtract from self
///
/// # Returns
/// A new tensor containing the element-wise difference
///
/// # Broadcasting
/// * Supports broadcasting when subtracting a 1D tensor from each row of a 2D tensor
impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, _other: Self) -> Self::Output {
        OpsLayer::sub(&self, &_other).unwrap()
    }
}

/// Multiply trait implementation for owned tensors
///
/// # Arguments
/// * `_other` - The tensor to multiply with self
///
/// # Returns
/// A new tensor containing the element-wise product (Hadamard product)
///
/// # Note
/// * This performs element-wise multiplication, not matrix multiplication
/// * For matrix multiplication, use `matmul()` instead
impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, _other: Self) -> Self::Output {
        OpsLayer::mul(&self, &_other).unwrap()
    }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `_other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl Div for Tensor {
    type Output = Tensor;

    fn div(self, _other: Self) -> Self::Output {
        OpsLayer::div(&self, &_other).unwrap()
    }
}

/// Add trait implementation for tensor references
///
/// # Arguments
/// * `_other` - Reference to the tensor to add to self
///
/// # Returns
/// A new tensor containing the element-wise sum
///
/// # Broadcasting
/// * Supports broadcasting when adding a 1D tensor to each row of a 2D tensor
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, _other: &Tensor) -> Self::Output {
        OpsLayer::add(self, _other).unwrap()
    }
}

/// Subtract trait implementation for tensor references
///
/// # Arguments
/// * `_other` - Reference to the tensor to subtract from self
///
/// # Returns
/// A new tensor containing the element-wise difference
///
/// # Broadcasting
/// * Supports broadcasting when subtracting a 1D tensor from each row of a 2D tensor
impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, _other: &Tensor) -> Self::Output {
        OpsLayer::sub(self, _other).unwrap()
    }
}

/// Multiply trait implementation for tensor references
///
/// # Arguments
/// * `_other` - Reference to the tensor to multiply with self
///
/// # Returns
/// A new tensor containing the element-wise product
///
/// # Note
/// * This performs element-wise multiplication, not matrix multiplication
/// * For matrix multiplication, use `matmul()` instead
impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, _other: &Tensor) -> Self::Output {
        OpsLayer::mul(self, _other).unwrap()
    }
}

/// Divide trait implementation for tensor references
///
/// # Arguments
/// * `_other` - Reference to the tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, _other: &Tensor) -> Self::Output {
        OpsLayer::div(self, _other).unwrap()
    }
}

