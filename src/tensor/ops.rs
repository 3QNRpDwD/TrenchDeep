use std::sync::Arc;
use crate::{backend, MlError, MlResult};
use crate::backend::{Backend, Device};
use crate::tensor::{Abs, Add, Div, Exp, Log, Matmax, Matmul, Mul, Neg, Pow, Sub, Sqrt, Square, Topk, Tensor, TensorError};
use crate::tensor::{TensorBase, Function};


impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Abs<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: S) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.backend().exp(self.tensor.data()), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Exp<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: S) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.tensor.data().iter().map(|&x| x.abs()).collect(), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F:, S> Function<F, S> for Log<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: S) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.tensor.data().iter().map(|&x| x.ln()).collect(), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Neg<F> {
    type Forwarded = MlResult<Tensor<F>>;
    type Gradiant = MlResult<(Box<dyn TensorBase<f32>>, Box<dyn TensorBase<f32>>)>;

    fn new(tensor: F, _: S) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.tensor.data().iter().map(|&x| -x).collect(), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Sqrt<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: S) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.backend().sqrt(self.tensor.data()), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Square<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: S) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.tensor.data().iter().map(|&x| x * x).collect(), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Add<F, S> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(first_tensor: F, second_tensor: Option<S>) -> MlResult<Self>  {
        Ok(Self {
            first_tensor,
            second_tensor: second_tensor.unwrap(),
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&self) -> Self::Forwarded {
        if self.first_tensor.shape().len() == 2 && self.second_tensor.shape().len() == 1 && self.first_tensor.shape()[1] == self.second_tensor.shape()[0] {
            let (_, features) = (self.first_tensor.shape()[0], self.first_tensor.shape()[1]);
            let mut data = vec![0.0; self.first_tensor.data().len()];

            for (i, chunk) in data.chunks_mut(features).enumerate() {
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = self.first_tensor.data()[i * features + j] + self.second_tensor.data()[j];
                }
            }
            return Tensor::from_vec(data, self.first_tensor.shape())
        }
        match self.chk_shape(&self.second_tensor) {
            Err(e) => Err(e),
            _ => Tensor::from_vec(
                self.backend()
                    .add(
                    self.first_tensor.data(),
                    self.second_tensor.data()
                    ),
                self.first_tensor.shape()
            )
        }
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Sub<F, S> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(first_tensor: F, second_tensor: Option<S>) -> MlResult<Self> {
        Ok(Self {
            first_tensor,
            second_tensor: second_tensor.unwrap(),
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from_vec the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&self) -> Self::Forwarded {
        if self.first_tensor.shape().len() == 2 && self.second_tensor.shape().len() == 1 && self.first_tensor.shape()[1] == self.second_tensor.shape()[0] {
            let (batch_size, features) = (self.first_tensor.shape()[0], self.first_tensor.shape()[1]);
            let mut data = vec![0.0; self.first_tensor.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = self.first_tensor.data()[i * features + j] - self.second_tensor.data()[j];
                }
            }
            return Tensor::from_vec(data, &self.first_tensor.shape());
        }

        match self.chk_shape(&self.second_tensor) {
            Err(e) => Err(e),
            _ => Tensor::from_vec(self.backend().sub(self.first_tensor.data(), self.second_tensor.data()), self.first_tensor.shape())
        }
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Mul<F, S> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(first_tensor: F, second_tensor: Option<S>) -> MlResult<Self> {
        Ok(Self {
            first_tensor,
            second_tensor: second_tensor.unwrap(),
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn forward(&self) -> Self::Forwarded {
        match self.chk_shape(&self.second_tensor) {
            Err(e) => Err(e),
            _ => Tensor::from_vec(self.backend().multiply(self.first_tensor.data(), self.second_tensor.data()), self.first_tensor.shape())
        }
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Div<F, S> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(first_tensor: F, second_tensor: Option<S>) -> MlResult<Self> {
        Ok(Self {
            first_tensor,
            second_tensor: second_tensor.unwrap(),
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&self) -> Self::Forwarded {

        match self.chk_shape(&self.second_tensor) {
            Err(e) => Err(e),
            _ => Tensor::from_vec(self.backend().div(self.first_tensor.data(), self.second_tensor.data()), self.first_tensor.shape())
        }
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Pow<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: Option<S>) -> MlResult<Self> {
        Ok(Self {
            tensor,
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self) -> Self::Forwarded {
        Tensor::from_vec(self.backend().pow(self.tensor.data(), self.tensor.power()), self.tensor.shape())
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Matmul<F, S> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(first_tensor: F, second_tensor: S) -> MlResult<Self> {
        Ok(Self {
            first_tensor,
            second_tensor: second_tensor.unwrap(),
            backend: Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Performs matrix multiplication on two tensors
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the matrix multiplication
    // Handle empty tensors
    fn forward(&self) -> Self::Forwarded {
        if self.first_tensor.data().is_empty() || self.second_tensor.data().is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let a = self.first_tensor.shape().len();
        let b = self.second_tensor.shape().len();

        match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                match self.chk_shape(&self.second_tensor) {
                    Err(e) => Err(e),
                    _ => Tensor::from_vec(
                        vec![self.first_tensor.data().iter().zip(self.second_tensor.data().iter()).map(|(&a, &b)| a * b).sum::<f32>()],
                        &vec![]
                    )
                }
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if self.shape()[1] != self.second_tensor.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.first_tensor.shape().to_vec(),
                            right_shape: self.second_tensor.shape().to_vec(),
                        },
                    ));
                }
                let m = self.first_tensor.shape()[0];
                let k = self.first_tensor.shape()[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += self.first_tensor.data()[i * k + j] * self.second_tensor.data()[j];
                    }
                    data[i] = sum;
                }
                Tensor::from_vec(data, &[m].to_vec())
            }

            (1, 2) => {
                if self.first_tensor.shape()[0] != self.second_tensor.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.first_tensor.shape().to_vec(),
                            right_shape: self.second_tensor.shape().to_vec(),
                        },
                    ));
                }
                let k = self.first_tensor.shape()[0];
                let n = self.second_tensor.shape()[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += self.first_tensor.data()[i] * self.second_tensor.data()[i * n + j];
                    }
                    data[j] = sum;
                }
                Tensor::from_vec(data, &[n].to_vec())
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    self.first_tensor.shape()[..a - 2].iter().product()
                } else {
                    1
                };
                let m = self.first_tensor.shape()[a - 2];
                let k = self.first_tensor.shape()[a - 1];
                let n = self.second_tensor.shape()[b - 1];

                if k != self.second_tensor.shape()[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.first_tensor.shape().to_vec(),
                            right_shape: self.second_tensor.shape().to_vec(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    self.second_tensor.shape()[..b - 2].iter().product()
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
                            left_shape: self.first_tensor.shape().to_vec(),
                            right_shape: self.second_tensor.shape().to_vec()
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
                                    self.first_tensor.data()[start1 + i * k + l] * self.second_tensor.data()[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&self.first_tensor.shape()[..a - 2]);
                    } else {
                        shape.extend_from_slice(&self.second_tensor.shape()[..b - 2]);
                    }
                }
                shape.push(m);
                shape.push(n);

                Tensor::from_vec(data, &shape)
            }
        }
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Topk<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: Option<S>) ->MlResult<Self>{
        Ok(Self {
            tensor,
            backend : Arc::new(backend::CpuBackend::new()?),
        })
    }

    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn forward(&self) -> Self::Forwarded {
        if self.tensor.topk().0 == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = self.tensor.shape().len() - 1;
        let last_dim_size = self.tensor.shape()[last_dim];

        if self.tensor.topk().0 > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    self.tensor.topk().0, last_dim_size
                ),
            }));
        }


        let slice_size = last_dim_size;
        let num_slices: usize = self.tensor.shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * self.tensor.topk().0);
        let mut indices = Vec::with_capacity(num_slices * self.tensor.topk().0);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &self.tensor.data()[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..self.tensor.topk().0];
            let mut selected = top_k.to_vec();
            if !self.tensor.topk().1 {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = self.tensor.shape().to_vec();
        new_shape[last_dim] = self.tensor.topk().0;

        Ok((
            Tensor::from_vec(values, &new_shape)?,
            Tensor::from_vec(indices, &new_shape)?,
        ))
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

impl<F: TensorBase<_>, S: TensorBase<_>> Function<F, S> for Matmax<F> {
    type Forwarded = MlResult<F>;
    type Gradiant = MlResult<(F, F)>;

    fn new(tensor: F, _: Option<S>) ->MlResult<Self>{
        Ok(Self {
            tensor,
            backend : Arc::new(backend::CpuBackend::new()?),
        })
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
    fn forward(&self) -> Self::Forwarded {
        match self.tensor.matmax().0 {
            None => {
                // Find global maximum
                let max_val = self.tensor.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                Ok((Tensor::from_vec(vec![max_val], &vec![1])?, None))

            }
            Some(d) => {
                let dim = if d < 0 {
                    (self.tensor.shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= self.tensor.shape().len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: self.tensor.shape().to_vec(),
                    }));
                }

                let mut new_shape = self.tensor.shape().to_vec();
                if !self.tensor.matmax().1 {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = self.tensor.shape()[dim + 1..].iter().product();
                let outer_stride: usize = self.tensor.shape()[dim..].iter().product();
                let outer_dims: usize = self.tensor.shape()[..dim].iter().product();
                let dim_size = self.tensor.shape()[dim];

                let mut max_values = Vec::with_capacity(self.tensor.data().len() / dim_size);
                let mut max_indices = Vec::with_capacity(self.tensor.data().len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = self.tensor.data()[idx];
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
                    Tensor::from_vec(max_values, &new_shape)?,
                    Some(Tensor::from_vec(max_indices, &new_shape)?),
                ))
            }
        }
    }

    fn backward(&self, grad: F) -> Self::Gradiant {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
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
impl std::ops::Add for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Self) -> Self::Output {
        Add::new(self, Some(other))?.forward()?
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
impl std::ops::Sub for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: Self) -> Self::Output {
        Sub::new(self, Some(other))?.forward()?
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
impl std::ops::Mul for Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: Self) -> Self::Output {
        Mul::new(self, Some(other))?.forward()?
    }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `_other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div for Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: Self) -> Self::Output {
        Div::new(self, Some(other))?.forward()?
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
impl std::ops::Add for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: &Tensor<f32>) -> Self::Output {
        Add::new(self, Some(other))?.forward()?
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
impl std::ops::Sub for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: &Tensor<f32>) -> Self::Output {
        Sub::new(self, Some(other))?.forward()?
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
impl std::ops::Mul for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: &Tensor<f32>) -> Self::Output {
        Mul::new(self, Some(other))?.forward()?
    }
}

/// Divide trait implementation for tensor references
///
/// # Arguments
/// * `_other` - Reference to the tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: &Tensor<f32>) -> Self::Output {
        Div::new(self, Some(other))?.forward()?
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
        let tensor: Tensor<f32> = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        ops!(tensor, Topk<Tensor<f32>>, 3)?;
        let (values, indices) = ops!(&tensor, Topk<Tensor<f32>>, &3, true)?;
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let tensor = Tensor::from_vec(
            vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0],
            &[2, 5],
        )?;
        let (values, indices) = ops!(&tensor, Topk<Tensor<f32>>, &2, true)?;
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = ops!(&tensor, Topk<Tensor<f32>>, &3, false)?;
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
    #[test]
    fn test_max() -> MlResult<()> {
        // Test global maximum
        let a = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let (max_all, _) = ops!(&a, Matmax, None, false)?;

        assert_eq!(max_all.data(), &[6.0]);

        // Test maximum along dimension 0
        let (max_dim0, indices0) = ops!(a, Matmax, 0, true)?;
        assert_eq!(max_dim0.shape(), &[1, 3]);
        assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(indices0.unwrap().data(), &[1.0, 1.0, 1.0]);

        // Test maximum along dimension 1
        let (max_dim1, indices1) = ops!(a.clone(), Matmax, 1, true)?;
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
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;
        let c = ops!(a, Matmul, b)?;


        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_2d() -> MlResult<()> {
        // Case 2: 1D * 2D (Vector-Matrix Multiplication)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[40.0, 46.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_2d_1d() -> MlResult<()> {
        // Case 3: 2D * 1D (Matrix-Vector Multiplication)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0], &[3])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[50.0, 122.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_3d() -> MlResult<()> {
        // Case 4: 3D * 3D (Batch Matrix Multiplication)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::from_vec(
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
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0], &[2])?;

        // This should return an error since the shapes are incompatible
        assert!(ops!(a, Matmul, b).is_err());

        // Test incompatible batch dimensions
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

        // This should return an error since the batch dimensions don't match
        assert!(ops!(a, Matmul, b).is_err());

        Ok(())
    }

    #[test]
    fn test_matmul_1x1() -> MlResult<()> {
        // Case 5: 1x1 Matrix Multiplication
        let a = Tensor::from_vec(vec![2.0], &[1, 1])?;
        let b = Tensor::from_vec(vec![3.0], &[1, 1])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c.data(), &[6.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_1d() -> MlResult<()> {
        // Case 6: 1D * 1D (Dot Product)
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[]); // scalar output
        assert_eq!(c.data(), &[32.0]); // 1*4 + 2*5 + 3*6 = 32
        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d_broadcasting() -> MlResult<()> {
        // Case 7: 3D * 2D Broadcasting
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;
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
        let a = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            ],
            &[2, 2, 2, 2],
        )?;
        let b = Tensor::from_vec(
            vec![
                5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
            ],
            &[2, 2, 2, 2],
        )?;
        let c = ops!(a, Matmul, b)?;

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
        let a = Tensor::from_vec(vec![], &[0, 2])?;
        let b = Tensor::from_vec(vec![], &[2, 0])?;

        // This should return an error for empty tensors
        assert!(ops!(a, Matmul, b).is_err());
        Ok(())
    }

    #[test]
    fn test_matmul_broadcast_batch_dims() -> MlResult<()> {
        // Case 10: Broadcasting with Different Batch Dimensions
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
        let b = Tensor::from_vec(
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