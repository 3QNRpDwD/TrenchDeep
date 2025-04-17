use super::*;


impl Function<f32> for Abs {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| x.abs()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Exp {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(self.backend().exp(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)|  target_data.exp() * grad_data)
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradiant, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Log {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| x.ln()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Neg {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Sum {
    fn new() -> MlResult<Self> {
        todo!()
    }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }
}

impl Function<f32> for Sqrt {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(self.backend().sqrt(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Square {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|x| x * x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)| 2.0  * target_data * grad_data )
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradiant, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Add {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let first_shape = first_target.shape();
        let second_shape = second_target.shape();

        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_target.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_target.data()[i * features + j] + second_target.data()[j];
                }
            }
            return Ok(vec![Tensor::<f32>::from_vec(data, first_shape)?])
        }

        match first_target.chk_shape(second_target) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().add(first_target.data(), second_target.data()), first_target.shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![grad.clone(), grad.clone()])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Sub {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from_vec the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![Tensor::<f32>::from_vec(data, &targets[0].shape())?])
        }

        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![grad.clone(), -grad.clone()])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Mul {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().multiply(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![
            self.forward(&[grad, targets[1]])?.remove(0),
            self.forward(&[grad, targets[0]])?.remove(0)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Div {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::<f32>::from_vec(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape())?])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let x1 = targets[1];

        Ok(vec![
            self.forward(&[grad, x1])?.remove(0), // grad / x2
            grad * self.forward(&[&-targets[0], &(x1 * x1)])?.remove(0) // grad * (-x0 / x1^2)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Pow {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), power: None }) }
    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(self.backend().pow(targets[0].data(), self.power.unwrap()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let power = self.power.unwrap();
        let target = targets[0];
        let forwarded = Tensor::<f32>::from_vec(self.backend().pow(target.data(), power - 1.0), target.shape())?; // x ** (c - 1)
        let result = Tensor::from_vec(
            forwarded
                .data()
                .iter()
                .map(|&x| power * x)
                .collect(), target.shape())?; // c * x ** (c - 1)
        Ok(vec![result * grad]) // c * x ** (c -1) * gy
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

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

        let a = targets[0].shape().len();
        let b = targets[1].shape().len();

        let buffer = match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                match targets[0].chk_shape(targets[1]) {
                    Err(e) => return Err(e),
                    _ => Tensor::<f32>::from_vec(vec![targets[0].data().iter().zip(targets[1].data().iter()).map(|(&a, &b)| a * b).sum::<f32>()], &vec![])?
                }
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if targets[0].shape()[1] != targets[1].shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: targets[0].shape().to_vec(),
                            right_shape: targets[1].shape().to_vec(),
                        },
                    ));
                }
                let m = targets[0].shape()[0];
                let k = targets[0].shape()[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += targets[0].data()[i * k + j] * targets[1].data()[j];
                    }
                    data[i] = sum;
                }
                Tensor::<f32>::from_vec(data, &[m].to_vec())?
            }

            (1, 2) => {
                if targets[0].shape()[0] != targets[1].shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: targets[0].shape().to_vec(),
                            right_shape: targets[1].shape().to_vec(),
                        },
                    ));
                }
                let k = targets[0].shape()[0];
                let n = targets[1].shape()[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += targets[0].data()[i] * targets[1].data()[i * n + j];
                    }
                    data[j] = sum;
                }
                Tensor::<f32>::from_vec(data, &[n].to_vec())?
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    targets[0].shape()[..a - 2].iter().product()
                } else {
                    1
                };
                let m = targets[0].shape()[a - 2];
                let k = targets[0].shape()[a - 1];
                let n = targets[1].shape()[b - 1];

                if k != targets[1].shape()[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: targets[0].shape().to_vec(),
                            right_shape: targets[1].shape().to_vec(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    targets[1].shape()[..b - 2].iter().product()
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
                            left_shape: targets[0].shape().to_vec(),
                            right_shape: targets[1].shape().to_vec()
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
                                    targets[0].data()[start1 + i * k + l] * targets[1].data()[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&targets[0].shape()[..a - 2]);
                    } else {
                        shape.extend_from_slice(&targets[1].shape()[..b - 2]);
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

impl Function<f32> for Topk {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), topk: None }) }
    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        if self.topk.unwrap().0 == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = targets[0].shape().len() - 1;
        let last_dim_size = targets[0].shape()[last_dim];

        if self.topk.unwrap().0 > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    self.topk.unwrap().0, last_dim_size
                ),
            }));
        }

        let slice_size = last_dim_size;
        let num_slices: usize = targets[0].shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * self.topk.unwrap().0);
        let mut indices = Vec::with_capacity(num_slices * self.topk.unwrap().0);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &targets[0].data()[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..self.topk.unwrap().0];
            let mut selected = top_k.to_vec();
            if !self.topk.unwrap().1 {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = targets[0].shape().to_vec();
        new_shape[last_dim] = self.topk.unwrap().0;

        Ok(vec![Tensor::<f32>::from_vec(values, &new_shape)?, Tensor::<f32>::from_vec(indices, &new_shape)?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Matmax {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), matmax: None }) }
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
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let buffer = match self.matmax.unwrap().0 {
            None => {
                // Find global maximum
                let max_val = targets[0].data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                vec![Tensor::<f32>::from_vec(vec![max_val], &vec![1])?, Tensor::<f32>::zeros()]
            }
            Some(d) => {
                let dim = if d < 0 {
                    (targets[0].shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= targets[0].shape().len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: targets[0].shape().to_vec(),
                    }));
                }

                let mut new_shape = targets[0].shape().to_vec();
                if !self.matmax.unwrap().1 {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = targets[0].shape()[dim + 1..].iter().product();
                let outer_stride: usize = targets[0].shape()[dim..].iter().product();
                let outer_dims: usize = targets[0].shape()[..dim].iter().product();
                let dim_size = targets[0].shape()[dim];

                let mut max_values = Vec::with_capacity(targets[0].data().len() / dim_size);
                let mut max_indices = Vec::with_capacity(targets[0].data().len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = targets[0].data()[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = k;
                            }
                        }

                        max_values.push(max_val);
                        max_indices.push(max_idx as f32);
                    }
                }

                vec![Tensor::<f32>::from_vec(max_values, &new_shape)?, Tensor::<f32>::from_vec(max_indices, &new_shape)?]
            }
        };

        Ok(buffer)
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

/// `Sin` 함수는 입력 텐서의 각 요소에 사인 함수를 적용합니다.
/// 입력 텐서는 각도로, 출력 텐서는 해당 각도의 사인 값을 포함합니다.
impl Function<f32> for Sin {
    /// 새로운 `Sin` 인스턴스를 생성합니다.
    /// CPU 백엔드를 사용합니다.
    fn new() -> MlResult<Self> {
        Ok(Self { backend: Arc::new(CpuBackend::new()? )} )
    }

    /// 입력 텐서에 사인 함수를 요소별로 적용하여, 동일한 모양을 가진 새로운 텐서를 반환합니다.
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|x| x.sin()).collect(), targets[0].shape())?])
    }

    /// 사인 함수의 기울기를 계산합니다.
    /// 반환되는 기울기 텐서는 입력 텐서와 동일한 모양을 가집니다.
    /// 각 요소는 입력 텐서의 해당 요소의 코사인 값과 다음 계층의 기울기 값의 곱입니다.
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradient = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target)|  target.cos() * grad_data)
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradient, targets[0].shape())?])
    }

    /// 연산에 사용되는 백엔드 객체의 참조를 반환
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

/// `Cos` 함수는 입력 텐서의 각 요소에 코사인 함수를 적용합니다.
/// 입력 텐서는 각도로, 출력 텐서는 해당 각도의 코사인 값을 포함합니다.
impl Function<f32> for Cos {
    /// 새로운 `Cos` 인스턴스를 생성합니다.
    /// CPU 백엔드를 사용합니다.
    fn new() -> MlResult<Self> {
        Ok(Self { backend: Arc::new(CpuBackend::new()? )} )
    }

    /// 입력 텐서에 코사인 함수를 요소별로 적용하여, 동일한 모양을 가진 새로운 텐서를 반환합니다.
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|x| x.cos()).collect(), targets[0].shape())?])
    }

    /// 코사인 함수의 기울기를 계산합니다.
    /// 반환되는 기울기 텐서는 입력 텐서와 동일한 모양을 가집니다.
    /// 각 요소는 입력 텐서의 해당 요소의 음수 사인 값과 다음 계층의 기울기 값의 곱입니다.
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradient = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target)|  -target.sin() * grad_data)
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradient, targets[0].shape())?])
    }

    /// 연산에 사용되는 백엔드 객체의 참조를 반환
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for ApproxSin {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
            threshold: 0.0001
        })
    }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let x = targets[0];
        let x_data = x.data();
        let mut result = x_data.to_vec(); // Start with x (first term of series)

        // Calculate powers for the series approximation
        let mut term_sign = -1.0;
        let mut current_power = 3;
        let mut x_power = self.backend.multiply(x_data, x_data); // x²
        x_power = self.backend.multiply(&x_power, x_data); // x³
        let mut factorial = 6.0; // 3!

        // Continue adding terms until desired threshold
        while current_power <= 15 { // Limiting to reasonable number of terms

            // Calculate term: x^n/n! with alternating sign
            let term_value = self.backend.div(&x_power, &vec![factorial; x_power.len()]);

            // Apply sign and add/subtract from result
            let term = self.backend.multiply(&term_value, &vec![term_sign; term_value.len()]);
            result = self.backend.add(&result, &term);

            // Prepare for next iteration
            term_sign *= -1.0;

            // Update power: x^n -> x^(n+2)
            x_power = self.backend.multiply(&x_power, x_data);
            x_power = self.backend.multiply(&x_power, x_data);

            // Update factorial: n! -> (n+2)!
            factorial *= (current_power + 1) as f32 * (current_power + 2) as f32;
            current_power += 2;
        }

        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        // The derivative of sin(x) is cos(x)
        // We can use the ApproxCos implementation for this
        let cos = ApproxCos {
            backend: Arc::clone(&self.backend),
            threshold: self.threshold,
        };

        let cos_output = cos.forward(targets)?;

        // Multiply the cos result with the incoming gradient
        let x = targets[0];
        cos_output[0].chk_shape(grad)?;

        let grad_data = grad.data();
        let cos_data = cos_output[0].data();
        let result = self.backend.multiply(cos_data, grad_data);
        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for ApproxCos {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
            threshold: 0.0001
        })
    }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let x = targets[0];
        let x_data = x.data();
        let mut result = vec![1.0; x_data.len()];

        // Calculate x²
        let x_squared = self.backend.multiply(x_data, x_data);
        let mut term_sign = -1.0;
        let mut current_power = 2;
        let mut x_power = x_squared.clone(); // Start with x²
        let mut factorial = 2.0; // 2!

        // Continue adding terms until desired threshold
        while current_power <= 14 { // Limiting to reasonable number of terms
            // Calculate term: x^n/n! with alternating sign
            let term_value = self.backend.div(&x_power, &vec![factorial; x_power.len()]);

            // Apply sign and add/subtract from result
            let term = self.backend.multiply(&term_value, &vec![term_sign; term_value.len()]);
            result = self.backend.add(&result, &term);

            // Prepare for next iteration
            term_sign *= -1.0;

            // Update power: x^n -> x^(n+2)
            x_power = self.backend.multiply(&x_power, x_squared.as_slice());

            // Update factorial: n! -> (n+2)!
            factorial *= (current_power + 1) as f32 * (current_power + 2) as f32;
            current_power += 2;
        }

        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        // The derivative of cos(x) is -sin(x)
        // We can use the ApproxSin implementation for this
        let sin = ApproxSin {
            backend: Arc::clone(&self.backend),
            threshold: self.threshold,
        };

        let sin_output = sin.forward(targets)?;
        // Multiply the -sin result with the incoming gradient
        let x = targets[0];
        sin_output[0].chk_shape(grad)?;

        let grad_data = grad.data();
        let sin_data = sin_output[0].data();

        // Apply negative sign to sin result
        let neg_sin = self.backend.multiply(sin_data, &vec![-1.0; sin_data.len()]);
        // Then multiply by gradient
        let result = self.backend.multiply(&neg_sin, grad_data);

        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

#[cfg(test)]
mod tests {
    use crate::{MlResult, tensor_ops};
    use crate::tensor::{Tensor, TensorBase};
    use crate::tensor::operators::{Function, Matmul, Topk, Matmax};

    #[test]
    fn test_topk() -> MlResult<()> {
        // Test 1: Basic 1D tensor
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor_ops!(buffer, Topk, 3, true);
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0], &[2, 5], )?;
        let (values, indices) = tensor_ops!(buffer, Topk, 2, true);
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor_ops!(buffer, Topk ,3, false);
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
    #[test]
    fn test_max() -> MlResult<()> {
        // Test global maximum
        let buffer = Tensor::<f32>::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let (max_all, _) = tensor_ops!(buffer, Matmax, None, false);
        assert_eq!(max_all.data(), &[6.0]);

        // Test maximum along dimension 0
        let (max_dim0, indices0) = tensor_ops!(buffer, Matmax, Some(0), true);
        assert_eq!(max_dim0.shape(), &[1, 3]);
        assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(indices0.data(), &[1.0, 1.0, 1.0]);

        // Test maximum along dimension 1
        let (max_dim1, indices1) = tensor_ops!(buffer, Matmax, Some(1), true);
        assert_eq!(max_dim1.shape(), &[2, 1]);
        assert_eq!(max_dim1.data(), &[3.0, 6.0]);
        assert_eq!(indices1.data(), &[2.0, 2.0]);

        // Test maximum with negative dimension
        let (max_neg, indices_neg) = tensor_ops!(buffer, Matmax, Some(-1), true);
        assert_eq!(max_neg.data(), &[3.0, 6.0]);
        assert_eq!(indices_neg.data(), &[2.0, 2.0]);

        Ok(())
    }

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