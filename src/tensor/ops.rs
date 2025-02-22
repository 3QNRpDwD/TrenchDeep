use std::ops::Deref;
use std::sync::Arc;
use crate::{binary, MlError, MlResult};
use crate::backend::{CpuBackend, Device};
use crate::tensor::{Abs, Add, Div, Exp, Log, Matmax, Matmul, Mul, Neg, Pow, Sub, Sqrt, Square, Topk, Tensor, TensorError, ArcTensor, Operator, TensorBase, Function, UnaryOp, BinaryOp, SpecialOp};

impl Function<f32> for Abs<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: Self::Operator) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.op.tensor.data().iter().map(|&x| x.abs()).collect(), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }
        // self.op.output = Some(buffer.tensor.clone()); stmt expr 에 대한 구현이 아직 추가되지 않음.
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Exp<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = Self::Forwarded;

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: UnaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.backend.exp(&self.op.tensor.data()), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        let grad = grad.data().iter()
            .zip(self.op.output.as_ref().unwrap().data().iter())
            .map(|(grad_val, data)| grad_val  * data)
            .collect();

        Tensor::<f32>::from_vec(grad, self.op.output.as_ref().unwrap().shape())
    }
}

impl Function<f32> for Log<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: UnaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.op.tensor.data().iter().map(|&x| x.ln()).collect(), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }

        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Neg<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: UnaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Negates each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the negation of tensor_element
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.op.tensor.data().iter().map(|&x| -x).collect(), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Sqrt<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;

    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: UnaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.backend.sqrt(self.op.tensor.data()), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Square<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = Self::Forwarded;

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: UnaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.op.tensor.data().iter().map(|x| x * x).collect(), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        let grad = grad.data().iter()
            .zip(self.op.tensor.data().iter())
            .map(|(grad_val, data)| 2.0 * grad_val  * data)
            .collect();

        Tensor::<f32>::from_vec(grad, self.op.tensor.shape())
    }
}

impl Function<f32> for Add<f32> {
    type Operator = BinaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = (ArcTensor<f32>, ArcTensor<f32>);

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: BinaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&mut self) -> Self::Forwarded {
        if self.op.first_tensor.shape().len() == 2 && self.op.second_tensor.shape().len() == 1 && self.op.first_tensor.shape()[1] == self.op.second_tensor.shape()[0] {
            let (_, features) = (self.op.first_tensor.shape()[0], self.op.first_tensor.shape()[1]);
            let mut data = vec![0.0; self.op.first_tensor.data().len()];

            for (i, chunk) in data.chunks_mut(features).enumerate() {
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = self.op.first_tensor.data()[i * features + j] + self.op.second_tensor.data()[j];
                }
            }
             let buffer = Tensor::<f32>::from_vec(data, self.op.first_tensor.shape())?;
            #[cfg(feature = "enable_backpropagation")]
                    {
            self.op.output = Some(buffer.tensor.clone());
        }
            return Ok(buffer)
        }

        match self.op.first_tensor.chk_shape(self.op.second_tensor.deref()) {
            Err(e) => Err(e),
            _ => {
                let buffer = Tensor::<f32>::from_vec(self.backend.add(self.op.first_tensor.data(), self.op.second_tensor.data()), self.op.first_tensor.shape())?;
                #[cfg(feature = "enable_backpropagation")]
                        {
            self.op.output = Some(buffer.tensor.clone());
        }
                Ok(buffer)
            }
        }
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        (grad.clone(), grad.clone())
    }
}

impl Function<f32> for Sub<f32> {
    type Operator = BinaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }
    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: BinaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Subtracts two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from_vec the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise subtraction
    fn forward(&mut self) -> Self::Forwarded {
        let buffer: ArcTensor<f32>;
        if self.op.first_tensor.shape().len() == 2 && self.op.second_tensor.shape().len() == 1 && self.op.first_tensor.shape()[1] == self.op.second_tensor.shape()[0] {
            let (batch_size, features) = (self.op.first_tensor.shape()[0], self.op.first_tensor.shape()[1]);
            let mut data = vec![0.0; self.op.first_tensor.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = self.op.first_tensor.data()[i * features + j] - self.op.second_tensor.data()[j];
                }
            }
            buffer = Tensor::<f32>::from_vec(data, &self.op.first_tensor.shape())?;
            #[cfg(feature = "enable_backpropagation")]
                    {
            self.op.output = Some(buffer.tensor.clone());
        }
            return Ok(buffer)
        }

        match self.op.first_tensor.chk_shape(self.op.second_tensor.deref()) {
            Err(e) => Err(e),
            _ => {
                buffer = Tensor::<f32>::from_vec(self.backend.sub(self.op.first_tensor.data(), self.op.second_tensor.data()), self.op.first_tensor.shape())?;
                #[cfg(feature = "enable_backpropagation")]
                        {
            self.op.output = Some(buffer.tensor.clone());
        }
                Ok(buffer)
            }
        }
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Mul<f32> {
    type Operator = BinaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: BinaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn forward(&mut self) -> Self::Forwarded {
        match self.op.first_tensor.chk_shape(self.op.second_tensor.deref()) {
            Err(e) => Err(e),
            _ => {
                let buffer = Tensor::<f32>::from_vec(self.backend.multiply(self.op.first_tensor.data(), self.op.second_tensor.data()), self.op.first_tensor.shape())?;
                #[cfg(feature = "enable_backpropagation")]
                        {
            self.op.output = Some(buffer.tensor.clone());
        }
                Ok(buffer)
            }
        }
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Div<f32> {
    type Operator = BinaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: BinaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Divides two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to divide the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise division
    fn forward(&mut self) -> Self::Forwarded {
        match self.op.first_tensor.chk_shape(self.op.second_tensor.deref()) {
            Err(e) => Err(e),
            _ => {
                 let buffer = Tensor::<f32>::from_vec(self.backend.div(self.op.first_tensor.data(), self.op.second_tensor.data()), self.op.first_tensor.shape())?;
                #[cfg(feature = "enable_backpropagation")]
                        {
            self.op.output = Some(buffer.tensor.clone());
        }
                return Ok(buffer)
            }
        }
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Pow<f32> {
    type Operator = UnaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?), power: None })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: UnaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
            power: None,
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?), power: None };
        new.op.start = true;
        Ok(new)
    }

    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&mut self) -> Self::Forwarded {
        let buffer = Tensor::<f32>::from_vec(self.backend.pow(self.op.tensor.data(), self.power.unwrap()), self.op.tensor.shape())?;
        #[cfg(feature = "enable_backpropagation")]
                {
            self.op.output = Some(buffer.tensor.clone());
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

impl Function<f32> for Matmul<f32> {
    type Operator = BinaryOp<f32>;
    type Forwarded = MlResult<ArcTensor<f32>>;
    type Gradiant = ();

    fn new(op: Self::Operator) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: BinaryOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
        })
    }

    fn start(op: Self::Operator)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?) };
        new.op.start = true;
        Ok(new)
    }

    /// Performs matrix multiplication on two tensors
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the matrix multiplication
    // Handle empty tensors
    fn forward(&mut self) -> Self::Forwarded {
        if self.op.first_tensor.data().is_empty() || self.op.second_tensor.data().is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let a = self.op.first_tensor.shape().len();
        let b = self.op.second_tensor.shape().len();

        let buffer =  match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                match self.op.first_tensor.chk_shape(self.op.second_tensor.deref()) {
                    Err(e) => return Err(e),
                    _ => {
                        Tensor::<f32>::from_vec(
                            vec![self.op.first_tensor.data().iter().zip(self.op.second_tensor.data().iter()).map(|(&a, &b)| a * b).sum::<f32>()],
                            &vec![]
                        )?
                    }
                }
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if self.op.first_tensor.shape()[1] != self.op.second_tensor.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.op.first_tensor.shape().to_vec(),
                            right_shape: self.op.second_tensor.shape().to_vec(),
                        },
                    ));
                }
                let m = self.op.first_tensor.shape()[0];
                let k = self.op.first_tensor.shape()[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += self.op.first_tensor.data()[i * k + j] * self.op.second_tensor.data()[j];
                    }
                    data[i] = sum;
                }
                Tensor::<f32>::from_vec(data, &[m].to_vec())?
            }

            (1, 2) => {
                if self.op.first_tensor.shape()[0] != self.op.second_tensor.shape()[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.op.first_tensor.shape().to_vec(),
                            right_shape: self.op.second_tensor.shape().to_vec(),
                        },
                    ));
                }
                let k = self.op.first_tensor.shape()[0];
                let n = self.op.second_tensor.shape()[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += self.op.first_tensor.data()[i] * self.op.second_tensor.data()[i * n + j];
                    }
                    data[j] = sum;
                }
                Tensor::<f32>::from_vec(data, &[n].to_vec())?
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    self.op.first_tensor.shape()[..a - 2].iter().product()
                } else {
                    1
                };
                let m = self.op.first_tensor.shape()[a - 2];
                let k = self.op.first_tensor.shape()[a - 1];
                let n = self.op.second_tensor.shape()[b - 1];

                if k != self.op.second_tensor.shape()[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.op.first_tensor.shape().to_vec(),
                            right_shape: self.op.second_tensor.shape().to_vec(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    self.op.second_tensor.shape()[..b - 2].iter().product()
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
                            left_shape: self.op.first_tensor.shape().to_vec(),
                            right_shape: self.op.second_tensor.shape().to_vec()
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
                                    self.op.first_tensor.data()[start1 + i * k + l] * self.op.second_tensor.data()[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&self.op.first_tensor.shape()[..a - 2]);
                    } else {
                        shape.extend_from_slice(&self.op.second_tensor.shape()[..b - 2]);
                    }
                }
                shape.push(m);
                shape.push(n);
                Tensor::<f32>::from_vec(data, &shape)?
            }
        };
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some(buffer.tensor.clone());
        }
        Ok(buffer)
    }


    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> Self::Gradiant {
        todo!()
    }
}

// impl Function<f32> for Topk<f32> {
//     type Operator = UnaryOp<f32>;
//     type Forwarded = MlResult<(ArcTensor<f32>, ArcTensor<f32>)>;
//     #[cfg(feature = "enable_backpropagation")]
//     type Gradiant = ();
//
// }

// impl Function<f32> for Matmax<f32> {
//     type Operator = UnaryOp<f32>;
//     type Forwarded = MlResult<(ArcTensor<f32>, ArcTensor<f32>)>;
//     #[cfg(feature = "enable_backpropagation")]
//     type Gradiant = ();
// }

impl Topk<f32> {
    fn new(op: SpecialOp<f32>) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?), topk: None })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: SpecialOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
            topk: None,
        })
    }

    fn start(op: SpecialOp<f32>)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?), topk: None };
        new.op.start = true;
        Ok(new)
    }

    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn forward(&mut self) -> MlResult<(ArcTensor<f32>, ArcTensor<f32>)> {
        if self.topk.unwrap().0 == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = self.op.tensor.shape().len() - 1;
        let last_dim_size = self.op.tensor.shape()[last_dim];

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
        let num_slices: usize = self.op.tensor.shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * self.topk.unwrap().0);
        let mut indices = Vec::with_capacity(num_slices * self.topk.unwrap().0);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &self.op.tensor.data()[start_idx..end_idx];
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

        let mut new_shape = self.op.tensor.shape().to_vec();
        new_shape[last_dim] = self.topk.unwrap().0;

        let buffer = (Tensor::<f32>::from_vec(values, &new_shape)?, Tensor::<f32>::from_vec(indices, &new_shape)?);
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some((buffer.0.tensor.clone(), buffer.1.tensor.clone()));
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> () {
        todo!()
    }
}

impl Matmax<f32> {
    fn new(op: SpecialOp<f32>) -> MlResult<Self> {
        Ok(Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?), matmax: None })
    }

    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: SpecialOp<f32>) -> MlResult<Self> {
        Ok(Self {
            pr_fn: Some(pr_fn),
            op,
            backend: Arc::new(CpuBackend::new()?),
            matmax: None,
        })
    }

    fn start(op: SpecialOp<f32>)  -> MlResult<Self> {
        let mut new = Self { pr_fn: None, op, backend: Arc::new(CpuBackend::new()?), matmax: None };
        new.op.start = true;
        Ok(new)
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
    fn forward(&mut self) -> MlResult<(ArcTensor<f32>, ArcTensor<f32>)> {
        let buffer = match self.matmax.unwrap().0 {
            None => {
                // Find global maximum
                let max_val = self.op.tensor.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                (Tensor::<f32>::from_vec(vec![max_val], &vec![1])?, Tensor::<f32>::zeros())
            }
            Some(d) => {
                let dim = if d < 0 {
                    (self.op.tensor.shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= self.op.tensor.shape().len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: self.op.tensor.shape().to_vec(),
                    }));
                }

                let mut new_shape = self.op.tensor.shape().to_vec();
                if !self.matmax.unwrap().1 {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = self.op.tensor.shape()[dim + 1..].iter().product();
                let outer_stride: usize = self.op.tensor.shape()[dim..].iter().product();
                let outer_dims: usize = self.op.tensor.shape()[..dim].iter().product();
                let dim_size = self.op.tensor.shape()[dim];

                let mut max_values = Vec::with_capacity(self.op.tensor.data().len() / dim_size);
                let mut max_indices = Vec::with_capacity(self.op.tensor.data().len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = self.op.tensor.data()[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = k;
                            }
                        }

                        max_values.push(max_val);
                        max_indices.push(max_idx as f32);
                    }
                }

                (Tensor::<f32>::from_vec(max_values, &new_shape)?, Tensor::<f32>::from_vec(max_indices, &new_shape)?)
            }
        };
        #[cfg(feature = "enable_backpropagation")]
        {
            self.op.output = Some((buffer.0.tensor.clone(), buffer.1.tensor.clone()))
        }
        Ok(buffer)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<f32>) -> () {
        todo!()
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
impl std::ops::Add for ArcTensor<f32> {
    type Output = ArcTensor<f32>;

    fn add(self, other: ArcTensor<f32>) -> Self::Output {
        Add::<f32>::new(binary!(self, other).unwrap()).unwrap().forward().unwrap()
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
impl std::ops::Sub for ArcTensor<f32> {
    type Output = ArcTensor<f32>;

    fn sub(self, other: ArcTensor<f32>) -> Self::Output {
        Sub::<f32>::new(binary!(self, other).unwrap()).unwrap().forward().unwrap()
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
impl std::ops::Mul for ArcTensor<f32> {
    type Output = ArcTensor<f32>;

    fn mul(self, other: ArcTensor<f32>) -> Self::Output {
        Mul::<f32>::new(binary!(self, other).unwrap()).unwrap().forward().unwrap()
    }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `_other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div for ArcTensor<f32> {
    type Output = ArcTensor<f32>;

    fn div(self, other: ArcTensor<f32>) -> Self::Output {
        Div::<f32>::new(binary!(self, other).unwrap()).unwrap().forward().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{MlResult, ops, binary, special};
    use crate::tensor::*;

    #[test]
    fn test_topk() -> MlResult<()> {
        // Test 1: Basic 1D tensor
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = ops!(buffer, Topk, 3, true)?;
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0], &[2, 5], )?;
        let (values, indices) = ops!(buffer, Topk, 2, true)?;
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = ops!(buffer, Topk ,3, false)?;
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
    #[test]
    fn test_max() -> MlResult<()> {
        // Test global maximum
        let buffer = Tensor::<f32>::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let (max_all, _) = ops!(buffer.clone(), Matmax, None, false)?;
        assert_eq!(max_all.data(), &[6.0]);

        // Test maximum along dimension 0
        let (max_dim0, indices0) = ops!(buffer.clone(), Matmax, Some(0), true)?;
        assert_eq!(max_dim0.shape(), &[1, 3]);
        assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(indices0.data(), &[1.0, 1.0, 1.0]);

        // Test maximum along dimension 1
        let (max_dim1, indices1) = ops!(buffer.clone(), Matmax, Some(1), true)?;
        assert_eq!(max_dim1.shape(), &[2, 1]);
        assert_eq!(max_dim1.data(), &[3.0, 6.0]);
        assert_eq!(indices1.data(), &[2.0, 2.0]);

        // Test maximum with negative dimension
        let (max_neg, indices_neg) = ops!(buffer, Matmax, Some(-1), true)?;
        assert_eq!(max_neg.data(), &[3.0, 6.0]);
        assert_eq!(indices_neg.data(), &[2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_matmul_2d_2d() -> MlResult<()> {
        // Case 1: 2D * 2D Matrix Multiplication
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::<f32>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])?;
        let c = ops!(a, Matmul, b)?;


        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_2d() -> MlResult<()> {
        // Case 2: 1D * 2D (Vector-Matrix Multiplication)
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 2])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[40.0, 46.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_2d_1d() -> MlResult<()> {
        // Case 3: 2D * 1D (Matrix-Vector Multiplication)
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = Tensor::<f32>::from_vec(vec![7.0, 8.0, 9.0], &[3])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[50.0, 122.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_3d() -> MlResult<()> {
        // Case 4: 3D * 3D (Batch Matrix Multiplication)
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::<f32>::from_vec(vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[2, 2, 2], )?;
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
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0], &[2])?;

        // This should return an error since the shapes are incompatible
        assert!(ops!(a, Matmul, b).is_err());

        // Test incompatible batch dimensions
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

        // This should return an error since the batch dimensions don't match
        assert!(ops!(a, Matmul, b).is_err());

        Ok(())
    }

    #[test]
    fn test_matmul_1x1() -> MlResult<()> {
        // Case 5: 1x1 Matrix Multiplication
        let a = Tensor::<f32>::from_vec(vec![2.0], &[1, 1])?;
        let b = Tensor::<f32>::from_vec(vec![3.0], &[1, 1])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c.data(), &[6.0]);
        Ok(())
    }

    #[test]
    fn test_matmul_1d_1d() -> MlResult<()> {
        // Case 6: 1D * 1D (Dot Product)
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
        let c = ops!(a, Matmul, b)?;

        assert_eq!(c.shape(), &[]); // scalar output
        assert_eq!(c.data(), &[32.0]); // 1*4 + 2*5 + 3*6 = 32
        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d_broadcasting() -> MlResult<()> {
        // Case 7: 3D * 2D Broadcasting
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])?;
        let b = Tensor::<f32>::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2])?;
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
        let a = Tensor::<f32>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            ],
            &[2, 2, 2, 2],
        )?;
        let b = Tensor::<f32>::from_vec(
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
        let a = Tensor::<f32>::from_vec(vec![], &[0, 2])?;
        let b = Tensor::<f32>::from_vec(vec![], &[2, 0])?;

        // This should return an error for empty tensors
        assert!(ops!(a, Matmul, b).is_err());
        Ok(())
    }

    #[test]
    fn test_matmul_broadcast_batch_dims() -> MlResult<()> {
        // Case 10: Broadcasting with Different Batch Dimensions
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
        let b = Tensor::<f32>::from_vec(
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