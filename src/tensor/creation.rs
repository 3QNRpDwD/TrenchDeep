use std::sync::Arc;
use crate::{MlError, MlResult};
use crate::backend::{Backend, CpuBackend, Device};
use crate::tensor::{Abs, Add, Div, Exp, Log, Matmax, Matmul, Mul, Neg, Pow, Sub, Sqrt, Square, Topk, Tensor, TensorError, ArcTensor, UnaryOp, BinaryOp, SpecialOp, Operator, TensorBase};


impl  Tensor<f32> {
    pub fn zeros() -> ArcTensor<f32> {
        ArcTensor::new(Self {
            data: vec![],
            shape: vec![],
            requires_grad: cfg!(feature = "enable_backpropagation"),

            // #[cfg(feature = "enable_backpropagation")]
            // grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        })
    }

    pub fn scalar(scalar: f32) -> ArcTensor<f32> {
        ArcTensor::new(Self {
            data: vec![scalar],
            shape: vec![1],
            requires_grad: cfg!(feature = "enable_backpropagation"),

            // #[cfg(feature = "enable_backpropagation")]
            // grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        })
    }
}

impl TensorBase<f32> for Tensor<f32> {
    fn new(data: Vec<Vec<f32>>) -> ArcTensor<f32>  {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();

        ArcTensor::new(Self {
            data,
            shape,
            requires_grad: cfg!(feature = "enable_backpropagation"),

            // #[cfg(feature = "enable_backpropagation")]
            // grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        })
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<ArcTensor<f32>> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(ArcTensor::new(Self {
            data,
            shape: shape.to_vec(),
            requires_grad: cfg!(feature = "enable_backpropagation"),

            // #[cfg(feature = "enable_backpropagation")]
            // grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        }))
    }

    #[cfg(feature = "enable_backpropagation")]
    fn from_grad_fn(data: Vec<f32>, shape: &[usize], grad_fn: Arc<dyn Operator<f32>>) -> ArcTensor<f32> {
        ArcTensor::new(Self {
            data,
            shape: shape.to_vec(),
            requires_grad: false,

            #[cfg(feature = "enable_backpropagation")]
            grad_fn: Some(grad_fn)
        })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        &self.data
    }

    fn get(&self, indices: &[usize]) -> Option<&f32> {
        self.data.get(self.index(indices)?)
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        Some(
            indices
                .iter()
                .zip(&self.shape)
                .fold(0, |acc, (&i, &dim)| acc * dim + i),
        )
    }

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn chk_shape(&self, other: &dyn TensorBase<f32>) -> MlResult<()> {
        if self.shape != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[cfg(feature = "enable_backpropagation")]
    fn set_grad_fn(&mut self, grad_fn: Arc<dyn Operator<f32>>) {
        self.grad_fn = Some(grad_fn)
    }

    #[cfg(feature = "enable_backpropagation")]
    fn grad(&self) -> Option<&dyn TensorBase<f32>> {
        todo!()
    }

    // #[cfg(feature = "enable_backpropagation")]
    // fn set_grad_fn(&mut self, grad_fn: Box<dyn crate::tensor::Function<'static, f32, Forwarded=(), Gradiant=()>>) {
    //     self.grad_fn = Some(grad_fn);
    // }
    //
    // #[cfg(feature = "enable_backpropagation")]
    // fn grad(&self) -> Option<Arc<dyn TensorBase<f32>>> {
    //     self.grad.as_ref().map(|g| g.as_ref())
    // }
}

impl<T> UnaryOp<T> {
    pub fn new(tensor: Arc<dyn TensorBase<T>>) -> MlResult<UnaryOp<T>> {
        Ok(Self {
            tensor,
            backend: Arc::new(CpuBackend::new()?),
            #[cfg(feature = "enable_backpropagation")]
            output: None,
        })
    }

    pub fn update(&mut self, tensor: Arc<dyn TensorBase<T>>) {
        self.tensor = tensor;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.output = None;
        }
    }
}

impl<T> BinaryOp<T> {
    pub fn new(first_tensor: Arc<dyn TensorBase<T>>, second_tensor: Arc<dyn TensorBase<T>>) -> MlResult<BinaryOp<T>> {
        Ok(Self {
            first_tensor,
            second_tensor,
            backend: Arc::new(CpuBackend::new()?),
            #[cfg(feature = "enable_backpropagation")]
            output: None,
        })
    }

    pub fn update(&mut self, first_tensor: Arc<dyn TensorBase<T>>, second_tensor: Arc<dyn TensorBase<T>>) {
        self.first_tensor = first_tensor;
        self.second_tensor = second_tensor;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.output = None;
        }
    }
}

impl<T> SpecialOp<T> {
    pub fn new(tensor: Arc<dyn TensorBase<T>>) -> MlResult<SpecialOp<T>> {
        Ok(Self {
            tensor,
            backend: Arc::new(CpuBackend::new()?),
            #[cfg(feature = "enable_backpropagation")]
            output: None,
        })
    }

    pub fn update(&mut self, tensor: Arc<dyn TensorBase<T>>) {
        self.tensor = tensor;
        #[cfg(feature = "enable_backpropagation")]
        {
            self.output = None;
        }
    }
}

impl Operator<f32> for Abs<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Exp<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Log<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Neg<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Sqrt<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}


impl Operator<f32> for Square<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Add<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self>  {
        Ok(Self { op: BinaryOp::new(first, second.unwrap())? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first, second.unwrap())
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Sub<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self>  {
        Ok(Self { op: BinaryOp::new(first, second.unwrap())? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first, second.unwrap())
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Mul<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self>  {
        Ok(Self { op: BinaryOp::new(first, second.unwrap())? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first, second.unwrap())
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Div<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self>  {
        Ok(Self { op: BinaryOp::new(first, second.unwrap())? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first, second.unwrap())
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Matmul<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self>  {
        Ok(Self { op: BinaryOp::new(first, second.unwrap())? })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, second: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first, second.unwrap())
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Pow<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: UnaryOp::new(first)?, power: None })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Topk<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: SpecialOp::new(first)?, topk: None })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}

impl Operator<f32> for Matmax<f32> {
    fn new(first: Arc<dyn TensorBase<f32>>, _: Option<Arc<dyn TensorBase<f32>>>) -> MlResult<Self> {
        Ok(Self { op: SpecialOp::new(first)?, matmax: None })
    }

    // fn update(&mut self, first: Arc<dyn TensorBase<f32>>,_: Option<Arc<dyn TensorBase<f32>>>) {
    //     self.op.update(first)
    // }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.op.backend
    }
}