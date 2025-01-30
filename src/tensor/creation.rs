use std::sync::Arc;

use crate::{backend, MlError, MlResult};
use crate::backend::{Backend, Device};
use crate::tensor::{Add, TensorBase, Div, Function, Matmax, Matmul, Mul, Pow, Sub, Tensor, TensorError, Topk};

impl<T: TensorBase> Add<T>{
    pub fn new(first: T, second: T) -> MlResult<Self> {
        Ok(Add { first, second })
    }
}

impl<T: TensorBase> Sub<T>{
    fn new(first: T, second: T) -> MlResult<Self> {
        Ok(Sub { first, second })
    }
}

impl<T: TensorBase> Mul<T>{
    pub fn new(first: T, second: T) -> MlResult<Self> {
        Ok(Mul { first, second })
    }
}

impl<T: TensorBase> Div<T>{
    pub fn new(first: T, second: T) -> MlResult<Self> {
        Ok(Div { first, second })
    }
}

impl<T: TensorBase> Pow<T>{
    pub fn new(first: T, power: f32) -> MlResult<Self> {
        Ok(Pow { first, power })
    }
}

impl<T: TensorBase> Matmul<T>{
    pub fn new(first: T, second: T) -> MlResult<Self> {
        Ok(Matmul { first, second })
    }
}

impl<T: TensorBase> Topk<T>{
    pub fn new(first: T, k: usize, sorted: bool) -> MlResult<Self> {
        Ok(Topk { first, k, sorted })
    }
}

impl<T: TensorBase> Matmax<T>{
    pub fn new(first: T, dim: Option<i32>, keepdim: bool) -> MlResult<Self> {
        Ok(Matmax { first, dim, keepdim })
    }
}

impl TensorBase for Tensor {
    fn new(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();
        let backend: Arc<dyn Backend> =  Arc::new(backend::CpuBackend::new()?);

        Ok(Self {
            backend,
            data,
            shape,
            grad: None,
            grad_fn: None,
            requires_grad: false,
        })
    }

    fn from(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }
        let backend: Arc<dyn Backend> = Arc::new(backend::CpuBackend::new()?);
        Ok(Self {
            backend,
            data,
            shape: shape.to_vec(),
            grad: None,
            grad_fn: None,
            requires_grad: false,
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

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}


// impl<T: TensorBase Function<T> for Functions {
//     type Output = MlResult<T>;
//     type Gradient = f64;
//
//     fn forward(&self) -> Self::Output {
//         match self {
//             Functions::Abs      (F) => F.forward(),
//             Functions::Exp      (F) => F.forward(),
//             Functions::Log      (F) => F.forward(),
//             Functions::Neg      (F) => F.forward(),
//             Functions::Sqrt     (F) => F.forward(),
//             Functions::Square   (F) => F.forward(),
//
//             Functions::Add      (F) => F.forward(),
//             Functions::Sub      (F) => F.forward(),
//             Functions::Mul      (F) => F.forward(),
//             Functions::Div      (F) => F.forward(),
//             Functions::Pow      (F) => F.forward(),
//             Functions::Matmul   (F) => F.forward(),
//
//             Functions::Topk     (F) => F.forward(),
//             Functions::Matmax   (F) => F.forward(),
//         }
//     }
//     fn backward(&self, grad: Self::Gradient) -> Self::Output {
//         match self {
//             Functions::Abs      (F) =>  F.backword(grad),
//             Functions::Exp      (F) =>  F.backword(grad),
//             Functions::Log      (F) =>  F.backword(grad),
//             Functions::Neg      (F) =>  F.backword(grad),
//             Functions::Sqrt     (F) =>  F.backword(grad),
//             Functions::Square   (F) =>  F.backword(grad),
//
//             Functions::Add      (F) =>  F.backword(grad),
//             Functions::Sub      (F) =>  F.backword(grad),
//             Functions::Mul      (F) =>  F.backword(grad),
//             Functions::Div      (F) =>  F.backword(grad),
//             Functions::Pow      (F) =>  F.backword(grad),
//             Functions::Matmul   (F) =>  F.backword(grad),
//
//             Functions::Topk     (F) =>  F.backword(grad),
//             Functions::Matmax   (F) =>  F.backword(grad),
//         }
//         todo!("역전파 구현하기")
//     }
// }