use std::fmt::Display;
use std::sync::Arc;

use crate::{
    MlError,
    MlResult,
    backend::Backend,
    backend::Device,
    backend,
};

mod ops;
mod broadcast;

#[derive(Debug)]
pub struct Tensor {
    backend: Arc<dyn Backend>,
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidDataLength {
        expected: usize,
        got: usize,
    },
    InvalidOperation {
        op: &'static str,
        reason: String,
    },
    InvalidAxis {
        axis: usize,
        shape: Vec<usize>,
    },
    MatrixMultiplicationError {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    EmptyTensor,
}

impl std::error::Error for TensorError {}

impl Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidDataLength { expected, got } => {
                write!(f, "Invalid data length: expected {}, got {}", expected, got)
            }
            TensorError::InvalidOperation { op, reason } => {
                write!(f, "Invalid operation '{}': {}", op, reason)
            }
            TensorError::InvalidAxis { axis, shape } => {
                write!(f, "Invalid axis {} for tensor with shape {:?}", axis, shape)
            }
            TensorError::MatrixMultiplicationError {
                left_shape,
                right_shape,
            } => {
                write!(f, "Invalid dimensions for matrix multiplication: left shape {:?}, right shape {:?}", left_shape, right_shape)
            }
            TensorError::EmptyTensor => {
                write!(f, "Empty tensor")
            }
        }
    }
}

#[derive(Clone)]
struct GradFn(Arc<dyn Fn(&Tensor) -> MlResult<()>>);

impl std::fmt::Debug for GradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradFn")
    }
}


impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl Eq for Tensor {
    // Todo: 구현 필요
}

impl PartialOrd for Tensor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait DefaultLayer {
    fn new(data: Vec<Vec<f32>>)                 -> MlResult<Self> where Self: Sized;
    fn from(data: Vec<f32>, shape: &[usize])    -> MlResult<Self> where Self: Sized;
    fn shape(&self)                             -> &[usize];
    fn data(&self)                              -> &[f32];
    fn get(&self, indices: &[usize])            -> Option<&f32>;
    fn index(&self, indices: &[usize])          -> Option<usize>;
    fn backend(&self)                           -> &Arc<dyn Backend>;
}


pub trait BroadcastLayer {
    fn can_broadcast(&self, other: &Self) -> bool;
    fn broadcast_shape(&self, other: &Self) -> Vec<usize>;
    fn broadcasting<F>(self, other: Self, op: F) -> Option<Self>
    where
        F: Fn(f32, f32) -> f32,
        Self: Sized;
    fn calculate_broadcast_indices(&self, other: &Self, idx: usize, shape: &[usize]) -> Option<(usize, usize)>;
}

pub trait OpsLayer<T: DefaultLayer>{
    type Output;

    fn can_op(&self, other: &T) -> MlResult<()>;

    // 사칙연산
    fn add(&self, other: &T)                -> Self::Output;
    fn sub(&self, other: &T)                -> Self::Output;
    fn mul(&self, other: &T)                -> Self::Output;
    fn div(&self, other: &T)                -> Self::Output;

    // 텐서 & 스칼라 연산
    fn add_scalar(&self, scalar: f32)       -> Self::Output;
    fn sub_scalar(&self, scalar: f32)       -> Self::Output;
    fn mul_scalar(&self, scalar: f32)       -> Self::Output;
    fn div_scalar(&self, scalar: f32)       -> Self::Output;

    // 스칼라 & 텐서 연산
    fn scalar_sub(&self, scalar: f32)       -> Self::Output;
    fn scalar_div(&self, scalar: f32)       -> Self::Output;

    fn neg(&self)                           -> Self::Output;
    fn exp(&self)                           -> Self::Output;
    fn pow(&self, power: f32)               -> Self::Output;
    fn pow_scalar(&self, exponent: f32)     -> Self::Output;
    fn scalar_pow(&self, scalar: f32)       -> Self::Output;
    fn sqrt(&self)                          -> Self::Output;
    fn square(&self)                        -> Self::Output;
    fn log(&self)                           -> Self::Output;
    fn matmul(&self, other: &T)             -> Self::Output;
    fn eq_scalar(&self, scalar: f32)        -> Self::Output;
    fn abs(&self)                           -> Self::Output;
    fn topk(&self, k: usize, sorted: bool)              -> MlResult<(T, T)>;
    fn matmax(&self, dim: Option<i32>, keepdim: bool)   -> MlResult<(Tensor, Option<Tensor>)>;
}

impl DefaultLayer for Tensor {
    fn new(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();
        let backend: Arc<dyn Backend> =  Arc::new(backend::CpuBackend::new()?);

        Ok(Self {
            backend,
            data,
            shape,
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

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::{DefaultLayer, OpsLayer, Tensor};
    pub fn assert_tensor_eq(tensor: Tensor, expected_tensor: Tensor, ) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
        Ok(())
    }

    #[test]
    fn tensor() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]])?;
        assert_eq!(t1.data(), vec![1.0, 2.0]);
        assert_eq!(t1.shape(), vec![1, 2]);
        Ok(())
    }

    #[test]
    fn test_add_symbol() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]])?;
        let t2 = Tensor::new(vec![vec![3.0, 4.0]])?;
        let et = Tensor::new(vec![vec![4.0, 6.0]])?;
        assert_tensor_eq(t1 + t2, et)
    }
    #[test]
    fn test_sub_symbol() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]])?;
        let t2 = Tensor::new(vec![vec![3.0, 4.0]])?;
        let et = Tensor::new(vec![vec![-2.0, -2.0]])?;
        assert_tensor_eq(t1 - t2, et)
    }
    #[test]
    fn test_mul_symbol() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]])?;
        let t2 = Tensor::new(vec![vec![3.0, 4.0]])?;
        let et = Tensor::new(vec![vec![3.0, 8.0]])?;
        assert_tensor_eq(t1 * t2, et)
    }
    #[test]
    fn test_div_symbol() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]])?;
        let t2 = Tensor::new(vec![vec![2.0, 4.0]])?;
        let et = Tensor::new(vec![vec![0.5, 0.5]])?;
        assert_tensor_eq(t1 / t2, et)
    }

    #[test]
    fn tensor_ops_add_scalar() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();
        assert_tensor_eq(t1.add_scalar(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_sub_scalar() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![-1.0, 0.0]]).unwrap();
        assert_tensor_eq(t1.sub_scalar(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_mul_scalar() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![2.0, 4.0]]).unwrap();
        assert_tensor_eq(t1.mul_scalar(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_div_scalar() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![0.5, 1.0]]).unwrap();
        assert_tensor_eq(t1.div_scalar(2.0).unwrap(), et)
    }

    #[test]
    fn tensor_ops_scalar_sub() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();
        assert_tensor_eq(t1.scalar_sub(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_scalar_div() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![2.0, 1.0]]).unwrap();
        assert_tensor_eq(t1.scalar_div(2.0).unwrap(), et)
    }
}
