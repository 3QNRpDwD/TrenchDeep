use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};
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
    fn new(data: Vec<Vec<f32>>) -> MlResult<Self> where Self: Sized;
    fn from(data: Vec<f32>, shape: &Vec<usize>) -> MlResult<Self> where Self: Sized;
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[f32];
    fn get(&self, indices: &[usize]) -> Option<&f32>;
    fn index(&self, indices: &[usize]) -> Option<usize>;
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

pub trait OpsLayer<T: PartialEq> {
    fn can_op(&self, other: &Tensor) ->  MlResult<()> where Self: Sized;

    // 사칙연산
    fn add(&self, other: &Tensor)
           -> MlResult<Self> where T: Add<Output = T>, Self: Sized;
    fn sub(&self, other: &Tensor)
           -> MlResult<Self> where T: Sub<Output = T>, Self: Sized;
    fn mul(&self, other: &Tensor)
           -> MlResult<Self> where T: Mul<Output = T>, Self: Sized;
    fn div(&self, other: &Tensor)
           -> MlResult<Self> where T: Div<Output = T>, Self: Sized;

    // 텐서 & 스칼라 연산
    fn add_scalar(&self, scalar: f32)
                  -> MlResult<Self> where T: Add<Output = T>, Self: Sized;
    fn sub_scalar(&self, scalar: f32)
                  -> MlResult<Self> where T: Sub<Output = T>, Self: Sized;
    fn mul_scalar(&self, scalar: f32)
                  -> MlResult<Self> where T: Mul<Output = T>, Self: Sized;
    fn div_scalar(&self, scalar: f32)
                  -> MlResult<Self> where T: Div<Output = T>, Self: Sized;

    // 스칼라 & 텐서 연산
    fn scalar_sub(&self, scalar: f32)
                  -> MlResult<Self> where T: Sub<Output = T>, Self: Sized;
    fn scalar_div(&self, scalar: f32)
                  -> MlResult<Self> where T: Div<Output = T>, Self: Sized;

    fn neg(&self) -> MlResult<Tensor>;
    fn exp(&self) -> MlResult<Tensor>;
    fn pow(&self, power: f32) -> MlResult<Tensor>;
    fn pow_scalar(&self, exponent: f32) -> MlResult<Tensor>;
    fn scalar_pow(&self, scalar: f32) -> MlResult<Tensor>;
    fn sqrt(&self) -> MlResult<Tensor>;
    fn square(&self) -> MlResult<Self> where Self: Sized;
    fn log(&self) -> MlResult<Tensor>;
    fn matmul(&self, other: &Tensor) -> MlResult<Tensor>;
    fn eq_scalar(&self, scalar: f32) -> MlResult<Tensor>;
    fn topk(&self, k: usize, sorted: bool) -> MlResult<(Tensor, Tensor)>;
    fn abs(&self) -> MlResult<Self> where Self: Sized;
    fn matmax(&self, dim: Option<i32>, keepdim: bool) -> MlResult<(Tensor, Option<Tensor>)>;
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

    fn from(data: Vec<f32>, shape: &Vec<usize>) -> MlResult<Self> {
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
}

#[cfg(test)]
mod tests {
    use crate::tensor::{DefaultLayer, OpsLayer, Tensor};

    // Option<T>의 결과를 테스트하는 헬퍼 함수
    pub fn assert_tensor_eq(
        tensor: Tensor,
        expected_tensor: Tensor,
    ) {
        debug_assert_eq!(tensor.data(), expected_tensor.data());
        debug_assert_eq!(tensor.shape(), expected_tensor.shape());
    }

    #[test]
    fn tensor() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        debug_assert_eq!(t1.data(), vec![1.0, 2.0]);
        debug_assert_eq!(t1.shape(), vec![1, 2]);
    }

    #[test]
    fn tensor_ops_add() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();
        let et = Tensor::new(vec![vec![4.0, 6.0]]).unwrap();
        assert_tensor_eq(t1 + t2, et);
    }
    #[test]
    fn tensor_ops_sub() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();
        let et = Tensor::new(vec![vec![-2.0, -2.0]]).unwrap();
        assert_tensor_eq(t1 - t2, et);
    }
    #[test]
    fn tensor_ops_mul() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();
        let et = Tensor::new(vec![vec![3.0, 8.0]]).unwrap();
        assert_tensor_eq(t1 * t2, et);
    }
    #[test]
    fn tensor_ops_div() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let t2 = Tensor::new(vec![vec![2.0, 4.0]]).unwrap();
        let et = Tensor::new(vec![vec![0.5, 0.5]]).unwrap();
        assert_tensor_eq(t1 / t2, et);
    }

    #[test]
    fn tensor_ops_add_scalar() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();
        assert_tensor_eq(<Tensor as OpsLayer<f32>>::add_scalar(&t1, 2.0).unwrap(), et);
    }
    #[test]
    fn tensor_ops_sub_scalar() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![-1.0, 0.0]]).unwrap();
        assert_tensor_eq(<Tensor as OpsLayer<f32>>::sub_scalar(&t1, 2.0).unwrap(), et);
    }
    #[test]
    fn tensor_ops_mul_scalar() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![2.0, 4.0]]).unwrap();
        assert_tensor_eq(<Tensor as OpsLayer<f32>>::mul_scalar(&t1, 2.0).unwrap(), et);
    }
    #[test]
    fn tensor_ops_div_scalar() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![0.5, 1.0]]).unwrap();
        assert_tensor_eq(<Tensor as OpsLayer<f32>>::div_scalar(&t1, 2.0).unwrap(), et);
    }

    #[test]
    fn tensor_ops_scalar_sub() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();
        assert_tensor_eq(<Tensor as OpsLayer<f32>>::scalar_sub(&t1, 2.0).unwrap(), et);
    }
    #[test]
    fn tensor_ops_scalar_div() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![2.0, 1.0]]).unwrap();
        assert_tensor_eq(<Tensor as OpsLayer<f32>>::scalar_div(&t1, 2.0).unwrap(), et);
    }
}
