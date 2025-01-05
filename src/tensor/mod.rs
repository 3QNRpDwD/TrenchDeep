use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;
use crate::{MlError, MlResult};

mod ops;
mod broadcast;

#[derive(Debug, Default)]
pub struct Tensor {
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
    fn from(data: Vec<f32>, shape: Vec<usize>) -> MlResult<Self> where Self: Sized;
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[f32];
    fn get(&self, indices: &[usize]) -> Option<&f32>;
    fn index(&self, indices: &[usize]) -> Option<usize>;
}


pub trait BroadcastLayer<T> {
    fn can_broadcast(&self, other: &Self) -> bool;
    fn broadcast_shape(&self, other: &Self) -> Vec<usize>;
    fn broadcast_op<F>(self, other: Self, op: F) -> Option<Self>
    where
        F: Fn(T, T) -> T,
        Self: Sized;
    fn calculate_broadcast_indices(&self, other: &Self, idx: usize, shape: &[usize]) -> Option<(usize, usize)>;
}

pub trait OpsLayer<T: PartialEq> {
    // 사칙연산
    fn add(self, other: Tensor)
           -> Option<Self> where T: Add<Output = T>, Self: Sized;
    fn sub(self, other: Tensor)
           -> Option<Self> where T: Sub<Output = T>, Self: Sized;
    fn div(self, other: Tensor)
           -> Option<Self> where T: Div<Output = T>, Self: Sized;
    fn mul(self, other: Tensor)
           -> Option<Self> where T: Mul<Output = T>, Self: Sized;

    // 텐서 & 스칼라 연산
    fn add_scalar(self, other: Tensor)
                  -> Option<Self> where T: Add<Output = T>, Self: Sized;
    fn mul_scalar(self, other: Tensor)
                  -> Option<Self> where T: Mul<Output = T>, Self: Sized;
    fn sub_scalar(self, other: Tensor)
                  -> Option<Self> where T: Sub<Output = T>, Self: Sized;
    fn div_scalar(self, other: Tensor)
                  -> Option<Self> where T: Div<Output = T>, Self: Sized;

    // 스칼라 & 텐서 연산
    fn scalar_sub(self, other: Tensor)
                  -> Option<Self> where T: Sub<Output = T>, Self: Sized;
    fn scalar_div(self, other: Tensor)
                  -> Option<Self> where T: Div<Output = T>, Self: Sized;
}

impl DefaultLayer for Tensor {
    fn new(data: Vec<Vec<f32>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();

        Ok(Self {
            data,
            shape,
        })
    }

    fn from(data: Vec<f32>, shape: Vec<usize>) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(Self {
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
    use crate::tensor::{DefaultLayer, Tensor};

    // Option<T>의 결과를 테스트하는 헬퍼 함수
    pub fn assert_tensor_eq(
        result: Option<Tensor>,
        expected_data: Vec<f32>,
        expected_shape: Vec<usize>
    ) {
        let tensor = result.unwrap();
        debug_assert_eq!(tensor.data(), expected_data);
        debug_assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn tensor_ops_add() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]);
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]);

        assert_tensor_eq(t1.add(), vec![vec![6.0, 6.0, 6.0, 6.0]], vec![2, 2]);
    }
    #[test]
    fn tensor_ops_sub() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]);
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]);

        assert_tensor_eq(t1.sub(t2), vec![vec![2.0, 2.0, 2.0, 2.0]], vec![2, 2]);
    }
    #[test]
    fn tensor_ops_mul() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]);
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]);

        assert_tensor_eq(t1.mul(t2), vec![vec![8.0, 8.0, 8.0, 8.0]], vec![2, 2]);
    }
    #[test]
    fn tensor_ops_div() {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]);
        let t2 = Tensor::new(vec![vec![3.0, 4.0]]);

        assert_tensor_eq(t1.div(t2), vec![vec![2.0, 2.0, 2.0, 2.0]], vec![2, 2]);
    }

}
