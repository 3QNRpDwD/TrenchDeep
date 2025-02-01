use std::{
    sync::Arc,
    fmt::{Debug, Display, Formatter, Result}
};

use crate::{backend::Backend, MlResult};

mod ops;
mod broadcast;
mod creation;

/// A macro for simplifying tensor operations in Rust.
///
/// This macro provides a convenient way to perform unary, binary, and ternary operations
/// on tensors using custom operator structures. The macro matches different patterns
/// to handle various use cases for tensor computations.
/// Handles unary operations (e.g., Neg, Abs, Sqrt).
///
/// # Arguments
///
/// * `$x`: The input tensor.
/// * `$op`: The operator struct that implements the `forward` method.
///
/// # Returns
/// A new tensor resulting from_vec the unary operation.
///
///
/// Handles binary operations (e.g., Add, Sub, Mul, Div).
///
/// # Arguments
///
/// * `$x`: The first input tensor.
/// * `$op`: The operator struct that implements the `forward` method.
/// * `$y`: The second input tensor.
///
/// # Returns
/// A new tensor resulting from_vec the binary operation.
///
///
/// Handles ternary operations for specific operators like `Topk` and `Matmax`.
///
/// # Arguments
///
/// * `$x`: The first input tensor.
/// * `$op`: The operator struct (`Topk` or `Matmax`) that implements the `forward` method.
/// * `$y`: The second parameter (e.g., `k` for `Topk`, `dim` for `Matmax`).
/// * `$z`: The third parameter (e.g., `sorted` for `Topk`, `keepdim` for `Matmax`).
///
/// # Returns
/// A new tensor resulting from_vec the ternary operation. For `Topk`, returns the top-k values and their indices.
/// For `Matmax`, returns the maximum values and their indices (if applicable).
#[macro_export]
macro_rules! ops {
    ($x:expr, Matmul, $y:expr) => {
        Matmul::new($x.as_ref(), Some($y.as_ref())).unwrap().forward()
    };

    ($x:expr, Topk) => {
        Topk::new($x.as_ref(), None).unwrap().forward()
    };

    ($x:expr, Matmax) => {
        Matmax::new($x.as_ref(), None).unwrap().forward()
    };
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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
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
struct GradFn<T: Debug + 'static>(Arc<dyn Fn(&Tensor<T>) -> MlResult<()>>);

impl<T: Debug> Debug for GradFn<Vec<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "GradFn")
    }
}

#[derive(Debug)]
pub struct Tensor<Type: Debug + 'static> where GradFn<Vec<Type>>: Debug {
    data: Vec<Type>,
    shape: Vec<usize>,
    grad: Option<Box<Tensor<Type>>>,
    grad_fn: Option<GradFn<Vec<Type>>>,
    requires_grad: bool,
    power: Option<f32>,
    topk: Option<(usize, bool)>,
    matmax: Option<(Option<i32>, bool)>
}

impl<T> Debug for dyn TensorBase<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "TensorBase Debug")
    }
}

impl Display for dyn TensorBase<f32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "TensorBase Display")
    }
}

impl PartialEq for Tensor<f32> {
    fn eq(&self, other: &Self) -> bool {

        self.data == other.data && self.shape == other.shape
    }
}

impl Eq for Tensor<f32> {
    // Todo: 구현 필요
}

impl PartialOrd for Tensor<f32> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor<f32> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait TensorBase<Type: Debug + Display + 'static> {
    fn new(data: Vec<Vec<Type>>)                            -> Box<dyn TensorBase<Type>> where Self: Sized;
    fn from_vec(data: Vec<Type>, shape: &[usize])           -> MlResult<Box<dyn TensorBase<Type>>> where Self: Sized;
    fn shape(&self)                                         -> &[usize];
    fn data(&self)                                          -> &[Type];
    fn power(&self)                                         -> f32;
    fn topk(&self)                                          -> (usize, bool);
    fn matmax(&self)                                        -> (Option<i32>, bool);
    fn set_power(&mut self, exponent: f32)                      ;
    fn set_topk(&mut self, k: usize, sorted: bool)              ;
    fn set_matmax(&mut self, dim: Option<i32>, keepdim: bool)   ;
    fn get(&self, indices: &[usize])                        -> Option<&Type>;
    fn index(&self, indices: &[usize])                      -> Option<usize>;
    fn chk_shape(&self, other: &dyn TensorBase<Type>)   -> MlResult<()>;
    /// Enables gradient computation for the tensor
    fn requires_grad(&mut self, requires_grad: bool);

    /// Sets the gradient function for the tensor
    // fn set_grad_fn<F>(&mut self, grad_fn: F)
    // where
    //     F: Fn(&Tensor<Type>) -> MlResult<()> + 'static;

    /// Returns the gradient of the tensor
   fn grad(&self) -> Option<&Tensor<Type>>;
}

pub trait Function<T: Debug + Display + Clone> {
    type Forwarded;
    type Gradiant;

    fn new(first: &dyn TensorBase<T>, second: Option<&dyn TensorBase<T>>) -> MlResult<Self> where Self: Sized;
    fn forward(&self) -> Self::Forwarded;
    fn backward(&self, grad: Box<dyn TensorBase<T>>) -> Self::Gradiant;
    fn backend(&self) -> &Arc<dyn Backend>;
}

// pub trait BroadcastLayer {
//     fn can_broadcast(&self, other: &Self) -> bool;
//     fn broadcast_shape(&self, other: &Self) -> Vec<usize>;
//     fn broadcasting<F>(self, other: Self, op: F) -> Option<Self>
//     where
//         F: Fn(f32, f32) -> f32,
//         Self: Sized;
//     fn calculate_broadcast_indices(&self, other: &Self, idx: usize, shape: &[usize]) -> Option<(usize, usize)>;
// }

/// Structure representing an exponential operation.
pub struct Exp<'t, T>    { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a negation operation.
pub struct Neg<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a square root operation.
pub struct Sqrt<'t, T>    { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing an absolute value operation.
pub struct Abs<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a squaring operation.
pub struct Square<'t, T>  { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a logarithmic operation.
pub struct Log<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a power operation.
pub struct Pow<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a Top-k operation.
pub struct Topk<'t, T>    { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> } // k: second, sorted: third

/// Structure representing a matrix max operation along a dimension.
pub struct Matmax<'t, T>  { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> } // dim: second, keepdim: third

/// Structure representing an addition operation.
pub struct Add<'t, T>     { first_tensor: &'t dyn TensorBase<T>, second_tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a subtraction operation.
pub struct Sub<'t, T>     { first_tensor: &'t dyn TensorBase<T>, second_tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a multiplication operation.
pub struct Mul<'t, T>     { first_tensor: &'t dyn TensorBase<T>, second_tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a division operation.
pub struct Div<'t, T>     { first_tensor: &'t dyn TensorBase<T>, second_tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

/// Structure representing a matrix multiplication operation.
pub struct Matmul<'t, T>  { first_tensor: &'t dyn TensorBase<T>, second_tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend> }

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::{TensorBase, Tensor};

    pub fn assert_tensor_eq(tensor: Box<dyn TensorBase<f32>>, expected_tensor: Box<dyn TensorBase<f32>>) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
        Ok(())
    }

    #[test]
    fn tensor() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]]);
        assert_eq!(t1.data(), vec![1.0, 2.0]);
        assert_eq!(t1.shape(), vec![1, 2]);
        Ok(())
    }

    #[test]
    fn test_add_symbol() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let et = Tensor::<f32>::new(vec![vec![4.0, 6.0]]);

        assert_tensor_eq(first + second, et)
    }
    #[test]
    fn test_sub_symbol() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let et = Tensor::<f32>::new(vec![vec![-2.0, -2.0]]);

        assert_tensor_eq(first - second, et)
    }
    #[test]
    fn test_mul_symbol() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let et = Tensor::<f32>::new(vec![vec![3.0, 8.0]]);

        assert_tensor_eq(first * second, et)
    }
    #[test]
    fn test_div_symbol() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![2.0, 4.0]]);
        let et = Tensor::<f32>::new(vec![vec![0.5, 0.5]]);

        assert_tensor_eq(first / second, et)
    }

    // #[test]
    // fn tensor_ops_add_scalar() -> MlResult<()> {
    //     let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
    //     let et = Tensor::<f32>::new(vec![vec![3.0, 4.0]]).unwrap();
    //
    //     assert_tensor_eq(first + 2.0, et)
    // }
    // #[test]
    // fn tensor_ops_sub_scalar() -> MlResult<()> {
    //     let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
    //     let et = Tensor::<f32>::new(vec![vec![-1.0, 0.0]]).unwrap();
    //
    //     assert_tensor_eq(first - 2.0, et)
    // }
    // #[test]
    // fn tensor_ops_mul_scalar() -> MlResult<()> {
    //     let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
    //     let et = Tensor::<f32>::new(vec![vec![2.0, 4.0]]).unwrap();
    //
    //     assert_tensor_eq(first - 2.0, et)
    // }
    // #[test]
    // fn tensor_ops_div_scalar() -> MlResult<()> {
    //     let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
    //     let et = Tensor::<f32>::new(vec![vec![0.5, 1.0]]).unwrap();
    //
    //     assert_tensor_eq(first / 2.0, et)
    // }
    //
    // #[test]
    // fn tensor_ops_scalar_sub() -> MlResult<()> {
    //     let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
    //     let et = Tensor::<f32>::new(vec![vec![1.0, 0.0]]).unwrap();
    //
    //     assert_tensor_eq(2.0 - first, et)
    //
    // }
    // #[test]
    // fn tensor_ops_scalar_div() -> MlResult<()> {
    //     let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
    //     let et = Tensor::<f32>::new(vec![vec![2.0, 1.0]]).unwrap();
    //
    //     assert_tensor_eq(2.0 - first, et)
    // }
}
