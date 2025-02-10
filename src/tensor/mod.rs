use std::{
    sync::Arc,
    fmt::{Debug, Display, Formatter, Result}
};

use crate::{backend::Backend, MlResult};

mod ops;
mod broadcast;
mod creation;


#[macro_export]
macro_rules! ops {
    ($tensor:expr, Matmul, $second_tensor:expr) => {
        Matmul::new($tensor, Some($second_tensor)).unwrap().forward()
    };

    ($tensor:expr, Topk, $k:expr, $sorted:expr) => {{
        $tensor.set_topk($k, $sorted);
        Topk::new($tensor, None).unwrap().forward()
    }};

    ($tensor:expr, Matmax, $dim:expr, $keepdim:expr) => {{
        $tensor.set_matmax($dim, $keepdim);
        Matmax::new($tensor, None).unwrap().forward()
    }};

    ($tensor:expr, Add, $second_tensor:expr) => {
        Add::new($tensor, Some($second_tensor)).unwrap().forward()
    };

    ($tensor:expr, Sub, $second_tensor:expr) => {
        Sub::new($tensor, Some($second_tensor)).unwrap().forward()
    };

    ($tensor:expr, Mul, $second_tensor:expr) => {
        Mul::new($tensor, Some($second_tensor)).unwrap().forward()
    };

    ($tensor:expr, Div, $second_tensor:expr) => {
        Div::new($tensor, Some($second_tensor)).unwrap().forward()
    };

    ($tensor:expr, Exp) => {
        Exp::new($tensor, None).unwrap().forward()
    };

    ($tensor:expr, Neg) => {
        Neg::new($tensor, None).unwrap().forward()
    };

    ($tensor:expr, Sqrt) => {
        Sqrt::new($tensor, None).unwrap().forward()
    };

    ($tensor:expr, Abs) => {
        Abs::new($tensor, None).unwrap().forward()
    };

    ($tensor:expr, Square) => {
        Square::new($tensor, None).unwrap().forward()
    };

    ($tensor:expr, Log) => {
        Log::new($tensor, None).unwrap().forward()
    };

    ($tensor:expr, Pow, $exponent:expr) => {{
        $tensor.set_power($exponent);
        Pow::new($tensor, None).unwrap().forward()
    }};
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

#[derive(Debug)]
pub struct Tensor<Type: Debug + 'static>
{
    data: Vec<Type>,
    shape: Vec<usize>,
    power: Option<f32>,
    topk: Option<(usize, bool)>,
    matmax: Option<(Option<i32>, bool)>,
    requires_grad: bool,

    #[cfg(feature = "enable-backpropagation")]
    grad: Option<Box<dyn TensorBase<Type>>>,
    #[cfg(feature = "enable-backpropagation")]
    grad_fn: Option<&'static dyn Function<'static, Type, Forwarded=(), Gradiant=()>>
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

pub trait TensorBase<Type: Debug + 'static> {
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
    fn chk_shape(&self, other: &dyn TensorBase<Type>)       -> MlResult<()>;

    #[cfg(feature = "enable-backpropagation")]
    /// Enables gradient computation for the tensor
    fn requires_grad(&self) -> bool;

    #[cfg(feature = "enable-backpropagation")]
    //// Sets the gradient function for the tensor
    fn set_grad_fn(&mut self, grad_fn: &dyn Function<'static, Type, Forwarded=(), Gradiant=()>);

    #[cfg(feature = "enable-backpropagation")]
    //// Returns the gradient of the tensor
   fn grad(&self) -> Option<&dyn TensorBase<Type>>;
}

impl<T> Debug for dyn TensorBase<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "TensorBase Debug")
    }
}

pub trait Function<'t, T: Debug + Clone> {
    type Forwarded;
    #[cfg(feature = "enable-backpropagation")]
    type Gradiant;

    fn new(first: &'t dyn TensorBase<T>, second: Option<&'t dyn TensorBase<T>>) -> MlResult<Self> where Self: Sized;
    fn forward(&'t mut self) ->  Self::Forwarded;
    #[cfg(feature = "enable-backpropagation")]
    fn backward(&'t mut self, grad: &'t dyn TensorBase<T>) -> Self::Gradiant;
    fn backend(&self) -> &Arc<dyn Backend>;
}

// impl<T> Debug for dyn Function<'_, T, Forwarded=(), Gradiant=()> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> Result {
//         write!(f, "Function Debug")
//     }
// }


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
pub struct Exp<'t, T>    { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a negation operation.
pub struct Neg<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a square root operation.
pub struct Sqrt<'t, T>    { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing an absolute value operation.
pub struct Abs<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a squaring operation.
pub struct Square<'t, T>  { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a logarithmic operation.
pub struct Log<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a power operation.
pub struct Pow<'t, T>     { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a Top-k operation.
pub struct Topk<'t, T>    { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<(Box<dyn TensorBase<T>>, Box<dyn TensorBase<T>>)>
} // k: second, sorted: third

/// Structure representing a matrix max operation along a dimension.
pub struct Matmax<'t, T>  { tensor: &'t dyn TensorBase<T>, backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<(Box<dyn TensorBase<T>>, Box<dyn TensorBase<T>>)>
} // dim: second, keepdim: third

/// Structure representing an addition operation.
pub struct Add<'t, T>     {
    first_tensor: &'t dyn TensorBase<T>,
    second_tensor: &'t dyn TensorBase<T>,
    backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a subtraction operation.
pub struct Sub<'t, T>     {
    first_tensor: &'t dyn TensorBase<T>,
    second_tensor: &'t dyn TensorBase<T>,
    backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a multiplication operation.
pub struct Mul<'t, T> {
    first_tensor: &'t dyn TensorBase<T>,
    second_tensor: &'t dyn TensorBase<T>,
    backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a division operation.
pub struct Div<'t, T> {
    first_tensor: &'t dyn TensorBase<T>,
    second_tensor: &'t dyn TensorBase<T>,
    backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

/// Structure representing a matrix multiplication operation.
pub struct Matmul<'t, T> {
    first_tensor: &'t dyn TensorBase<T>,
    second_tensor: &'t dyn TensorBase<T>,
    backend: Arc<dyn Backend>,
    #[cfg(feature = "enable-backpropagation")]
    output: Option<&'t dyn TensorBase<T>>
}

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::*;

    pub fn assert_tensor_eq(tensor: &Box<dyn TensorBase<f32>>, expected_tensor: &Box<dyn TensorBase<f32>>) -> MlResult<()> {
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
    fn test_add() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let m_add = ops!(first.as_ref(), Add, second.as_ref())?;
        let s_add = first + second;
        let et = Tensor::<f32>::new(vec![vec![4.0, 6.0]]);

        assert_tensor_eq(&m_add, &et)?;
        assert_tensor_eq(&s_add, &et)
    }
    #[test]
    fn test_sub() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let m_sub = ops!(first.as_ref(), Sub, second.as_ref())?;
        let s_sub = first - second;
        let et = Tensor::<f32>::new(vec![vec![-2.0, -2.0]]);

        assert_tensor_eq(&m_sub, &et)?;
        assert_tensor_eq(&s_sub, &et)
    }
    #[test]
    fn test_mul_symbol() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let m_mul = ops!(first.as_ref(), Mul, second.as_ref())?;
        let s_mul = first * second;
        let et = Tensor::<f32>::new(vec![vec![3.0, 8.0]]);

        assert_tensor_eq(&m_mul, &et)?;
        assert_tensor_eq(&s_mul, &et)
    }
    #[test]
    fn test_div_symbol() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![2.0, 4.0]]);
        let m_div = ops!(first.as_ref(), Div, second.as_ref())?;
        let s_div = first / second;
        let et = Tensor::<f32>::new(vec![vec![0.5, 0.5]]);

        assert_tensor_eq(&m_div, &et)?;
        assert_tensor_eq(&s_div, &et)
    }

    #[test]
    fn test_macro_matmul() {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0], vec![4.0]]);
        let result = ops!(first.as_ref(), Matmul, second.as_ref()).unwrap();
        assert_eq!(result.data(), vec![11.0]);
    }

    #[test]
    fn test_macro_exp() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = ops!(tensor.as_ref(), Exp).unwrap();
        assert_eq!(result.data(), vec![std::f32::consts::E, 7.389056]);
    }

    #[test]
    fn test_macro_neg() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = ops!(tensor.as_ref(), Neg).unwrap();
        assert_eq!(result.data(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_macro_sqrt() {
        let tensor = Tensor::new(vec![vec![1.0, 4.0]]);
        let result = ops!(tensor.as_ref(), Sqrt).unwrap();
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_macro_abs() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = ops!(tensor.as_ref(), Abs).unwrap();
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_macro_square() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = ops!(tensor.as_ref(), Square).unwrap();
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_macro_log() {
        let tensor = Tensor::new(vec![vec![1.0, std::f32::consts::E]]);
        let result = ops!(tensor.as_ref(), Log).unwrap();
        assert_eq!(result.data(), vec![0.0, 0.99999994]);
    }

    #[test]
    fn test_macro_pow() {
        let mut tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = ops!(tensor.as_mut(), Pow, 2.0).unwrap();
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn tensor_ops_add_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::<f32>::new(vec![vec![3.0, 4.0]]).unwrap();

        assert_tensor_eq(first + 2.0, et)
    }
    #[test]
    fn tensor_ops_sub_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::<f32>::new(vec![vec![-1.0, 0.0]]).unwrap();

        assert_tensor_eq(first - 2.0, et)
    }
    #[test]
    fn tensor_ops_mul_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::<f32>::new(vec![vec![2.0, 4.0]]).unwrap();

        assert_tensor_eq(first - 2.0, et)
    }
    #[test]
    fn tensor_ops_div_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::<f32>::new(vec![vec![0.5, 1.0]]).unwrap();

        assert_tensor_eq(first / 2.0, et)
    }

    #[test]
    fn tensor_ops_scalar_sub() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::<f32>::new(vec![vec![1.0, 0.0]]).unwrap();

        assert_tensor_eq(2.0 - first, et)

    }
    #[test]
    fn tensor_ops_scalar_div() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::<f32>::new(vec![vec![2.0, 1.0]]).unwrap();

        assert_tensor_eq(2.0 - first, et)
    }
}
