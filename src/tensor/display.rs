use super::*;

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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

impl Display for MlError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            MlError::TensorError(e) => write!(f, "Tensor error: {}", e),
            MlError::StringError(s) => write!(f, "{}", s),
        }
    }
}

impl<Type: Debug> Debug for Variable<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("Variable");
        ds
            .field("tensor", &self.tensor)
            .field("requires_grad", &self.requires_grad);
        #[cfg(feature = "enableBackpropagation")]
        {
            ds.field("grad", &self.grad);
        }
        ds.finish()
    }
}

#[cfg(feature = "enableBackpropagation")]
impl<Type: Debug + Clone> Debug for ComputationGraph<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("ComputationGraph");
        ds
            .field("nodes", &self.nodes)
            .field("topo_sorted", &self.topo_sorted)
            .field("sorted", &self.sorted)
            .finish()
    }
}

#[cfg(feature = "enableBackpropagation")]
impl<Type: Debug + Clone> Debug for ComputationNode<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("ComputationNode");
        ds
            .field("id", &self.id)
            .field("variable", &self.variable)
            .field("function", &self.function.as_ref().map(|f| f.type_name()))
            .field("inputs", &self.inputs)
            .field("is_life", &self.is_life)
            .finish()
    }
}

impl<Type: Debug + Clone> Debug for &dyn TensorBase<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "data: {:?}, shape: {:?}",
            self.data(), self.shape()
        )
    }
}