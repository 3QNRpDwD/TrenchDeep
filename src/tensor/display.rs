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
            },
            &TensorError::TensorNotFound => {
                write!(f, "Tensor not found")
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

impl<Type: Debug + Clone> Debug for &dyn TensorBase<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "data: {:?}, shape: {:?}",
            self.data(), self.shape()
        )
    }
}

#[cfg(feature = "enableBackpropagation")]
impl Debug for ComputationGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("ComputationGraph");
        ds
            .field("nodes", &self.nodes)
            .field("node_map", &self.node_map)
            .field("adjacency_list", &self.adjacency_list)
            .field("reverse_adjacency", &self.reverse_adjacency)
            .field("topo_order", &self.topo_order)
            .field("is_sorted", &self.is_sorted)
            .finish()
    }
}

#[cfg(feature = "enableBackpropagation")]
impl Debug for ComputationNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("ComputationNode");
        ds
            .field("id", &self.id)
            .field("variable", &self.variable)
            .field("function", &self.function.as_ref().map(|f| f.type_name()))
            .field("inputs", &self.inputs)
            .field("is_leaf", &self.is_leaf)
            .finish()
    }
}