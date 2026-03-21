use super::*;

#[cfg(feature = "enableBackward")]
impl Debug for ComputationGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("ComputationGraph");
        ds
            .field("nodes", &self.nodes)
            .field("node_map", &self.node_map)
            .field("adjacency_list", &self.adjacency_list)
            .field("topo_order", &self.topo_order)
            .field("is_sorted", &self.is_sorted)
            .finish()
    }
}

#[cfg(feature = "enableBackward")]
impl Debug for ComputationNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut ds = f.debug_struct("ComputationNode");
        ds
            .field("id", &self.id)
            .field("variable", &self.variable)
            .field("function", &self.function.as_ref().unwrap())
            .field("inputs", &self.inputs)
            .field("is_leaf", &self.is_leaf)
            .finish()
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "data: {:?}, shape: {:?}",
            self.data(), self.shape()
        )
    }
}

impl Debug for &dyn TensorBase {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "data: {:?}, shape: {:?}",
            self.data(), self.shape()
        )
    }
}