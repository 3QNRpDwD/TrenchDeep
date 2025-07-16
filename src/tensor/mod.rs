use std::{
    fmt::{
        Debug,
        Display,
        Formatter,
        Result
    },
    sync::{
        Arc,
        atomic::Ordering
    },
    collections::{HashMap, HashSet},
    cell::RefCell,
};

use crate::{
    backend::{
        Backend,
        CpuBackend,
        Device
    },
    MlResult,
    MlError,
    tensor::{
        operators::{
            Function,
        }
    },
    nn::{Parameter, Variable},
    TensorError,
};

pub mod creation;
pub mod operators;
pub mod display;
pub mod graph;
pub mod visualization;
mod allocator;

#[macro_export]
macro_rules! tensor_ops {
    // Topk, Matmax 등 특수 반환 값을 갖는 연산자
    ($tensor:expr, Topk, $($field:ident = $value:expr),+) => {{
        let op = crate::tensor::operators::Topk::new($($value),+);
        let mut result = crate::tensor::operators::Topk::forward(&op, &[&$tensor]).unwrap();
        (result.remove(0), result.remove(0))
    }};
    ($tensor:expr, Matmax, $($field:ident = $value:expr),+) => {{
        let op = crate::tensor::operators::Matmax::new($($value),+);
        let mut result = crate::tensor::operators::Matmax::forward(&op, &[&$tensor]).unwrap();
        (result.remove(0), result.remove(0))
    }};
    // 파라미터가 있는 연산자 (e.g., Pow, power = 2.0)
    ($tensor:expr, $op:ident, $($field:ident = $value:expr),+) => {{
        let op = crate::tensor::operators::$op::new($($value),+);
        crate::tensor::operators::$op::forward(&op, &[&$tensor]).unwrap().remove(0)
    }};
    // 이항 연산자
    ($tensor:expr, $op:ident, $second_tensor:expr) => {{
        let op = crate::tensor::operators::$op::new();
        crate::tensor::operators::$op::forward(&op, &[&$tensor, &$second_tensor]).unwrap().remove(0)
    }};
    // 단항 연산자
    ($tensor:expr, $op:ident) => {{
        let op = crate::tensor::operators::$op::new();
        crate::tensor::operators::$op::forward(&op, &[&$tensor]).unwrap().remove(0)
    }};
}

#[macro_export]
macro_rules! scalar_ops {
    ($tensor:expr, Add, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x + $scalar).collect(), &$tensor.shape())
    };

    ($tensor:expr, Sub, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x - $scalar).collect(), &$tensor.shape())
    };

    ($tensor:expr, Mul, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x * $scalar).collect(), &$tensor.shape())
    };

    ($tensor:expr, Div, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x / $scalar).collect(), &$tensor.shape())
    };

    ($scalar:expr, buS, $tensor:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| $scalar - x).collect(), &$tensor.shape())
    };

    ($scalar:expr, viD, $tensor:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| $scalar / x).collect(), &$tensor.shape())
    };
}

#[macro_export]
macro_rules! scalar  {

    ($scalar:expr) => {
        {
            use crate::tensor::PooledTensor;
            {
                PooledTensor::new(vec![vec![$scalar]])
            }
        }
    };
}


#[derive(Debug, Clone)]
pub struct GlobalTensor<Type> {
    pub data: Vec<Type>,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
pub struct Tensor (HandleId);

impl Tensor {
    pub fn replace(&self, other_tensor: GlobalTensor<f32>) {
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.storage.insert(self.0, other_tensor)
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandleId(u64);


pub struct NodeIdGenerator {
    counter: std::sync::atomic::AtomicU64,
}


pub(crate) static NODE_ID_GEN: NodeIdGenerator = NodeIdGenerator::new();


impl NodeIdGenerator {
    pub const fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn next(&self) -> HandleId {
        HandleId(self.counter.fetch_add(1, Ordering::Relaxed))
    }

    pub fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }
}

pub(crate) struct ComputationNode {
    id: HandleId,
    variable: Variable,
    function: Option<Arc<dyn Function + Send + Sync>>,
    inputs: Vec<HandleId>,
    is_leaf: bool,
}

pub(crate) struct ComputationGraph {
    nodes: Vec<ComputationNode>,
    pub(crate) node_map: HashMap<HandleId, usize>,
    adjacency_list: Vec<Vec<usize>>,
    reverse_adjacency: Vec<Vec<usize>>,
    topo_order: Vec<usize>,
    is_sorted: bool,
    pub(crate) memory_pool: Vec<HandleId>
}

#[cfg(feature = "enableVisualization")]
#[derive(Debug, Clone)]
pub struct VisualizationGraph {
    pub nodes: HashSet<String>,
    pub edges: Vec<String>,
    pub node_types: HashMap<String, NodeType>,
    pub node_labels: HashMap<String, String>,
}

#[cfg(feature = "enableVisualization")]
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Variable,
    Function,
    Input,
    Weight,
    Bias,
    Loss,
    Activation,
    Output,
}

#[derive(Debug)]
pub struct TensorAllocator {
    storage: HashMap<HandleId, GlobalTensor<f32>>,
    pool: HashMap<Vec<usize>, Vec<HandleId>>,
}

#[derive(Debug, Clone)]
pub struct PooledTensor {
    node_id: HandleId,
    detached: bool
}

thread_local! {
    #[cfg(feature = "enableBackpropagation")]
    pub(crate) static   COMPUTATION_GRAPH   : std::sync::Mutex<ComputationGraph> = std::sync::Mutex::new(ComputationGraph::new());
    pub(crate) static   TENSOR_ALLOCATOR    : RefCell<TensorAllocator> = RefCell::new(TensorAllocator::new());
    #[cfg(feature = "enableVisualization")]
    pub(crate) static   VISUALIZATION_GRAPH : RefCell<VisualizationGraph> = RefCell::new(VisualizationGraph::new());
    #[cfg(feature = "enableVisualization")]
    static              LABEL_COUNTERS      : RefCell<HashMap<String, usize>> = RefCell::new(HashMap::new());
    #[cfg(feature = "enableVisualization")]
    static              SHAPE_REGISTRY      : RefCell<HashMap<String, usize>> = RefCell::new(HashMap::new());
}


impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data() && self.shape() == other.shape()
    }
}

impl Eq for Tensor {
    // Todo: 구현 필요
}

impl PartialOrd for Tensor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data().partial_cmp(&other.data())
    }
}

impl Ord for Tensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}


pub trait TensorBase {
    fn new(_data: Vec<Vec<f32>>) -> Self where Self: Sized {
        unimplemented!(" TensorBase::new() is not implemented ")
    }

    fn from_vec(_data: Vec<f32>, _shape: &[usize]) -> MlResult<Self> where Self: Sized {
        unimplemented!(" TensorBase::from_vec() is not implemented ")
    }

    fn as_ptr(&self) -> *const GlobalTensor<f32> {
        unimplemented!(" TensorBase::tensor_ptr() is not implemented ")
    }

    fn as_mut(&self) -> *mut GlobalTensor<f32> {
        unimplemented!(" TensorBase::tensor_mut() is not implemented ")
    }

    fn shape(&self) -> &[usize] {
        unimplemented!(" TensorBase::shape() is not implemented ")
    }

    fn data(&self) -> &[f32] {
        unimplemented!(" TensorBase::data() is not implemented ")
    }

    fn get(&self, _indices: &[usize]) -> Option<&f32> {
        unimplemented!(" TensorBase::get() is not implemented ")
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape().len() {
            return None;
        }
        let mut idx = 0;
        let shape = self.shape();
        for (i, &ind) in indices.iter().enumerate() {
            if ind >= shape[i] {
                return None;
            }
            idx = idx * shape[i] + ind;
        }
        Some(idx)
    }

    fn chk_shape(&self, other: &dyn TensorBase) -> MlResult<()> {
        if self.shape() == other.shape() {
            Ok(())
        } else {
            Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }))
        }
    }

    fn zeros(shape: &[usize]) -> Self where Self: Sized {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Self::from_vec(data, shape).unwrap()
    }

    fn zeros_like(&self) -> Self where Self: Sized {
        Self::zeros(&self.shape())
    }

    fn ones(shape: &[usize]) -> Self where Self: Sized {
        let size: usize = shape.iter().product();
        let data = vec![1.0; size];
        Self::from_vec(data, shape).unwrap()
    }

    fn ones_like(&self) -> Self where Self: Sized {
        Self::ones(&self.shape())
    }

    fn rand(shape: &[usize]) -> Self where Self: Sized {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
        Self::from_vec(data, shape).unwrap()
    }

    fn scalar(scalar: f32) -> Self where Self: Sized {
        Self::new(vec![vec![scalar]])
    }
}

pub trait AutogradFunction: Function + Send + Sync  + 'static {
    fn apply(self: &Arc<Self>, _inputs: &[&Variable]) -> MlResult<Variable> {
        unimplemented!(" AutogradFunction::apply() not implemented for this type")
    }

    fn apply_with_label(self: &Arc<Self>, inputs: &[&Variable], label: &str) -> MlResult<Variable> {
        unimplemented!(" AutogradFunction::apply_with_label() not implemented for this type")
    }
}

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::{Tensor, TensorBase};
    use crate::tensor::operators::{Abs, Add, Cos, Div, Exp, Function, Log, Matmax, Matmul, Mul, Neg, Pow, Sin, Sqrt, Square, Sub, Sum, Topk, Transpose};

    pub fn assert_tensor_eq(tensor: &dyn TensorBase, expected_tensor: &dyn TensorBase) -> MlResult<()> {
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
    fn test_add_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let m_add = tensor_ops!(first, Add, second);
        assert_tensor_eq(&m_add, &expected)
    }

    #[test]
    fn test_sub_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![-2.0, -2.0]]);
        let m_sub = tensor_ops!(first, Sub, second);
        assert_tensor_eq(&m_sub, &expected)
    }

    #[test]
    fn test_mul_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![3.0, 8.0]]);
        let m_mul = tensor_ops!(first, Mul, second);
        assert_tensor_eq(&m_mul, &expected)
    }

    #[test]
    fn test_div_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let m_div = tensor_ops!(first, Div, second);
        assert_tensor_eq(&m_div, &Tensor::new(vec![vec![0.5, 0.5]]))
    }

    #[test]
    fn test_matmul_macro() {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0], vec![4.0]]);
        let result = tensor_ops!(first, Matmul, second);
        assert_eq!(result.data(), vec![11.0]);
    }

    #[test]
    fn tes_macro_exp_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = tensor_ops!(tensor, Exp);
        assert_eq!(result.data(), vec![std::f32::consts::E, 7.389056]);
    }

    #[test]
    fn test_neg_macro() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = tensor_ops!(tensor, Neg);
        assert_eq!(result.data(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_sqrt_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 4.0]]);
        let result = tensor_ops!(tensor, Sqrt);
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_abs_macro() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = tensor_ops!(tensor, Abs);
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_square_macro() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = tensor_ops!(tensor, Square);
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_log_macro() {
        let tensor = Tensor::new(vec![vec![1.0, std::f32::consts::E]]);
        let result = tensor_ops!(tensor, Log);
        assert_eq!(result.data(), vec![0.0, 0.99999994]);
    }

    #[test]
    fn test_pow_macro() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = tensor_ops!(tensor, Pow, power = Some(2.0));
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_sum_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = tensor_ops!(tensor, Sum);
        assert_eq!(result.data(), vec![10.0]);
        assert_eq!(result.shape(), vec![1,1]);
    }

    #[test]
    fn test_sin_macro() {
        let tensor = Tensor::new(vec![vec![0.0, std::f32::consts::PI / 2.0]]);
        let result = tensor_ops!(tensor, Sin);
        assert_eq!(result.data(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_cos_macro() {
        let tensor = Tensor::new(vec![vec![0.0, std::f32::consts::PI]]);
        let result = tensor_ops!(tensor, Cos);
        assert_eq!(result.data(), vec![1.0, -1.0]);
    }

    #[test]
    fn test_transpose_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]); // shape [2, 2]
        let result = tensor_ops!(tensor, Transpose, dims = (0, 1));
        let expected = Tensor::new(vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
        assert_tensor_eq(&result, &expected).unwrap();
    }

    #[test]
    fn test_topk_macro() {
        let tensor = Tensor::from_vec(vec![1.0, 9.0, 4.0, 6.0], &[1, 4]).unwrap();
        let (values, indices) = tensor_ops!(tensor, Topk, topk = Some((2, true)));
        assert_eq!(values.data(), vec![9.0, 6.0]);
        assert_eq!(indices.data(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_matmax_macro() {
        let tensor = Tensor::from_vec(vec![1.0, 9.0, 4.0, 6.0], &[2, 2]).unwrap();
        let (values, indices) = tensor_ops!(tensor, Matmax, matmax = Some((Some(1), false)));
        assert_eq!(values.data(), vec![9.0, 6.0]);
        assert_eq!(values.shape(), vec![2]);
        assert_eq!(indices.data(), vec![1.0, 1.0]);
        assert_eq!(indices.shape(), vec![2]);
    }

    #[test]
    fn tensor_add_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Add, 2.0)?;
        assert_tensor_eq(&result, &Tensor::new(vec![vec![3.0, 4.0]]))
    }
    #[test]
    fn tensor_sub_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Sub, 2.0)?;
        assert_tensor_eq(&result, &Tensor::new(vec![vec![-1.0, 0.0]]))
    }
    #[test]
    fn tensor_mul_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Mul , 2.0)?;
        assert_tensor_eq(&result, &Tensor::new(vec![vec![2.0, 4.0]]))
    }
    #[test]
    fn tensor_div_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Div , 2.0)?;
        assert_tensor_eq(&result, &Tensor::new(vec![vec![0.5, 1.0]]))
    }

    #[test]
    fn tensor_scalar_sub() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(2.0, buS , first)?;
        assert_tensor_eq(&result, &Tensor::new(vec![vec![1.0, 0.0]]))

    }
    #[test]
    fn tensor_scalar_div() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(2.0, viD , first)?;
        assert_tensor_eq(&result, &Tensor::new(vec![vec![2.0, 1.0]]))
    }
}
