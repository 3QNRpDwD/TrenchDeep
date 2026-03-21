use std::{
    fmt::{
        Debug,
        Display,
        Formatter,
        Result
    },
    sync::Arc
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use crate::backend::{
    Backend,
    CpuBackend,
    Device
};


pub mod creation;
pub mod operators;
pub mod display;
pub mod graph;
pub mod visualization;

use crate::{MlError, MlResult, register_operator, tensor::operators::Function, TensorError};
use crate::nn::{Parameter, Variable};

#[macro_export]
macro_rules! tensor_ops {
    ($tensor:expr, Pow, $exponent:expr) => {{
        let op = crate::tensor::operators::Pow::new().unwrap();
        let power_t = Tensor::scalar($exponent);
        op.forward(&[&$tensor, &power_t]).unwrap().remove(0)
    }};

    ($tensor:expr, $op:ident, $second_tensor:expr) => {
        $op::new().unwrap().forward(&[&$tensor, &$second_tensor]).unwrap().remove(0)
    };

    ($tensor:expr, $op:ident) => {
        $op::new().unwrap().forward(&[&$tensor]).unwrap().remove(0)
    };

    ($tensor:expr, Topk, $k:expr, $sorted:expr) => {{
        let op = crate::tensor::operators::Topk::new().unwrap();
        let k_t = Tensor::scalar($k as f32);
        let sorted_t = Tensor::scalar(if $sorted { 1.0 } else { 0.0 });
        let mut result = op.forward(&[&$tensor, &k_t, &sorted_t]).unwrap();
        (result.remove(0), result.remove(0))
    }};

    ($tensor:expr, Matmax, $dim:expr, $keepdim:expr) => {{
        let op = crate::tensor::operators::Matmax::new().unwrap();
        let dim_val: Option<i32> = $dim;
        let dim_t = match dim_val {
            Some(d) => Tensor::scalar(d as f32),
            None => Tensor::scalar(f32::NAN),
        };
        let keepdim_t = Tensor::scalar(if $keepdim { 1.0 } else { 0.0 });
        let mut result = op.forward(&[&$tensor, &dim_t, &keepdim_t]).unwrap();
        (result.remove(0), result.remove(0))
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
            use crate::tensor::GlobalTensor;
            {
                GlobalTensor::new(vec![vec![$scalar]])
            }
        }
    };
}


#[derive(Debug, Clone)]
pub struct GlobalTensor<Type> {
    pub data: Vec<Type>,
    pub shape: Vec<usize>,
}

// pub struct Variable {
//     #[cfg(all(feature = "enableVisualization"))]
//     label: String,
//     #[cfg(all(feature = "enableVisualization"))]
//     node_type: NodeType,
//     tensor: Tensor,
//     requires_grad: RefCell<bool>,
//     grad: RefCell<Option<Tensor>>,
// }

#[derive(Clone, Debug)]
pub struct GlobalFunction {
    name: String,
    func_id: NodeId,
}

#[derive(Debug)]
pub struct TensorHandle {
    id: NodeId,
    label: String,
    owns_data: bool,
}

impl Drop for TensorHandle {
    fn drop(&mut self) {
        if !self.owns_data {
            return;
        }
        let id = self.id;
        let label = self.label.clone();
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            if storage.remove(&id).is_some() {
                if id.0 % 100 == 0 {
                    tracing::trace!("🔥 [Tensor Release] ID: {:?}, Label: '{}' - Memory freed.", id, label);
                }
            }
        });
    }
}

pub struct Tensor (Arc<TensorHandle>);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let new_ref = Self(self.0.clone());
        let rc = Arc::strong_count(&self.0);
        if self.id().0 % 100 == 0 {
            tracing::trace!("✨ [Tensor Clone] ID: {:?}, Label: '{}', New RC: {}", self.id(), self.0.label, rc);
        }
        new_ref
    }
}
// 기존의 텐서는 직접 variable 에 소유되는 구조로, 메모리 관리와 정적계산그래프 구현이 불가능하기 때문에 실제 텐서는 전역으로 관리하며 기존의 텐서는 아이디를 통해서 관리하도록 변경함.

impl Tensor {
    pub fn new_with_id(id: NodeId) -> Self {
        let label = format!("tensor_{:?}", id);
        if id.0 % 100 == 0 {
            tracing::trace!("🆕 [Tensor Create] ID: {:?}, Label: '{}'", id, label);
        }
        Self(Arc::new(TensorHandle { id, label, owns_data: true }))
    }

    pub fn new_with_label(id: NodeId, label: &str) -> Self {
        if id.0 % 100 == 0 {
            tracing::trace!("🆕 [Tensor Create] ID: {:?}, Label: '{}'", id, label);
        }
        Self(Arc::new(TensorHandle { id, label: label.to_string(), owns_data: true }))
    }

    pub fn new_ref(id: NodeId) -> Self {
        let label = format!("tensor_ref_{:?}", id);
        Self(Arc::new(TensorHandle { id, label, owns_data: false }))
    }

    pub fn id(&self) -> NodeId {
        self.0.id
    }

    pub fn replace(&self, other_tensor: GlobalTensor<f32>) {
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(self.id(), other_tensor)
        });
    }

    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
}

impl Function for GlobalFunction {
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        OPERATOR_STORAGE.with(|ops| {
            let mut ops = ops.borrow_mut();
            match ops.get_mut(self.name()) {
                Some(op) => op.forward(inputs),
                None => Err(MlError::StringError(format!("Function {} is not registered globally.", self.type_name())))
            }
        })
    }

    fn assign_forward(&self, inputs: &[&dyn TensorBase], node_id: NodeId) -> MlResult<Vec<Tensor>> {
        OPERATOR_STORAGE.with(|ops| {
            let mut ops = ops.borrow_mut();
            match ops.get_mut(self.name()) {
                Some(op) => op.assign_forward(inputs, node_id),
                None => Err(MlError::StringError(format!("Function {} is not registered globally.", self.type_name())))
            }
        })
    }

    #[cfg(feature = "enableBackward")]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        OPERATOR_STORAGE.with(|ops| {
            let mut ops = ops.borrow_mut();
            match ops.get_mut(self.name()) {
                Some(op) => op.backward(targets, grad),
                None => Err(MlError::StringError(format!("Function {} is not registered globally.", self.type_name())))
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);


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

    pub fn next(&self) -> NodeId {
        NodeId(self.counter.fetch_add(1, Ordering::Relaxed))
    }

    pub fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }
}

pub(crate) struct ComputationNode {
    id: NodeId,
    variable: Variable,
    function: Option<String>,
    inputs: Vec<NodeId>,
    is_leaf: bool,
}

pub(crate) struct ComputationGraph {
    nodes: Vec<ComputationNode>,
    pub(crate) node_map: HashMap<NodeId, usize>,
    adjacency_list: Vec<Vec<usize>>,
    reverse_adjacency: Vec<Vec<usize>>,
    topo_order: Vec<usize>,
    is_sorted: bool,
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

pub struct ExecutionContext {
    pub graph: ComputationGraph,
    pub tensor_storage: HashMap<NodeId, GlobalTensor<f32>>,
    pub node_id_generator: NodeIdGenerator, // NodeId 생성기도 컨텍스트에 포함
    // 필요하다면 시각화 그래프나 다른 상태도 여기에 추가할 수 있습니다.
    #[cfg(feature = "enableVisualization")]
    pub visualization_graph: VisualizationGraph,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            tensor_storage: HashMap::new(),
            node_id_generator: NodeIdGenerator::new(),
            #[cfg(feature = "enableVisualization")]
            visualization_graph: VisualizationGraph::new(),
        }
    }

    pub fn add_tensor(&mut self, tensor: GlobalTensor<f32>) -> Tensor {
        let node_id = self.node_id_generator.next();
        self.tensor_storage.insert(node_id, tensor);
        Tensor::new_with_id(node_id)
    }

    pub fn get_tensor_data(&self, tensor: &Tensor) -> Option<&GlobalTensor<f32>> {
        self.tensor_storage.get(&tensor.id())
    }

    // ... 기타 필요한 헬퍼 메서드들 ...
}

thread_local! {
    #[cfg(feature = "enableBackward")]
    pub(crate) static   COMPUTATION_GRAPH   : std::sync::Mutex<ComputationGraph> = std::sync::Mutex::new(ComputationGraph::new());
    pub(crate) static   OPERATOR_STORAGE    : RefCell<HashMap<String, Box<dyn Function>>> = RefCell::new(HashMap::new());
    pub(crate) static   TENSOR_STORAGE      : RefCell<HashMap<NodeId, GlobalTensor<f32>>> = RefCell::new(HashMap::new());
    // pub(crate) static   EXECUTION_CONTEXT    : RefCell<ExecutionContext> = RefCell::new(ExecutionContext::new());
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

    fn shape(&self) -> &[usize] {
        unimplemented!(" TensorBase::shape() is not implemented ")
    }

    fn data(&self) -> &[f32] {
        unimplemented!(" TensorBase::data() is not implemented ")
    }

    fn get(&self, _indices: &[usize]) -> Option<&f32> {
        unimplemented!(" TensorBase::get() is not implemented ")
    }

    fn index(&self, _indices: &[usize]) -> Option<usize> {
        unimplemented!(" TensorBase::index() is not implemented ")
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

pub trait AutogradFunction: Function {
    fn apply(&mut self, _inputs: &[&Variable]) -> MlResult<Variable> {
        unimplemented!(" AutogradFunction::apply() not implemented for this type")
    }

    fn apply_with_label(&mut self, inputs: &[&Variable], label: &str) -> MlResult<Variable> {
        unimplemented!(" AutogradFunction::apply_with_label() not implemented for this type")
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Abs, Add, Div, Exp, Function, Log, Matmul, Mul, Neg, Sqrt, Square, Sub};
    use crate::tensor::{Tensor, TensorBase};
    use crate::MlResult;

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
        let result = tensor_ops!(tensor, Pow, 2.0);
        assert_eq!(result.data(), vec![4.0, 9.0]);
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
