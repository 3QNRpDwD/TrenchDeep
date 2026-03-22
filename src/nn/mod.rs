pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;
mod parameter;
mod checkpoint;

use crate::{register_operator, var_act, var_bias, var_weight, backend::Backend, MlResult, TensorError, tensor::{
    operators::{Add, Div, Matmul, Mul, Sub},
    operators::Function,
    GlobalFunction,
    GlobalTensor,
    NodeId,
    Tensor,
    TensorBase
}, MlError};
use std::{
    cell::RefCell,
    fmt::{
        Formatter,
        Debug
    },
    ops::Deref,
    collections::{
        HashMap,
        HashSet
    },
    sync::Arc
};
use std::cell::Cell;
use crate::tensor::COMPUTATION_GRAPH;

#[macro_export]
macro_rules! variable {
    ($vec:expr) => {
        crate::nn::Variable::new(crate::tensor::Tensor::new($vec))
    };

    ($data:expr, $shape:expr) => {
        crate::nn::Variable::new(crate::tensor::Tensor::from_vec($data, $shape).unwrap())
    };

    ($data:expr, $shape:expr, $label:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::with_label(crate::tensor::Tensor::from_vec($data, $shape).unwrap(), $label)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new(crate::tensor::Tensor::from_vec($data, $shape).unwrap())
            }
        }
    };
}

pub trait Layer: Debug {
    fn apply(&mut self, input: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>>;
    fn params(&self) -> Vec<&dyn Parameter>;
    fn inputs_cache(&self) -> &HashSet<NodeId>;
    fn outputs_cache(&self) -> &HashMap<NodeId, NodeId>;
    fn inputs_cache_mut(&mut self) -> &mut HashSet<NodeId>;
    fn outputs_cache_mut(&mut self) -> &mut HashMap<NodeId, NodeId>;
    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    } // 레이어를 구현하는 구조체의 이름을 반환
    fn label(&self) -> &str;    // 유저가 설정한 레이어의 이름을 반환
}

pub trait Parameter: Debug {
    fn new(tensor: Tensor) -> Self where Self: Sized;

    #[cfg(feature = "enableBackward")]
    fn node_id(&self) -> NodeId;

    fn add_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Add::new()?.assign_forward(&[self.tensor(), &other_tensor], self.node_id())?;
        Ok(())
    }

    fn sub_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Sub::new()?.assign_forward(&[self.tensor(), &other_tensor], self.node_id())?;
        Ok(())
    }

    fn mul_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Mul::new()?.assign_forward(&[self.tensor(), &other_tensor], self.node_id())?;
        Ok(())
    }

    fn div_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Div::new()?.assign_forward(&[self.tensor(), &other_tensor], self.node_id())?;
        Ok(())
    }

    fn tensor(&self) -> &Tensor;
    fn is_retain_grad(&self) -> bool;

    fn retain_grad(&self);

    fn grad(&self) -> &Tensor;

    #[cfg(feature = "enableBackward")]
    fn set_grad(&self, grad: GlobalTensor<f32>);

    #[cfg(feature = "enableVisualization")]
    fn set_label(&mut self, new_label: &str);

    /// 현재 라벨 반환
    #[cfg(feature = "enableVisualization")]
    fn label(&self) -> &str;

    #[cfg(feature = "enableVisualization")]
    fn node_type(&self) -> &crate::tensor::NodeType;

    #[cfg(feature = "enableBackward")]
    fn clear_grad(&self);

    // 추가: O(1) dirty 조회
    // backward 루프에서 grad.is_empty() O(n) 스캔 대신 사용
    #[cfg(feature = "enableBackward")]
    fn is_grad_dirty(&self) -> bool;
    
    #[cfg(feature = "enableBackward")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()>;

    fn backward(&self) -> MlResult<()> {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            if graph.node_map.contains_key(&self.node_id()) {
                graph.ensure_topological_sort();
                graph.backward(self.node_id())
            } else {
                Err(MlError::StringError("계산 그래프가 생성되지 않았습니다.".to_string()))
            }
        })
    }

    /// Performs backpropagation and then automatically resets the computation graph
    /// to release memory for all intermediate tensors.
    fn backward_and_clear(&self) -> MlResult<()> {
        self.backward()?;
        crate::tensor::ComputationGraph::reset_graph();
        Ok(())
    }
}

// #[derive(Clone)]
// pub struct Variable {
//     #[cfg(all(feature = "enableVisualization"))]
//     label: String,
// #[cfg(all(feature = "enableVisualization"))]
//     node_type: crate::tensor::NodeType,
//     tensor: Tensor,
//     requires_grad: RefCell<bool>,
//     grad: Tensor,
// }

#[derive(Clone)]
pub struct Variable {
    #[cfg(feature = "enableVisualization")]
    label: Arc<String>,
    #[cfg(feature = "enableVisualization")]
    node_type: crate::tensor::NodeType,
    tensor: Tensor,
    requires_grad: Cell<bool>,
    grad: Tensor,
    // 추가 필드
    // grad에 실제 값이 기록됐는지 추적하는 O(1) 플래그.
    // Cell<bool>: Copy 타입 → Variable의 Clone derive 그대로 동작.
    // RefCell 불필요 — 대여 추적 없이 내부 가변성만 필요.
    // #[cfg] 로 enableBackward 외 빌드에서는 0바이트.
    #[cfg(feature = "enableBackward")]
    grad_dirty: Cell<bool>,
}

impl Debug for Variable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Variable");
        ds
            .field("tensor", &self.tensor)
            .field("requires_grad", &self.requires_grad);
        #[cfg(feature = "enableBackward")]
        {
            ds.field("grad", &self.grad);
        }
        ds.finish()
    }
}
#[derive(Debug)]
pub struct Linear    {
    label: String,
    inputs: HashSet<NodeId>,
    outputs: HashMap<NodeId, NodeId>,
    weight: Variable,
    bias: Variable
}

#[derive(Debug)]
pub struct Conv      {
    label: String,
    inputs: HashSet<NodeId>,
    outputs: HashMap<NodeId, NodeId>,
    weight: Variable,
    bias: Variable
}

#[derive(Debug)]
pub struct Pooling  {
    label: String,
    inputs: HashSet<NodeId>,
    outputs: HashMap<NodeId, NodeId>,
    weight: Arc<dyn Parameter>,
    bias: Arc<dyn Parameter>,
}

pub struct Sequential {
    label: String,
    layers: Vec<Box<dyn Layer>>, // Box<dyn Layer>를 사용하여 다양한 종류의 레이어를 하나의 Vec에 저장
    params: HashSet<NodeId>
}

impl Sequential {
    pub fn new() -> Self {
        Self { label: "Sequential".to_string(), layers: vec![], params: HashSet::new() }
    }

    pub fn from(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { label: "Sequential".to_string(), layers, params: HashSet::new() }
    }

    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
    
    pub fn remove(&mut self, index: usize) -> Box<dyn Layer> {
        self.layers.remove(index)
    }
}

impl Layer for Sequential {
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let mut current_output= variable!(vec![0.0], input.tensor().shape(), &input.label);
        for layer in &mut self.layers {
            current_output = layer.apply(&current_output)?
        };
        Ok(current_output)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let mut layer_iter = self.layers.iter_mut();
        let first_layer = match layer_iter.next() {
            Some(layer) => layer,
            None => return Err(MlError::StringError("Sequential 모델에 레이어가 없습니다.".to_string())),
        };

        let mut output = first_layer.predict(input)?;
        for layer in layer_iter {
            output = layer.predict(&output)?;
        }
        Ok(output)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        self.layers.iter().flat_map(|layer| layer.params()).collect()
    }

    fn inputs_cache(&self) -> &HashSet<NodeId> {
        todo!()
    }

    fn outputs_cache(&self) -> &HashMap<NodeId, NodeId> {
        todo!()
    }

    fn inputs_cache_mut(&mut self) -> &mut HashSet<NodeId> {
        todo!()
    }

    fn outputs_cache_mut(&mut self) -> &mut HashMap<NodeId, NodeId> {
        todo!()
    }

    fn label(&self) -> &str { &self.label }
}

impl Debug for Sequential {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Sequential");
        ds
            .field("label", &self.label)
            .field("layers", &self.layers)
            .field("params", &self.params)
            .finish()
    }
}

