pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;
mod parameter;
mod checkpoint;

use crate::{register_operator, var_bias, var_weight, backend::Backend, MlResult, tensor::{
    operators::{Add, Matmul, Sub},
    operators::Function,
    GlobalFunction,
    GlobalTensor,
    NodeId,
    Tensor,
    TensorBase,
    AutogradFunction,
}, MlError, TensorError};
use std::{
    fmt::{
        Formatter,
        Debug
    },
    sync::Arc
};

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
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>>;
    fn params(&self) -> Vec<&dyn Parameter>;
    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    } // 레이어를 구현하는 구조체의 이름을 반환
    fn label(&self) -> &str;    // 유저가 설정한 레이어의 이름을 반환
}

pub trait Parameter: Debug {
    fn new(tensor: Tensor) -> Self where Self: Sized;
    fn node_id(&self) -> NodeId;
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
    // TENSOR_STORAGE의 GlobalTensor.dirty 플래그를 조회 (O(1))
    // backward 루프에서 grad.is_empty() O(n) 스캔 대신 사용
    // 모든 Variable 클론이 동일한 STORAGE 항목을 공유하므로 원본 Variable에서도 정확히 동작
    #[cfg(feature = "enableBackward")]
    fn is_grad_dirty(&self) -> bool;
    #[cfg(feature = "enableBackward")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()>;
    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self) -> MlResult<()> {
        crate::tensor::COMPUTATION_GRAPH.with(|graph| {
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
    #[cfg(all(feature = "enableBackward"))]
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
    requires_grad: std::cell::Cell<bool>,
    grad: Tensor,
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
pub struct Linear {
    label: String,
    weight: Variable,
    bias: Variable
}

#[derive(Debug)]
pub struct Conv {
    label: String,
    weight: Variable,
    bias: Variable
}

#[derive(Debug)]
pub struct Pooling {
    label: String,
    weight: Arc<dyn Parameter>,
    bias: Arc<dyn Parameter>,
}

pub struct Sequential {
    label: String,
    layers: Vec<Box<dyn Layer>>, // Box<dyn Layer>를 사용하여 다양한 종류의 레이어를 하나의 Vec에 저장
}

impl Sequential {
    pub fn new() -> Self {
        Self { label: "Sequential".to_string(), layers: vec![] }
    }

    pub fn from(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { label: "Sequential".to_string(), layers }
    }

    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
    
    pub fn remove(&mut self, index: usize) -> Box<dyn Layer> {
        self.layers.remove(index)
    }
}

impl Layer for Sequential {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        if self.layers.is_empty() {
            return Err(MlError::StringError("Sequential has no layers".to_string()));
        }
        let mut iter = self.layers.iter_mut();
        let mut current = iter.next().unwrap().apply(input)?;
        for layer in iter {
            current = layer.apply(&current)?;
        }
        Ok(current)
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

    fn label(&self) -> &str { &self.label }
}

impl Debug for Sequential {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Sequential");
        ds
            .field("label", &self.label)
            .field("layers", &self.layers)
            .finish()
    }
}

