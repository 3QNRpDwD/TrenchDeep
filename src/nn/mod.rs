use std::{
    cell::RefCell,
    collections::{
        HashMap,
        HashSet
    },
    fmt::{
        Debug,
        Formatter
    },
    ops::Deref,
    sync::Arc
};

use crate::{
    backend::Backend,
    MlError,
    MlResult,
    register_operator,
    tensor::{
        GlobalFunction,
        GlobalTensor,
        HandleId,
        operators::{
            Add,
            Div,
            Matmul,
            Mul,
            Sub,
            Function
        },
        Tensor,
        TensorBase,
        PooledTensor,
        TENSOR_ALLOCATOR
    },
    TensorError,
    var_act,
    var_bias,
    var_weight,
};

pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;
mod parameter;

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
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>>;
    fn params(&self) -> Vec<&dyn Parameter>;
    fn type_name(&self) -> &str { std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown") } // 레이어를 구현하는 구조체의 이름을 반환
    fn label(&self) -> &str;    // 유저가 설정한 레이어의 이름을 반환
}

pub trait Parameter: Debug {
    fn new(tensor: Tensor) -> Self where Self: Sized;

    fn node_id(&self) -> HandleId;

    fn add_tensor(&self, other_tensor: &dyn TensorBase) -> MlResult<()> {
        Add::new()?.assign_forward(&[self.tensor(), other_tensor], self.node_id())?;
        Ok(())
    }

    fn sub_tensor(&self, other_tensor: &dyn TensorBase) -> MlResult<()> {
        Sub::new()?.assign_forward(&[self.tensor(), other_tensor], self.node_id())?;
        Ok(())
    }

    fn mul_tensor(&self, other_tensor: &dyn TensorBase) -> MlResult<()> {
        Mul::new()?.assign_forward(&[self.tensor(), other_tensor], self.node_id())?;
        Ok(())
    }

    fn div_tensor(&self, other_tensor: &dyn TensorBase) -> MlResult<()> {
        Div::new()?.assign_forward(&[self.tensor(), other_tensor], self.node_id())?;
        Ok(())
    }

    fn tensor(&self) -> &Tensor;
    fn is_retain_grad(&self) -> bool;

    fn retain_grad(&self);

    fn grad(&self) -> &Tensor;

    #[cfg(feature = "enableBackpropagation")]
    fn set_grad(&self, grad: GlobalTensor<f32>);

    #[cfg(feature = "enableVisualization")]
    fn set_label(&mut self, new_label: &str);

    /// 현재 라벨 반환
    #[cfg(feature = "enableVisualization")]
    fn label(&self) -> &str;

    #[cfg(feature = "enableVisualization")]
    fn node_type(&self) -> &crate::tensor::NodeType;

    #[cfg(feature = "enableBackpropagation")]
    fn clear_grad(&self);

    #[cfg(feature = "enableBackpropagation")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()>;

    #[cfg(feature = "enableBackpropagation")]
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

    fn tpye_name(&self) -> String {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown").replace("<f32>", "")
    }
}

#[derive(Clone)]
pub struct Variable {
    #[cfg(all(feature = "enableVisualization"))]
    label: String,
    #[cfg(all(feature = "enableVisualization"))]
    node_type: crate::tensor::NodeType,
    tensor: Tensor,
    grad: Tensor,
    requires_grad: RefCell<bool>,
    is_persistent: RefCell<bool>,
}

impl Variable {
    pub fn is_persistent(&self) -> bool {
        *self.is_persistent.borrow()
    }
}

impl Debug for Variable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Variable");
        ds
            .field("tensor", &self.tensor.id())
            .field("requires_grad", &self.requires_grad.take());
        #[cfg(feature = "enableBackpropagation")]
        {
            ds.field("grad", &self.grad.id());
        }
        ds.finish()
    }
}
#[derive(Debug)]
pub struct Linear    {
    label: String,
    weight: Variable,
    bias: Variable,
    matmul: GlobalFunction,
    add: GlobalFunction
}

#[derive(Debug)]
pub struct Conv      {
    label: String,
    inputs: HashSet<HandleId>,
    outputs: HashMap<HandleId, HandleId>,
    weight: Variable,
    bias: Variable
}

#[derive(Debug)]
pub struct Pooling  {
    label: String,
    inputs: HashSet<HandleId>,
    outputs: HashMap<HandleId, HandleId>,
    weight: Arc<dyn Parameter>,
    bias: Arc<dyn Parameter>,
}

pub struct Sequential {
    label: String,
    layers: Vec<Box<dyn Layer>>, // Box<dyn Layer>를 사용하여 다양한 종류의 레이어를 하나의 Vec에 저장
    params: HashSet<HandleId>
}

#[macro_export]
macro_rules! sequential {
    ($($layer:expr),* $(,)?) => {
        {
            let mut seq = Sequential::new();
            $(seq.push($layer);)*
            seq
        }
    };
}

impl Sequential {
    pub fn new() -> Self {
        Self { label: "Sequential".to_string(), layers: vec![], params: HashSet::new() }
    }

    pub fn from<T: Layer+ 'static>(layers: Vec<T>) -> Self {
        // 각 레이어를 Box로 감싸서 트레이트 객체로 변환
        let boxed_layers: Vec<Box<dyn Layer>> = layers
            .into_iter()
            .map(|layer| Box::new(layer) as Box<dyn Layer>)
            .collect();

        Self {
            label: "Sequential".to_string(),
            layers: boxed_layers,
            params: HashSet::new(),
        }
    }

    pub fn add_layer<T: Layer+ 'static>(mut self, layer: T) -> Self {
        self.layers.push(Box::new(layer) as Box<dyn Layer>);
        self
    }

    
    pub fn remove(&mut self, index: usize) -> Box<dyn Layer> {
        self.layers.remove(index)
    }
}

impl Layer for Sequential {
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let mut current_output= input.clone();
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
    fn params(&self) -> Vec<&dyn Parameter> { self.layers.iter().flat_map(|layer| layer.params()).collect() }
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

