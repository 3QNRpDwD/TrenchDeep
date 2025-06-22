pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;

use crate::{
    backend::{
        Backend,
    },
    tensor::{
        operators::Function,
        TensorBase,
        Variable,
        GlobalFunction,
        GlobalTensor,
        NodeId,
    },
    MlResult,
};
use std::{
    fmt::Debug,
    sync::Arc,
    collections::HashSet,
    collections::HashMap
};

#[macro_export]
macro_rules! register_layer {
    ($name:ident) => {
        {
        use crate::tensor::NODE_ID_GEN;
        use crate::tensor::OPERATOR_STORAGE;
        use crate::backend::CpuBackend;
        use crate::backend::Device;
        use std::collections::HashSet;
        use std::collections::HashMap;
            {
                OPERATOR_STORAGE.with(|ops| {
                    let my = stringify!($name);
                    let mut ops = ops.borrow_mut();
                    match ops.contains_key(my) {
                        true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                        false => {
                            ops.insert(
                                String::from(my),
                                Box::new($name {
                                    backend: Arc::new(CpuBackend::new()?),
                                    node_id: NODE_ID_GEN.next(),
                                    inputs: HashSet::new(),
                                    outputs: HashMap::new(),
                                    label: my.to_string()
                                })
                            );
                            Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                        }
                    }
                })
            }
        }
    };
}

pub trait Layer {
    fn forward(&mut self, input: Arc<Variable>) -> MlResult<Arc<Variable>>;
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

pub trait Parameter {}
impl Parameter for Variable {}

pub struct Linear    {
    label: String,
    inputs: HashSet<NodeId>,
    params: HashSet<NodeId>
}
pub struct Conv      {
    label: String,
    inputs: HashSet<NodeId>,
    params: HashSet<NodeId>
}
pub struct Pooling  {
    label: String,
    inputs: HashSet<NodeId>,
    params: HashSet<NodeId>
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
    fn forward(&mut self, input: Arc<Variable>) -> MlResult<Arc<Variable>> {
        let mut current_output= input.clone();
        for layer in &mut self.layers {
            current_output = layer.forward(current_output)?
        };
        Ok(current_output)
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

