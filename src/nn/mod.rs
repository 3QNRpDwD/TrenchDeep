pub mod activation;
pub mod conv;
pub mod pooling;
pub mod linear;

use crate::{
    backend::{
        Backend,
        CpuBackend,
        Device
    },
    tensor::{
        operators::Function,
        AutogradFunction,
        Tensor,
        TensorBase,
        Variable,
        OPERATOR_STORAGE,
        GlobalFunction,
        GlobalTensor,
        NodeId,
        NODE_ID_GEN
    },
    MlResult,
    register_operator
};
use std::{
    fmt::Debug,
    sync::Arc,
    collections::HashSet
};

pub trait Layer {
    fn forward(&self, input: Arc<Variable>) -> MlResult<Arc<Variable>>;
    fn params(&self) -> Vec<&dyn Parameter>;
    fn type_name(&self) -> &str; // 레이어를 구현하는 구조체의 이름을 반환
    fn label(&self) -> &str;    // 유저가 설정한 레이어의 이름을 반환
}

pub trait Parameter {}
impl Parameter for Variable {}

pub struct Linear    {
    label: String
}
pub struct Conv      {
    label: String
}
pub struct Pooling  { 
    label: String
}

pub struct Sequential {
    label: String,
    layers: Vec<Box<dyn Layer>> // Box<dyn Layer>를 사용하여 다양한 종류의 레이어를 하나의 Vec에 저장
}

impl Sequential {
    pub fn new() -> Self {
        Self { label: "Sequential".to_string(), layers: vec![] }
    }

    pub fn from(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { label: "Sequential".to_string(), layers }
    }

    // 바로 이 부분입니다! 레이어 관리는 Sequential 같은 컨테이너의 고유 기능으로 정의합니다.
    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
    
    pub fn remove(&mut self, index: usize) -> Box<dyn Layer> {
        self.layers.remove(index)
    }
}

impl Layer for Sequential {
    fn forward(&self, input: Arc<Variable>) -> MlResult<Arc<Variable>> {
        let mut current_output= input.clone();
        for layer in &self.layers {
            current_output = layer.forward(current_output)?;
        }
        Ok(current_output)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        self.layers.iter().flat_map(|layer| layer.params()).collect()
    }

    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    fn label(&self) -> &str {
        &self.label
    }
}

