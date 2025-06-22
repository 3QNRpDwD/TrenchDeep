pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;

use super::*;
use crate::register_layer;

#[macro_export] // 다른 모듈에서도 사용할 수 있도록 export
macro_rules! impl_activation_layer {
    // 매크로는 `$struct_name`이라는 이름의 '식별자(identifier)'를 인자로 받습니다.
    // 예: impl_activation_layer!(ReLU); -> $struct_name은 ReLU가 됩니다.
    ($struct_name:ident) => {
        // `impl Layer for $struct_name` 블록을 자동으로 생성합니다.
        impl Layer for $struct_name {
            fn forward(&mut self, input: Arc<Variable>) -> MlResult<Arc<Variable>> {
                let input_id = input.node_id();
                let output_tensor = <Self as Function>::forward(self, &[input.tensor()])?.remove(0);
                let output = match self.inputs.contains(&input_id) {
                    true => Arc::new(Variable::new(output_tensor.with_id(*self.outputs.get(&input_id).unwrap())?)),
                    false => {
                        let temp = Arc::new(Variable::new(output_tensor.to_id()?));
                        self.inputs.insert(input_id);
                        self.outputs.insert(input_id, temp.node_id());
                        temp
                    }
                };
                output.clone().with_grad_fn(<Self as Function>::type_name(self), &[&input]);
                return Ok(output)
            }
            fn inputs_cache(&self) -> &std::collections::HashSet<NodeId> { &self.inputs }
            fn outputs_cache(&self) -> &std::collections::HashMap<NodeId, NodeId> { &self.outputs }
            fn inputs_cache_mut(&mut self) -> &mut std::collections::HashSet<NodeId> { &mut self.inputs }
            fn outputs_cache_mut(&mut self) -> &mut std::collections::HashMap<NodeId, NodeId> { &mut self.outputs }
            fn params(&self) -> Vec<&dyn Parameter> { vec![] }
            fn type_name(&self) -> &str { stringify!($struct_name) }
            fn label(&self) -> &str { &self.label }
        }
    };
}

pub trait Activation: Function + Layer {}

impl<T: Function + Layer> Activation for T {}

#[derive(Debug, Clone)]
pub struct Sigmoid { 
    backend: Arc<dyn Backend>, node_id: NodeId, inputs: HashSet<NodeId>, outputs: HashMap<NodeId, NodeId>, label: String
}

#[derive(Debug, Clone)]
pub struct Tanh    { 
    backend: Arc<dyn Backend>, node_id: NodeId, inputs: HashSet<NodeId>, outputs: HashMap<NodeId, NodeId>, label: String
}

#[derive(Debug, Clone)]
pub struct ReLU { 
    backend: Arc<dyn Backend>, node_id: NodeId, inputs: HashSet<NodeId>, outputs: HashMap<NodeId, NodeId>, label: String
}

#[derive(Debug, Clone)]
pub struct Softmax { 
    backend: Arc<dyn Backend>, node_id: NodeId, inputs: HashSet<NodeId>, outputs: HashMap<NodeId, NodeId>, label: String
}