pub mod sigmoid;
pub mod tanh;
pub mod relu;
pub mod softmax;
pub mod identity;
mod silu;

use super::*;

pub trait Activation: Layer {}

impl<T: Layer> Activation for T {}

/// 활성화 함수의 연산자 구조체를 생성합니다.
macro_rules! define_activation_op {
    ($name:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            backend: Arc<dyn Backend>,
            node_id: NodeId,
        }
    };
}

/// 활성화 레이어의 struct 정의 + 생성자 + Layer impl 전체를 자동 생성합니다.
/// 각 활성화 `.rs` 파일에는 `impl Function for $Op`(forward/backward)만 남습니다.
///
/// # 사용법
/// ```ignore
/// activation_layer!(SigmoidLayer, Sigmoid);
/// ```
macro_rules! activation_layer {
    ($layer:ident, $op:ident) => {
        #[derive(Debug, Clone)]
        pub struct $layer {
            label: String,
            operator: GlobalFunction,
        }

        impl $layer {
            pub fn new(label: &str) -> MlResult<Self> {
                Ok(Self { label: label.to_string(), operator: $op::new()? })
            }
        }

        impl Layer for $layer {
            #[cfg(all(feature = "enableBackward"))]
            fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
                self.operator.apply(&[input])
            }

            fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
                Ok(self.operator.forward(&[input])?.remove(0))
            }

            fn params(&self) -> Vec<&dyn Parameter> { vec![] }

            fn label(&self) -> &str { &self.label }
        }
    };
}

define_activation_op!(Sigmoid);
define_activation_op!(Tanh);
define_activation_op!(ReLU);
define_activation_op!(Softmax);
define_activation_op!(Identity);
define_activation_op!(SiLU);

activation_layer!(SigmoidLayer, Sigmoid);
activation_layer!(TanhLayer, Tanh);
activation_layer!(ReLULayer, ReLU);
activation_layer!(SoftmaxLayer, Softmax);
activation_layer!(IdentityLayer, Identity);
activation_layer!(SiLULayer, SiLU);
