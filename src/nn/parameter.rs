use super::*;

impl Parameter for Variable {
    fn new(tensor: Tensor) -> Self {
        #[cfg(feature = "enableVisualization")]
        let label = crate::tensor::creation::LabelGenerator::generate_label(&tensor, None);

        Variable {
            #[cfg(feature = "enableVisualization")]
            label,
            #[cfg(feature = "enableVisualization")]
            node_type: crate::tensor::NodeType::Variable,
            grad: tensor.zeros_like(),
            tensor,
            requires_grad: cfg!(feature = "requiresGrad").into()
        }
    }

    fn node_id(&self) -> NodeId {
        self.tensor.id()
    }

    fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    fn is_retain_grad(&self) -> bool {
        *self.requires_grad.borrow().deref()
    }

    fn retain_grad(&self) {
        self.requires_grad.replace(true);
        self.grad.replace(GlobalTensor::zeros(self.tensor.shape()))
    }

    fn grad(&self) -> &Tensor {
        &self.grad
    }

    #[cfg(feature = "enableBackpropagation")]
    fn set_grad(&self, grad: GlobalTensor<f32>) {
        self.grad.replace(grad);
    }

    #[cfg(feature = "enableVisualization")]
    fn set_label(&mut self, new_label: &str) {
        self.label = crate::tensor::creation::LabelGenerator::get_unique_label(new_label);
    }

    /// 현재 라벨 반환
    #[cfg(feature = "enableVisualization")]
    fn label(&self) -> &str {
        &self.label
    }

    #[cfg(feature = "enableVisualization")]
    fn node_type(&self) -> &crate::tensor::NodeType {
        &self.node_type
    }

    ///
    #[cfg(feature = "enableBackpropagation")]
    fn clear_grad(&self) {
            // TENSOR_STORAGE.with_borrow_mut(|storage| {
            //     storage.remove(&self.grad().id()) // 만약 스토리지가 분리되면 그냥 그래프를 초기화하면 되기 때문에 성능이 더욱 향상될듯함
            // });
            // 기존에 Variable 이 텐서를 소유하던 구조에서 기울기를 지우던 로직을 그대로 사용해서
            // 텐서 스토리지에 있던 기울기가 사라지지 않고 그대로 남아있던 문제가 있었음.
            // 따라서 해당 부분을 지우는 로직을 추가함.
            // 하지만 현재는 텐서 스토리지와 분리되어있지 않아, 게산그래프에서 추가되는 모든 텐서가 텐서 스토리지에 등록되어,
            // 성능이 저하되는 문제가 있음. 따라서 텐서 스토리지와 계산그래프 전용 텐서 스토리지를 만들어서 완전히 분리하던가,
            // 배치별로 다른 스토리지를 만들어서 관리하도록 하던가하는 방법으로 최적화 해야할듯함.
            // 최종적으로 정적 계산그래프로 전환한다면 더욱 성능향상이 기대됨.
        if !self.grad().is_empty() || !self.is_retain_grad() {
            self.grad.replace(GlobalTensor::zeros(self.tensor.shape()));
        }
    }

    #[cfg(feature = "enableBackpropagation")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()> {
        // 차원 검증 추가
        if self.grad.shape() != new_grad.shape() {
            return Err(TensorError::InvalidShape {
                expected: self.grad.shape().to_vec(),
                got: new_grad.shape().to_vec(),
            }.into());
        }

        // 가능하다면 in-place 연산을 사용하여 효율성 개선
        let mut accumulated_data = self.grad.data().to_vec();
        for (i, &val) in new_grad.data().iter().enumerate() {
            accumulated_data[i] += val;
        }

        Tensor::with_id(accumulated_data, self.grad.shape(), self.grad().id())
            .map_err(|e| format!("Failed gradient accumulation: {:?}", e))?;
        Ok(())
    }
}

impl Variable {
    /// 사용자 정의 라벨로 변수 생성
    pub fn with_label(tensor: Tensor, label_hint: &str) -> Self {
        let label = "unlabeled".to_string();
        let retains_grad = cfg!(feature = "requiresGrad");

        #[cfg(feature = "enableVisualization")]
        {
            use crate::tensor::NodeType;
            let label = crate::tensor::creation::LabelGenerator::generate_label(&tensor, Some(label_hint));
            let node_type = if label.contains("input") {
                NodeType::Input
            } else if label.contains("weight") {
                NodeType::Weight
            } else if label.contains("bias") {
                NodeType::Bias
            } else if label.contains("output") {
                NodeType::Output
            } else if label.contains("act") {
                NodeType::Activation
            } else if label.contains("loss") {
                NodeType::Loss
            } else {
                NodeType::Variable
            };

            return Variable {
                #[cfg(feature = "enableVisualization")]
                label,
                #[cfg(feature = "enableVisualization")]
                node_type: crate::tensor::NodeType::Variable,
                grad: tensor.zeros_like(),
                tensor,
                requires_grad: cfg!(feature = "requiresGrad").into()
            }
        }

        Variable {
            #[cfg(feature = "enableVisualization")]
            label,
            #[cfg(feature = "enableVisualization")]
            node_type: crate::tensor::NodeType::Variable,
            grad: tensor.zeros_like(),
            tensor,
            requires_grad: cfg!(feature = "requiresGrad").into()
        }
    }

    /// 특정 용도에 맞는 변수 생성자들
    #[cfg(feature = "enableVisualization")]
    pub fn new_input(tensor: Tensor) -> Self {
        Self::with_label(tensor, "input")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_weight(tensor: Tensor) -> Self {
        Self::with_label(tensor, "weight")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_bias(tensor: Tensor) -> Self {
        Self::with_label(tensor, "bias")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_output(tensor: Tensor) -> Self {
        Self::with_label(tensor, "output")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_hidden(tensor: Tensor) -> Self {
        Self::with_label(tensor, "hidden")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_conv_weight(tensor: Tensor, layer_idx: usize) -> Self {
        Self::with_label(tensor, &format!("conv{}_weight", layer_idx))
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_linear_weight(tensor: Tensor, layer_idx: usize) -> Self {
        Self::with_label(tensor, &format!("fc{}_weight", layer_idx))
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_activation(tensor: Tensor, activation_type: &str) -> Self {
        Self::with_label(tensor, &format!("{}_act", activation_type))
    }

    /// 라벨 변경
    #[cfg(not(feature = "enableVisualization"))]
    pub fn label(&self) -> &str {
        "unlabeled"
    }

    /// 텐서 정보와 함께 디버그 정보 출력
    pub fn debug_info(&self) -> String {
        format!(
            "Variable '{}': tensor={:?}, requires_grad={:?}, grad={:?}",
            self.label(),
            self.tensor(),
            self.is_retain_grad(),
            self.grad(),
        )
    }
}

#[cfg(feature = "enableBackpropagation")]
impl PartialEq for &Variable {
    fn eq(&self, other: &&Variable) -> bool {
        self.tensor == other.tensor &&
            self.requires_grad == other.requires_grad &&
            self.grad == other.grad
    }
}


#[macro_export]
macro_rules! var_input {
    ($tensor:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::new_input($tensor)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new($tensor)
            }
        }
    };
}

#[macro_export]
macro_rules! var_output {
    ($tensor:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::new_output($tensor)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new($tensor)
            }
        }
    };
}

#[macro_export]
macro_rules! var_act {
    ($tensor:expr, $type_name:expr) => {
        {
            use std::sync::Arc;
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::new_activation($tensor, $type_name)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new($tensor)
            }
        }
    };
}

#[macro_export]
macro_rules! var_weight {
    ($tensor:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::new_weight($tensor)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new($tensor)
            }
        }
    };
}

#[macro_export]
macro_rules! var_bias {
    ($tensor:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::new_bias($tensor)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new($tensor)
            }
        }
    };
}

#[macro_export]
macro_rules! var_with_label {
    ($tensor:expr, $label:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::with_label($tensor, $label)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new($tensor)
            }
        }
    };
}