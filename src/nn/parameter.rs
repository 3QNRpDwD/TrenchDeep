use super::*;

impl Parameter for Variable {
    fn new(tensor: Tensor) -> Self {
        Variable {
            #[cfg(feature = "enableVisualization")]
            label: crate::tensor::creation::LabelGenerator::generate_label(&tensor, None),
            #[cfg(feature = "enableVisualization")]
            node_type: crate::tensor::NodeType::Variable,
            grad: tensor.zeros_like(),
            tensor,
            requires_grad: RefCell::new(false),
            is_persistent: RefCell::new(false),
        }
    }

    fn node_id(&self) -> HandleId {
        self.tensor.id()
    }

    fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    fn is_retain_grad(&self) -> bool {
        *self.requires_grad.borrow()
    }

    fn retain_grad(&self) {
        self.requires_grad.replace(true);
    }

    fn grad(&self) -> &Tensor {
        &self.grad
    }

    fn mut_grad(&mut self) -> &mut Tensor {
        &mut self.grad
    }

    #[cfg(feature = "enableBackpropagation")]
    fn set_grad(&self, grad_data: GlobalTensor<f32>) {
        self.grad.replace(grad_data);
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
        if !self.is_retain_grad() {
            TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
                if let Some(grad_tensor) = allocator.get_tensor_mut(&self.grad.id()) {
                    grad_tensor.data.fill(0.0);
                }
            });
        }
    }

    #[cfg(feature = "enableBackpropagation")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()> {
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
        let mut var = Variable::new(tensor);
        var.is_persistent = RefCell:: new(false);

        #[cfg(feature = "enableVisualization")]
        {
            use crate::tensor::NodeType;
            let label = crate::tensor::creation::LabelGenerator::generate_label(&var.tensor, Some(label_hint));
            var.label = label;
            var.node_type = if var.label.contains("input") {
                NodeType::Input
            } else if var.label.contains("weight") {
                NodeType::Weight
            } else if var.label.contains("bias") {
                NodeType::Bias
            } else if var.label.contains("output") {
                NodeType::Output
            } else if var.label.contains("act") {
                NodeType::Activation
            } else if var.label.contains("loss") {
                NodeType::Loss
            } else {
                NodeType::Variable
            };
        }
        var
    }

    pub fn new_persistent(tensor: Tensor, label_hint: &str) -> Self {
        let mut var = Self::with_label(tensor, label_hint);
        var.is_persistent = RefCell::new(true);
        var
    }

    /// 특정 용도에 맞는 변수 생성자들
    #[cfg(feature = "enableVisualization")]
    pub fn new_input(tensor: Tensor) -> Self { Self::new_persistent(tensor, "input") }

    #[cfg(feature = "enableVisualization")]
    pub fn new_weight(tensor: Tensor) -> Self { Self::new_persistent(tensor, "weight") }

    #[cfg(feature = "enableVisualization")]
    pub fn new_bias(tensor: Tensor) -> Self {
        Self::new_persistent(tensor, "bias")
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
    pub fn new_loss(tensor: Tensor) -> Self { Self::with_label(tensor, "loss") }
    
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
                crate::nn::Variable::new_persistent($tensor, "input")
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
                crate::nn::Variable::new_persistent($tensor, "weight")
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
                crate::nn::Variable::new_persistent($tensor, "bias")
            }
        }
    };
}

#[macro_export]
macro_rules! var_loss {
    ($tensor:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::new_loss($tensor)
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