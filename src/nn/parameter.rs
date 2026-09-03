use super::*;
use crate::tensor::GlobalTensor;

// Variable in-place operator overloading (no graph registration, for parameter updates)
impl std::ops::SubAssign<GlobalTensor<f32>> for Variable {
    fn sub_assign(&mut self, rhs: GlobalTensor<f32>) {
        Sub::new().unwrap().assign_forward(&[self.tensor(), &rhs], self.node_id()).unwrap();
    }
}

impl std::ops::AddAssign<GlobalTensor<f32>> for Variable {
    fn add_assign(&mut self, rhs: GlobalTensor<f32>) {
        Add::new().unwrap().assign_forward(&[self.tensor(), &rhs], self.node_id()).unwrap();
    }
}

impl Parameter for Variable {
    fn new(tensor: Tensor) -> Self {
        Variable {
            #[cfg(feature = "enableVisualization")]
            label: None,
            #[cfg(feature = "enableVisualization")]
            node_type: None,
            tensor,
            requires_grad: false.into(),
            //  zeros_like() → new_empty()
            //   zeros_like(): shape 크기만큼 Vec<f32> 힙 할당 + TENSOR_STORAGE insert
            //   new_empty() : data=[], shape=[] — capacity=0, 힙 할당 없음
            grad: Tensor::new_empty(),
        }
    }
    
    fn node_id(&self) -> NodeId {
        self.tensor.id()
    }

    fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    fn is_retain_grad(&self) -> bool {
        self.requires_grad.get()
    }

    fn retain_grad(&self) {
        // retain_grad는 학습 전 명시적으로 grad 버퍼를 미리 할당할 때 사용.
        // zeros로 초기화해 버퍼를 확보하되, dirty는 건드리지 않음.
        //   - dirty=false 상태에서 버퍼가 비어있지 않으면 accumulate_grad가
        //     in-place 덧셈 경로를 타므로 동작상 문제 없다.
        //   - clear_grad는 dirty=true일 때만 제로화하므로,
        //     retain_grad만 호출하고 backward를 돌리지 않으면 clear_grad는 no-op.
        self.requires_grad.replace(true);
        self.grad.replace(GlobalTensor::zeros(self.tensor.shape()));
    }

    fn grad(&self) -> &Tensor {
        &self.grad
    }

    #[cfg(feature = "enableBackward")]
    fn set_grad(&self, grad: GlobalTensor<f32>) {
        // 출력 노드의 grad를 1.0으로 주입할 때 사용.
        // replace()는 TENSOR_STORAGE의 기존 항목을 덮어쓰므로 힙 할당 1회.
        // (backward 시작마다 출력 노드 1개에만 호출됨)
        // dirty 플래그도 GlobalTensor에 포함되어 있으므로 replace 시 함께 전달.
        let mut grad = grad;
        grad.dirty = true;
        self.grad.replace(grad);
    }

    #[cfg(feature = "enableVisualization")]
    fn set_label(&mut self, new_label: &str) {
        self.label = Some(Arc::new(new_label.to_string()));
    }

    /// 현재 라벨 반환
    #[cfg(feature = "enableVisualization")]
    fn label(&self) -> &str {
        self.label.as_deref().map(String::as_str).unwrap_or("unlabeled")
    }

    #[cfg(feature = "enableVisualization")]
    fn node_type(&self) -> &crate::visualization::NodeRole {
        static VARIABLE: crate::visualization::NodeRole = crate::visualization::NodeRole::Variable;
        self.node_type.as_ref().unwrap_or(&VARIABLE)
    }

    ///
    #[cfg(feature = "enableBackward")]
    fn clear_grad(&self) {
        // dirty 플래그로 불필요한 작업을 완전히 건너뜀
        //
        // dirty=false인 두 가지 경우:
        //   (a) new_empty() 직후 — 버퍼 자체가 없음, 제로화 불필요
        //   (b) 이전 clear_grad로 이미 제로화됨 — 다시 제로화 불필요
        //
        // dirty=true인 경우: accumulate_grad 또는 set_grad가 실제 값을 기록함
        //   → 버퍼를 in-place 제로화하고 재사용 준비
        //
        // dirty 플래그는 GlobalTensor 안에 저장되므로 모든 Variable 클론이
        // 동일한 상태를 공유한다.
        crate::tensor::TENSOR_STORAGE.with_borrow_mut(|storage| {
            if let Some(gt) = storage.get_mut(&self.grad.id()) {
                if !gt.dirty {
                    return;
                }
                gt.data.iter_mut().for_each(|x| *x = 0.0);
                gt.dirty = false;
            }
        });
    }

    #[cfg(feature = "enableBackward")]
    fn is_grad_dirty(&self) -> bool {
        crate::tensor::TENSOR_STORAGE.with(|storage| {
            storage.borrow()
                .get(&self.grad.id())
                .map(|gt| gt.dirty)
                .unwrap_or(false)
        })
    }
    
    #[cfg(feature = "enableBackward")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()> {
        // 버퍼 존재 여부에 따라 두 경로로 분기
        //
        // data().is_empty() == true  → 전체 학습 통틀어 첫 번째 grad 기록
        //                              버퍼를 새로 할당하고 값 복사 (힙 할당 1회)
        //
        // data().is_empty() == false → 버퍼가 이미 존재 (에폭 2 이후, 또는
        //                              retain_grad()로 미리 할당된 경우)
        //                              in-place 덧셈만 수행 (힙 할당 0회)
        //
        // [주의] 첫 번째 경로에서만 shape 검사를 생략.
        //   new_empty() 상태의 grad는 shape=[]이므로 new_grad와 항상 불일치.
        //   버퍼가 존재하는 경우에만 shape 검사가 의미 있다.
        if self.grad.data().is_empty() {
            // ── 첫 번째 기록: 버퍼 할당 ─────────────────────────────────────
            // 이후 에폭에서는 이 경로를 타지 않는다.
            // (clear_grad는 버퍼를 해제하지 않고 0으로만 채우기 때문)
            // dirty=true를 GlobalTensor에 포함하여 replace.
            let mut buf = GlobalTensor::from_vec(
                new_grad.data().to_vec(),
                new_grad.shape(),
            )?;
            buf.dirty = true;
            self.grad.replace(buf);
        } else {
            // ── 이후 기록: in-place 덧셈 ────────────────────────────────────
            if self.grad.shape() != new_grad.shape() {
                return Err(TensorError::InvalidShape {
                    expected: self.grad.shape().to_vec(),
                    got: new_grad.shape().to_vec(),
                }.into());
            }
            // new_grad.data()는 raw pointer를 통해 TENSOR_STORAGE에 접근한다.
            // .with()의 불변 대여는 as_ptr() 반환 시 즉시 해제되므로
            // 이후 with_borrow_mut()과 충돌하지 않는다.
            // grad.id() != new_grad.id() 조건은 backward 구조상 항상 성립한다.
            let new_data: &[f32] = new_grad.data();
            crate::tensor::TENSOR_STORAGE.with_borrow_mut(|storage| {
                if let Some(gt) = storage.get_mut(&self.grad.id()) {
                    gt.data.iter_mut()
                        .zip(new_data.iter())
                        .for_each(|(d, &v)| *d += v);
                    gt.dirty = true;
                }
            });
        }
        Ok(())
    }
}

impl Variable {
    #[cfg(feature = "enableVisualization")]
    pub(crate) fn visualization_metadata(
        &self,
    ) -> (Option<&str>, Option<&crate::visualization::NodeRole>) {
        (self.label.as_deref().map(String::as_str), self.node_type.as_ref())
    }

    /// 사용자 정의 라벨로 변수 생성
    pub fn with_label(tensor: Tensor, label_hint: &str) -> Self {
        Self::with_persistent_label(tensor, label_hint)
    }

    #[cfg(feature = "enableVisualization")]
    fn with_transient_label(tensor: Tensor, label_hint: &str) -> Self {
        if crate::visualization::recording::is_active() {
            Self::with_persistent_label(tensor, label_hint)
        } else {
            Variable {
                label: None,
                node_type: None,
                grad: tensor.zeros_like(),
                tensor,
                requires_grad: false.into(),
            }
        }
    }

    fn with_persistent_label(tensor: Tensor, label_hint: &str) -> Self {
        let mut tensor = tensor;

        #[cfg(feature = "enableVisualization")]
        {
            use crate::visualization::NodeRole;
            let label = Arc::new(label_hint.to_string());

            // 라벨을 Tensor 핸들에 반영
            tensor.set_label(&label);

            let node_type = if label.contains("input") {
                NodeRole::Input
            } else if label.contains("weight") {
                NodeRole::Weight
            } else if label.contains("bias") {
                NodeRole::Bias
            } else if label.contains("output") {
                NodeRole::Output
            } else if label.contains("act") {
                NodeRole::Activation
            } else if label.contains("loss") {
                NodeRole::Loss
            } else {
                NodeRole::Variable
            };

            return Variable {
                #[cfg(feature = "enableVisualization")]
                label: Some(label),
                #[cfg(feature = "enableVisualization")]
                node_type: Some(node_type),
                grad: tensor.zeros_like(),
                tensor,
                requires_grad: false.into(),
            }
        }

        Variable {
            #[cfg(feature = "enableVisualization")]
            label: None,
            #[cfg(feature = "enableVisualization")]
            node_type: None,
            grad: tensor.zeros_like(),
            tensor,
            requires_grad: false.into(),
        }
    }

    /// 특정 용도에 맞는 변수 생성자들
    #[cfg(feature = "enableVisualization")]
    pub fn new_input(tensor: Tensor) -> Self {
        Self::with_transient_label(tensor, "input")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_weight(tensor: Tensor) -> Self {
        Self::with_persistent_label(tensor, "weight")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_bias(tensor: Tensor) -> Self {
        Self::with_persistent_label(tensor, "bias")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_output(tensor: Tensor) -> Self {
        Self::with_transient_label(tensor, "output")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_hidden(tensor: Tensor) -> Self {
        Self::with_transient_label(tensor, "hidden")
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_conv_weight(tensor: Tensor, layer_idx: usize) -> Self {
        Self::with_persistent_label(tensor, &format!("conv{}_weight", layer_idx))
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_linear_weight(tensor: Tensor, layer_idx: usize) -> Self {
        Self::with_persistent_label(tensor, &format!("fc{}_weight", layer_idx))
    }

    #[cfg(feature = "enableVisualization")]
    pub fn new_activation(tensor: Tensor, activation_type: &str) -> Self {
        Self::with_transient_label(tensor, &format!("{}_act", activation_type))
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

#[cfg(feature = "enableBackward")]
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
