use super::*;
use std::collections::HashMap;
use std::ops::{Deref, DivAssign, MulAssign, SubAssign};
use crate::tensor::operators::{Add, Div, Mul, Sub};

impl TensorBase for GlobalTensor<f32> {
    fn new(data: Vec<Vec<f32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();
        GlobalTensor { data, shape }
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(GlobalTensor { data, shape: shape.to_vec() })
    }

    fn as_ptr(&self) -> *const GlobalTensor<f32> {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        &self.data
    }

    fn get(&self, indices: &[usize]) -> Option<&f32> {
        self.data().get(self.index(indices)?)
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape().len() {
            return None;
        }
        let mut idx = 0;
        let shape = self.shape();
        for (i, &ind) in indices.iter().enumerate() {
            if ind >= shape[i] {
                return None;
            }
            idx = idx * shape[i] + ind;
        }
        Some(idx)
    }

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn chk_shape(&self, other: &dyn TensorBase) -> MlResult<()> {
        if self.shape() != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }
}

// 매번 생성된 텐서를 전역 그래프에 저장하느라 심각한 성능저하와 메모리 낭비 발생. 아마도 연산자등에서 생성하는 텐서는 저장되지 않도록 조치를 취해야 할듯.
impl TensorBase for Tensor {
    fn new(data: Vec<Vec<f32>>) -> Tensor {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();

        let node_id = NODE_ID_GEN.next();
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(node_id, GlobalTensor { data, shape })
        });
        Tensor(node_id)
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Tensor> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        let node_id = NODE_ID_GEN.next();
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(node_id, GlobalTensor { data, shape: shape.to_vec() })
        });

        Ok(Tensor(node_id))
    }

    fn as_ptr(&self) -> *const GlobalTensor<f32> {
        TENSOR_STORAGE.with(|storage| {
            storage.borrow().get(&self.0).map(|gt| gt as *const GlobalTensor<f32>).unwrap()
        })
    }
    
    fn shape(&self) -> &[usize] {
        unsafe { &self.as_ptr().as_ref().unwrap().shape }
    }

    fn data(&self) -> &[f32] {
        unsafe { &self.as_ptr().as_ref().unwrap().data }
    }

    fn get(&self, indices: &[usize]) -> Option<&f32> {
        self.data().get(self.index(indices)?)
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape().len() {
            return None;
        }
        let mut idx = 0;
        let shape = self.shape();
        for (i, &ind) in indices.iter().enumerate() {
            if ind >= shape[i] {
                return None;
            }
            idx = idx * shape[i] + ind;
        }
        Some(idx)
    }

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn chk_shape(&self, other: &dyn TensorBase) -> MlResult<()> {
        if self.shape() != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }
}

impl GlobalTensor<f32> {

    pub fn with_id(self) -> MlResult<Tensor> {
        let expected_len: usize = self.shape.iter().product();
        if self.data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: self.data.len(),
            }));
        }

        let node_id = NODE_ID_GEN.next();
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(node_id, self)
        });

        Ok(Tensor(node_id))
    }
}

impl Tensor {
    pub fn with_id(data: Vec<f32>, shape: &[usize], node_id: NodeId) -> MlResult<Tensor> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(node_id, GlobalTensor { data, shape: shape.to_vec() })
        });

        Ok(Tensor(node_id))
    }
}

#[cfg(feature = "enableVisualization")]
pub struct LabelGenerator;

#[cfg(feature = "enableVisualization")]
impl LabelGenerator {
    /// 텐서의 특성을 기반으로 직관적인 라벨 생성
    pub fn generate_label(tensor: &Tensor, hint: Option<&str>) -> String {
        // 힌트가 제공된 경우 우선 사용
        if let Some(hint) = hint {
            return Self::get_unique_label(hint);
        }

        // 텐서 모양 기반 라벨 생성
        let shape_label = Self::shape_to_label(tensor.shape());
        let context_label = Self::infer_context_from_shape(tensor.shape());

        // 컨텍스트가 있으면 컨텍스트 우선, 없으면 모양 기반
        if !context_label.is_empty() {
            Self::get_unique_label(&context_label)
        } else {
            Self::get_unique_label(&shape_label)
        }
    }

    /// 텐서 모양을 기반으로 컨텍스트 추론
    fn infer_context_from_shape(shape: &[usize]) -> String {
        match shape.len() {
            0 => "scalar".to_string(),
            1 => {
                match shape[0] {
                    1 => "bias".to_string(),
                    2..=10 => "small_vec".to_string(),
                    11..=100 => "vector".to_string(),
                    101..=1000 => "embedding".to_string(),
                    _ => "large_vec".to_string(),
                }
            },
            2 => {
                let (rows, cols) = (shape[0], shape[1]);
                match (rows, cols) {
                    (1, 1) => "scalar".to_string(),
                    (1, _) => "row_vec".to_string(),
                    (_, 1) => "col_vec".to_string(),
                    (r, c) if r == c && r <= 10 => "small_matrix".to_string(),
                    (r, c) if r == c => "square_matrix".to_string(),
                    (r, c) if r > c * 2 => "tall_matrix".to_string(),
                    (r, c) if c > r * 2 => "wide_matrix".to_string(),
                    _ => "matrix".to_string(),
                }
            },
            3 => {
                let (d1, d2, d3) = (shape[0], shape[1], shape[2]);
                match (d1, d2, d3) {
                    (_, _, 1) => "feature_map".to_string(),
                    (_, _, 3) => "rgb_image".to_string(),
                    (_, _, 4) => "rgba_image".to_string(),
                    (1, _, _) => "batch_1".to_string(),
                    (_, h, w) if h == w => "square_tensor".to_string(),
                    _ => "tensor_3d".to_string(),
                }
            },
            4 => {
                let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
                match (batch, channels, height, width) {
                    (1, 1, _, _) => "single_channel".to_string(),
                    (1, 3, _, _) => "rgb_batch_1".to_string(),
                    (_, 1, _, _) => "grayscale_batch".to_string(),
                    (_, 3, _, _) => "rgb_batch".to_string(),
                    (_, c, _, _) if c > 64 => "deep_feature".to_string(),
                    _ => "conv_tensor".to_string(),
                }
            },
            _ => format!("tensor_{}d", shape.len()),
        }
    }

    /// 텐서 모양을 문자열로 변환
    fn shape_to_label(shape: &[usize]) -> String {
        match shape.len() {
            0 => "scalar".to_string(),
            1 => format!("vec_{}", shape[0]),
            2 => format!("mat_{}x{}", shape[0], shape[1]),
            3 => format!("t3d_{}x{}x{}", shape[0], shape[1], shape[2]),
            4 => format!("t4d_{}x{}x{}x{}", shape[0], shape[1], shape[2], shape[3]),
            _ => format!("t{}d_{}", shape.len(), shape.iter().map(|&s| s.to_string()).collect::<Vec<_>>().join("x")),
        }
    }

    /// 고유한 라벨 생성 (중복 방지)
    fn get_unique_label(base_label: &str) -> String {
        LABEL_COUNTERS.with(|counters| {
            let mut counters = counters.borrow_mut();
            let count = counters.entry(base_label.to_string()).or_insert(0);
            *count += 1;

            if *count == 1 {
                base_label.to_string()
            } else {
                format!("{}_{}", base_label, count)
            }
        })
    }

    /// 라벨 카운터 초기화
    pub fn reset_counters() {
        LABEL_COUNTERS.with(|counters| {
            counters.borrow_mut().clear();
        });
        SHAPE_REGISTRY.with(|registry| {
            registry.borrow_mut().clear();
        });
    }

    /// 현재 등록된 라벨들의 통계 정보
    pub fn get_label_stats() -> HashMap<String, usize> {
        LABEL_COUNTERS.with(|counters| {
            counters.borrow().clone()
        })
    }
}

impl Variable {
    /// 기본 생성자 - 텐서 모양 기반 자동 라벨링
    pub fn new(tensor: Tensor) -> Self {
        #[cfg(feature = "enableVisualization")]
        let label = LabelGenerator::generate_label(&tensor, None);

        Variable {
            #[cfg(feature = "enableVisualization")]
            label,
            #[cfg(feature = "enableVisualization")]
            node_type: NodeType::Variable,
            tensor,
            requires_grad: cfg!(feature = "requiresGrad").into(),
            grad: None.into(),
        }
    }

    #[cfg(feature = "enableBackpropagation")]
    pub fn node_id(&self) -> NodeId {
        self.tensor.0
    }

    /// 사용자 정의 라벨로 변수 생성
    pub fn with_label(tensor: Tensor, label_hint: &str) -> Self {
        #[cfg(feature = "enableVisualization")]
        let label = LabelGenerator::generate_label(&tensor, Some(label_hint));
        #[cfg(feature = "enableVisualization")]
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

        Variable {
            #[cfg(feature = "enableVisualization")]
            label,
            #[cfg(feature = "enableVisualization")]
            node_type,
            tensor,
            requires_grad: cfg!(feature = "requiresGrad").into(),
            grad: None.into(),
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
    #[cfg(feature = "enableVisualization")]
    pub fn set_label(&mut self, new_label: &str) {
        self.label = LabelGenerator::get_unique_label(new_label);
    }

    /// 현재 라벨 반환
    #[cfg(feature = "enableVisualization")]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[cfg(feature = "enableVisualization")]
    pub fn node_type(&self) -> &NodeType {
        &self.node_type
    }
    
    #[cfg(not(feature = "enableVisualization"))]
    pub fn label(&self) -> &str {
        "unlabeled"
    }

    /// 텐서 정보와 함께 디버그 정보 출력
    pub fn debug_info(&self) -> String {
        format!(
            "Variable '{}': tensor={:?}, requires_grad={:?}, has_grad={}",
            self.label(),
            self.tensor(),
            self.is_retain_grad(),
            self.grad().is_some(),
        )
    }

    pub fn add_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Add::new()?.assign_forward(&[&self.tensor, &other_tensor], self.node_id())?;
        Ok(())
    }

    pub fn sub_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Sub::new()?.assign_forward(&[&self.tensor, &other_tensor], self.node_id())?;
        Ok(())
    }

    pub fn mul_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Mul::new()?.assign_forward(&[&self.tensor, &other_tensor], self.node_id())?;
        Ok(())
    }

    pub fn div_tensor(&self, other_tensor: GlobalTensor<f32>) -> MlResult<()> {
        Div::new()?.assign_forward(&[&self.tensor, &other_tensor], self.node_id())?;
        Ok(())
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn is_retain_grad(&self) -> bool {
        *self.requires_grad.borrow().deref()
    }

    pub fn retain_grad(&self) {
        self.requires_grad.replace(true);
    }

    pub fn grad(&self) -> Option<&Tensor> {
        let ptr: *const Option<Tensor> = self.grad.as_ptr();
        unsafe { ptr.as_ref().unwrap().as_ref() }
    }

    #[cfg(feature = "enableBackpropagation")]
    pub fn set_grad(&self, grad: Tensor) {
        self.grad.replace(Some(grad));
    }

    ///
    #[cfg(feature = "enableBackpropagation")]
    pub fn clear_grad(&self) {
        if !self.grad().is_none() && !self.is_retain_grad() {
            TENSOR_STORAGE.with_borrow_mut(|storage| {
                storage.remove(&self.grad().unwrap().0) // 만약 스토리지가 분리되면 그냥 그래프를 초기화하면 되기 때문에 성능이 더욱 향상될듯함
            }); 
            // 기존에 Variable 이 텐서를 소유하던 구조에서 기울기를 지우던 로직을 그대로 사용해서
            // 텐서 스토리지에 있던 기울기가 사라지지 않고 그대로 남아있던 문제가 있었음.
            // 따라서 해당 부분을 지우는 로직을 추가함.
            // 하지만 현재는 텐서 스토리지와 분리되어있지 않아, 게산그래프에서 추가되는 모든 텐서가 텐서 스토리지에 등록되어,
            // 성능이 저하되는 문제가 있음. 따라서 텐서 스토리지와 계산그래프 전용 텐서 스토리지를 만들어서 완전히 분리하던가,
            // 배치별로 다른 스토리지를 만들어서 관리하도록 하던가하는 방법으로 최적화 해야할듯함.
            // 최종적으로 정적 계산그래프로 전환한다면 더욱 성능향상이 기대됨.
            self.grad.replace(None);
        }
    }

    /// 그래디언트 값 누적 추가
    ///
    /// # 특징 동작
    /// - `enableBackpropagation` 기능 전용 메소드
    /// - 기존 그래디언트와 새로운 그래디언트를 요소별 합산
    ///
    /// # 오류 사항
    /// - 텐서 모양 불일치 시 에러 반환
    ///
    /// # 파라미터
    /// - new_grad: 추가할 그래디언트 텐서
    #[cfg(feature = "enableBackpropagation")]
    pub fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()> {
        if let Some(existing_grad) = self.grad() {
            // 차원 검증 추가
            if existing_grad.shape() != new_grad.shape() {
                return Err(TensorError::InvalidShape {
                    expected: existing_grad.shape().to_vec(),
                    got: new_grad.shape().to_vec(),
                }.into());
            }

            // 가능하다면 in-place 연산을 사용하여 효율성 개선
            let mut accumulated_data = existing_grad.data().to_vec();
            for (i, &val) in new_grad.data().iter().enumerate() {
                accumulated_data[i] += val;
            }

            Tensor::with_id(accumulated_data, existing_grad.shape(), self.grad().unwrap().0)
                .map_err(|e| format!("Failed gradient accumulation: {:?}", e))?;
        } else {
            self.set_grad(new_grad);
        }


        Ok(())
    }
}

impl GlobalFunction {
    pub fn new(name: String, node_id: NodeId) -> Self {
        Self {
            name,
            func_id: node_id,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn func_id(&self) -> &NodeId {
        &self.func_id
    }
}

#[macro_export]
macro_rules! var_input {
    ($tensor:expr) => {
        {
            use std::sync::Arc;
            use crate::tensor::Variable;
            #[cfg(feature = "enableVisualization")]
            {
                Arc::new(Variable::new_input($tensor))
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                Arc::new(Variable::new($tensor))
            }
        }
    };
}

#[macro_export]
macro_rules! var_output {
    ($tensor:expr) => {
        {
            use std::sync::Arc;
            use crate::tensor::Variable;
            #[cfg(feature = "enableVisualization")]
            {
                Arc::new(Variable::new_output($tensor))
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                Arc::new(Variable::new($tensor))
            }
        }
    };
}

#[macro_export]
macro_rules! var_weight {
    ($tensor:expr) => {
        {
            use std::sync::Arc;
            use crate::tensor::Variable;
            #[cfg(feature = "enableVisualization")]
            {
                Arc::new(Variable::new_weight($tensor))
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                Arc::new(Variable::new($tensor))
            }
        }
    };
}

#[macro_export]
macro_rules! var_bias {
    ($tensor:expr) => {
        {
            use std::sync::Arc;
            use crate::tensor::Variable;
            #[cfg(feature = "enableVisualization")]
            {
                Arc::new(Variable::new_bias($tensor))
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                Arc::new(Variable::new($tensor))
            }
        }
    };
}

#[macro_export]
macro_rules! var_with_label {
    ($tensor:expr, $label:expr) => {
        {
            use std::sync::Arc;
            use crate::tensor::Variable;
            #[cfg(feature = "enableVisualization")]
            {
                Arc::new(Variable::with_label($tensor, $label))
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                Arc::new(Variable::new($tensor))
            }
        }
    };
}

// 사용 예시를 위한 테스트 함수들
// #[cfg(feature = "enableVisualization")]
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_intuitive_labeling() -> MlResult<()> {
//         let tensor = &Tensor::from_vec(vec![1.0], &[])?;
//         let scalar = Variable::new(&tensor);
//         assert_eq!(scalar.label(), "scalar");
//
//         // 벡터들
//         let small_vec = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2])?);
//         assert_eq!(small_vec.label(), "small_vec");
//
//         let bias = Variable::new(Tensor::from_vec(vec![1.0], &[1])?);
//         assert_eq!(bias.label(), "bias");
//
//         // 행렬들
//         let square = Variable::new(Tensor::from_vec(vec![1.0; 9], &[3, 3])?);
//         assert_eq!(square.label(), "small_matrix");
//
//         let wide = Variable::new(Tensor::from_vec(vec![1.0; 20], &[2, 10])?);
//         assert_eq!(wide.label(), "wide_matrix");
//
//         // RGB 이미지
//         let rgb = Variable::new(Tensor::from_vec(vec![1.0; 192], &[8, 8, 3])?);
//         assert_eq!(rgb.label(), "rgb_image");
//
//         // 배치 RGB
//         let rgb_batch = Variable::new(Tensor::from_vec(vec![1.0; 768], &[4, 3, 8, 8])?);
//         assert_eq!(rgb_batch.label(), "rgb_batch");
//
//         Ok(())
//     }
//
//     #[test]
//     fn test_custom_labels() -> MlResult<()> {
//         let input = Variable::new_input(Tensor::from_vec(vec![1.0; 10], &[10])?);
//         assert_eq!(input.label(), "input");
//
//         let weight = Variable::new_weight(Tensor::from_vec(vec![1.0; 20], &[4, 5])?);
//         assert_eq!(weight.label(), "weight");
//
//         let conv_weight = Variable::new_conv_weight(Tensor::from_vec(vec![1.0; 36], &[3, 3, 2, 2])?, 1);
//         assert_eq!(conv_weight.label(), "conv1_weight");
//
//         Ok(())
//     }
//
//     #[test]
//     fn test_unique_labeling() -> MlResult<()> {
//         let input1 = Variable::new_input(Tensor::from_vec(vec![1.0; 10], &[10])?);
//         let input2 = Variable::new_input(Tensor::from_vec(vec![2.0; 10], &[10])?);
//
//         assert_eq!(input1.label(), "input");
//         assert_eq!(input2.label(), "input_2");
//
//         Ok(())
//     }
// }