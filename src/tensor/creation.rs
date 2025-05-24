use super::*;
use std::collections::HashMap;

impl Tensor<f32> {
    pub fn zeros(shape: &[usize]) -> Tensor<f32> {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];
        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self::zeros(&self.shape)
    }

    pub fn scalar(scalar: f32) -> Tensor<f32> {
        Self {
            data: vec![scalar],
            shape: vec![],
        }
    }
}

impl TensorBase<f32> for Tensor<f32> {
    fn new(data: Vec<Vec<f32>>) -> Tensor<f32> {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();

        Self {
            data,
            shape,
        }
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Tensor<f32>> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        &self.data
    }

    fn get(&self, indices: &[usize]) -> Option<&f32> {
        self.data.get(self.index(indices)?)
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        Some(
            indices
                .iter()
                .zip(&self.shape)
                .fold(0, |acc, (&i, &dim)| acc * dim + i),
        )
    }

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn chk_shape(&self, other: &dyn TensorBase<f32>) -> MlResult<()> {
        if self.shape != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }
}

// 라벨링을 위한 전역 카운터들
#[cfg(feature = "enableVisualization")]
thread_local! {
    static LABEL_COUNTERS: std::cell::RefCell<HashMap<String, usize>> = std::cell::RefCell::new(HashMap::new());
    static SHAPE_REGISTRY: std::cell::RefCell<HashMap<String, usize>> = std::cell::RefCell::new(HashMap::new());
}

#[cfg(feature = "enableVisualization")]
pub struct LabelGenerator;

#[cfg(feature = "enableVisualization")]
impl LabelGenerator {
    /// 텐서의 특성을 기반으로 직관적인 라벨 생성
    pub fn generate_label(tensor: &Tensor<f32>, hint: Option<&str>) -> String {
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
                    (b, h, w) if h == w => "square_tensor".to_string(),
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

impl Variable<f32> {
    /// 기본 생성자 - 텐서 모양 기반 자동 라벨링
    pub fn new(tensor: Tensor<f32>) -> Self {
        #[cfg(feature = "enableVisualization")]
        let label = LabelGenerator::generate_label(&tensor, None);

        Variable {
            #[cfg(feature = "enableVisualization")]
            label,
            tensor,
            requires_grad: cfg!(feature = "requiresGrad"),

            #[cfg(feature = "enableBackpropagation")]
            grad: std::cell::RefCell::new(None),
        }
    }

    /// 사용자 정의 라벨로 변수 생성
    pub fn with_label(tensor: Tensor<f32>, label_hint: &str) -> Self {
        #[cfg(feature = "enableVisualization")]
        let label = LabelGenerator::generate_label(&tensor, Some(label_hint));

        Variable {
            #[cfg(feature = "enableVisualization")]
            label,
            tensor,
            requires_grad: cfg!(feature = "requiresGrad"),

            #[cfg(feature = "enableBackpropagation")]
            grad: std::cell::RefCell::new(None),
        }
    }

    /// 특정 용도에 맞는 변수 생성자들
    pub fn new_input(tensor: Tensor<f32>) -> Self {
        Self::with_label(tensor, "input")
    }

    pub fn new_weight(tensor: Tensor<f32>) -> Self {
        Self::with_label(tensor, "weight")
    }

    pub fn new_bias(tensor: Tensor<f32>) -> Self {
        Self::with_label(tensor, "bias")
    }

    pub fn new_output(tensor: Tensor<f32>) -> Self {
        Self::with_label(tensor, "output")
    }

    pub fn new_hidden(tensor: Tensor<f32>) -> Self {
        Self::with_label(tensor, "hidden")
    }

    /// 신경망 레이어별 변수 생성자들
    pub fn new_conv_weight(tensor: Tensor<f32>, layer_idx: usize) -> Self {
        Self::with_label(tensor, &format!("conv{}_weight", layer_idx))
    }

    pub fn new_linear_weight(tensor: Tensor<f32>, layer_idx: usize) -> Self {
        Self::with_label(tensor, &format!("fc{}_weight", layer_idx))
    }

    pub fn new_activation(tensor: Tensor<f32>, activation_type: &str) -> Self {
        Self::with_label(tensor, &format!("{}_out", activation_type))
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

    #[cfg(not(feature = "enableVisualization"))]
    pub fn label(&self) -> &str {
        "unlabeled"
    }

    /// 텐서 정보와 함께 디버그 정보 출력
    pub fn debug_info(&self) -> String {
        format!(
            "Variable '{}': shape={:?}, requires_grad={}, has_grad={}",
            self.label(),
            self.tensor.shape(),
            self.requires_grad,
            self.grad().is_some(),
        )
    }

    pub fn update_tensor(&mut self, tensor: Tensor<f32>) {
        self.tensor = tensor
    }

    /// 변수가 보유한 텐서의 참조 반환
    ///
    /// # 반환 값
    /// - 내부 텐서 데이터의 불변 참조
    pub fn tensor(&self) -> &Tensor<f32> {
        &self.tensor
    }

    /// 그래디언트 보존 여부 확인
    ///
    /// # 반환 값
    /// - 현재 변수의 requires_grad 플래그 상태
    pub fn retain_grad(&self) -> bool {
        self.requires_grad
    }

    /// 저장된 그래디언트 값 조회
    ///
    /// # 특징 동작
    /// - `enableBackpropagation` 기능 전용 메소드
    ///
    /// # 반환 값
    /// - Option<Tensor<f32>>: 현재 저장된 그래디언트 또는 None
    #[cfg(feature = "enableBackpropagation")]
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.grad.borrow().clone()
    }

    /// 그래디언트 값 직접 설정
    ///
    /// # 특징 동작
    /// - `enableBackpropagation` 기능 전용 메소드
    /// - 기존 그래디언트 값을 완전히 대체
    ///
    /// # 파라미터
    /// - grad: 설정할 새로운 그래디언트 텐서
    #[cfg(feature = "enableBackpropagation")]
    pub fn set_grad(&self, grad: Tensor<f32>) {
        *self.grad.borrow_mut() = Some(grad);
    }

    /// 그래디언트 값 초기화
    ///
    /// # 특징 동작
    /// - `enableBackpropagation` 기능 전용 메소드
    /// - 기존 그래디언트 값을 삭제
    ///
    #[cfg(feature = "enableBackpropagation")]
    pub fn clear_grad(&self) {
        *self.grad.borrow_mut() = None;
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
    pub fn accumulate_grad(&self, new_grad: Tensor<f32>) -> MlResult<()> {
        let mut grad_ref = self.grad.borrow_mut();

        if let Some(ref existing_grad) = *grad_ref {
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

            let accumulated_grad = Tensor::from_vec(accumulated_data, existing_grad.shape())
                .map_err(|e| format!("Failed gradient accumulation: {:?}", e))?;

            *grad_ref = Some(accumulated_grad);
        } else {
            *grad_ref = Some(new_grad);
        }

        Ok(())
    }
}

#[macro_export]
macro_rules! var_input {
    ($tensor:expr) => {
        Variable::new_input($tensor)
    };
}

#[macro_export]
macro_rules! var_weight {
    ($tensor:expr) => {
        Variable::new_weight($tensor)
    };
}

#[macro_export]
macro_rules! var_bias {
    ($tensor:expr) => {
        Variable::new_bias($tensor)
    };
}

#[macro_export]
macro_rules! var_with_label {
    ($tensor:expr, $label:expr) => {
        Variable::with_label($tensor, $label)
    };
}

// 사용 예시를 위한 테스트 함수들
#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intuitive_labeling() -> MlResult<()> {
        // 스칼라
        let scalar = Variable::new(Tensor::from_vec(vec![1.0], &[])?);
        assert_eq!(scalar.label(), "scalar");

        // 벡터들
        let small_vec = Variable::new(Tensor::from_vec(vec![1.0, 2.0], &[2])?);
        assert_eq!(small_vec.label(), "small_vec");

        let bias = Variable::new(Tensor::from_vec(vec![1.0], &[1])?);
        assert_eq!(bias.label(), "bias");

        // 행렬들
        let square = Variable::new(Tensor::from_vec(vec![1.0; 9], &[3, 3])?);
        assert_eq!(square.label(), "small_matrix");

        let wide = Variable::new(Tensor::from_vec(vec![1.0; 20], &[2, 10])?);
        assert_eq!(wide.label(), "wide_matrix");

        // RGB 이미지
        let rgb = Variable::new(Tensor::from_vec(vec![1.0; 192], &[8, 8, 3])?);
        assert_eq!(rgb.label(), "rgb_image");

        // 배치 RGB
        let rgb_batch = Variable::new(Tensor::from_vec(vec![1.0; 768], &[4, 3, 8, 8])?);
        assert_eq!(rgb_batch.label(), "rgb_batch");

        Ok(())
    }

    #[test]
    fn test_custom_labels() -> MlResult<()> {
        let input = Variable::new_input(Tensor::from_vec(vec![1.0; 10], &[10])?);
        assert_eq!(input.label(), "input");

        let weight = Variable::new_weight(Tensor::from_vec(vec![1.0; 20], &[4, 5])?);
        assert_eq!(weight.label(), "weight");

        let conv_weight = Variable::new_conv_weight(Tensor::from_vec(vec![1.0; 36], &[3, 3, 2, 2])?, 1);
        assert_eq!(conv_weight.label(), "conv1_weight");

        Ok(())
    }

    #[test]
    fn test_unique_labeling() -> MlResult<()> {
        let input1 = Variable::new_input(Tensor::from_vec(vec![1.0; 10], &[10])?);
        let input2 = Variable::new_input(Tensor::from_vec(vec![2.0; 10], &[10])?);

        assert_eq!(input1.label(), "input");
        assert_eq!(input2.label(), "input_2");

        Ok(())
    }
}