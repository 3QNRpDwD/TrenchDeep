use super::*;

impl TensorBase for PooledTensor {
    fn new(data: Vec<Vec<f32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();
        TENSOR_ALLOCATOR.with_borrow_mut(|alloc| {
            let unit =  alloc.alloc_temporary(&shape);
            alloc.storage.get_mut(&unit.id()).unwrap().data = data;
            unit
        })
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }
        
        TENSOR_ALLOCATOR.with_borrow_mut(|alloc| {
            let unit =  alloc.alloc_temporary(&shape);
            alloc.storage.get_mut(&unit.id()).unwrap().data = data;
            Ok(unit)
        })
    }
    
    fn as_ptr(&self) -> *const GlobalTensor<f32> {
        TENSOR_ALLOCATOR.with_borrow(|allocator| {
            // `Ref`의 수명을 연장하기 위해 raw 포인터를 사용.
            // allocator의 borrow가 끝난 후에도 포인터가 유효하다고 가정.
            let tensor_ref = allocator.get_tensor_ref(&self.node_id).unwrap();
            tensor_ref as *const GlobalTensor<f32>
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
}

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
}

// 매번 생성된 텐서를 전역 그래프에 저장하느라 심각한 성능저하와 메모리 낭비 발생. 아마도 연산자등에서 생성하는 텐서는 저장되지 않도록 조치를 취해야 할듯.
impl TensorBase for Tensor {
    fn new(data: Vec<Vec<f32>>) -> Tensor {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();

        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.alloc_permanent(data, shape).unwrap()
        })
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Tensor> {
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.alloc_permanent(data, shape.to_vec())
        })
    }

    fn as_ptr(&self) -> *const GlobalTensor<f32> {
        TENSOR_ALLOCATOR.with_borrow(|allocator| {
            // `Ref`의 수명을 연장하기 위해 raw 포인터를 사용.
            // allocator의 borrow가 끝난 후에도 포인터가 유효하다고 가정.
            let tensor_ref = allocator.get_tensor_ref(&self.0).unwrap();
            tensor_ref as *const GlobalTensor<f32>
        })
    }

    fn as_mut(&self) -> *mut GlobalTensor<f32> {
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            // `Mut`의 수명을 연장하기 위해 raw 포인터를 사용.
            // allocator의 borrow가 끝난 후에도 포인터가 유효하다고 가정.
            let tensor_mut = allocator.get_tensor_mut(&self.0).unwrap();
            tensor_mut as *mut GlobalTensor<f32>
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
}

impl PooledTensor {
    pub fn to_id(mut self, detached: bool) -> MlResult<Tensor> {
        self.detached = detached;
        Ok(Tensor(self.node_id))
    }
}

impl GlobalTensor<f32> {
    pub fn to_id(self) -> MlResult<Tensor> {
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.alloc_permanent(self.data, self.shape)
        })
    }

    pub fn with_id(self, node_id: HandleId) -> MlResult<Tensor> {
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.storage.insert(node_id, self);
            Ok(Tensor(node_id))
        })
    }
    
    pub fn new_empty() -> GlobalTensor<f32> { 
        GlobalTensor { data: vec![], shape: vec![] }
    }
}

impl Tensor {
    pub fn with_id(data: Vec<f32>, shape: &[usize], node_id: HandleId) -> MlResult<Tensor> {
        let global_tensor = GlobalTensor::from_vec(data, shape)?;
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.storage.insert(node_id, global_tensor);
            Ok(Tensor(node_id))
        })
    }
    
    pub fn to_id(self) -> MlResult<HandleId> {
        Ok(self.0)
    }

    pub fn id(&self) -> HandleId {
        self.0
    }

    pub fn new_empty() -> Tensor {
        let node_id = NODE_ID_GEN.next();
        TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
            allocator.storage.insert(node_id, GlobalTensor::new_empty());
        });
        Tensor(node_id)
    }
    
    pub fn is_empty(&self) -> bool {
        self.data().iter().all(|&d| d == 0.0) || self.data().len() == 0
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
    pub(crate) fn infer_context_from_shape(shape: &[usize]) -> String {
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
    pub(crate) fn shape_to_label(shape: &[usize]) -> String {
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
    pub(crate) fn get_unique_label(base_label: &str) -> String {
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

impl GlobalFunction {
    pub fn new(name: String, node_id: HandleId) -> Self {
        Self {
            name,
            func_id: node_id,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn func_id(&self) -> &HandleId {
        &self.func_id
    }
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