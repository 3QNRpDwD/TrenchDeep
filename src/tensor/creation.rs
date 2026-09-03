use super::*;

impl TensorBase for GlobalTensor<f32> {
    fn new(data: Vec<Vec<f32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();
        GlobalTensor { data, shape, dirty: false }
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(GlobalTensor { data, shape: shape.to_vec(), dirty: false })
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
            storage.insert(node_id, GlobalTensor { data, shape, dirty: false })
        });
        Tensor::new_with_id(node_id)
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
            storage.insert(node_id, GlobalTensor { data, shape: shape.to_vec(), dirty: false })
        });

        Ok(Tensor::new_with_id(node_id))
    }

    fn as_ptr(&self) -> *const GlobalTensor<f32> {
        TENSOR_STORAGE.with(|storage| {
            storage.borrow().get(&self.id()).map(|gt| gt as *const GlobalTensor<f32>).unwrap()
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
    pub fn to_id(self) -> MlResult<Tensor> {
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

        Ok(Tensor::new_with_id(node_id))
    }

    pub fn with_id(self, node_id: NodeId) -> MlResult<Tensor> {
        let expected_len: usize = self.shape.iter().product();
        if self.data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: self.data.len(),
            }));
        }

        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(node_id, self)
        });

        Ok(Tensor::new_ref(node_id))
    }
    
    pub fn new_empty() -> GlobalTensor<f32> {
        GlobalTensor { data: vec![], shape: vec![], dirty: false }
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
            storage.insert(node_id, GlobalTensor { data, shape: shape.to_vec(), dirty: false })
        });

        Ok(Tensor::new_ref(node_id))
    }
    
    pub fn to_id(self) -> NodeId {
        self.id()
    }

    pub fn new_empty() -> Tensor {
        let node_id = NODE_ID_GEN.next();
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            storage.insert(node_id, GlobalTensor { data: vec![], shape: vec![], dirty: false })
        });
        Tensor::new_with_id(node_id)
    }
    
    pub fn is_empty(&self) -> bool {
        self.data().iter().all(|&d| d == 0.0) || self.data().len() == 0
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
