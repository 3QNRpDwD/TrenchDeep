use super::*;

impl Tensor<f32> {
    pub fn zeros() -> Tensor<f32> {
        Self {
            data: vec![],
            shape: vec![],
        }
    }

    pub fn scalar(scalar: f32) -> Tensor<f32> {
        Self {
            data: vec![scalar],
            shape: vec![1, 1],
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

impl Variable<f32> {
    pub fn new(tensor: Tensor<f32>) -> Self {
        Variable {
            tensor,
            requires_grad: cfg!(feature = "requiresGrad"),

            #[cfg(feature = "enableBackpropagation")]
            grad: RefCell::new(None),
        }
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