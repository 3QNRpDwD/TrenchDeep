use super::*;

impl TensorAllocator {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
            pool: HashMap::new(),
        }
    }

    /// ID를 통해 텐서의 불변 참조를 얻습니다.
    pub fn get_tensor_ref(&self, node_id: &HandleId) -> Option<&GlobalTensor<f32>> {
        self.storage.get(node_id)
    }

    pub fn get_tensor_mut(&mut self, node_id: &HandleId) -> Option<&mut GlobalTensor<f32>> {
        self.storage.get_mut(node_id)
    }


    /// 영구 텐서를 할당합니다. 이 텐서는 풀에 반환되지 않습니다.
    pub fn alloc_permanent(&mut self, data: Vec<f32>, shape: Vec<usize>) -> MlResult<Tensor> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        let node_id = NODE_ID_GEN.next();
        self.storage.insert(node_id, GlobalTensor { data, shape });
        Ok(Tensor(node_id))
    }

    /// 임시 텐서를 할당하거나 풀에서 가져옵니다. PooledTensor로 반환됩니다.
    pub fn alloc_temporary(&mut self, shape: &[usize]) -> PooledTensor {
        let node_id = if let Some(nodes) = self.pool.get_mut(shape) {
            nodes.pop()
        } else {
            None
        };

        let tensor_id = if let Some(id) = node_id {
            id
        } else {
            // 풀에 없으면 새로 생성
            let new_id = NODE_ID_GEN.next();
            let size = shape.iter().product();
            let mut data = Vec::with_capacity(size);
            unsafe { data.set_len(size); }
            self.storage.insert(new_id, GlobalTensor { data, shape: shape.to_vec() });
            new_id
        };

        PooledTensor { node_id: tensor_id, detached: false }
    }
    
    fn release(&mut self, node_id: HandleId) {
        if let Some(tensor) = self.storage.get(&node_id) {
            let shape = tensor.shape().to_vec();
            self.pool.entry(shape).or_default().push(node_id);
        }
    }
}

impl Drop for PooledTensor {
    fn drop(&mut self) {
        if !self.detached {
            TENSOR_ALLOCATOR.with_borrow_mut(|allocator| {
                allocator.release(self.node_id);
            });
        }
    }
}

// PooledTensor를 일반 Tensor처럼 사용하기 위해 Deref/DerefMut 구현
impl std::ops::Deref for PooledTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        // 이 변환은 수명(lifetime) 관점에서 안전하지 않을 수 있으나,
        // PooledTensor가 살아있는 동안 Tensor도 유효하다고 가정합니다.
        // 더 안전한 구현을 위해서는 TENSOR_ALLOCATOR 접근이 필요합니다.
        // 여기서는 간단한 사용을 위해 NodeId를 Tensor로 변환합니다.
        // 하지만 PooledTensor는 Tensor가 아니라 그 자체로 사용되어야 합니다.
        unsafe { &*(self as *const PooledTensor as *const Tensor) }
    }
}
