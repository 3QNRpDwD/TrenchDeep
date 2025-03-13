use crate::{MlError, MlResult};
use crate::MlError::StringError;
use crate::tensor::*;


impl  Tensor<f32> {
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

// 전역 계산 그래프 (스레드 로컬)
#[cfg(feature = "enable_backpropagation")]
thread_local! {
    static COMPUTATION_GRAPH: RefCell<ComputationGraph<f32>> = RefCell::new(ComputationGraph::new());
}

impl Variable<f32> {
    pub fn new(tensor: Tensor<f32>) -> Self {
        Variable {
            tensor,
            requires_grad: cfg!(feature = "enable_backpropagation"),

            #[cfg(feature = "enable_backpropagation")]
            grad: RefCell::new(None),
        }
    }

    pub fn tensor(&self) -> &Tensor<f32> {
        &self.tensor
    }

    pub fn retain_grad(&self) -> bool {
        self.requires_grad
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.grad.borrow().clone()
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn set_grad(&self, grad: Tensor<f32>) {
        *self.grad.borrow_mut() = Some(grad);
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn clear_grad(&self) {
        *self.grad.borrow_mut() = None;
    }


    #[cfg(feature = "enable_backpropagation")]
    pub fn accumulate_grad(&self, new_grad: Tensor<f32>) -> MlResult<()> {
        let mut grad_ref = self.grad.borrow_mut();

        if let Some(ref existing_grad) = *grad_ref {
            let mut new_data = Vec::with_capacity(existing_grad.data().len());
            for (a, b) in existing_grad.data().iter().zip(new_grad.data().iter()) {
                new_data.push(a + b);
            }

            let accumulated_grad = Tensor::from_vec(new_data, existing_grad.shape())
                .map_err(|e| format!("그래디언트 누적 실패: {:?}", e))?;

            *grad_ref = Some(accumulated_grad);
        } else {
            *grad_ref = Some(new_grad);
        }

        Ok(())
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn with_grad_fn(self: Arc<Self>, function: Arc<dyn Function<f32>>, inputs: &[&Arc<Variable<f32>>]) {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.borrow_mut();

            // 입력 노드 ID 찾기 또는 추가
            let input_ids =
                inputs.iter().map(|&input_var| {
                // 이미 그래프에 있는지 확인
                for (id, node) in &graph.nodes {
                    if node.variable.as_ref() == input_var {
                        return *id;
                    }
                }
                // 없으면 추가
                graph.add_input(input_var.clone())
            }).collect();

            // 연산 노드 추가
            graph.add_operation(self, function, input_ids);
        });
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn backward(&self) -> MlResult<()> {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.borrow_mut();

            let node_id = graph.nodes.iter()
                .find(|(_, &ref node)| node.variable.as_ref() == self)
                .map(|(id, _)| *id);

            match node_id {
                Some(id) => { graph.backward(id) },
                None => Err(StringError("계산 그래프가 생성되지 않았습니다.".to_string())),
            }
        })
    }
}

#[cfg(feature = "enable_backpropagation")]
impl ComputationGraph<f32> {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            topo_sorted: Vec::new(),
            sorted: false,
        }
    }

    // 입력 변수 노드 추가
    #[cfg(feature = "enable_backpropagation")]
    fn add_input(&mut self, variable: Arc<Variable<f32>>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let node = ComputationNode {
            id,
            variable,
            function: None,
            // output: None,
            inputs: Vec::new(),
        };

        self.nodes.insert(id, node);
        self.sorted = false;
        id
    }

    // 연산 노드 추가
    #[cfg(feature = "enable_backpropagation")]
    fn add_operation(&mut self, variable: Arc<Variable<f32>>, function: Arc<dyn Function<f32>>,  inputs: Vec<NodeId>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let node = ComputationNode {
            id,
            variable,
            function: Some(function),
            // output,
            inputs,
        };

        self.nodes.insert(id, node);
        self.sorted = false;
        id
    }

    // 위상정렬 수행
    fn topological_sort(&mut self) {
        if self.sorted {
            return;
        }

        let mut result = Vec::new();
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();

        // 진입차수 초기화
        for (&node_id, node) in &self.nodes {
            let degree = node.inputs.len();
            in_degree.insert(node_id, degree);

            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            // 원래 위상정렬 알고리즘에서 중복 노드를 고려하지 않은 설계 때문에 같은 노드를 여러번 사용하는 계산에서 오류가 발생했음.
            // 현재는 구조를 개선한 상태임.
            result.push(node_id);

            // 이 노드를 입력으로 사용하는 노드들 찾기
            for (&next_id, next_node) in &self.nodes {
                let count = next_node.inputs.iter().filter(|&&input_id| input_id == node_id).count();

                // 해당 노드가 입력으로 사용된 횟수만큼 진입 차수 감소
                if count > 0 {
                    let degree = in_degree.get_mut(&next_id).unwrap();
                    *degree -= count; //

                    if *degree == 0 {
                        queue.push_back(next_id);
                    }
                }
            }
        }

        self.topo_sorted = result;
        self.sorted = true;
    }

    // 역전파 수행
    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, output_id: NodeId) -> MlResult<()> {
        if !self.sorted {
            self.topological_sort();
        }

        for (_, node) in &self.nodes {
            node.variable.clear_grad();
        }

        let output_var = &self.nodes.get(&output_id).ok_or("출력 노드를 찾을 수 없습니다.")?.variable;
        if output_var.grad().is_none() {
            let grad = Tensor::from_vec(
                vec![1.0; output_var.tensor.shape().iter().product()],
                output_var.tensor.shape()
            )?;
            output_var.set_grad(grad);
        }

        for &node_id in self.topo_sorted.iter().rev() {
            let node = self.nodes.get(&node_id).unwrap();

            if node.variable.grad().is_none() || node.function.is_none() {
                continue;
            }

            if let Some(function) = &node.function {
                for &input_id in &node.inputs {
                    let input_node = self.nodes.get(&input_id).unwrap();
                    let input_grads = function.backward(input_node.variable.tensor(), &node.variable.grad().unwrap())
                        .map_err(|e| format!("역전파 실패: {:?}", e))?;
                    input_node.variable.accumulate_grad(input_grads[0].clone())?;
                }
            }
        }

        Ok(())
    }
}


pub trait AutogradFunction: Function<f32> + Clone where Self: 'static {
    fn apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>> {
        let tensors: Vec<&Tensor<f32>> = inputs.iter().map(|&var| var.tensor()).collect();
        let mut results = self.forward(&tensors)?;

        #[cfg(feature = "enable_backpropagation")]
        {
            let result = Arc::new(results.remove(0));
            result.clone().with_grad_fn(Arc::new(self.clone()), inputs);
            return Ok(result)
        }

        Ok(Arc::new(results.remove(0)))
    }
}

impl<F: Function<f32> + Clone + 'static> AutogradFunction for F {}