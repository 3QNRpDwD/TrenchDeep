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
    // #[cfg(feature = "enable_backpropagation")]
    // fn from_grad_fn(data: Vec<f32>, shape: &[usize], grad_fn: &mut dyn Operator<f32>) -> Tensor<f32> {
    //     Tensor::new(Self {
    //         data,
    //         shape: shape.to_vec(),
    //         requires_grad: false,
    //
    //         // #[cfg(feature = "enable_backpropagation")]
    //         // grad_fn: Some(Arc::new(grad_fn))
    //     })
    // }

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
            grad: None,
        }
    }

    pub fn tensor(&self) -> &Tensor<f32> {
        &self.tensor
    }

    pub fn retain_grad(&self) -> bool {
        self.requires_grad
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn grad(&self) -> Option<&Tensor<f32>> {
        self.grad.as_ref()
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn set_grad(&mut self, grad: Tensor<f32>) {
        self.grad = Some(grad);
    }

    // 계산 그래프에 연산 추가
    #[cfg(feature = "enable_backpropagation")]
    pub fn with_grad_fn(self, function: Arc<dyn Function<f32>>, inputs: &[&Arc<Variable<f32>>]) -> Self {
        let self_arc = Arc::new(self);

        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.borrow_mut();

            // 입력 노드 ID 찾기 또는 추가
            let input_ids = inputs.iter().map(|&input_var| {
                // 이미 그래프에 있는지 확인
                for (id, node) in &graph.nodes {
                    if Arc::ptr_eq(&node.variable, input_var) {
                        return *id;
                    }
                }
                // 없으면 추가
                graph.add_input(input_var.clone())
            }).collect();

            // 연산 노드 추가
            graph.add_operation(self_arc.clone(), function, input_ids);
        });

        // 안전하게 Arc에서 추출 (실제로는 좀 더 복잡한 방법 필요)
        let result = (*self_arc).clone();
        result
    }

    #[cfg(feature = "enable_backpropagation")]
    pub fn backward(&self) -> MlResult<()> {
        let self_arc = Arc::new(self.clone());

        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.borrow_mut();

            let node_id = graph.nodes.iter()
                .find(|(_, &ref node)| Arc::ptr_eq(&node.variable, &self_arc))
                .map(|(id, _)| *id);

            match node_id {
                Some(id) => graph.backward(id),
                None => Err(StringError("위상정렬된 노드를 찾을수 없습니다.".to_string())),
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
    fn add_input(&mut self, var: Arc<Variable<f32>>) -> NodeId {
        let node_id = self.next_id;
        self.next_id += 1;

        let node = ComputationNode {
            id: node_id,
            variable: var,
            function: None,
            output: None,
            inputs: Vec::new(),
        };

        self.nodes.insert(node_id, node);
        self.sorted = false;
        node_id
    }

    // 연산 노드 추가
    #[cfg(feature = "enable_backpropagation")]
    fn add_operation(&mut self, var: Arc<Variable<f32>>, function: Arc<dyn Function<f32>>,  inputs: Vec<NodeId>) -> NodeId {
        let node_id = self.next_id;
        self.next_id += 1;
        let output = match var.grad.is_none() {
            true => None,
            false => Some(var.grad.clone().unwrap())
        };

        let node = ComputationNode {
            id: node_id,
            variable: var,
            function: Some(function),
            output,
            inputs,
        };

        self.nodes.insert(node_id, node);
        self.sorted = false;
        node_id
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

        // 위상정렬 수행
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            // 이 노드를 입력으로 사용하는 노드들 찾기
            for (&next_id, next_node) in &self.nodes {
                if next_node.inputs.contains(&node_id) {
                    let degree = in_degree.get_mut(&next_id).unwrap();
                    *degree -= 1;

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

        // 각 노드의 그래디언트 초기화
        let mut grads = HashMap::new();

        // 출력 노드의 그래디언트를 1.0으로 초기화
        let output_node = self.nodes.get(&output_id).ok_or("출력 노드를 찾을 수 없습니다.")?;
        let shape = &output_node.variable.tensor.shape;
        let data = vec![1.0; shape.iter().product()];
        let grad = Tensor::from_vec(data, &shape).map_err(|e| format!("그래디언트 초기화 실패: {:?}", e))?;

        grads.insert(output_id, grad);

        // 역순으로 순회하며 역전파 수행
        for &node_id in self.topo_sorted.iter().rev() {
            if let Some(grad) = grads.get(&node_id).cloned() {
                let node = self.nodes.get(&node_id).unwrap();

                if let Some(function) = &node.function {
                    // 함수의 backward 호출
                    let input_grads = function.backward(node.output.as_ref().unwrap(), &grad).map_err(|e| format!("역전파 실패: {:?}", e))?;

                    // 입력 노드들에 그래디언트 전파
                    for (i, &input_id) in node.inputs.iter().enumerate() {
                        let input_grad = input_grads.get(i).ok_or("입력 그래디언트가 부족합니다.")?;

                        if let Some(existing_grad) = grads.get_mut(&input_id) {
                            // 그래디언트 누적 (여러 경로에서 그래디언트가 들어올 수 있음)
                            let mut new_data = Vec::with_capacity(existing_grad.data().len());
                            for (a, b) in existing_grad.data().iter().zip(input_grad.data().iter()) {
                                new_data.push(a + b);
                            }
                            *existing_grad = Tensor::from_vec(new_data, existing_grad.shape())
                                .map_err(|e| format!("그래디언트 누적 실패: {:?}", e))?;
                        } else {
                            grads.insert(input_id, input_grad.clone());
                        }
                    }
                }
            }
        }

        // 각 변수에 그래디언트 설정
        for (node_id, grad) in grads {
            let node = self.nodes.get(&node_id).unwrap();
            if node.variable.requires_grad {
                unsafe {
                    let var_ptr = Arc::as_ptr(&node.variable) as *mut Variable<f32>;
                    (*var_ptr).set_grad(grad);
                }
            }
        }

        Ok(())
    }
}

// Function 트레이트 확장
#[cfg(feature = "enable_backpropagation")]
pub trait AutogradFunction<T: Debug + Clone>: Function<f32> + Clone where Self: 'static {
    // 자동 미분을 위한 함수 실행 (계산 그래프에 노드 추가)
    fn apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>> {
        let tensors: Vec<&Tensor<f32>> = inputs.iter().map(|&var| var.tensor()).collect();
        let mut results = self.forward(&tensors)?;

        if results.len() != 1 {
            return Err("현재는 단일 출력 함수만 지원합니다.".into());
        }

        let result = results.remove(0);

        #[cfg(feature = "enable_backpropagation")]
        if inputs.iter().any(|var| var.requires_grad) {
            return Ok(Arc::new(result.with_grad_fn(Arc::new(self.clone()), inputs)))
        }

        Ok(Arc::new(result))
    }
}

// 모든 Function 구현체에 AutogradFunction 자동 구현
#[cfg(feature = "enable_backpropagation")]
impl<F: Function<f32> + Clone + 'static> AutogradFunction<f32> for F {}