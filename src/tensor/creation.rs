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

    /// Attaches gradient computation information to the variable by creating a computation graph node.
    ///
    /// This method performs three main tasks:
    /// 1. Registers input variables in the global computation graph
    /// 2. Creates an operation node representing the mathematical function
    /// 3. Links the operation to its input variables in the graph
    ///
    /// # Arguments
    /// * `function` - The mathematical operation to record for backward pass (must implement `Function<f32>`)
    /// * `inputs` - Reference to input variables used in this operation (must already exist in computation graph)
    ///
    /// # Returns
    /// return a new `Variable` instance with gradient computation capabilities:
    /// - Maintains same tensor data as original variable
    /// - Contains backreference to the operation in computation graph
    ///
    /// # Safety
    /// - Uses thread-local storage for computation graph (not thread-safe)
    /// - Clones Arc references internally - ensure proper ownership management
    /// - All inputs must belong to the same computation graph context
    ///
    /// # Panics
    /// Will panic if:
    /// - There's mutable borrow conflict in thread-local graph storage
    /// - Input variables exist in different computation graph contexts (TOCTOU violation)
    ///
    /// # Implementation Notes
    /// - Uses Arc pointer equality checks for existing graph node detection
    /// - Maintains DAG structure through node ID tracking
    /// - Operation nodes store backward function and input relationships
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

        let result = (*self_arc).clone();
        result
    }


    /// Performs backward propagation of gradients through the computation graph starting from this variable.
    ///
    /// This method initiates the reverse-mode automatic differentiation process by:
    /// 1. Locating this variable's node in the computation graph
    /// 2. Executing topological sort-based gradient calculation
    /// 3. Accumulating gradients through chain rule applications
    ///
    /// # Returns
    /// return `Ok(())` on successful gradient propagation:
    /// - All upstream variables will have their `.grad` fields updated
    /// - Gradient calculation follows reverse execution order
    ///
    /// # Errors
    /// Returns `Err` with:
    /// - "위상정렬된 노드를 찾을수 없습니다" if variable isn't registered in computation graph
    /// - Any errors occurring during gradient calculation steps
    ///
    /// # Panics
    /// Will panic if:
    /// - Mutex borrow fails on thread-local computation graph storage
    /// - Graph contains cycles (violates DAG requirement)
    /// - Numerical errors occur during gradient computation
    ///
    /// # Safety
    /// - Requires all preceding operations to be properly registered in computation graph
    /// - Should typically be called only once per backward pass from root variable
    /// - Not re-entrant due to thread-local storage usage
    ///
    /// # Implementation Details
    /// - Uses thread-local computation graph storage
    /// - Relies on topological ordering stored during forward pass
    /// - Gradient accumulation uses += operator (users should zero gradients when needed)
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

        // 출력 노드의 그래디언트를 1.0으로 초기화
        let mut grads = HashMap::new();
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
                    let input_grads = function.backward(node.output.as_ref().unwrap(), &grad).map_err(|e| format!("역전파 실패: {:?}", e))?;

                    // 입력 노드들에 그래디언트 전파
                    for (i, &input_id) in node.inputs.iter().enumerate() {
                        let input_grad = input_grads.get(i).ok_or("입력 그래디언트가 부족합니다.")?;

                        if let Some(existing_grad) = grads.get_mut(&input_id) {
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


pub trait AutogradFunction<T: Debug + Clone>: Function<f32> + Clone where Self: 'static {
    /// Applies the operation defined by this struct to the given input variables.
    ///
    /// This method performs the following steps:
    /// 1. Extracts tensors from input variables
    /// 2. Executes the forward pass of the operation using these tensors
    /// 3. Handles backpropagation setup when enabled
    ///
    /// # Arguments
    /// * `inputs` - A slice of reference-counted variables containing input tensors.
    ///              All inputs must have matching shapes that are compatible with this operation.
    ///
    /// # Returns
    /// return `Ok(Arc<Variable<f32>>)` containing:
    /// - The output tensor wrapped in a Variable
    /// - Optional gradient computation graph when backpropagation is enabled
    ///
    /// # Errors
    /// Returns `Err` if:
    /// - The forward pass produces multiple outputs (current implementation only supports single-output operations)
    /// - Any operation-specific validation fails during the forward pass
    ///
    /// # Backpropagation Behavior
    /// When compiled with `enable_backpropagation` feature:
    /// - Automatically constructs gradient computation graph if any input requires gradients
    /// - Attaches backpropagation information using the operation's clone and input references
    /// - Maintains the ownership of result tensors while enabling gradient tracking
    // #[cfg(feature = "enable_backpropagation")]
    /// When backpropagation is disabled:
    /// - Simply returns the output tensor without any gradient information
    /// - All gradient-related flags are ignored
    fn apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>> {
        let tensors: Vec<&Tensor<f32>> = inputs.iter().map(|&var| var.tensor()).collect();
        let mut results = self.forward(&tensors)?;

        #[cfg(feature = "enable_backpropagation")]
        {
            if results.len() != 1 {
                return Err("자동 역전파는 현재는 단일 출력 함수만 지원합니다.".into());
            }

            let result = results.remove(0);

            if inputs.iter().any(|var| var.requires_grad) {
                return Ok(Arc::new(result.with_grad_fn(Arc::new(self.clone()), inputs)))
            }

            return Ok(Arc::new(result))
        }

        Ok(Arc::new(results.remove(0)))
    }
}

impl<F: Function<f32> + Clone + 'static> AutogradFunction<f32> for F {}