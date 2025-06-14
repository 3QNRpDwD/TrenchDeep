use super::*;

#[cfg(feature = "enableBackpropagation")]
impl Variable {
    pub fn tpye_name(&self) -> String {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown").replace("<f32>", "")
    }

    /// Attaches gradient computation information to the variable by creating a computation graph node.
    ///
    /// This method performs three main tasks:
    /// 1. Registers input variables in the global computation graph
    /// 2. Creates an operation node representing the mathematical function
    /// 3. Links the operation to its input variables in the graph
    ///
    /// # Arguments
    /// * `function` - The mathematical operation to record for backward pass (must implement `Function`)
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
    pub fn with_grad_fn(self: Arc<Self>, operator_name: &str, inputs: &[&Arc<Variable>]) {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            let input_ids: Vec<NodeId> = inputs.iter().map(|&input_var| {
                let input_id = input_var.node_id();
                if !graph.node_map.contains_key(&input_id) {
                    graph.add_input(input_var.clone());
                }
                input_id
            }).collect();
            graph.add_operation(self, operator_name, input_ids)
        });

        // 입력 노드 ID 찾기 또는 추가
        // 없으면 추가
        // 현재 경사하강법등의 기존 텐서의 수정이 불가피한 메서드를 사용할때 계속해서 새로운 텐서를 만들기 때문에,
        // 기존의 생성된 텐서는 더이상 사용되지 않음에도, 계산그래프상에 남아있으며, 이로 인해 계산 그래프 자체가 거대해지고 검색자체도 굉장히 느려지는 현상이 발생함.
        // 이를 해결하려면 단순히 텐서를 비교하는것이 아니라, 메모리값을 비교후. 메모리값이 같은데 내부값이 다를 경우, 업데이트하는 방식을 사용하거나,
        // 텐서 자체를 복사하는것이 아닌 메모리값을 계산그래프에 추가하는등의 방식으로, 텐서와 계산그래프의 수정과 연동이 가능하도록 개선해야될듯함.
        // 이에 대한 자세한 해결책을 시급히 만들어야함.


        // 원래 고유한 아이디를 만들어서 계산그래프를 구성했으나, 현재 연산구조의 특성상 텐서의 포인터를 노드의 키값으로 설정하는것이
        // 같은 효과를 내면서도, 훨신 강력한 성능을 이끌어낼것으로 생각되어, 변경했으며, 기존보다 약 1.8배가량 성능이 향상된것으로 보임.
        // 또한, 이같은 변화로, 향후 개선돠어야할 계산그래프의 쓰레기 텐서(더이상 연산에 사용되지 않는 텐서)의 발생을 줄이는데 도움이 될것으로 보이며,
        // 계산그래프의 수정또한 더욱 쉽게 가능할것으로 보임.
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
    pub fn backward(self: &Arc<Self>) -> MlResult<()> {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            if graph.node_map.contains_key(&self.node_id()) {
                graph.ensure_topological_sort();
                graph.backward(self.node_id())
            } else {
                Err(MlError::StringError("계산 그래프가 생성되지 않았습니다.".to_string()))
            }
        })
    }
}

#[cfg(feature = "enableBackpropagation")]
impl ComputationGraph {
    /// 새로운 계산 그래프를 생성합니다.
    ///
    /// 이 메서드는 노드와 관련 데이터를 저장할 빈 `ComputationGraph` 인스턴스를 초기화합니다.
    ///
    /// # 반환값
    /// - `Self`: 초기화된 `ComputationGraph` 인스턴스
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: std::collections::HashMap::new(),
            adjacency_list: Vec::new(),
            reverse_adjacency: Vec::new(),
            topo_order: Vec::new(),
            is_sorted: false,
            // memory_pool: TensorPool::new(),
        }
    }

    /// 입력 변수 노드를 계산 그래프에 추가합니다.
    ///
    /// 이 메서드는 새로운 입력 노드를 생성하고, 고유한 ID를 부여하여 그래프에 추가합니다.
    ///
    /// # 파라미터
    /// - `variable`: `Arc<Variable>` 타입의 입력 변수 (스마트 포인터로 감싸진 변수)
    ///
    /// # 반환값
    /// - `NodeId`: 추가된 노드의 고유 식별자
    pub(crate) fn add_input(&mut self, variable: Arc<Variable>) -> NodeId {
        let node_id = variable.node_id();
        let node_idx = self.nodes.len();
        
        #[cfg(feature = "enableVisualization")]
        {
            VISUALIZATION_GRAPH.with(|viz_graph| {
                let mut viz = viz_graph.borrow_mut();
                let id_str = format!("{:?}",node_id);
                viz.add_variable_node(&id_str, variable.label(), variable.node_type());
            });
        }
        
        let node = ComputationNode {
            id: node_id,
            variable,
            function: None,
            inputs: Vec::new(),
            is_leaf: true,
        };

        self.nodes.push(node);
        self.node_map.insert(node_id, node_idx);
        self.adjacency_list.push(Vec::new());
        self.reverse_adjacency.push(Vec::new());
        self.is_sorted = false;

        node_id
    }

    /// 연산 노드를 계산 그래프에 추가합니다.
    ///
    /// 이 메서드는 연산을 나타내는 노드를 생성하고, 입력 노드들과 연산 함수를 연결하여 그래프에 추가합니다.
    ///
    /// # 파라미터
    /// - `variable`: `Arc<Variable>` 타입의 연산 결과 변수
    /// - `function`: `Arc<dyn Function>` 타입의 연산 함수 (동적 디스패치 지원)
    /// - `inputs`: 이 연산의 입력 노드 ID 목록
    ///
    /// # 반환값
    /// - `NodeId`: 추가된 연산 노드의 고유 식별자
    pub(crate) fn add_operation(&mut self, variable: Arc<Variable>, operator_name: &str, inputs: Vec<NodeId>) -> NodeId {
        #[cfg(feature = "enableVisualization")]
        VISUALIZATION_GRAPH.with(|viz_graph| {
            let mut viz = viz_graph.borrow_mut();
            viz.register_operation(operator_name, &inputs, variable.node_id());
            viz.add_variable_node(&format!("{:?}", variable.node_id()), &variable.label, &variable.node_type);
        });


        let output_id = variable.node_id();
        let output_idx = self.nodes.len();

        // 입력 노드들의 인덱스 찾기
        let input_indices: Vec<usize> = inputs.iter()
            .map(|id| *self.node_map.get(&id).unwrap())
            .collect();

        let node = ComputationNode {
            id: output_id,
            variable,
            function: Some(operator_name.to_string()),
            inputs: inputs
                .iter()
                .map(|&var| var).collect(),
            is_leaf: false,
        };

        self.nodes.push(node);
        self.node_map.insert(output_id, output_idx);
        self.adjacency_list.push(Vec::new());
        self.reverse_adjacency.push(Vec::new());

        // 인접 리스트 업데이트
        for &input_idx in &input_indices {
            self.adjacency_list[input_idx].push(output_idx);
            self.reverse_adjacency[output_idx].push(input_idx);
        }

        self.is_sorted = false;
        output_id
    }

    pub fn reset_graph() {
        COMPUTATION_GRAPH.with(|graph| { 
            graph.lock().unwrap().clear();
        });
        #[cfg(feature = "enableVisualization")]
        {
            VISUALIZATION_GRAPH.with(|viz_graph| {
                viz_graph.borrow_mut().clear();
            });
        }
    }

    fn ensure_topological_sort(&mut self) {
        if !self.is_sorted {
            self.topological_sort();
        }
    }

    /// 계산 그래프의 노드들을 위상 정렬(Topological Sort)합니다.
    ///
    /// 이 메서드는 그래프의 노드들을 의존성 순서대로 정렬하여 역전파를 위한 준비를 합니다.
    /// 이미 정렬된 경우에는 아무 작업도 수행하지 않습니다.
    pub(crate) fn topological_sort(&mut self) {
        let mut in_degree = vec![0; self.nodes.len()];
        let mut queue = std::collections::VecDeque::new();

        // 진입 차수 계산
        for adj_list in &self.adjacency_list {
            for &neighbor in adj_list {
                in_degree[neighbor] += 1;
            }
        }

        // 진입 차수가 0인 노드들을 큐에 추가
        for (idx, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(idx);
            }
        }

        self.topo_order.clear();
        while let Some(node_idx) = queue.pop_front() {
            self.topo_order.push(node_idx);

            for &neighbor in &self.adjacency_list[node_idx] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        self.is_sorted = true;
    }

    /// 역전파(Backpropagation)를 수행합니다.
    ///
    /// 이 메서드는 주어진 출력 노드에서 시작하여 그래프를 따라 그래디언트를 계산하고 전파합니다.
    /// 위상 정렬이 아직 수행되지 않았다면 먼저 정렬을 실행합니다.
    ///
    /// # 파라미터
    /// - `output_id`: 역전파를 시작할 출력 노드의 ID
    ///
    /// # 반환값
    /// - `MlResult<()>`: 성공 시 `Ok(())`, 실패 시 오류 메시지를 포함한 `Err`
    ///
    /// # 오류
    /// - 출력 노드가 존재하지 않을 경우
    /// - 그래디언트 초기화 실패 시
    /// - 역전파 계산 실패 시
    #[cfg(feature = "enableBackpropagation")]
    pub(crate) fn backward(&mut self, output_id: NodeId) -> MlResult<()> {
        for node in &self.nodes {
            node.variable.clear_grad();
        }
        // Set output node's gradient to 1.0
        let output_idx = *self.node_map.get(&output_id)
            .ok_or_else(|| MlError::StringError("Output node not found".to_string()))?;
        let output_var = &self.nodes[output_idx].variable;
        if output_var.grad().is_none() {
            let grad = Tensor::from_vec(
                vec![1.0; output_var.tensor.shape().iter().product()],
                output_var.tensor.shape()
            )?;
            output_var.set_grad(grad);
        }

        // 위상 정렬된 순서의 역순으로 순회
        for &node_idx in self.topo_order.iter().rev() {
            let node = &self.nodes[node_idx];
            let var = &node.variable;
            let grad = var.grad();
            if node.function.is_none() || grad.is_none() { continue; }
            let function = node.function.as_ref().unwrap();
            let input_tensors: Vec<&Tensor> = node.inputs
                .iter()
                .map(|&input_id| {
                    let input_idx = self.node_map[&input_id];
                    self.nodes[input_idx].variable.tensor()
                })
                .collect::<Vec<&Tensor>>();

            let output_grad = grad.unwrap();
            let input_grads = OPERATOR_STORAGE.with(|ops| ops.borrow().get(function).unwrap().backward(&input_tensors, &output_grad)
                .map_err(|e| MlError::StringError(format!("Failed to compute backward for function {:?}: {}", function, e))))?;

            for (input_id, grad) in node.inputs.iter().zip(input_grads) {
                let input_idx = self.node_map[input_id];
                let input_node = &self.nodes[input_idx];
                input_node.variable.accumulate_grad(grad)?;
            }

            if !var.is_retain_grad() { var.clear_grad(); }
        }
        Ok(())
    }

    // 역전파 메서드와 반대로 기록된 노드의 순서대로 실행하고 해당 값을 노드에 저장하는 메서드
    // pub fn forward(&mut self, input_id: NodeId) -> MlResult<()> {}

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.node_map.clear();
        self.adjacency_list.clear();
        self.reverse_adjacency.clear();
        self.topo_order.clear();
        self.is_sorted = false;
        // self.memory_pool.clear();
    }
    
    pub fn get_graph_stats() -> (usize, bool) {
        COMPUTATION_GRAPH.with(|compute_graph| {
            let graph = compute_graph.lock().unwrap();
            (graph.nodes.len(), graph.is_sorted)
        })
    }

    pub fn print_graph_details(&self) {
        println!("=== Computation Graph Details ===");
        for (order, &node_idx) in self.topo_order.iter().enumerate() {
            let node = &self.nodes[node_idx];
            let var = &node.variable;
            let tensor = var.tensor();
            let first_data = tensor.data();
            let shape = tensor.shape();

            let func_name = node.function
                .as_ref()
                .map(|f| OPERATOR_STORAGE.with(|ops| ops.borrow().get(f).unwrap().type_name().to_string()))
                .unwrap_or_else(|| String::from("Input"));

            println!(
                "[{}] Func: {:<12} | First data: {:?} | Shape: {:?}",
                order, func_name, first_data, shape
            );
        }
        println!("=================================");
    }
}

impl AutogradFunction for GlobalFunction {
    fn apply(&self, inputs: &[&Arc<Variable>]) -> MlResult<Arc<Variable>> {
        let tensors: Vec<&Tensor> = inputs
            .iter()
            .map(|&var| var.tensor()).collect();
        let output = Arc::new(Variable::new(self.forward(&tensors)?.remove(0)));

        #[cfg(feature = "enableBackpropagation")]
        {
            output.clone().with_grad_fn(self.name(), inputs);
            return Ok(output)
        }
        // 정적계산 그래프를 통해서 메모리 효율성을 증대하려 했으나, 사전에 텐서의 정보가 주입되지 않으면 메모리 관리가 어려워,
        // 무산될것으로 예상되며, 정적, 동적계산그래프를 전환 가능하도록 향후 추가될것으로 생각하고있음.
        // 따라서 매 계산마다 계산그래프를 갱신하는 현재 구조를 유지하게될것 같은데, 이는 계산그래프 갱신으로 인한 오버헤드가 예상됨.
        // 솔직히 어느 방식을 선택해야할지잘 모르겠음.

        Ok(output)
    }

    fn apply_with_label(&self, inputs: &[&Arc<Variable>], label: &str) -> MlResult<Arc<Variable>> {
        let tensors: Vec<&Tensor> = inputs
            .iter()
            .map(|&var| var.tensor()).collect();
        let output = crate::var_with_label!(self.forward(&tensors)?.remove(0), label);
        #[cfg(feature = "enableBackpropagation")]
        {
            output.clone().with_grad_fn(self.name(), inputs);
            return Ok(output)
        }
        // 정적계산 그래프를 통해서 메모리 효율성을 증대하려 했으나, 사전에 텐서의 정보가 주입되지 않으면 메모리 관리가 어려워,
        // 무산될것으로 예상되며, 정적, 동적계산그래프를 전환 가능하도록 향후 추가될것으로 생각하고있음.
        // 따라서 매 계산마다 계산그래프를 갱신하는 현재 구조를 유지하게될것 같은데, 이는 계산그래프 갱신으로 인한 오버헤드가 예상됨.
        // 솔직히 어느 방식을 선택해야할지잘 모르겠음.

        Ok(output)
    }
}