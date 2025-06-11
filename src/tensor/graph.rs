use std::fmt::format;
use super::*;

#[cfg(feature = "enableBackpropagation")]
impl Variable<f32> {
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
    pub fn with_grad_fn(self: Arc<Self>, operator_name: &str, inputs: &[&Arc<Variable<f32>>]) {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            let input_ids: Vec<NodeId> = inputs.iter().map(|&input_var| {
                let input_id = input_var.node_id();
                if !graph.node_map.contains_key(&input_id) {
                    graph.add_input(input_var.clone());
                }
                (input_id)
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
impl ComputationGraph<f32> {
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
    pub(crate) fn add_input(&mut self, variable: Arc<Variable<f32>>) -> NodeId {
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
    pub(crate) fn add_operation(&mut self, variable: Arc<Variable<f32>>, operator_name: &str, inputs: Vec<NodeId>) -> NodeId {
        #[cfg(feature = "enableVisualization")]
        VISUALIZATION_GRAPH.with(|viz_graph| {
            let mut viz = viz_graph.borrow_mut();
            viz.register_operation(operator_name, &inputs, variable.var_id);
            viz.add_variable_node(&format!("{:?}", variable.var_id), &variable.label, &variable.node_type);
        });


        let output_id = variable.var_id;
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

        self.topo_order.reverse(); // backward 순서로 변경
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
        // Set output node's gradient to 1.0
        let output_idx = *self.node_map.get(&output_id)
            .ok_or_else(|| MlError::StringError("Output node not found".to_string()))?;
        let output_var = &self.nodes[output_idx].variable;
        if output_var.grad().is_none() {
            let grad = Tensor::from_vec(
                vec![1.0; output_var.tensor.borrow().shape().iter().product()],
                output_var.tensor.borrow().shape()
            )?;
            output_var.set_grad(grad);
        }

        // 위상 정렬된 순서의 역순으로 순회
        for &node_idx in &self.topo_order {
            let node = &self.nodes[node_idx];

            let var = &node.variable;
            let grad = var.grad();
            if node.function.is_none() || grad.is_none() {
                continue;
            }

            if let Some(function) = &node.function {
                let input_tensors: Vec<&Tensor<f32>> = node.inputs
                    .iter()
                    .map(|&input_id| {
                        let input_idx = self.node_map[&input_id];
                        unsafe { self.nodes[input_idx].variable.tensor() }
                    })
                    .collect::<Vec<&Tensor<f32>>>();

                if let Some(output_grad) = grad {
                    let input_grads = OPERATOR_STORAGE.with(|ops| ops.borrow().get(function).unwrap().backward(&input_tensors, &output_grad)
                        .map_err(|e| MlError::StringError(format!("Failed to compute backward for function {:?}: {}", function, e))))?;

                    for (input_id, grad) in node.inputs.iter().zip(input_grads) {
                        let input_idx = self.node_map[input_id];
                        self.nodes[input_idx].variable.accumulate_grad(grad)?;
                    }


                    if !node.variable.requires_grad { node.variable.clear_grad(); }
                }
            }
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
            let tensor = unsafe{ var.tensor() };
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

#[cfg(feature = "enableVisualization")]
impl VisualizationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: Vec::new(),
            node_types: std::collections::HashMap::new(),
            node_labels: std::collections::HashMap::new(),
        }
    }

    fn compute_node_layers(&self) -> HashMap<String, usize> {
        use std::collections::{HashMap, HashSet, VecDeque};

        let mut layers = HashMap::new();
        let mut in_degree = HashMap::new();
        let mut adjacency = HashMap::new();

        // 그래프 구조 파싱
        for node in &self.nodes {
            in_degree.insert(node.clone(), 0);
            adjacency.insert(node.clone(), Vec::new());
        }

        // 엣지로부터 인접리스트와 입차수 계산
        for edge in &self.edges {
            if let Some(captures) = self.parse_edge(edge) {
                let from = captures.0;
                let to = captures.1;

                if let Some(adj) = adjacency.get_mut(&from) {
                    adj.push(to.clone());
                }
                if let Some(degree) = in_degree.get_mut(&to) {
                    *degree += 1;
                }
            }
        }

        // 위상 정렬을 통한 계층 할당
        let mut queue = VecDeque::new();

        // 입차수가 0인 노드들을 시작점으로 설정
        for (node, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back((node.clone(), 0));
                layers.insert(node.clone(), 0);
            }
        }

        while let Some((current, layer)) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            let new_layer = layer + 1;
                            layers.insert(neighbor.clone(), new_layer);
                            queue.push_back((neighbor.clone(), new_layer));
                        }
                    }
                }
            }
        }

        layers
    }

    // 엣지 문자열에서 from과 to 노드를 추출하는 헬퍼 메서드
    fn parse_edge(&self, edge: &str) -> Option<(String, String)> {
        // 정규표현식을 사용하지 않고 간단하게 파싱
        if let Some(arrow_pos) = edge.find(" -> ") {
            let from_part = &edge[..arrow_pos];
            let to_part = &edge[arrow_pos + 4..];

            // 큰따옴표 제거
            let from = from_part.trim().trim_start_matches("    \"").trim_end_matches("\"");
            let to = to_part.split_whitespace().next()?.trim_start_matches("\"").trim_end_matches("\"");

            Some((from.to_string(), to.to_string()))
        } else {
            None
        }
    }

    // 계층적 랭킹을 추가하는 메서드
    fn add_layered_ranking(&self, dot: &mut String, layers: &HashMap<String, usize>,
                           function_nodes: &[String], variable_nodes: &[String],
                           weight_nodes: &[String], bias_nodes: &[String],
                           activation_nodes: &[String], loss_nodes: &[String]) {
        // 최대 계층 수 계산
        let max_layer = layers.values().max().copied().unwrap_or(0);

        // 각 계층별로 노드 타입을 분리하여 배치
        for layer in 0..=max_layer {
            let mut layer_functions = Vec::new();
            let mut layer_variables = Vec::new();
            let mut layer_others = Vec::new();

            // 현재 계층의 노드들을 타입별로 분류
            for (node, &node_layer) in layers {
                if node_layer == layer {
                    if function_nodes.contains(node) || activation_nodes.contains(node) {
                        layer_functions.push(node.clone());
                    } else if variable_nodes.contains(node) {
                        layer_variables.push(node.clone());
                    } else if weight_nodes.contains(node) || bias_nodes.contains(node) || loss_nodes.contains(node) {
                        layer_others.push(node.clone());
                    }
                }
            }

            // 함수 노드들을 먼저 배치 (상단)
            if !layer_functions.is_empty() {
                dot.push_str(&format!(
                    "    {{ rank=same; {}; }}\n",
                    layer_functions.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join("; ")
                ));
            }

            // 변수 노드들을 중간에 배치
            if !layer_variables.is_empty() {
                dot.push_str(&format!(
                    "    {{ rank=same; {}; }}\n",
                    layer_variables.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join("; ")
                ));
            }

            // 기타 노드들을 하단에 배치
            // if !layer_others.is_empty() {
            //     dot.push_str(&format!(
            //         "    {{ rank=same; {}; }}\n",
            //         layer_others.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join("; ")
            //     ));
            // }

            // 같은 계층 내에서 함수 -> 변수 -> 기타 순서로 배치하는 보이지 않는 엣지 추가
            if layer_functions.len() > 0 && layer_variables.len() > 0 {
                dot.push_str(&format!(
                    "    \"{}\" -> \"{}\" [style=invis, weight=10];\n",
                    layer_functions[0], layer_variables[0]
                ));
            }
            if layer_variables.len() > 0 && layer_others.len() > 0 {
                dot.push_str(&format!(
                    "    \"{}\" -> \"{}\" [style=invis, weight=10];\n",
                    layer_variables[0], layer_others[0]
                ));
            }
        }
    }

    pub fn add_variable_node(&mut self, id: &str, label: &str, node_type: &NodeType) {
        self.nodes.insert(id.to_string());
        self.node_labels.insert(id.to_string(), label.to_string());
        self.node_types.insert(id.to_string(), node_type.clone());
    }

    pub fn add_weight_node(&mut self, id: &str, label: &str) {
        self.nodes.insert(id.to_string());
        self.node_labels.insert(id.to_string(), label.to_string());
        self.node_types.insert(id.to_string(), NodeType::Weight);
    }

    pub fn add_bias_node(&mut self, id: &str, label: &str) {
        self.nodes.insert(id.to_string());
        self.node_labels.insert(id.to_string(), label.to_string());
        self.node_types.insert(id.to_string(), NodeType::Bias);
    }

    pub fn add_loss_node(&mut self, id: &str, label: &str) {
        self.nodes.insert(id.to_string());
        self.node_labels.insert(id.to_string(), label.to_string());
        self.node_types.insert(id.to_string(), NodeType::Loss);
    }

    pub fn add_activation_node(&mut self, id: &str, label: &str) {
        self.nodes.insert(id.to_string());
        self.node_labels.insert(id.to_string(), label.to_string());
        self.node_types.insert(id.to_string(), NodeType::Activation);
    }

    // 핵심 개선: 고유한 함수 노드 ID 생성
    pub fn add_function_node(&mut self, id: &str, label: &str) {
        // 기존 방식대로 노드 추가
        self.nodes.insert(id.to_string());
        self.node_labels.insert(id.to_string(), label.to_string());

        match label {
            "Sigmoid" | "ReLU" | "Tanh" | "Softmax" | "Linear"
            => self.node_types.insert(id.to_string(), NodeType::Activation),
            "MSE"
            => self.node_types.insert(id.to_string(), NodeType::Loss),
            _
            => self.node_types.insert(id.to_string(), NodeType::Function)
        };
    }

    // 고유한 함수 노드 ID를 생성하는 새로운 메서드
    pub fn add_unique_function_node(&mut self, operator_name: &str, inputs: &[NodeId]) -> String {
        // 입력 노드들의 해시를 기반으로 고유 ID 생성
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        operator_name.hash(&mut hasher);
        for input in inputs {
            input.hash(&mut hasher);
        }
        let hash = hasher.finish();

        let unique_id = format!("{}_{:x}", operator_name, hash);

        // 중복 방지를 위한 추가 체크
        let final_id = if self.nodes.contains(&unique_id) {
            // 만약 해시 충돌이 발생하면 카운터 추가
            LABEL_COUNTERS.with(|counters| {
                let mut counters = counters.borrow_mut();
                let counter = counters.entry(operator_name.to_string()).or_insert(0);
                *counter += 1;
                format!("{}_{}_{}", operator_name, hash, counter)
            })
        } else {
            unique_id
        };

        self.add_function_node(&final_id, operator_name);
        final_id
    }

    // 연산자와 변수들 간의 관계를 등록하는 편의 메서드
    pub fn register_operation(&mut self, operator_name: &str, inputs: &[NodeId], output: NodeId) -> String {
        let function_id = self.add_unique_function_node(operator_name, inputs);

        // 입력들에서 함수로의 엣지 추가
        for input in inputs {
            self.add_edge(&format!("{:?}", input), &function_id, "data_flow");
        }

        // 함수에서 출력으로의 엣지 추가
        self.add_edge(&function_id, &format!("{:?}", output), "data_flow");

        function_id
    }

    // 백프로파게이션 그래디언트 플로우를 위한 메서드
    pub fn add_gradient_edge(&mut self, from: &str, to: &str) {
        self.add_edge(from, to, "gradient_flow");
    }

    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: &str) {
        let style = match edge_type {
            "data_flow" => "style=solid, color=\"#2E86AB\", penwidth=2",
            "gradient_flow" => "style=dashed, color=\"#A23B72\", penwidth=2",
            "control_flow" => "style=dotted, color=\"#F18F01\", penwidth=1",
            _ => "style=solid, color=black, penwidth=1",
        };

        self.edges.push(format!("    \"{}\" -> \"{}\" [{}];", from, to, style));
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.node_types.clear();
        self.node_labels.clear();

        // 카운터도 초기화
        LABEL_COUNTERS.with(|counters| {
            counters.borrow_mut().clear();
        });
    }


    // DOT 그래프 생성 (개선된 스타일)
    pub fn generate_dot(&self) -> String {
        let mut dot = String::from(
            "digraph ComputationGraph {\n\
            // 모던한 전체 스타일링\n\
            bgcolor=\"#FAFBFC\";\n\
            rankdir=LR;\n\
            splines=curved;\n\
            nodesep=1.0;\n\
            ranksep=2.5;\n\
            pad=0.5;\n\
            \n\
            // 기본 노드/엣지 스타일\n\
            node [fontname=\"SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial\", fontsize=11, margin=0.1];\n\
            edge [fontname=\"SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial\", fontsize=9, color=\"#6B7280\", penwidth=1.5];\n\n"
        );

        // 위상 정렬을 위한 준비
        let mut layers = self.compute_node_layers();

        // 노드 타입별 그룹화
        let mut input_nodes = Vec::new();
        let mut output_nodes = Vec::new();
        let mut function_nodes = Vec::new();
        let mut variable_nodes = Vec::new();
        let mut weight_nodes = Vec::new();
        let mut bias_nodes = Vec::new();
        let mut loss_nodes = Vec::new();
        let mut activation_nodes = Vec::new();

        for node_id in &self.nodes {
            let label = self.node_labels.get(node_id).unwrap_or(node_id);
            let node_type = self.node_types.get(node_id).unwrap_or(&NodeType::Variable);

            let (shape, style, color, font_color) = match node_type {
                NodeType::Input => {
                    input_nodes.push(node_id.clone());
                    ("house", "filled", "#10B981", "white")           // 에메랄드 그린
                },
                NodeType::Output => {
                    output_nodes.push(node_id.clone());
                    ("invhouse", "filled", "#EF4444", "white")        // 모던 레드
                },
                NodeType::Function => {
                    function_nodes.push(node_id.clone());
                    ("hexagon", "filled", "#3B82F6", "white")         // 모던 블루
                },
                NodeType::Variable => {
                    variable_nodes.push(node_id.clone());
                    ("ellipse", "filled", "#F59E0B", "white")         // 앰버 옐로우
                },
                NodeType::Weight => {
                    weight_nodes.push(node_id.clone());
                    ("diamond", "filled", "#8B5CF6", "white")         // 바이올렛 퍼플
                },
                NodeType::Bias => {
                    bias_nodes.push(node_id.clone());
                    ("circle", "filled", "#F97316", "white")          // 오렌지
                },
                NodeType::Loss => {
                    loss_nodes.push(node_id.clone());
                    ("octagon", "filled", "#EC4899", "white")         // 핑크
                },
                NodeType::Activation => {
                    activation_nodes.push(node_id.clone());
                    ("doublecircle", "filled", "#06B6D4", "white")    // 사이안
                }
            };

            // 함수 노드의 경우 원래 연산자 이름만 표시
            let display_label = if matches!(node_type, NodeType::Function | NodeType::Activation) {
                // ID에서 연산자 이름만 추출 (예: "Add_abc123_1" -> "Add")
                if let Some(underscore_pos) = node_id.find('_') {
                    &node_id[..underscore_pos]
                } else {
                    label
                }
            } else {
                label
            };

            dot.push_str(&format!(
                "    \"{}\" [label=\"{}\", shape={}, style=\"{}\", fillcolor=\"{}\", fontcolor=\"{}\"];\n",
                node_id, display_label, shape, style, color, font_color
            ));
        }

        // 계층적 레이아웃 설정
        self.add_layered_ranking(&mut dot, &layers, &function_nodes, &variable_nodes,
                                 &weight_nodes, &bias_nodes, &activation_nodes, &loss_nodes);

        // 입력과 출력은 명시적으로 처음과 끝에 배치
        if !input_nodes.is_empty() {
            dot.push_str(&format!(
                "    {{ rank=source; {}; }}\n",
                input_nodes.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join("; ")
            ));
        }

        if !output_nodes.is_empty() {
            dot.push_str(&format!(
                "    {{ rank=sink; {}; }}\n",
                output_nodes.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join("; ")
            ));
        }

        // 범례 추가
        dot.push_str("\n    // 범례 - 모던 스타일\n");
        dot.push_str("    subgraph cluster_legend {\n");
        dot.push_str("        label=<<B>Node Types</B>>;\n");
        dot.push_str("        style=\"filled,rounded\";\n");
        dot.push_str("        fillcolor=\"#FFFFFF\";\n");
        dot.push_str("        color=\"#E5E7EB\";\n");
        dot.push_str("        penwidth=1;\n");
        dot.push_str("        fontsize=14;\n");
        dot.push_str("        fontname=\"SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial\";\n");
        dot.push_str("        fontcolor=\"#374151\";\n");
        dot.push_str("        margin=12;\n");
        dot.push_str("        \n");

        // 범례 노드들을 두 열로 배치
        dot.push_str("        // 첫 번째 열\n");
        dot.push_str("        legend_input [label=<<B>Input</B>>, shape=house, style=\"filled\", fillcolor=\"#10B981\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        legend_func [label=<<B>Function</B>>, shape=hexagon, style=\"filled\", fillcolor=\"#3B82F6\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        legend_var [label=<<B>Variable</B>>, shape=ellipse, style=\"filled\", fillcolor=\"#F59E0B\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        legend_output [label=<<B>Output</B>>, shape=invhouse, style=\"filled\", fillcolor=\"#EF4444\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        \n");

        dot.push_str("        // 두 번째 열\n");
        dot.push_str("        legend_weight [label=<<B>Weight</B>>, shape=diamond, style=\"filled\", fillcolor=\"#8B5CF6\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        legend_bias [label=<<B>Bias</B>>, shape=circle, style=\"filled\", fillcolor=\"#F97316\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        legend_loss [label=<<B>Loss</B>>, shape=octagon, style=\"filled\", fillcolor=\"#EC4899\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        legend_activation [label=<<B>Activation</B>>, shape=doublecircle, style=\"filled\", fillcolor=\"#06B6D4\", fontcolor=\"white\", width=1.2, height=0.6];\n");
        dot.push_str("        \n");

        // 범례 레이아웃 - 그리드 형태로 배치
        dot.push_str("        // 범례 레이아웃\n");
        dot.push_str("        { rank=same; legend_input; legend_weight; }\n");
        dot.push_str("        { rank=same; legend_func; legend_bias; }\n");
        dot.push_str("        { rank=same; legend_var; legend_loss; }\n");
        dot.push_str("        { rank=same; legend_output; legend_activation; }\n");
        dot.push_str("        \n");

        dot.push_str("        legend_input -> legend_func -> legend_var -> legend_output [style=invis];\n");
        dot.push_str("        legend_weight -> legend_bias -> legend_loss -> legend_activation [style=invis];\n");
        dot.push_str("    }\n\n");

        // 엣지 추가
        for edge in &self.edges {
            dot.push_str(edge);
            dot.push('\n');
        }

        dot.push_str("}\n");
        dot
    }

    // 그래프 통계 정보 제공
    #[cfg(feature = "enableVisualization")]
    pub fn get_graph_stats() -> (usize, usize) {
        VISUALIZATION_GRAPH.with(|viz_graph| {
            let viz = viz_graph.borrow();
            (viz.nodes.len(), viz.edges.len())
        })
    }

    // 개선된 DOT 그래프 생성
    #[cfg(feature = "enableVisualization")]
    pub fn get_dot_graph() -> String {
        VISUALIZATION_GRAPH.with(|viz_graph| {
            viz_graph.borrow().generate_dot()
        })
    }

    // DOT 그래프를 파일로 저장 (SVG도 지원)
    #[cfg(feature = "enableVisualization")]
    pub fn save_graph<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
        let dot = Self::get_dot_graph();
        std::fs::write(path, dot)
    }

    // SVG로 직접 렌더링 (graphviz가 설치된 경우)
    #[cfg(feature = "enableVisualization")]
    pub fn render_to_svg<P: AsRef<std::path::Path>>(output_path: P) -> std::io::Result<()> {
        let dot = Self::get_dot_graph();

        // 임시 DOT 파일 생성
        let temp_dot_path = std::env::temp_dir().join("computation_graph.dot");
        std::fs::write(&temp_dot_path, dot)?;

        // graphviz로 SVG 렌더링
        let output = std::process::Command::new("dot")
            .arg("-Tsvg")
            .arg(&temp_dot_path)
            .arg("-o")
            .arg(output_path.as_ref())
            .output();

        // 임시 파일 정리
        let _ = std::fs::remove_file(&temp_dot_path);

        match output {
            Ok(output) if output.status.success() => Ok(()),
            Ok(output) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Graphviz error: {}", String::from_utf8_lossy(&output.stderr))
            )),
            Err(e) => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Graphviz not found. Please install graphviz: {}", e)
            )),
        }
    }
}



impl<F: Function<f32> + Clone +  'static> AutogradFunction<f32> for F {
    fn apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>> {
        let tensors: Vec<&Tensor<f32>> = inputs
            .iter()
            .map(|&var| unsafe{ var.tensor() }).collect();
        let input = Arc::new(Variable::new(self.forward(&tensors)?.remove(0)));

        #[cfg(feature = "enableBackpropagation")]
        {
            OPERATOR_STORAGE.with(|g_ops| {
                let mut ops = g_ops.borrow_mut();
                let type_name = self.type_name().to_string();
                match ops.contains_key(&type_name) { 
                    true => input.clone().with_grad_fn(self.type_name(), inputs),
                    false => {
                        input.clone().with_grad_fn(&type_name, inputs);
                        ops.insert(type_name, Arc::new(self.clone()));
                    }
                }
            });
            return Ok(input)
        }
        // 정적계산 그래프를 통해서 메모리 효율성을 증대하려 했으나, 사전에 텐서의 정보가 주입되지 않으면 메모리 관리가 어려워,
        // 무산될것으로 예상되며, 정적, 동적계산그래프를 전환 가능하도록 향후 추가될것으로 생각하고있음.
        // 따라서 매 계산마다 계산그래프를 갱신하는 현재 구조를 유지하게될것 같은데, 이는 계산그래프 갱신으로 인한 오버헤드가 예상됨.
        // 솔직히 어느 방식을 선택해야할지잘 모르겠음.

        Ok(input)
    }
}