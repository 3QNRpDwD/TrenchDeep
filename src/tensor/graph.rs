use super::*;


// 전역 계산 그래프 (스레드 로컬)
#[cfg(feature = "enableBackpropagation")]
thread_local! {
    pub(crate) static COMPUTATION_GRAPH: std::sync::Mutex<ComputationGraph<f32>> = std::sync::Mutex::new(ComputationGraph::new());
    #[cfg(feature = "enableVisualization")]
    pub(crate) static VISUALIZATION_GRAPH: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());

}

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
    pub fn with_grad_fn(self: Arc<Self>, function: Arc<dyn Function<f32>>, inputs: &[&Arc<Variable<f32>>]) {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            // 입력 노드 ID 찾기 또는 추가
            let input_ids: Vec<NodeId<f32>> = inputs.iter().map(|&input_var| {
                let input_id = &Arc::as_ptr(input_var);
                // println!("graph: {:?}", graph);
                // println!("{:?} - eq: {:?}", input_id, graph.nodes.contains_key(input_id));
                // 이미 그래프에 있는지 확인
                if graph.nodes.contains_key(input_id) {
                    return *input_id;
                }

                // 없으면 추가
                // 현재 경사하강법등의 기존 텐서의 수정이 불가피한 메서드를 사용할때 계속해서 새로운 텐서를 만들기 때문에,
                // 기존의 생성된 텐서는 더이상 사용되지 않음에도, 계산그래프상에 남아있으며, 이로 인해 계산 그래프 자체가 거대해지고 검색자체도 굉장히 느려지는 현상이 발생함.
                // 이를 해결하려면 단순히 텐서를 비교하는것이 아니라, 메모리값을 비교후. 메모리값이 같은데 내부값이 다를 경우, 업데이트하는 방식을 사용하거나,
                // 텐서 자체를 복사하는것이 아닌 메모리값을 계산그래프에 추가하는등의 방식으로, 텐서와 계산그래프의 수정과 연동이 가능하도록 개선해야될듯함.
                // 이에 대한 자세한 해결책을 시급히 만들어야함.

                graph.add_input(input_var.clone(), *input_id)
            }).collect();


            // 원래 고유한 아이디를 만들어서 계산그래프를 구성했으나, 현재 연산구조의 특성상 텐서의 포인터를 노드의 키값으로 설정하는것이
            // 같은 효과를 내면서도, 훨신 강력한 성능을 이끌어낼것으로 생각되어, 변경했으며, 기존보다 약 1.8배가량 성능이 향상된것으로 보임.
            // 또한, 이같은 변화로, 향후 개선돠어야할 계산그래프의 쓰레기 텐서(더이상 연산에 사용되지 않는 텐서)의 발생을 줄이는데 도움이 될것으로 보이며,
            // 계산그래프의 수정또한 더욱 쉽게 가능할것으로 보임.
            graph.add_operation(self, function, input_ids)

        });
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
            let node_id= Arc::as_ptr(&self);

            match graph.nodes.contains_key(&node_id) {
                true => {
                    if !graph.sorted { graph.topological_sort(); }
                    graph.backward(node_id)
                },
                false => Err(MlError::StringError("계산 그래프가 생성되지 않았습니다.".to_string())),
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
    pub(crate) fn new() -> Self {
        Self {
            nodes: std::collections::HashMap::new(),
            topo_sorted: Vec::new(),
            sorted: false,
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
    pub(crate) fn add_input(&mut self, variable: Arc<Variable<f32>>, id: NodeId<f32>) -> NodeId<f32> {
        #[cfg(feature = "enableVisualization")]
        self._dot_var(id, variable.tpye_name().as_str());
        let node = ComputationNode {
            id,
            variable,
            function: None,
            inputs: Vec::new(),
        };

        self.nodes.insert(id, node);
        self.sorted = false;
        id
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
    pub(crate) fn add_operation(&mut self, variable: Arc<Variable<f32>>, function: Arc<dyn Function<f32>>,  inputs: Vec<NodeId<f32>>) -> NodeId<f32> {
        let func_id = Arc::as_ptr(&function);
        let output_id = Arc::as_ptr(&variable);

        #[cfg(feature = "enableVisualization")]
        {
            self._dot_func(func_id, function.type_name());
            self._dot_var(output_id, variable.tpye_name().as_str());
            VISUALIZATION_GRAPH.with(|dot_graph| {
                for input_id in &inputs {
                    dot_graph.borrow_mut().push_str(&format!("    \"{:?}\" -> \"{:?}\" [style=dashed, color=blue, arrowhead=vee];\n", input_id, func_id));
                }
                dot_graph.borrow_mut().push_str(&format!("    \"{:?}\" -> \"{:?}\" [style=solid, color=red, arrowtail=vee];\n", func_id, output_id));
            });
        }

        let node = ComputationNode {
            id: output_id,
            variable,
            function: Some(function),
            inputs,
        };

        self.nodes.insert(output_id, node);
        self.sorted = false;
        output_id
    }

    // DOT 노드 정보 추가
    // When adding nodes/functions, continue building VISUALIZATION_GRAPH with raw lines:
    #[cfg(feature = "enableVisualization")]
    pub(crate) fn _dot_var(&self, dot_id: NodeId<f32>, label: &str) {
        VISUALIZATION_GRAPH.with(|graph| {
            graph.borrow_mut().push_str(&format!(
                "    \"{:?}\" [label=\"{}\", style=filled, fillcolor=orange];\n",
                dot_id, label
            ));
        });
    }

    #[cfg(feature = "enableVisualization")]
    pub(crate) fn _dot_func(&self, dot_id: FuncId<f32>, label: &str) {
        VISUALIZATION_GRAPH.with(|graph| {
            graph.borrow_mut().push_str(&format!(
                "    \"{:?}\" [label=\"{}\", shape=box, style=filled, fillcolor=lightblue];\n",
                dot_id, label
            ));
        });
    }


    /// Returns a DOT-format string for the current graph
    #[cfg(feature = "enableVisualization")]
    pub fn get_dot_graph() -> String {
        VISUALIZATION_GRAPH.with(|dot_graph| {
            let body = dot_graph.borrow();
            // Build header with global graph attributes
            let mut dot = String::from(
                "digraph ComputationGraph {\n      // start graph
                    splines=ortho;\n                  // orthogonal edges
                    node [shape=ellipse, style=filled, fillcolor=lightgoldenrod1, fontsize=10];\n"
            );

            // Identify leaf nodes (no inputs) and pin them at the top
            COMPUTATION_GRAPH.with(|graph| {
                let leaves: Vec<String> = graph.lock().unwrap().nodes.values()
                    .filter(|n| n.inputs.is_empty())
                    .map(|n| format!("\"{:?}\"; ", n.id))
                    .collect();
                if !leaves.is_empty() {
                    dot.push_str("    { rank=source; ");
                    dot.push_str(&leaves.concat());
                    dot.push_str("}\n");
                }
            });

            // Append the body of node/edge definitions
            dot.push_str(&body);

            // Close graph
            dot.push_str("}\n");
            dot
        })
    }

    /// Save the DOT graph to a file
    #[cfg(feature = "enableVisualization")]
    pub fn save_graph<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
        let dot = ComputationGraph::get_dot_graph();
        std::fs::write(path, dot)
    }

    /// 그래프 초기화 (필요 시)
    pub fn reset_graph() {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph_guard = graph.lock().unwrap();
            *graph_guard = ComputationGraph::new();
        });
        #[cfg(feature = "enableVisualization")]
        {
            VISUALIZATION_GRAPH.with(|dot_graph| {
                *dot_graph.borrow_mut() = String::new();
            });
        }
    }

    /// 계산 그래프의 노드들을 위상 정렬(Topological Sort)합니다.
    ///
    /// 이 메서드는 그래프의 노드들을 의존성 순서대로 정렬하여 역전파를 위한 준비를 합니다.
    /// 이미 정렬된 경우에는 아무 작업도 수행하지 않습니다.
    pub(crate) fn topological_sort(&mut self) {
        if self.sorted {
            return;
        }

        let mut result = Vec::new();
        let mut in_degree = std::collections::HashMap::new();
        let mut queue = std::collections::VecDeque::new();

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
    pub(crate) fn backward(&self, output_id: NodeId<f32>) -> MlResult<()> {
        // Clear gradients for all nodes
        for (_, node) in &self.nodes {
            node.variable.clear_grad();
        }

        // Set output node's gradient to 1.0
        let output_var = &self.nodes.get(&output_id).ok_or("Output node not found.")?.variable;
        if output_var.grad().is_none() {
            let grad = Tensor::from_vec(
                vec![1.0; output_var.tensor.shape().iter().product()],
                output_var.tensor.shape()
            )?;
            output_var.set_grad(grad);
        }

        // 위상 정렬된 순서의 역순으로 순회
        for &node_id in self.topo_sorted.iter().rev() {
            let node = self.nodes.get(&node_id).unwrap();
            if node.variable.grad().is_none() || node.function.is_none() { continue; }

            if let Some(function) = &node.function {
                let inputs_tensor: Vec<&Tensor<f32>> = node.inputs
                    .iter()
                    .map(|&input_id| self.nodes.get(&input_id).unwrap().variable.tensor())
                    .collect();

                let gradients = function.backward(&inputs_tensor, &node.variable.grad().unwrap())
                    .map_err(|e| format!("Backward failure: {:?}", e))?;

                for (input_id, grad) in node.inputs.iter().zip(gradients) {
                    self.nodes.get(input_id).unwrap().variable.accumulate_grad(grad)?;
                    if !node.variable.requires_grad { node.variable.clear_grad(); }
                }
            }
        }
        Ok(())
    }
}

impl<F: Function<f32> + Clone +  'static> AutogradFunction<f32> for F {
    fn apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>> {
        let tensors: Vec<&Tensor<f32>> = inputs.iter().map(|&var| var.tensor()).collect();
        let results = Variable::new(self.forward(&tensors)?.remove(0));

        #[cfg(feature = "enableBackpropagation")]
        {
            let result = Arc::new(results);
            result.clone().with_grad_fn(Arc::new(self.clone()), inputs);
            return Ok(result)
        }
        // 정적계산 그래프를 통해서 메모리 효율성을 증대하려 했으나, 사전에 텐서의 정보가 주입되지 않으면 메모리 관리가 어려워,
        // 무산될것으로 예상되며, 정적, 동적계산그래프를 전환 가능하도록 향후 추가될것으로 생각하고있음.
        // 따라서 매 계산마다 계산그래프를 갱신하는 현재 구조를 유지하게될것 같은데, 이는 계산그래프 갱신으로 인한 오버헤드가 예상됨.
        // 솔직히 어느 방식을 선택해야할지잘 모르겠음.

        Ok(Arc::new(results))
    }
}