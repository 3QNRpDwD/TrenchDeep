use super::*;

#[cfg(feature = "enableBackward")]
impl Variable {
    pub fn tpye_name(&self) -> String {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown").replace("<f32>", "")
    }

    pub fn with_grad_fn(&self, operator_name: &str, inputs: &[&Variable]) {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            let input_ids: Vec<NodeId> = inputs.iter().map(|&input_var| {
                let input_id = input_var.node_id();
                if !graph.node_map.contains_key(&input_id) {
                    graph.add_input(input_var.clone());
                }
                input_id
            }).collect();
            graph.add_operation(self.clone(), operator_name, input_ids);
        })
    }
}

#[cfg(feature = "enableBackward")]
impl ComputationGraph {

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


    pub(crate) fn add_input(&mut self, variable: Variable) -> NodeId {
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

    pub(crate) fn add_operation(&mut self, variable: Variable, operator_name: &str, inputs: Vec<NodeId>) -> NodeId {
        #[cfg(feature = "enableVisualization")]
        VISUALIZATION_GRAPH.with(|viz_graph| {
            let mut viz = viz_graph.borrow_mut();
            viz.register_operation(operator_name, &inputs, variable.node_id());
            viz.add_variable_node(&format!("{:?}", variable.node_id()), variable.label(), variable.node_type());
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
            let mut g = graph.lock().unwrap();
            g.clear();
        });
        #[cfg(feature = "enableVisualization")]
        {
            VISUALIZATION_GRAPH.with(|viz_graph| {
                viz_graph.borrow_mut().clear();
            });
            // 시각화 라벨 카운터도 리셋하여 다음 그래프 생성 시 레이블이 깨끗하게 시작되도록 함
        }
        // tracing::debug!("Computation graph and visualization state have been reset.");
    }

    pub(crate) fn ensure_topological_sort(&mut self) {
        if !self.is_sorted {
            self.topological_sort();
        }
    }

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

    #[cfg(feature = "enableBackward")]
    pub(crate) fn backward(&mut self, output_id: NodeId) -> MlResult<()> {
        // ── 1. 전체 그래디언트 초기화 ────────────────────────────────────────
        // lear_grad()는 dirty=false이면 즉시 반환(no-op)하므로,
        // new_empty() 상태의 노드들에 대한 불필요한 TENSOR_STORAGE 조회가 없다.
        for node in &self.nodes {
            node.variable.clear_grad();
        }

        // ── 2. 출력 노드 그래디언트 주입 ────────────────────────────────────
        let output_idx = *self.node_map.get(&output_id)
            .ok_or_else(|| MlError::StringError("Output node not found".to_string()))?;
        let output_var = &self.nodes[output_idx].variable;

        // 원래 코드: if output_var.grad().is_empty() { ... }
        //   → clear_grad() 직후 data는 0으로 채워져 있어 is_empty()가 false를
        //     반환할 수 있음 (원소가 모두 0이어도 len > 0이면 false)
        //   → 조건 분기 자체가 불안정
        //
        // 변경 후: 조건 없이 항상 set_grad()
        //   → clear_grad()가 dirty=false로 리셋했으므로 항상 새로 주입이 맞다.
        //   → 출력 노드는 backward당 1개, 오버헤드 무시 가능.
        let ones = GlobalTensor::from_vec(
            vec![1.0; output_var.tensor().shape().iter().product()],
            output_var.tensor().shape()
        )?;
        output_var.set_grad(ones); // dirty = true

        // ── 3. 역전파 루프 ───────────────────────────────────────────────────
        for &node_idx in self.topo_order.iter().rev() {
            let node = &self.nodes[node_idx];
            let var = &node.variable;

            // 원래 코드: grad.is_empty() — 원소 전체 순회 O(n)
            //   data.iter().all(|&d| d == 0.0)
            //
            // 변경 후: !var.is_grad_dirty() — Cell<bool> 읽기 O(1)
            //   set_grad()와 accumulate_grad() 호출 시 true
            //   clear_grad() 호출 시 false
            //   → 출력 노드에서 도달하지 못한 노드는 dirty=false로 자동 스킵
            if node.function.is_none() || !var.is_grad_dirty() {
                continue;
            }

            let grad = var.grad();
            let function = node.function.as_ref().unwrap();

            let input_tensors: Vec<&dyn TensorBase> = node.inputs
                .iter()
                .map(|&input_id| {
                    let input_idx = self.node_map[&input_id];
                    self.nodes[input_idx].variable.tensor() as &dyn TensorBase
                })
                .collect();

            let input_grads = OPERATOR_STORAGE.with(|ops| {
                ops.borrow_mut()
                    .get_mut(function)
                    .unwrap()
                    .backward(&input_tensors, grad)
                    .map_err(|e| MlError::StringError(
                        format!("Failed to compute backward for function {:?}: {}", function, e)
                    ))
            })?;

            for (input_id, grad) in node.inputs.iter().zip(input_grads) {
                let input_idx = self.node_map[input_id];
                let input_node = &self.nodes[input_idx];
                // accumulate_grad: 첫 호출이면 버퍼 할당, 이후 in-place 덧셈
                // dirty = true 로 설정됨
                input_node.variable.accumulate_grad(grad.to_id()?)?;
            }

            // retain_grad=false인 중간 노드는 backward 직후 grad 버퍼를 제로화.
            // 버퍼는 해제하지 않으므로 다음 backward에서 재사용된다.
            if !var.is_retain_grad() {
                var.clear_grad(); // dirty = false
            }
        }
        Ok(())
    }
    
    // #[cfg(feature = "enableBackward")]
    // pub(crate) fn backward(&mut self, output_id: NodeId) -> MlResult<()> {
    //     for node in &self.nodes {
    //         node.variable.clear_grad();
    //     }
    //     // Set output node's gradient to 1.0
    //     let output_idx = *self.node_map.get(&output_id)
    //         .ok_or_else(|| MlError::StringError("Output node not found".to_string()))?;
    //     let output_var = &self.nodes[output_idx].variable;
    //     if output_var.grad().is_empty() {
    //         let grad = GlobalTensor::from_vec(
    //             vec![1.0; output_var.tensor().shape().iter().product()],
    //             output_var.tensor().shape()
    //         )?;
    //         output_var.set_grad(grad);
    //     }
    // 
    //     // 위상 정렬된 순서의 역순으로 순회
    //     for &node_idx in self.topo_order.iter().rev() {
    //         let node = &self.nodes[node_idx];
    //         let var = &node.variable;
    //         let grad = var.grad();
    //         if node.function.is_none() || grad.is_empty() { continue; }
    //         let function = node.function.as_ref().unwrap();
    //         let input_tensors: Vec<&dyn TensorBase> = node.inputs
    //             .iter()
    //             .map(|&input_id| {
    //                 let input_idx = self.node_map[&input_id];
    //                 self.nodes[input_idx].variable.tensor() as &dyn TensorBase
    //             })
    //             .collect::<Vec<&dyn TensorBase>>();
    // 
    //         let input_grads = OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut(function).unwrap().backward(&input_tensors, grad)
    //             .map_err(|e| MlError::StringError(format!("Failed to compute backward for function {:?}: {}", function, e))))?;
    // 
    //         for (input_id, grad) in node.inputs.iter().zip(input_grads) {
    //             let input_idx = self.node_map[input_id];
    //             let input_node = &self.nodes[input_idx];
    //             input_node.variable.accumulate_grad(grad.to_id()?)?;
    //         }
    //         
    //         if !var.is_retain_grad() {
    //             var.clear_grad();
    //         }
    //     }
    //     Ok(())
    // }

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
    fn apply(&mut self, inputs: &[&Variable]) -> MlResult<Variable> {
        let tensors: Vec<&dyn TensorBase> = inputs
            .iter()
            .map(|&var| var.tensor() as &dyn TensorBase)
            .collect::<Vec<&dyn TensorBase>>();
        let output = Variable::new(self.forward(&tensors)?.remove(0).to_id()?);

        #[cfg(feature = "enableBackward")]
        {
            output.clone().with_grad_fn(self.name(), inputs);
            return Ok(output)
        }

        Ok(output)
    }

    fn apply_with_label(&mut self, inputs: &[&Variable], label: &str) -> MlResult<Variable> {
        let tensors: Vec<&dyn TensorBase> = inputs
            .iter()
            .map(|&var| var.tensor() as &dyn TensorBase)
            .collect::<Vec<&dyn TensorBase>>();
        let output = crate::var_with_label!(self.forward(&tensors)?.remove(0).to_id()?, label);
        #[cfg(feature = "enableBackward")]
        {
            output.clone().with_grad_fn(self.name(), inputs);
            return Ok(output)
        }

        Ok(output)
    }
}