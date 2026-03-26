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
                    graph.add_input(input_var);
                }
                input_id
            }).collect();
            graph.add_operation(self, operator_name, input_ids);
        })
    }
}

#[cfg(feature = "enableBackward")]
impl ComputationGraph {

    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            adjacency_list: Vec::new(),
            reverse_adjacency: Vec::new(),
            topo_order: Vec::new(),
            is_sorted: false,
        }
    }


    pub(crate) fn add_input(&mut self, variable: &Variable) -> NodeId {
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
            tensor: variable.tensor().clone(),
            grad: variable.grad().clone(),
            requires_grad: variable.is_retain_grad(),
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

    pub(crate) fn add_operation(&mut self, variable: &Variable, operator_name: &str, inputs: Vec<NodeId>) -> NodeId {
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
            tensor: variable.tensor().clone(),
            grad: variable.grad().clone(),
            requires_grad: variable.is_retain_grad(),
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
        }
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
        for node in &self.nodes {
            Self::clear_node_grad(&node.grad);
        }

        // ── 2. 출력 노드 그래디언트 주입 ────────────────────────────────────
        let output_idx = *self.node_map.get(&output_id)
            .ok_or_else(|| MlError::StringError("Output node not found".to_string()))?;
        let output_node = &self.nodes[output_idx];

        let mut ones = GlobalTensor::from_vec(
            vec![1.0; output_node.tensor.shape().iter().product()],
            output_node.tensor.shape()
        )?;
        ones.dirty = true;
        output_node.grad.replace(ones);

        // ── 3. 역전파 루프 ───────────────────────────────────────────────────
        for &node_idx in self.topo_order.iter().rev() {
            let node = &self.nodes[node_idx];

            if node.function.is_none() || !Self::is_node_grad_dirty(&node.grad) {
                continue;
            }

            let grad = &node.grad;
            let function = node.function.as_ref().unwrap();

            let input_tensors: Vec<&dyn TensorBase> = node.inputs
                .iter()
                .map(|&input_id| {
                    let input_idx = self.node_map[&input_id];
                    &self.nodes[input_idx].tensor as &dyn TensorBase
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

            for (input_id, input_grad) in node.inputs.iter().zip(input_grads) {
                let input_idx = self.node_map[input_id];
                let input_node = &self.nodes[input_idx];
                Self::accumulate_node_grad(&input_node.grad, input_grad)?;
            }

            if !node.requires_grad {
                Self::clear_node_grad(&node.grad);
            }
        }
        Ok(())
    }

    #[cfg(feature = "enableBackward")]
    fn clear_node_grad(grad: &Tensor) {
        TENSOR_STORAGE.with_borrow_mut(|storage| {
            if let Some(gt) = storage.get_mut(&grad.id()) {
                if !gt.dirty { return; }
                gt.data.iter_mut().for_each(|x| *x = 0.0);
                gt.dirty = false;
            }
        });
    }

    #[cfg(feature = "enableBackward")]
    fn is_node_grad_dirty(grad: &Tensor) -> bool {
        TENSOR_STORAGE.with(|storage| {
            storage.borrow()
                .get(&grad.id())
                .map(|gt| gt.dirty)
                .unwrap_or(false)
        })
    }

    #[cfg(feature = "enableBackward")]
    fn accumulate_node_grad(grad: &Tensor, new_grad: GlobalTensor<f32>) -> MlResult<()> {
        if grad.data().is_empty() {
            let mut buf = GlobalTensor::from_vec(
                new_grad.data.clone(),
                &new_grad.shape,
            )?;
            buf.dirty = true;
            grad.replace(buf);
        } else {
            if grad.shape() != new_grad.shape.as_slice() {
                return Err(MlError::TensorError(TensorError::InvalidShape {
                    expected: grad.shape().to_vec(),
                    got: new_grad.shape.to_vec(),
                }));
            }
            let new_data: &[f32] = &new_grad.data;
            TENSOR_STORAGE.with_borrow_mut(|storage| {
                if let Some(gt) = storage.get_mut(&grad.id()) {
                    gt.data.iter_mut()
                        .zip(new_data.iter())
                        .for_each(|(d, &v)| *d += v);
                    gt.dirty = true;
                }
            });
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.node_map.clear();
        self.adjacency_list.clear();
        self.reverse_adjacency.clear();
        self.topo_order.clear();
        self.is_sorted = false;
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
            let tensor = &node.tensor;
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
            output.with_grad_fn(self.name(), inputs);
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
            output.with_grad_fn(self.name(), inputs);
            return Ok(output)
        }

        Ok(output)
    }

    /// forward()[0] 을 출력으로, forward()[1..] 을 saved tensors로 처리.
    ///
    /// with_grad_fn에 넘기는 inputs = 원래 inputs + saved tensors 이므로
    /// backward(targets, grad) 호출 시 targets 슬라이스가
    /// [원래 inputs..., saved tensors...] 순서로 구성된다.
    fn apply_with_saved(&mut self, inputs: &[&Variable]) -> MlResult<Variable> {
        let tensors: Vec<&dyn TensorBase> = inputs
            .iter()
            .map(|&var| var.tensor() as &dyn TensorBase)
            .collect();

        let mut outputs = self.forward(&tensors)?;

        // forward()[0] = 최종 출력
        let primary = Variable::new(outputs.remove(0).to_id()?);

        #[cfg(feature = "enableBackward")]
        {
            // forward()[1..] = saved tensors → Variable로 변환 후 그래프에 등록
            let saved: Vec<Variable> = outputs
                .into_iter()
                .map(|gt| Variable::new(gt.to_id().unwrap()))
                .collect();

            // [원래 inputs..., saved tensors...] 로 확장해서 with_grad_fn 호출
            let mut extended: Vec<&Variable> = inputs.to_vec();
            extended.extend(saved.iter());

            primary.with_grad_fn(self.name(), &extended);
            return Ok(primary);
        }

        Ok(primary)
    }
}
