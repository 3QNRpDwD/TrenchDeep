use std::sync::Mutex;
use crate::{MlError, MlResult};
use crate::MlError::StringError;
use crate::tensor::*;
use crate::tensor::operators::Function;

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
    pub(crate) static COMPUTATION_GRAPH: Mutex<ComputationGraph<f32>> = Mutex::new(ComputationGraph::new());
}

impl Variable<f32> {
    pub fn new(tensor: Tensor<f32>) -> Self {
        Variable {
            tensor,
            requires_grad: cfg!(feature = "requires_grad"),

            #[cfg(feature = "enable_backpropagation")]
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
    /// - `enable_backpropagation` 기능 전용 메소드
    ///
    /// # 반환 값
    /// - Option<Tensor<f32>>: 현재 저장된 그래디언트 또는 None
    #[cfg(feature = "enable_backpropagation")]
    pub fn grad(&self) -> Option<Tensor<f32>> {
        self.grad.borrow().clone()
    }

    /// 그래디언트 값 직접 설정
    ///
    /// # 특징 동작
    /// - `enable_backpropagation` 기능 전용 메소드
    /// - 기존 그래디언트 값을 완전히 대체
    ///
    /// # 파라미터
    /// - grad: 설정할 새로운 그래디언트 텐서
    #[cfg(feature = "enable_backpropagation")]
    pub fn set_grad(&self, grad: Tensor<f32>) {
        *self.grad.borrow_mut() = Some(grad);
    }

    /// 그래디언트 값 초기화
    ///
    /// # 특징 동작
    /// - `enable_backpropagation` 기능 전용 메소드
    /// - 기존 그래디언트 값을 삭제
    ///
    #[cfg(feature = "enable_backpropagation")]
    pub fn clear_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// 그래디언트 값 누적 추가
    ///
    /// # 특징 동작
    /// - `enable_backpropagation` 기능 전용 메소드
    /// - 기존 그래디언트와 새로운 그래디언트를 요소별 합산
    ///
    /// # 오류 사항
    /// - 텐서 모양 불일치 시 에러 반환
    ///
    /// # 파라미터
    /// - new_grad: 추가할 그래디언트 텐서
    #[cfg(feature = "enable_backpropagation")]
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
    #[cfg(feature = "enable_backpropagation")]
    pub fn backward(self: &Arc<Self>) -> MlResult<()> {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();
            let node_id= Arc::as_ptr(&self);

            match graph.nodes.contains_key(&node_id) {
                true => {
                    if !graph.sorted { graph.topological_sort(); }
                    graph.backward(node_id)
                },
                false => Err(StringError("계산 그래프가 생성되지 않았습니다.".to_string())),
            }
        })
    }
}

#[cfg(feature = "enable_backpropagation")]
impl ComputationGraph<f32> {
    /// 새로운 계산 그래프를 생성합니다.
    ///
    /// 이 메서드는 노드와 관련 데이터를 저장할 빈 `ComputationGraph` 인스턴스를 초기화합니다.
    ///
    /// # 반환값
    /// - `Self`: 초기화된 `ComputationGraph` 인스턴스
    pub(crate) fn new() -> Self {
        Self {
            nodes: HashMap::new(),
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
    #[cfg(feature = "enable_backpropagation")]
    pub(crate) fn add_input(&mut self, variable: Arc<Variable<f32>>, id: NodeId<f32>) -> NodeId<f32> {
        let node = ComputationNode {
            id,
            ref_count: 0,
            variable,
            function: None,
            // output: None,
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
    #[cfg(feature = "enable_backpropagation")]
    pub(crate) fn add_operation(&mut self, variable: Arc<Variable<f32>>, function: Arc<dyn Function<f32>>,  inputs: Vec<NodeId<f32>>) -> NodeId<f32> {
        let id = Arc::as_ptr(&variable);

        let node = ComputationNode {
            id,
            ref_count: 0,
            variable,
            function: Some(function),
            // output,
            inputs,
        };

        self.nodes.insert(id, node);
        self.sorted = false;
        id
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
    #[cfg(feature = "enable_backpropagation")]
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

        // Traverse in reverse topological order
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


/// 자동 미분(autograd)을 지원하는 함수 트레잇
///
/// 이 트레잇은 Function<f32>와 Clone을 구현하는 타입에 자동 미분 기능을 추가합니다.
/// 신경망의 순전파(forward pass)와 역전파(backward pass)를 연결하는 함수를 생성하는 역할을 수행합니다.
///
/// # 주요 기능
/// - 입력 변수들로부터 계산 결과 생성
/// - enable_backpropagation 기능 활성화 시 자동으로 역전파 그래프 구성
/// - 연산 결과에 그라데이션 함수 연결
///
/// # 메서드
///
/// ## apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>>
/// 입력 변수 슬라이스를 받아 다음 단계를 처리합니다:
/// 1. 모든 입력 변수에서 텐서 추출
/// 2. forward 메소드를 호출하여 순전파 수행
/// 3. (조건부) 역전파를 위한 계산 그래프 구성
///
/// ### 기능 플래그 동작
/// - #[cfg(feature = "enable_backpropagation")] 활성화 시:
/// - 단일 입력/출력 연산만 지원 (입력 슬라이스 길이 == 1)
/// - 계산 결과에 그라데이션 함수와 입력 변수들을 연결
/// - 기능 비활성화 시 기본 텐서 연산만 수행
///
/// # 구현 참고사항
/// - Function<f32> + Clone + 'static을 구현하는 모든 타입에 자동 구현 제공
/// - 사용자 정의 연산 구현 시 forward 메소드의 정확한 구현이 필요
///
/// # 제약 사항
/// - 현재 버전에서는 다중 입력/출력에 대한 역전파를 지원하지 않음
/// - f32 데이터 타입 전용으로 특화됨
pub trait AutogradFunction: Function<f32> + Clone where Self: 'static {
    fn apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>> {
        let tensors: Vec<&Tensor<f32>> = inputs.iter().map(|&var| var.tensor()).collect();
        let results = Variable::new(self.forward(&tensors)?.remove(0));

        #[cfg(feature = "enable_backpropagation")]
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

impl<F: Function<f32> + Clone +  'static> AutogradFunction for F {}