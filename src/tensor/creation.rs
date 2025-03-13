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
            let mut new_data = Vec::with_capacity(existing_grad.data().len());
            for (a, b) in existing_grad.data().iter().zip(new_grad.data().iter()) {
                new_data.push(a + b);
            }

            let accumulated_grad = Tensor::from_vec(new_data, existing_grad.shape())
                .map_err(|e| format!("failed gradient accumulation: {:?}", e))?;

            *grad_ref = Some(accumulated_grad);
        } else {
            *grad_ref = Some(new_grad);
        }

        Ok(())
    }

    /// 계산 그래프에 연산 노드 연결
    ///
    /// # 특징 동작
    /// - `enable_backpropagation` 기능 전용 메소드
    /// - 역전파 시 사용할 함수와 입력 변수를 그래프에 등록
    ///
    /// # 파라미터
    /// - function: 연결할 함수 오브젝트
    /// - inputs: 이 연산의 입력으로 사용된 변수들
    #[cfg(feature = "enable_backpropagation")]
    pub fn with_grad_fn(self: Arc<Self>, function: Arc<dyn Function<f32>>, inputs: &[&Arc<Variable<f32>>]) {
        COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.borrow_mut();

            // 입력 노드 ID 찾기 또는 추가
            let input_ids = inputs.iter().map(|&input_var| {
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

    /// 역전파 수행 메인 엔트리 포인트
    ///
    /// # 특징 동작
    /// - `enable_backpropagation` 기능 전용 메소드
    /// - 계산 그래프를 따라 그래디언트 전파 시작
    ///
    /// # 오류 사항
    /// - 계산 그래프에 등록되지 않은 변수 사용 시 에러 반환
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
    /// 새로운 계산 그래프를 생성합니다.
    ///
    /// 이 메서드는 노드와 관련 데이터를 저장할 빈 `ComputationGraph` 인스턴스를 초기화합니다.
    ///
    /// # 반환값
    /// - `Self`: 초기화된 `ComputationGraph` 인스턴스
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
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

    /// 계산 그래프의 노드들을 위상 정렬(Topological Sort)합니다.
    ///
    /// 이 메서드는 그래프의 노드들을 의존성 순서대로 정렬하여 역전파를 위한 준비를 합니다.
    /// 이미 정렬된 경우에는 아무 작업도 수행하지 않습니다.
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
    fn backward(&mut self, output_id: NodeId) -> MlResult<()> {
        if !self.sorted {
            self.topological_sort();
        }

        // 출력 노드의 그래디언트를 1.0으로 초기화
        let output_node = self.nodes.get(&output_id).ok_or("output node not found.")?;
        let shape = &output_node.variable.tensor.shape;
        let data = vec![1.0; shape.iter().product()];
        let grad = Tensor::from_vec(data, &shape).map_err(|e| format!("failed init gradient: {:?}", e))?;
        output_node.variable.set_grad(grad);

        for &node_id in self.topo_sorted.iter().rev() {
            let node = self.nodes.get(&node_id).unwrap();
            if let Some(function) = &node.function {
                // 입력 노드들에 그래디언트 전파
                for (_, &input_id) in node.inputs.iter().enumerate() {
                    let input_node = self.nodes.get(&input_id).unwrap();
                    input_node.variable.accumulate_grad(
                        function.backward(&input_node.variable.tensor, &node.variable.grad().unwrap())
                            .map_err(|e| format!("failed backpropagation: {:?}", e))?.remove(0)
                    )?;
                }
            }
        }

        Ok(())
    }
}


/// 자동 미분(autograd)을 지원하는 함수 트레잇
///
/// 이 트레잇은 Function<f32>와 Clone을 구현하는 타입에 자동 미분 기능을 추가합니다.
/// 신경망의 순전파(forward pass)와 역전파(backward pass)를 연결하는 그라데이션 함수를 생성하는 역할을 수행합니다.
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
        let mut results = self.forward(&tensors)?;

        #[cfg(feature = "enable_backpropagation")]
        {
            if inputs.len() != 1 {
                return Err("자동 역전파는 현재는 단일 입출력 함수만 지원합니다.".into());
            }
            let result = Arc::new(results.remove(0));
            result.clone().with_grad_fn(Arc::new(self.clone()), inputs);
            return Ok(result)
        }

        Ok(Arc::new(results.remove(0)))
    }
}

impl<F: Function<f32> + Clone + 'static> AutogradFunction for F {}