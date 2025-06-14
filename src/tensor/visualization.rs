use std::collections::{HashMap, HashSet};
#[cfg(feature = "enableVisualization")]
use crate::tensor::{LABEL_COUNTERS, NodeId, NodeType, VISUALIZATION_GRAPH, VisualizationGraph};

#[cfg(feature = "enableVisualization")]
impl VisualizationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: Vec::new(),
            node_types: HashMap::new(),
            node_labels: HashMap::new(),
        }
    }

    fn compute_node_layers(&self) -> HashMap<String, usize> {
        use std::collections::{HashMap, VecDeque};

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
        let layers = self.compute_node_layers();

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