/// 파라미터가 있는 레이어의 체크포인트(save/load) 보일러플레이트를 자동 생성합니다.
///
/// # 사용법
/// ```ignore
/// layer_params!(Linear, "Linear", [weight, bias], |s| serde_json::json!({
///     "in_features": s.weight.tensor().shape()[0],
///     "out_features": s.weight.tensor().shape()[1],
/// }));
/// ```
///
/// `|s|` 는 `self`를 바인딩하는 식별자입니다 (매크로 위생 규칙 때문에 `self` 직접 사용 불가).
///
/// 생성되는 메서드: `_params()`, `_save_state()`, `_load_state()`
/// Layer impl에서 위임하여 사용:
/// ```ignore
/// impl Layer for Linear {
///     fn params(&self) -> Vec<&dyn Parameter> { self._params() }
///     fn save_state(&self) -> LayerState { self._save_state() }
///     fn load_state(&mut self, state: &LayerState) -> MlResult<()> { self._load_state(state) }
/// }
/// ```
macro_rules! layer_params {
    ($layer:ty, $type_name:expr, [$($param:ident),+], |$s:ident| $config:expr) => {
        impl $layer {
            fn _params(&self) -> Vec<&dyn Parameter> {
                vec![$(&self.$param),+]
            }

            fn _save_state(&self) -> LayerState {
                let $s = self;
                LayerState {
                    layer_type: $type_name.to_string(),
                    label: $s.label.clone(),
                    config: $config,
                    params: vec![$(
                        {
                            let t = $s.$param.tensor();
                            ParamState {
                                name: stringify!($param).to_string(),
                                shape: t.shape().to_vec(),
                                data: t.data().to_vec(),
                                blob_offset: None,
                                blob_length: None,
                            }
                        },
                    )+],
                }
            }

            fn _load_state(&mut self, state: &LayerState) -> MlResult<()> {
                if state.layer_type != $type_name {
                    return Err(MlError::StringError(format!(
                        "레이어 타입 불일치: 파일='{}', 현재='{}'", state.layer_type, $type_name
                    )));
                }
                $(
                    let p = crate::nn::checkpoint::find_param(&state.params, stringify!($param))?;
                    crate::nn::checkpoint::validate_shape(p, self.$param.tensor().shape())?;
                    self.$param.tensor().replace(GlobalTensor::from_vec(p.data.clone(), &p.shape)?);
                )+
                Ok(())
            }
        }
    };
}

pub mod activation;
pub mod conv2d;
pub mod pooling;
pub mod linear;
pub mod group_norm;
mod parameter;
pub mod checkpoint;
mod reshape;

use crate::{
    register_operator,
    var_bias,
    var_weight,
    backend::Backend,
    MlResult,
    tensor::{
        operators::{
            Add,
            Matmul,
            Sub,
            Function
        },
        GlobalFunction,
        GlobalTensor,
        NodeId,
        Tensor,
        TensorBase,
        AutogradFunction,
    },
    MlError,
    TensorError,
    nn::activation::Softmax
};
use std::{
    fmt::{
        Formatter,
        Debug
    },
    sync::Arc
};
pub use checkpoint::{LayerState, ModelState, ParamState};

#[macro_export]
macro_rules! variable {
    ($vec:expr) => {
        crate::nn::Variable::new(crate::tensor::Tensor::new($vec))
    };

    ($data:expr, $shape:expr) => {
        crate::nn::Variable::new(crate::tensor::Tensor::from_vec($data, $shape).unwrap())
    };

    ($data:expr, $shape:expr, $label:expr) => {
        {
            #[cfg(feature = "enableVisualization")]
            {
                crate::nn::Variable::with_label(crate::tensor::Tensor::from_vec($data, $shape).unwrap(), $label)
            }

            #[cfg(not(feature = "enableVisualization"))]
            {
                crate::nn::Variable::new(crate::tensor::Tensor::from_vec($data, $shape).unwrap())
            }
        }
    };
}

pub trait Model {
    #[cfg(feature = "enableBackward")]
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, optimizer: &mut dyn crate::optimizer::Optimizer, tolerance: f32) -> MlResult<()>;
    #[cfg(feature = "enableBackward")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, test_data: &dyn TensorBase) -> MlResult<GlobalTensor<f32>>;
    fn save(&self, path: &str) -> MlResult<()>;
    fn load(&mut self, path: &str) -> MlResult<()>;
    fn get_loss(&self) -> f32;
    fn compute_total_error(&mut self, x_set: &[&Variable], t_set: &[&Variable]) -> MlResult<f32>;
    fn evaluate_model(&mut self, x_test: &[&Variable], t_test: &[&Variable]) -> MlResult<f32> {
        let n_val = x_test.len();
        let mut correct_predictions = 0;
        for i in 0..n_val {
            let y = self.predict(x_test[i].tensor())?;
            let predicted_class = y.data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let true_class = t_test[i].tensor().data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);
            if predicted_class == true_class {
                correct_predictions += 1;
            }
        }
        let accuracy = correct_predictions as f32 / n_val as f32 * 100.0;
        Ok(accuracy)
    }
}

pub trait Layer: Debug {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>>;
    fn params(&self) -> Vec<&dyn Parameter>;
    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    } // 레이어를 구현하는 구조체의 이름을 반환
    fn label(&self) -> &str;    // 유저가 설정한 레이어의 이름을 반환

    /// 현재 레이어의 파라미터를 LayerState로 직렬화.
    /// 학습 파라미터가 없는 레이어(활성화 함수, Pooling 등)는 기본 구현을 사용.
    fn save_state(&self) -> LayerState {
        LayerState {
            layer_type: self.type_name().to_string(),
            label: self.label().to_string(),
            config: serde_json::Value::Null,
            params: vec![],
        }
    }

    /// LayerState에서 파라미터를 복원.
    /// 학습 파라미터가 없는 레이어는 기본 구현(no-op)을 사용.
    fn load_state(&mut self, _state: &LayerState) -> MlResult<()> {
        Ok(())
    }
}

pub trait Parameter: Debug {
    fn new(tensor: Tensor) -> Self where Self: Sized;
    fn node_id(&self) -> NodeId;
    fn tensor(&self) -> &Tensor;
    fn is_retain_grad(&self) -> bool;
    fn retain_grad(&self);
    fn grad(&self) -> &Tensor;
    #[cfg(feature = "enableBackward")]
    fn set_grad(&self, grad: GlobalTensor<f32>);
    #[cfg(feature = "enableVisualization")]
    fn set_label(&mut self, new_label: &str);
    /// 현재 라벨 반환
    #[cfg(feature = "enableVisualization")]
    fn label(&self) -> &str;
    #[cfg(feature = "enableVisualization")]
    fn node_type(&self) -> &crate::tensor::NodeType;
    #[cfg(feature = "enableBackward")]
    fn clear_grad(&self);
    // TENSOR_STORAGE의 GlobalTensor.dirty 플래그를 조회 (O(1))
    // backward 루프에서 grad.is_empty() O(n) 스캔 대신 사용
    // 모든 Variable 클론이 동일한 STORAGE 항목을 공유하므로 원본 Variable에서도 정확히 동작
    #[cfg(feature = "enableBackward")]
    fn is_grad_dirty(&self) -> bool;
    #[cfg(feature = "enableBackward")]
    fn accumulate_grad(&self, new_grad: Tensor) -> MlResult<()>;
    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self) -> MlResult<()> {
        crate::tensor::COMPUTATION_GRAPH.with(|graph| {
            let mut graph = graph.lock().unwrap();

            if graph.node_map.contains_key(&self.node_id()) {
                graph.ensure_topological_sort();
                graph.backward(self.node_id())
            } else {
                Err(MlError::StringError("계산 그래프가 생성되지 않았습니다.".to_string()))
            }
        })
    }
    /// Performs backpropagation and then automatically resets the computation graph
    /// to release memory for all intermediate tensors.
    #[cfg(all(feature = "enableBackward"))]
    fn backward_and_clear(&self) -> MlResult<()> {
        self.backward()?;
        crate::tensor::ComputationGraph::reset_graph();
        Ok(())
    }
}

// #[derive(Clone)]
// pub struct Variable {
//     #[cfg(all(feature = "enableVisualization"))]
//     label: String,
// #[cfg(all(feature = "enableVisualization"))]
//     node_type: crate::tensor::NodeType,
//     tensor: Tensor,
//     requires_grad: RefCell<bool>,
//     grad: Tensor,
// }

#[derive(Clone)]
pub struct Variable {
    #[cfg(feature = "enableVisualization")]
    label: Arc<String>,
    #[cfg(feature = "enableVisualization")]
    node_type: crate::tensor::NodeType,
    tensor: Tensor,
    requires_grad: std::cell::Cell<bool>,
    grad: Tensor,
}

impl Debug for Variable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Variable");
        ds
            .field("tensor", &self.tensor)
            .field("requires_grad", &self.requires_grad);
        #[cfg(feature = "enableBackward")]
        {
            ds.field("grad", &self.grad);
        }
        ds.finish()
    }
}

#[derive(Debug)]
pub struct Linear {
    label: String,
    weight: Variable,
    bias: Variable,
}

#[derive(Debug)]
pub struct Conv2D {
    label: String,
    weight: Variable,           // [C_out, C_in, kH, kW]
    bias: Variable,             // [C_out]
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolingMode {
    Max,
    Average,
}

#[derive(Debug)]
pub struct Pooling {
    label: String,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub mode: PoolingMode,
}

/// Group Normalization 레이어
/// 학습 파라미터: γ (scale) [C], β (bias) [C]
#[derive(Debug)]
pub struct GroupNorm {
    label:        String,
    pub gamma:    Variable,   // [C]  — 초기값 1
    pub beta:     Variable,   // [C]  — 초기값 0
    pub num_groups:   usize,
    pub num_channels: usize,
    pub eps:      f32,
}

/// Reshape 레이어 — 입력 텐서의 shape를 `target_shape`로 변환.
/// `target_shape`에 `-1`을 하나만 쓰면 해당 차원을 자동 추론합니다.
#[derive(Debug)]
pub struct Reshape {
    label: String,
    target_shape: Vec<isize>,
    operator: GlobalFunction,
}

pub struct Sequential {
    label: String,
    layers: Vec<Box<dyn Layer>>, // Box<dyn Layer>를 사용하여 다양한 종류의 레이어를 하나의 Vec에 저장
}

impl Sequential {
    pub fn new(label: &str) -> Self {
        Self { label: label.to_string(), layers: vec![] }
    }

    pub fn from(layers: Vec<Box<dyn Layer>>, label: &str) -> Self {
        Self { label: label.to_string(), layers }
    }

    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
    
    pub fn remove(&mut self, index: usize) -> Box<dyn Layer> {
        self.layers.remove(index)
    }

    /// 레이어들의 파라미터를 파일로 저장 (확장자에 따라 JSON/binary 자동 선택).
    ///
    /// # 예시
    /// ```no_run
    /// net.save("checkpoints/model.json")?;  // JSON 포맷
    /// net.save("checkpoints/model.tdw")?;   // binary 포맷
    /// ```
    pub fn save(&self, path: &str) -> MlResult<()> {
        let state = ModelState::new(
            self.layers.iter().map(|l| l.save_state()).collect()
        );
        state.save(path)
    }

    /// 파일에서 파라미터를 로드. 레이블이 일치하는 레이어에 덮어씀.
    /// 레이블이 없는 레이어는 경고를 출력하고 건너뜀 (부분 로드 지원).
    ///
    /// # 예시
    /// ```no_run
    /// net.load("checkpoints/model.json")?;
    /// ```
    pub fn load(&mut self, path: &str) -> MlResult<()> {
        let state = ModelState::load(path)?;
        for layer in &mut self.layers {
            let label = layer.label().to_string();
            match state.layers.iter().find(|s| s.label == label) {
                Some(ls) => layer.load_state(ls)?,
                None => tracing::warn!(
                    "[Checkpoint] 레이어 '{}' 를 '{}' 에서 찾지 못함, 건너뜀",
                    label, path
                ),
            }
        }
        Ok(())
    }
}

impl Layer for Sequential {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        if self.layers.is_empty() {
            return Err(MlError::StringError("Sequential has no layers".to_string()));
        }
        let mut iter = self.layers.iter_mut();
        let mut current = iter.next().unwrap().apply(input)?;
        for layer in iter {
            current = layer.apply(&current)?;
        }
        Ok(current)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let mut layer_iter = self.layers.iter_mut();
        let first_layer = match layer_iter.next() {
            Some(layer) => layer,
            None => return Err(MlError::StringError("Sequential 모델에 레이어가 없습니다.".to_string())),
        };

        let mut output = first_layer.predict(input)?;
        for layer in layer_iter {
            output = layer.predict(&output)?;
        }
        Ok(output)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        self.layers.iter().flat_map(|layer| layer.params()).collect()
    }

    fn label(&self) -> &str { &self.label }

    fn save_state(&self) -> LayerState {
        let sub_layers: Vec<LayerState> = self.layers.iter()
            .map(|l| l.save_state())
            .collect();
        LayerState {
            layer_type: "Sequential".to_string(),
            label: self.label.clone(),
            config: serde_json::json!({ "sub_layers": sub_layers }),
            params: vec![],
        }
    }

    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        if state.layer_type != "Sequential" {
            return Err(MlError::StringError(format!(
                "레이어 타입 불일치: 파일='{}', 현재='Sequential'", state.layer_type
            )));
        }
        let sub_layers: Vec<LayerState> = serde_json::from_value(
            state.config["sub_layers"].clone()
        ).map_err(|e| MlError::StringError(
            format!("Sequential 하위 레이어 파싱 실패: {}", e)
        ))?;

        for layer in &mut self.layers {
            let label = layer.label().to_string();
            match sub_layers.iter().find(|s| s.label == label) {
                Some(ls) => layer.load_state(ls)?,
                None => tracing::warn!(
                    "[Checkpoint] Sequential 하위 레이어 '{}' 를 체크포인트에서 찾지 못함, 건너뜀",
                    label
                ),
            }
        }
        Ok(())
    }
}

impl Debug for Sequential {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Sequential");
        ds
            .field("label", &self.label)
            .field("layers", &self.layers)
            .finish()
    }
}

