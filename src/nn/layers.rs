use crate::{ContextId, ExecutionContext, MlError, MlResult, Tensor, TensorError, Variable};
use crate::{Parameter, TensorBuffer};

use super::checkpoint::{find_param, validate_shape};
use super::{LayerState, ModelState, ParamState};

pub trait Layer: std::fmt::Debug {
    fn forward(&self, input: &Variable) -> MlResult<Variable>;
    /// Compatibility entry point during the public API migration.
    fn apply(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        self.forward(input)
    }

    fn predict(&self, input: &Tensor) -> MlResult<Tensor> {
        self.validate_input(input)?;
        input.execution_context()?.no_grad(|| {
            let variable = input.as_variable()?;
            let output = self.forward(&variable)?;
            self.validate_input(output.tensor())?;
            Ok(output.tensor().clone())
        })
    }

    fn validate_input(&self, input: &Tensor) -> MlResult<()> {
        if input.context_id() != self.context_id() {
            return Err(crate::ContextError::Mismatch.into());
        }
        input.numel()?;
        for parameter in self.parameters() {
            if parameter.context_id() != self.context_id() {
                return Err(crate::ContextError::Mismatch.into());
            }
            parameter.tensor().numel()?;
        }
        Ok(())
    }
    fn parameters(&self) -> Vec<&Parameter>;
    fn context_id(&self) -> ContextId;
    fn label(&self) -> &str;

    fn save_state(&self) -> MlResult<LayerState> {
        if !self.parameters().is_empty() {
            return Err(crate::MlError::UnsupportedCapability {
                module: "layer",
                capability: "checkpoint save",
                operation: "save_state",
            });
        }
        Ok(LayerState {
            layer_type: std::any::type_name::<Self>()
                .split("::")
                .last()
                .unwrap_or("Layer")
                .trim_start_matches("Context")
                .to_string(),
            label: self.label().to_string(),
            config: serde_json::Value::Null,
            params: Vec::new(),
        })
    }

    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        if !self.parameters().is_empty() || !state.params.is_empty() {
            return Err(crate::MlError::UnsupportedCapability {
                module: "layer",
                capability: "checkpoint restore",
                operation: "load_state",
            });
        }
        Ok(())
    }
}

fn parameter_state(name: &str, parameter: &Parameter) -> MlResult<ParamState> {
    Ok(ParamState {
        name: name.to_string(),
        shape: parameter.tensor().shape()?,
        data: parameter.tensor().to_vec()?,
        blob_offset: None,
        blob_length: None,
    })
}

fn restore_parameter(
    context: &ExecutionContext,
    parameter: &Parameter,
    state: &LayerState,
    name: &str,
) -> MlResult<()> {
    let saved = find_param(&state.params, name)?;
    validate_shape(saved, &parameter.tensor().shape()?)?;
    context.replace_parameter(
        parameter.variable(),
        TensorBuffer::from_vec(saved.data.clone(), &saved.shape)?,
    )
}

fn validate_layer_type(state: &LayerState, expected: &str) -> MlResult<()> {
    if state.layer_type == expected {
        Ok(())
    } else {
        Err(MlError::StringError(format!(
            "layer type mismatch: checkpoint='{}', current='{}'",
            state.layer_type, expected
        )))
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    context: ExecutionContext,
    label: String,
    weight: Parameter,
    bias: Parameter,
}

impl Linear {
    pub fn new(
        context: &ExecutionContext,
        in_features: usize,
        out_features: usize,
        label: impl Into<String>,
    ) -> MlResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(crate::TensorError::InvalidOperation {
                op: "linear",
                reason: "in_features and out_features must be greater than zero".into(),
            }
            .into());
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight = context.initialization_uniform(
            in_features
                .checked_mul(out_features)
                .ok_or_else(|| TensorError::InvalidOperation {
                    op: "linear",
                    reason: "dimension overflow".into(),
                })?,
            bound,
        )?;
        Ok(Self {
            context: context.clone(),
            label: label.into(),
            weight: context.parameter(weight, &[in_features, out_features])?,
            bias: context.parameter(vec![0.0; out_features], &[out_features])?,
        })
    }

    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    pub fn bias(&self) -> &Parameter {
        &self.bias
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        let projected = input.matmul(self.weight.tensor())?;
        projected.add(self.bias.tensor())
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }

    fn context_id(&self) -> ContextId {
        self.context.id()
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn save_state(&self) -> MlResult<LayerState> {
        Ok(LayerState {
            layer_type: "Linear".into(),
            label: self.label.clone(),
            config: serde_json::json!({
                "in_features": self.weight.tensor().shape()?[0],
                "out_features": self.weight.tensor().shape()?[1],
            }),
            params: vec![
                parameter_state("weight", &self.weight)?,
                parameter_state("bias", &self.bias)?,
            ],
        })
    }

    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        validate_layer_type(state, "Linear")?;
        restore_parameter(&self.context, &self.weight, state, "weight")?;
        restore_parameter(&self.context, &self.bias, state, "bias")
    }
}

#[derive(Clone, Debug)]
pub struct Conv2D {
    context: ExecutionContext,
    label: String,
    weight: Parameter,
    bias: Parameter,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2D {
    pub fn new(
        context: &ExecutionContext,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        label: impl Into<String>,
    ) -> MlResult<Self> {
        if in_channels == 0 || out_channels == 0 || kernel.0 == 0 || kernel.1 == 0 {
            return Err(crate::TensorError::InvalidOperation {
                op: "conv2d",
                reason: "channels and kernel dimensions must be greater than zero".into(),
            }
            .into());
        }
        let fan_in = in_channels
            .checked_mul(kernel.0)
            .and_then(|n| n.checked_mul(kernel.1))
            .ok_or_else(|| TensorError::InvalidOperation {
                op: "conv2d",
                reason: "dimension overflow".into(),
            })?;
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = context.initialization_uniform(
            out_channels
                .checked_mul(fan_in)
                .ok_or_else(|| TensorError::InvalidOperation {
                    op: "conv2d",
                    reason: "dimension overflow".into(),
                })?,
            bound,
        )?;
        Ok(Self {
            context: context.clone(),
            label: label.into(),
            weight: context.parameter(weight, &[out_channels, in_channels, kernel.0, kernel.1])?,
            bias: context.parameter(vec![0.0; out_channels], &[out_channels])?,
            stride,
            padding,
        })
    }
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }
}

impl Layer for Conv2D {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        input.conv2d(
            self.weight.tensor(),
            self.bias.tensor(),
            self.stride,
            self.padding,
        )
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        let shape = self.weight.tensor().shape()?;
        Ok(LayerState {
            layer_type: "Conv2D".into(),
            label: self.label.clone(),
            config: serde_json::json!({
                "in_channels": shape[1], "out_channels": shape[0],
                "kernel_h": shape[2], "kernel_w": shape[3],
                "stride_h": self.stride.0, "stride_w": self.stride.1,
                "padding_h": self.padding.0, "padding_w": self.padding.1,
            }),
            params: vec![
                parameter_state("weight", &self.weight)?,
                parameter_state("bias", &self.bias)?,
            ],
        })
    }
    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        validate_layer_type(state, "Conv2D")?;
        restore_parameter(&self.context, &self.weight, state, "weight")?;
        restore_parameter(&self.context, &self.bias, state, "bias")
    }
}

#[derive(Clone, Debug)]
pub struct GroupNorm {
    context: ExecutionContext,
    label: String,
    gamma: Parameter,
    beta: Parameter,
    groups: usize,
    epsilon: f32,
}

impl GroupNorm {
    pub fn new(
        context: &ExecutionContext,
        groups: usize,
        channels: usize,
        epsilon: f32,
        label: impl Into<String>,
    ) -> MlResult<Self> {
        if groups == 0
            || channels == 0
            || channels % groups != 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(crate::TensorError::InvalidOperation {
                op: "group_norm",
                reason:
                    "channels must be divisible by non-zero groups and epsilon must be positive"
                        .into(),
            }
            .into());
        }
        Ok(Self {
            context: context.clone(),
            label: label.into(),
            gamma: context.parameter(vec![1.0; channels], &[channels])?,
            beta: context.parameter(vec![0.0; channels], &[channels])?,
            groups,
            epsilon,
        })
    }
    pub fn gamma(&self) -> &Parameter {
        &self.gamma
    }
    pub fn beta(&self) -> &Parameter {
        &self.beta
    }
}

impl Layer for GroupNorm {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        input.group_norm(
            self.gamma.tensor(),
            self.beta.tensor(),
            self.groups,
            self.epsilon,
        )
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.gamma, &self.beta]
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        Ok(LayerState {
            layer_type: "GroupNorm".into(),
            label: self.label.clone(),
            config: serde_json::json!({
                "num_groups": self.groups,
                "num_channels": self.gamma.tensor().shape()?[0],
                "eps": self.epsilon,
            }),
            params: vec![
                parameter_state("gamma", &self.gamma)?,
                parameter_state("beta", &self.beta)?,
            ],
        })
    }
    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        validate_layer_type(state, "GroupNorm")?;
        restore_parameter(&self.context, &self.gamma, state, "gamma")?;
        restore_parameter(&self.context, &self.beta, state, "beta")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingMode {
    Max,
    Average,
}

#[derive(Clone, Debug)]
pub struct Pooling {
    context: ExecutionContext,
    label: String,
    kernel: (usize, usize),
    stride: (usize, usize),
    mode: PoolingMode,
}

impl Pooling {
    pub fn max(
        context: &ExecutionContext,
        kernel: (usize, usize),
        stride: (usize, usize),
        label: impl Into<String>,
    ) -> Self {
        Self {
            context: context.clone(),
            label: label.into(),
            kernel,
            stride,
            mode: PoolingMode::Max,
        }
    }
    pub fn average(
        context: &ExecutionContext,
        kernel: (usize, usize),
        stride: (usize, usize),
        label: impl Into<String>,
    ) -> Self {
        Self {
            context: context.clone(),
            label: label.into(),
            kernel,
            stride,
            mode: PoolingMode::Average,
        }
    }
}

impl Layer for Pooling {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        match self.mode {
            PoolingMode::Max => input.max_pool2d(self.kernel, self.stride),
            PoolingMode::Average => input.avg_pool2d(self.kernel, self.stride),
        }
    }

    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        Ok(LayerState {
            layer_type: "Pooling".into(),
            label: self.label.clone(),
            config: serde_json::json!({
                "mode": if self.mode == PoolingMode::Max { "max" } else { "avg" },
                "kernel_h": self.kernel.0, "kernel_w": self.kernel.1,
                "stride_h": self.stride.0, "stride_w": self.stride.1,
            }),
            params: Vec::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Upsample2D {
    context: ExecutionContext,
    label: String,
    scale: (usize, usize),
}

impl Upsample2D {
    pub fn nearest(
        context: &ExecutionContext,
        scale: (usize, usize),
        label: impl Into<String>,
    ) -> MlResult<Self> {
        if scale.0 == 0 || scale.1 == 0 {
            return Err(TensorError::InvalidOperation {
                op: "nearest_upsample2d",
                reason: "scale dimensions must be greater than zero".into(),
            }
            .into());
        }
        Ok(Self {
            context: context.clone(),
            label: label.into(),
            scale,
        })
    }
}

impl Layer for Upsample2D {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        input.nearest_upsample2d(self.scale)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        Ok(LayerState {
            layer_type: "Upsample2D".into(),
            label: self.label.clone(),
            config: serde_json::json!({ "mode": "nearest", "scale_h": self.scale.0, "scale_w": self.scale.1 }),
            params: Vec::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Reshape {
    context: ExecutionContext,
    label: String,
    target_shape: Vec<isize>,
}

impl Reshape {
    pub fn new(
        context: &ExecutionContext,
        target_shape: &[isize],
        label: impl Into<String>,
    ) -> MlResult<Self> {
        if target_shape
            .iter()
            .filter(|&&dimension| dimension < 0)
            .count()
            > 1
        {
            return Err(TensorError::InvalidOperation {
                op: "reshape",
                reason: "at most one inferred dimension is allowed".into(),
            }
            .into());
        }
        Ok(Self {
            context: context.clone(),
            label: label.into(),
            target_shape: target_shape.to_vec(),
        })
    }

    fn resolve_shape(&self, input_shape: &[usize]) -> MlResult<Vec<usize>> {
        let total = input_shape
            .iter()
            .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
            .ok_or_else(|| TensorError::InvalidOperation {
                op: "reshape",
                reason: "input element count overflow".into(),
            })?;
        let mut result = Vec::with_capacity(self.target_shape.len());
        let mut inferred = None;
        let mut known = 1usize;
        for (index, &dimension) in self.target_shape.iter().enumerate() {
            let resolved = match dimension {
                value if value < 0 => {
                    inferred = Some(index);
                    1
                }
                0 => *input_shape
                    .get(index)
                    .ok_or_else(|| TensorError::InvalidOperation {
                        op: "reshape",
                        reason: format!(
                            "dimension {index} cannot be copied from rank {}",
                            input_shape.len()
                        ),
                    })?,
                value => usize::try_from(value).map_err(|_| TensorError::InvalidOperation {
                    op: "reshape",
                    reason: "target dimension is out of range".into(),
                })?,
            };
            known = known
                .checked_mul(resolved)
                .ok_or_else(|| TensorError::InvalidOperation {
                    op: "reshape",
                    reason: "target element count overflow".into(),
                })?;
            result.push(resolved);
        }
        if let Some(index) = inferred {
            if known == 0 || total % known != 0 {
                return Err(TensorError::InvalidShape {
                    expected: vec![total],
                    got: vec![known],
                }
                .into());
            }
            result[index] = total / known;
        } else if known != total {
            return Err(TensorError::InvalidShape {
                expected: vec![total],
                got: vec![known],
            }
            .into());
        }
        Ok(result)
    }
}

impl Layer for Reshape {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        let shape = self.resolve_shape(&input.tensor().shape()?)?;
        input.reshape(&shape)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        Ok(LayerState {
            layer_type: "Reshape".into(),
            label: self.label.clone(),
            config: serde_json::json!({ "target_shape": self.target_shape }),
            params: Vec::new(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    SiLU,
    Softmax { axis: usize },
}

#[derive(Clone, Debug)]
pub struct Activation {
    context: ExecutionContext,
    label: String,
    kind: ActivationKind,
}

impl Activation {
    pub fn new(context: &ExecutionContext, kind: ActivationKind, label: impl Into<String>) -> Self {
        Self {
            context: context.clone(),
            label: label.into(),
            kind,
        }
    }
}

impl Layer for Activation {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        match self.kind {
            ActivationKind::Identity => Ok(input.clone()),
            ActivationKind::ReLU => input.relu(),
            ActivationKind::Sigmoid => input.sigmoid(),
            ActivationKind::Tanh => input.tanh(),
            ActivationKind::SiLU => input.silu(),
            ActivationKind::Softmax { axis } => input.softmax(axis),
        }
    }

    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        let (name, axis) = match self.kind {
            ActivationKind::Identity => ("identity", None),
            ActivationKind::ReLU => ("relu", None),
            ActivationKind::Sigmoid => ("sigmoid", None),
            ActivationKind::Tanh => ("tanh", None),
            ActivationKind::SiLU => ("silu", None),
            ActivationKind::Softmax { axis } => ("softmax", Some(axis)),
        };
        Ok(LayerState {
            layer_type: "Activation".into(),
            label: self.label.clone(),
            config: serde_json::json!({ "kind": name, "axis": axis }),
            params: Vec::new(),
        })
    }
}

#[derive(Debug)]
pub struct Sequential {
    context: ExecutionContext,
    label: String,
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(context: &ExecutionContext, label: impl Into<String>) -> Self {
        Self {
            context: context.clone(),
            label: label.into(),
            layers: Vec::new(),
        }
    }
    pub fn push(&mut self, layer: Box<dyn Layer>) -> MlResult<()> {
        if layer.context_id() != self.context.id() {
            return Err(crate::ContextError::Mismatch.into());
        }
        self.layers.push(layer);
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn save(&self, path: &str) -> MlResult<()> {
        ModelState::new(vec![self.save_state()?]).save(path)
    }

    pub fn load(&mut self, path: &str) -> MlResult<()> {
        let model = ModelState::load(path)?;
        let state = model
            .layers
            .iter()
            .find(|state| state.label == self.label)
            .ok_or_else(|| {
                MlError::StringError(format!(
                    "sequential layer '{}' was not found in checkpoint",
                    self.label
                ))
            })?;
        self.load_state(state)
    }
}

impl Layer for Sequential {
    fn forward(&self, input: &Variable) -> MlResult<Variable> {
        self.validate_input(input.tensor())?;
        if input.tensor().context_id() != self.context.id() {
            return Err(crate::ContextError::Mismatch.into());
        }
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.apply(&current)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
    fn context_id(&self) -> ContextId {
        self.context.id()
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn save_state(&self) -> MlResult<LayerState> {
        let sub_layers = self
            .layers
            .iter()
            .map(|layer| layer.save_state())
            .collect::<MlResult<Vec<_>>>()?;
        Ok(LayerState {
            layer_type: "Sequential".into(),
            label: self.label.clone(),
            config: serde_json::json!({ "sub_layers": sub_layers }),
            params: Vec::new(),
        })
    }
    fn load_state(&mut self, state: &LayerState) -> MlResult<()> {
        validate_layer_type(state, "Sequential")?;
        let saved_layers: Vec<LayerState> =
            serde_json::from_value(state.config.get("sub_layers").cloned().ok_or_else(|| {
                MlError::StringError("sequential checkpoint has no sub_layers".into())
            })?)
            .map_err(|error| {
                MlError::StringError(format!("failed to decode sequential layers: {error}"))
            })?;
        for layer in &mut self.layers {
            if let Some(saved) = saved_layers
                .iter()
                .find(|saved| saved.label == layer.label())
            {
                layer.load_state(saved)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/nn/layers_tests.rs"]
mod tests;
