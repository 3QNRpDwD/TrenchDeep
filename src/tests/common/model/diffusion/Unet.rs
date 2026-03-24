use super::*;

#[derive(Debug)]
struct Unet {
    label: String,
    layers: Sequential,
}

#[derive(Debug)]
struct ResNetBlock {
    label: String,
    layers: Sequential,
}

#[derive(Debug)]
struct SelfAttentionBlock {
    label: String,
    layers: Sequential,
    query: Variable,
    key: Variable,
    value: Variable,
}

impl Unet {
    fn new() -> Self {
        let model = Sequential::from(vec![], "U-net");

        Unet { label: "".to_string(), layers: model }
    }
}

impl Layer for Unet {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        todo!()
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        todo!()
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }

    fn label(&self) -> &str {
        todo!()
    }
}

impl ResNetBlock {
    fn new() -> Self {
        let model = Sequential::from(vec![], "ResNetBlock");

        ResNetBlock { label: "".to_string(), layers: model }
    }

}

impl Layer for ResNetBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        todo!()
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        todo!()
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }

    fn label(&self) -> &str {
        self.label.as_str()
    }
}

impl SelfAttentionBlock {
    pub fn new(label: &str, ) -> Self {
        let model = Sequential::from(vec![], label);
        let query = variable!(vec![vec![]]);
        let key = variable!(vec![vec![]]);
        let value = variable!(vec![vec![]]);

        Self {
            label: label.to_string(),
            layers: model,
            query,
            key,
            value,
        }
    }
}

impl Layer for SelfAttentionBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        todo!()
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        todo!()
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }

    fn label(&self) -> &str {
        todo!()
    }
}