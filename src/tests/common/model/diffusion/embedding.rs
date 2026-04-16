use super::*;

#[derive(Debug)]
struct TimeEmbeddingMLP {
    dim: usize,           // 기본 차원 (예: 256)
    max_period: f32,      // 10000.0 (Sinusoidal 주깃값)
    mlp: Sequential,      // (dim) -> (dim * 4) -> (dim * 4)
}

impl TimeEmbeddingMLP {
    pub fn new(layer_params: &[usize], max_period: f32) -> MlResult<Self> {
        // parm = t(입력값), d_model(layer_params[0]), d_hidden(layer_params[1]), C(채널 차원, layer_params[2]), t_emb(이전 레이어 출력값)
        // layer = SinusoidalPE → Linear → SiLU → Linear

        info!(
            "Network Structure: {}(SinusoidalPE){} -> {}(Linear){} -> {}(SiLULayer){} -> {}(Linear){}",
            layer_params[0], layer_params[0], layer_params[0],
            layer_params[1], layer_params[1], layer_params[1],
            layer_params[1], layer_params[2],
        );
        let mlp = Sequential::from(vec![
            Box::new(SinusoidalPE::new(layer_params[0], "SinusoidalPE(t)")?),
            Box::new(Linear::new(layer_params[0], layer_params[1], "Linear(d_model, d_hidden)(t_emb)")?),
            Box::new(SiLU::new("SiLU(t_emb)")?),
            Box::new(Linear::new(layer_params[1], layer_params[2], "Linear(d_hidden, C)(t_emb)")?),
        ], "TimeEmbeddingMLP",);

        info!("TimeEmbeddingMLP model created successfully.");

        Ok(Self {
            dim: layer_params[0],
            max_period,
            mlp,
        })
    }
}

impl Layer for TimeEmbeddingMLP {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        self.mlp.apply(input)
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        self.mlp.predict(input)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        self.mlp.params()
    }

    fn label(&self) -> &str {
        self.mlp.label()
    }
}
