use super::*;

#[derive(Debug)]
struct Unet {
    channels: usize,
    self_condition: bool,
    init_conv: Box< dyn Layer>,
    self_random_or_learned_sinusoidal_cond: bool,
    time_mlp: Sequential,
    downs: Sequential,
    ups: Sequential,
    mid_block1: Sequential,
    mid_block2: Sequential,
    mid_attn: Sequential,
    out_dim: Option<usize>,
    final_res_block: Sequential,
    final_conv: Sequential,
    label: String,
    layers: Sequential,
}

#[derive(Debug)]
struct ResNetBlock {
    label:      String,
    branch1:    Sequential,               // gn1 → act1 → conv1
    branch2:    Sequential,               // gn2 → act2 → conv2
    skip_conv:  Option<Conv2D>,      // 1×1, in_channels != out_channels 일 때
    t_emb_proj: Option<Linear>,           // Linear(t_emb_dim → out_channels)
    #[cfg(feature = "enableBackward")]
    t_emb_cache: Option<Variable>,
}

#[derive(Debug)]
struct SelfAttentionBlock {
    label:    String,
    query:    Linear,
    key:      Linear,
    value:    Linear,
    out_proj: Linear,
    layers: Sequential,
}

impl Unet {
    fn new(
        dim: usize, init_dim: Option<usize>, out_dim: Option<usize>, dim_mults: &[usize],
        channels: usize,
        self_condition: bool,
        resnet_block_groups: usize,
        learned_variance: bool, learned_sinusoidal_cond: bool, learned_sinusoidal_dim: usize, sinusoidal_pos_emb_theta: usize,
        random_fourier_features: bool,
        attn_dim_head: usize, attn_heads: usize, full_attn: Option<bool>, flash_attn: bool,
    ) -> Self {
        todo!("Unet::new — DownBlock/UpBlock/MidBlock 조립 후 구현")
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
    /// * `in_channels`  - 입력 채널 수
    /// * `out_channels` - 출력 채널 수
    /// * `num_groups`   - GroupNorm 그룹 수 (in/out 채널 모두 나누어 떨어져야 함)
    /// * `t_emb_dim`    - time embedding 차원 (None 이면 주입 없음)
    fn new(
        in_channels:  usize,
        out_channels: usize,
        num_groups:   usize,
        t_emb_dim:    Option<usize>,
        label:        &str,
    ) -> MlResult<Self> {
        let branch1 = Sequential::from(vec![
            Box::new(GroupNorm::new(num_groups, in_channels,  1e-5, "norm1")?) as Box<dyn Layer>,
            Box::new(SiLULayer::new("act1")?),
            Box::new(Conv2D::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), "conv1")?),
        ], "branch1");

        let branch2 = Sequential::from(vec![
            Box::new(GroupNorm::new(num_groups, out_channels, 1e-5, "norm2")?) as Box<dyn Layer>,
            Box::new(SiLULayer::new("act2")?),
            Box::new(Conv2D::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), "conv2")?),
        ], "branch2");

        let skip_conv = if in_channels != out_channels {
            Some(Conv2D::new(in_channels, out_channels, (1, 1), (1, 1), (0, 0), "skip_conv")?)
        } else {
            None
        };

        let t_emb_proj = match t_emb_dim {
            Some(dim) => Some(Linear::new(dim, out_channels, "t_emb_proj")?),
            None      => None,
        };

        Ok(Self {
            label: label.to_string(),
            branch1,
            branch2,
            skip_conv,
            t_emb_proj,
            #[cfg(feature = "enableBackward")]
            t_emb_cache: None,
        })
    }

    /// time embedding을 다음 forward에 주입합니다.
    #[cfg(feature = "enableBackward")]
    pub fn set_time_emb(&mut self, t_emb: Variable) {
        self.t_emb_cache = Some(t_emb);
    }
}

impl Layer for ResNetBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        // gn1 → act1 → conv1
        let h = self.branch1.apply(input)?;

        // t_emb 주입: proj(t) [N, C_out] → 수동 broadcast-add → [N, C_out, H, W]
        let h = if let (Some(proj), Some(t)) = (self.t_emb_proj.as_mut(), self.t_emb_cache.take()) {
            let t_proj = proj.apply(&t)?;
            let h_shape = h.tensor().shape().to_vec();
            let (n, c, hh, w) = (h_shape[0], h_shape[1], h_shape[2], h_shape[3]);
            let t_data = t_proj.tensor().data().to_vec();
            let mut out = h.tensor().data().to_vec();
            for ni in 0..n {
                for ci in 0..c {
                    let v = t_data[ni * c + ci];
                    for yi in 0..hh {
                        for xi in 0..w {
                            out[ni*c*hh*w + ci*hh*w + yi*w + xi] += v;
                        }
                    }
                }
            }
            Variable::new(Tensor::from_vec(out, &h_shape)?)
        } else {
            h
        };

        // gn2 → act2 → conv2
        let h = self.branch2.apply(&h)?;

        // skip connection
        let skip = if let Some(ref mut sc) = self.skip_conv {
            sc.apply(input)?
        } else {
            input.clone()
        };

        Ok(&h + &skip)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        // gn1 → act1 → conv1
        let h = self.branch1.predict(input)?;

        // gn2 → act2 → conv2
        let h = self.branch2.predict(&h)?;

        // skip connection
        let skip: GlobalTensor<f32> = if let Some(ref mut sc) = self.skip_conv {
            sc.predict(input)?
        } else {
            GlobalTensor::from_vec(input.data().to_vec(), input.shape())?
        };

        let add_op = crate::tensor::operators::Add::new()?;
        Ok(add_op.forward(&[&h, &skip])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p = self.branch1.params();
        p.extend(self.branch2.params());
        if let Some(ref proj) = self.t_emb_proj { p.extend(proj.params()); }
        if let Some(ref sc)   = self.skip_conv   { p.extend(sc.params()); }
        p
    }

    fn label(&self) -> &str {
        self.label.as_str()
    }
}

impl SelfAttentionBlock {
    pub fn new(channels: usize, label: &str) -> MlResult<Self> {
        let layers = Sequential::from(vec![
            Box::new(GroupNorm::new(32, channels, 1e-5, "gn")?),
            // Box::new(Reshape::new()? as &dyn Layer), // [N, C, H, W] → [N, H*W, C] (채널이 마지막 차원)
        ], "attn_layers");


        Ok(Self {
            label:    label.to_string(),
            query:    Linear::new(channels, channels, "query")?, // 키, 쿼리, 벨류는 모두 channels 차원에서 선형 변환,
            key:      Linear::new(channels, channels, "key")?,   // 가중치와 편향값이 존재하는 선형함수.
            value:    Linear::new(channels, channels, "value")?,
            out_proj: Linear::new(channels, channels, "out_proj")?,
            layers
        })
    }
}

impl Layer for SelfAttentionBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        todo!("SelfAttentionBlock::apply — spatial flatten/reshape 필요")
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        todo!("SelfAttentionBlock::predict — spatial flatten/reshape 필요")
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p: Vec<&dyn Parameter> = Vec::new();
        p.extend(self.query.params());
        p.extend(self.key.params());
        p.extend(self.value.params());
        p.extend(self.out_proj.params());
        p
    }

    fn label(&self) -> &str { &self.label }
}
