use super::*;

#[derive(Debug)]
pub struct Unet {
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
    t_emb_cache: Option<Variable>,
}

#[derive(Debug)]
struct SelfAttentionBlock {
    label:    String,
    gn:       GroupNorm,
    query:    Linear,
    key:      Linear,
    value:    Linear,
    out_proj: Linear,
    scale:    Variable,   // [1,1], 1/sqrt(C) — Mul 의 스칼라 브로드캐스트 경로
    channels: usize,
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

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
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
            Box::new(SiLU::new("act1")?),
            Box::new(Conv2D::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), "conv1")?),
        ], "branch1");

        let branch2 = Sequential::from(vec![
            Box::new(GroupNorm::new(num_groups, out_channels, 1e-5, "norm2")?) as Box<dyn Layer>,
            Box::new(SiLU::new("act2")?),
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
            t_emb_cache: None,
        })
    }

    /// time embedding을 다음 forward에 주입합니다.
    pub fn set_time_emb(&mut self, t_emb: Variable) {
        self.t_emb_cache = Some(t_emb);
    }
}

impl Layer for ResNetBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        // gn1 → act1 → conv1
        let h = self.branch1.apply(input)?;

        // t_emb 주입: proj(t) [N, C_out] → reshape [N, C_out, 1, 1] → broadcast-add
        let h = if let (Some(proj), Some(t)) = (self.t_emb_proj.as_mut(), self.t_emb_cache.take()) {
            let t_proj = proj.apply(&t)?; // [N, C_out]
            let h_shape = h.tensor().shape();
            let (n, c) = (h_shape[0], h_shape[1]);
            let shape_4d = Variable::new(Tensor::from_vec(vec![0.0; n * c], &[n, c, 1, 1])?);
            let t_4d = ReshapeOp::new()?.apply(&[&t_proj, &shape_4d])?; // [N, C_out, 1, 1]
            &h + &t_4d // Add 브로드캐스트: [N, C, H, W] + [N, C, 1, 1] → autograd 유지
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

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        // gn1 → act1 → conv1
        let h = self.branch1.predict(input)?.to_id()?;

        let h = if let (Some(proj), Some(t)) = (&self.t_emb_proj, &self.t_emb_cache) {
            let t_proj = proj.predict(t.tensor())?; // [N, C_out]
            let h_shape = h.shape();
            let (n, c) = (h_shape[0], h_shape[1]);
            let shape_4d = Tensor::from_vec(vec![0.0; n * c], &[n, c, 1, 1])?;
            let t_4d = ReshapeOp::new()?.forward(&[&t_proj, &shape_4d])?.remove(0).to_id()?; // [N, C_out, 1, 1]
            &h + &t_4d // Add 브로드캐스트: [N, C, H, W] + [N, C, 1, 1] → autograd 유지
        } else {
            h
        };

        // gn2 → act2 → conv2
        let h = self.branch2.predict(&h)?;

        // skip connection
        let skip: GlobalTensor<f32> = if let Some(ref sc) = self.skip_conv {
            sc.predict(input)?
        } else {
            GlobalTensor::from_vec(input.data().to_vec(), input.shape())?
        };

        let add_op = Add::new()?;
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
    /// * `channels`   - 입력/출력 채널 C.
    /// * `num_groups` - GroupNorm 그룹 수 (C % num_groups == 0 필요).
    /// * `label`      - 레이어 레이블.
    pub fn new(channels: usize, num_groups: usize, label: &str) -> MlResult<Self> {
        let scale_val = (channels as f32).sqrt().recip();
        Ok(Self {
            label:    label.to_string(),
            gn:       GroupNorm::new(num_groups, channels, 1e-5, "gn")?,
            query:    Linear::new(channels, channels, "query")?,
            key:      Linear::new(channels, channels, "key")?,
            value:    Linear::new(channels, channels, "value")?,
            out_proj: Linear::new(channels, channels, "out_proj")?,
            scale:    Variable::new(Tensor::from_vec(vec![scale_val], &[1, 1])?),
            channels,
        })
    }
}

/// 마지막 축 기준 softmax (row-wise). `[..., L]` 를 L 단위로 정규화.
fn softmax_lastdim_rowwise(data: &[f32], shape: &[usize]) -> Vec<f32> {
    let last = *shape.last().unwrap();
    let rows = data.len() / last;
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        let base = r * last;
        let row = &data[base..base + last];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (i, e) in exps.iter().enumerate() {
            out[base + i] = e / sum;
        }
    }
    out
}

impl Layer for SelfAttentionBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let shape = input.tensor().shape().to_vec();
        if shape.len() != 4 {
            return Err(crate::MlError::StringError(
                "SelfAttentionBlock: 입력은 4D [N, C, H, W] 여야 합니다".into(),
            ));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let hw = h * w;

        let d1 = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1])?);
        let d2 = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1])?);

        // 새 shape 를 전달하기 위한 더미 Variable 생성 헬퍼.
        let mk_shape = |new_shape: &[usize]| -> MlResult<Variable> {
            let sz: usize = new_shape.iter().product();
            Ok(Variable::new(Tensor::from_vec(vec![0.0; sz], new_shape)?))
        };

        // 1) GroupNorm
        let h_ = self.gn.apply(input)?;

        // 2) [N, C, H, W] → [N, C, HW]
        let sh_bcl = mk_shape(&[n, c, hw])?;
        let x_bcl = ReshapeOp::new()?.apply(&[&h_, &sh_bcl])?;

        // 3) transpose(1,2) → [N, HW, C]
        let x_nhwc = Transpose::new()?.apply(&[&x_bcl, &d1, &d2])?;

        // 4) flatten to [N*HW, C] — Linear 의 bias broadcast 는 2D 입력만 지원.
        let sh_flat = mk_shape(&[n * hw, c])?;
        let x_flat = ReshapeOp::new()?.apply(&[&x_nhwc, &sh_flat])?;

        // 5) Q / K / V
        let q_flat = self.query.apply(&x_flat)?;
        let k_flat = self.key.apply(&x_flat)?;
        let v_flat = self.value.apply(&x_flat)?;

        // 6) [N*HW, C] → [N, HW, C]
        let sh_nhwc = mk_shape(&[n, hw, c])?;
        let q = ReshapeOp::new()?.apply(&[&q_flat, &sh_nhwc])?;
        let sh_nhwc_k = mk_shape(&[n, hw, c])?;
        let k = ReshapeOp::new()?.apply(&[&k_flat, &sh_nhwc_k])?;
        let sh_nhwc_v = mk_shape(&[n, hw, c])?;
        let v = ReshapeOp::new()?.apply(&[&v_flat, &sh_nhwc_v])?;

        // 7) K^T: [N, HW, C] → [N, C, HW]
        let k_t = Transpose::new()?.apply(&[&k, &d1, &d2])?;

        // 8) scores = Q · K^T → [N, HW, HW]
        let scores = Matmul::new()?.apply(&[&q, &k_t])?;

        // 9) * 1/sqrt(C)  (Mul 의 [1,1] 스칼라 브로드캐스트)
        let scores_scaled = Mul::new()?.apply(&[&scores, &self.scale])?;

        // 10) softmax
        //  NOTE(TODO): 현 Softmax 연산자는 전역 정규화를 수행하므로 row-wise 와 다름.
        //  axis-aware Softmax 가 도입되면 교체. 단일 배치·단일 공간 위치 케이스만 수학적으로 정확.
        let attn = SoftmaxOp::new()?.apply(&[&scores_scaled])?;

        // 11) attn · V → [N, HW, C]
        let ctx = Matmul::new()?.apply(&[&attn, &v])?;

        // 12) out_proj: flatten → Linear → unflatten
        let sh_flat2 = mk_shape(&[n * hw, c])?;
        let ctx_flat = ReshapeOp::new()?.apply(&[&ctx, &sh_flat2])?;
        let proj_flat = self.out_proj.apply(&ctx_flat)?;
        let sh_nhwc2 = mk_shape(&[n, hw, c])?;
        let proj = ReshapeOp::new()?.apply(&[&proj_flat, &sh_nhwc2])?;

        // 13) [N, HW, C] → [N, C, HW]
        let proj_bcl = Transpose::new()?.apply(&[&proj, &d1, &d2])?;

        // 14) [N, C, HW] → [N, C, H, W]
        let sh_bchw = mk_shape(&[n, c, h, w])?;
        let out = ReshapeOp::new()?.apply(&[&proj_bcl, &sh_bchw])?;

        // 15) residual
        Ok(&out + input)
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let shape = input.shape().to_vec();
        if shape.len() != 4 {
            return Err(crate::MlError::StringError(
                "SelfAttentionBlock: 입력은 4D [N, C, H, W] 여야 합니다".into(),
            ));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let hw = h * w;

        let d1 = Tensor::from_vec(vec![1.0], &[1, 1])?;
        let d2 = Tensor::from_vec(vec![2.0], &[1, 1])?;

        // 1) GroupNorm
        let h_ = self.gn.predict(input)?;

        // 2) [N, C, H, W] → [N, C, HW]
        let x_bcl = GlobalTensor::from_vec(h_.data().to_vec(), &[n, c, hw])?;

        // 3) transpose(1,2) → [N, HW, C]
        let x_nhwc = Transpose::new()?.forward(&[&x_bcl, &d1, &d2])?.remove(0);

        // 4) flatten to [N*HW, C]
        let x_flat = GlobalTensor::from_vec(x_nhwc.data().to_vec(), &[n * hw, c])?;

        // 5) Q / K / V
        let q_flat = self.query.predict(&x_flat)?;
        let k_flat = self.key.predict(&x_flat)?;
        let v_flat = self.value.predict(&x_flat)?;

        // 6) unflatten to [N, HW, C]
        let q = GlobalTensor::from_vec(q_flat.data().to_vec(), &[n, hw, c])?;
        let k = GlobalTensor::from_vec(k_flat.data().to_vec(), &[n, hw, c])?;
        let v = GlobalTensor::from_vec(v_flat.data().to_vec(), &[n, hw, c])?;

        // 7) K^T
        let k_t = Transpose::new()?.forward(&[&k, &d1, &d2])?.remove(0);

        // 8) scores = Q · K^T → [N, HW, HW]
        let scores = Matmul::new()?.forward(&[&q, &k_t])?.remove(0);

        // 9) * 1/sqrt(C)  — data 레벨 직접 스칼라 곱 (브로드캐스트 우회)
        let scale_val = (self.channels as f32).sqrt().recip();
        let scaled: Vec<f32> = scores.data().iter().map(|&x| x * scale_val).collect();

        // 10) row-wise softmax on last dim [N, HW, HW]
        let attn_data = softmax_lastdim_rowwise(&scaled, scores.shape());
        let attn = GlobalTensor::from_vec(attn_data, scores.shape())?;

        // 11) attn · V → [N, HW, C]
        let ctx = Matmul::new()?.forward(&[&attn, &v])?.remove(0);

        // 12) out_proj
        let ctx_flat = GlobalTensor::from_vec(ctx.data().to_vec(), &[n * hw, c])?;
        let proj_flat = self.out_proj.predict(&ctx_flat)?;
        let proj = GlobalTensor::from_vec(proj_flat.data().to_vec(), &[n, hw, c])?;

        // 13) transpose back → [N, C, HW]
        let proj_bcl = Transpose::new()?.forward(&[&proj, &d1, &d2])?.remove(0);

        // 14) → [N, C, H, W]
        let out = GlobalTensor::from_vec(proj_bcl.data().to_vec(), &[n, c, h, w])?;

        // 15) residual
        let add = Add::new()?;
        Ok(add.forward(&[&out, input])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p: Vec<&dyn Parameter> = Vec::new();
        p.extend(self.gn.params());
        p.extend(self.query.params());
        p.extend(self.key.params());
        p.extend(self.value.params());
        p.extend(self.out_proj.params());
        p
    }

    fn label(&self) -> &str { &self.label }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_attention_predict_shape() -> MlResult<()> {
        // C=16, num_groups=8 (16 % 8 == 0), spatial 4x4
        let mut block = SelfAttentionBlock::new(16, 8, "attn")?;
        let n = 2;
        let data: Vec<f32> = (0..(n * 16 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[n, 16, 4, 4])?;
        let out = block.predict(&x)?;
        assert_eq!(out.shape(), &[n, 16, 4, 4]);
        Ok(())
    }

    #[test]
    fn self_attention_new_rejects_invalid_groups() {
        // 16 % 5 != 0 → GroupNorm::new 실패가 전파되어야 함.
        assert!(SelfAttentionBlock::new(16, 5, "attn").is_err());
    }

    #[cfg(feature = "enableBackward")]
    #[test]
    fn self_attention_apply_shape() -> MlResult<()> {
        let mut block = SelfAttentionBlock::new(8, 4, "attn")?;
        let data: Vec<f32> = (0..(1 * 8 * 2 * 2)).map(|i| i as f32 * 0.01).collect();
        let x = Variable::new(Tensor::from_vec(data, &[1, 8, 2, 2])?);
        let out = block.apply(&x)?;
        assert_eq!(out.tensor().shape(), &[1, 8, 2, 2]);
        Ok(())
    }
}
