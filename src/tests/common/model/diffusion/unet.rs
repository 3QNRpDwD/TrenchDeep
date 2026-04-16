use crate::nn::Reshape;
use super::*;

// ── U-Net 하위 블록 ─────────────────────────────────────────────────────────

/// Down path 한 단계: ResNet×2 → (optional) Attention → Downsample(stride-2 conv)
#[derive(Debug)]
struct DownBlock {
    resnet1:    ResNetBlock,
    resnet2:    ResNetBlock,
    attn:       Option<SelfAttentionBlock>,
    downsample: Conv2D,  // stride=2 conv → 공간 해상도 1/2
}

/// Mid (bottleneck): ResNet → Attention → ResNet
#[derive(Debug)]
struct MidBlock {
    resnet1: ResNetBlock,
    attn:    SelfAttentionBlock,
    resnet2: ResNetBlock,
}

/// Up path 한 단계: ResNet×2 → (optional) Attention → NearestUpsample(2×) → Conv2D
/// UpBlock의 resnet1.in_channels = skip_channels + up_channels (concat 후)
#[derive(Debug)]
struct UpBlock {
    resnet1:      ResNetBlock,
    resnet2:      ResNetBlock,
    attn:         Option<SelfAttentionBlock>,
    upsample_conv: Conv2D,  // upsample 후 3×3 conv (채널 보정)
}

#[derive(Debug)]
pub struct Unet {
    channels: usize,
    self_condition: bool,
    init_conv: Box<dyn Layer>,
    time_mlp: Sequential,
    downs: Vec<DownBlock>,
    mid: MidBlock,
    ups: Vec<UpBlock>,
    out_dim: Option<usize>,
    final_res_block: ResNetBlock,
    final_conv: Conv2D,
    label: String,
}

#[derive(Debug)]
struct ResNetBlock {
    label:      String,
    branch1:    Sequential,               // gn1 → act1 → conv1
    branch2:    Sequential,               // gn2 → act2 → conv2
    skip_conv:  Option<Conv2D>,      // 1×1, in_channels != out_channels 일 때
    t_emb_proj: Option<Linear>,           // Linear(t_emb_dim → out_channels)
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

// ── DownBlock impl ───────────────────────────────────────────────────────────

impl DownBlock {
    /// * `in_ch`      - 입력 채널
    /// * `out_ch`     - 출력 채널 (resnet + downsample 모두 이 채널로 출력)
    /// * `num_groups` - GroupNorm 그룹 수
    /// * `t_emb_dim`  - time embedding 차원
    /// * `use_attn`   - SelfAttention 포함 여부
    fn new(
        in_ch: usize, out_ch: usize, num_groups: usize,
        t_emb_dim: Option<usize>, use_attn: bool, label: &str,
    ) -> MlResult<Self> {
        Ok(Self {
            resnet1: ResNetBlock::new(in_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res1", label))?,
            resnet2: ResNetBlock::new(out_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res2", label))?,
            attn: if use_attn {
                Some(SelfAttentionBlock::new(out_ch, num_groups, &format!("{}_attn", label))?)
            } else {
                None
            },
            downsample: Conv2D::new(out_ch, out_ch, (3, 3), (2, 2), (1, 1), &format!("{}_down", label))?,
        })
    }

    #[cfg(feature = "enableBackward")]
    fn forward(&mut self, x: &Variable, t_emb: &Variable) -> MlResult<Variable> {
        let mut h = self.resnet1.forward_with_t(x, Some(t_emb))?;
        h = self.resnet2.forward_with_t(&h, Some(t_emb))?;
        if let Some(ref mut attn) = self.attn {
            h = attn.apply(&h)?;
        }
        self.downsample.apply(&h)
    }

    fn predict_forward(&self, x: &dyn TensorBase, t_emb: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let mut h = self.resnet1.predict_with_t(x, Some(t_emb))?;
        h = self.resnet2.predict_with_t(&h, Some(t_emb))?;
        if let Some(ref attn) = self.attn {
            h = attn.predict(&h)?;
        }
        self.downsample.predict(&h)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p = self.resnet1.params();
        p.extend(self.resnet2.params());
        if let Some(ref attn) = self.attn { p.extend(attn.params()); }
        p.extend(self.downsample.params());
        p
    }
}

// ── MidBlock impl ───────────────────────────────────────────────────────────

impl MidBlock {
    fn new(channels: usize, num_groups: usize, t_emb_dim: Option<usize>, label: &str) -> MlResult<Self> {
        Ok(Self {
            resnet1: ResNetBlock::new(channels, channels, num_groups, t_emb_dim, &format!("{}_res1", label))?,
            attn:    SelfAttentionBlock::new(channels, num_groups, &format!("{}_attn", label))?,
            resnet2: ResNetBlock::new(channels, channels, num_groups, t_emb_dim, &format!("{}_res2", label))?,
        })
    }

    #[cfg(feature = "enableBackward")]
    fn forward(&mut self, x: &Variable, t_emb: &Variable) -> MlResult<Variable> {
        let mut h = self.resnet1.forward_with_t(x, Some(t_emb))?;
        h = self.attn.apply(&h)?;
        self.resnet2.forward_with_t(&h, Some(t_emb))
    }

    fn predict_forward(&self, x: &dyn TensorBase, t_emb: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let mut h = self.resnet1.predict_with_t(x, Some(t_emb))?;
        h = self.attn.predict(&h)?;
        self.resnet2.predict_with_t(&h, Some(t_emb))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p = self.resnet1.params();
        p.extend(self.attn.params());
        p.extend(self.resnet2.params());
        p
    }
}

// ── UpBlock impl ────────────────────────────────────────────────────────────

impl UpBlock {
    /// * `in_ch`      - concat 후 입력 채널 (skip_ch + prev_up_ch)
    /// * `out_ch`     - 출력 채널
    /// * `num_groups` - GroupNorm 그룹 수
    /// * `t_emb_dim`  - time embedding 차원
    /// * `use_attn`   - SelfAttention 포함 여부
    fn new(
        in_ch: usize, out_ch: usize, num_groups: usize,
        t_emb_dim: Option<usize>, use_attn: bool, label: &str,
    ) -> MlResult<Self> {
        Ok(Self {
            resnet1: ResNetBlock::new(in_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res1", label))?,
            resnet2: ResNetBlock::new(out_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res2", label))?,
            attn: if use_attn {
                Some(SelfAttentionBlock::new(out_ch, num_groups, &format!("{}_attn", label))?)
            } else {
                None
            },
            upsample_conv: Conv2D::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), &format!("{}_up_conv", label))?,
        })
    }

    /// NearestUpsample(2×) 적용 후 conv.
    #[cfg(feature = "enableBackward")]
    fn upsample_apply(&mut self, x: &Variable) -> MlResult<Variable> {
        let scale_h = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1])?);
        let scale_w = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1])?);
        let upsampled = NearestUpsample2d::new()?.apply(&[x, &scale_h, &scale_w])?;
        self.upsample_conv.apply(&upsampled)
    }

    /// NearestUpsample(2×) 적용 후 conv (추론).
    fn upsample_predict(&self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let scale_h = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;
        let scale_w = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;
        let upsampled = NearestUpsample2d::new()?.forward(&[x, &scale_h, &scale_w])?.remove(0);
        self.upsample_conv.predict(&upsampled)
    }

    #[cfg(feature = "enableBackward")]
    fn forward(&mut self, x: &Variable, t_emb: &Variable) -> MlResult<Variable> {
        let mut h = self.resnet1.forward_with_t(x, Some(t_emb))?;
        h = self.resnet2.forward_with_t(&h, Some(t_emb))?;
        if let Some(ref mut attn) = self.attn {
            h = attn.apply(&h)?;
        }
        self.upsample_apply(&h)
    }

    fn predict_forward(&self, x: &dyn TensorBase, t_emb: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let mut h = self.resnet1.predict_with_t(x, Some(t_emb))?;
        h = self.resnet2.predict_with_t(&h, Some(t_emb))?;
        if let Some(ref attn) = self.attn {
            h = attn.predict(&h)?;
        }
        self.upsample_predict(&h)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p = self.resnet1.params();
        p.extend(self.resnet2.params());
        if let Some(ref attn) = self.attn { p.extend(attn.params()); }
        p.extend(self.upsample_conv.params());
        p
    }
}

// ── Unet impl ───────────────────────────────────────────────────────────────

impl Unet {
    fn new(
        dim: usize, init_dim: Option<usize>, out_dim: Option<usize>, dim_mults: &[usize],
        channels: usize,
        self_condition: bool,
        resnet_block_groups: usize,
        learned_variance: bool, learned_sinusoidal_cond: bool, learned_sinusoidal_dim: usize, sinusoidal_pos_emb_theta: usize,
        random_fourier_features: bool,
        full_attn: Option<bool>, flash_attn: bool,
    ) -> Self {
        todo!("Unet::new — DownBlock/UpBlock/MidBlock 파라미터 조립 후 구현")
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
        let mut p = self.init_conv.params();
        p.extend(self.time_mlp.params());
        for d in &self.downs { p.extend(d.params()); }
        p.extend(self.mid.params());
        for u in &self.ups { p.extend(u.params()); }
        p.extend(self.final_res_block.params());
        p.extend(self.final_conv.params());
        p
    }

    fn label(&self) -> &str {
        &self.label
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
        })
    }

    /// 학습 경로: time embedding을 직접 받아 forward 실행.
    #[cfg(feature = "enableBackward")]
    pub fn forward_with_t(&mut self, x: &Variable, t_emb: Option<&Variable>) -> MlResult<Variable> {
        // gn1 → act1 → conv1
        let h = self.branch1.apply(x)?;

        // t_emb 주입: proj(t) [N, C_out] → reshape [N, C_out, 1, 1] → broadcast-add
        let h = if let (Some(proj), Some(t)) = (self.t_emb_proj.as_mut(), t_emb) {
            let t_proj = proj.apply(t)?; // [N, C_out]
            let h_shape = h.tensor().shape();
            let (n, c) = (h_shape[0], h_shape[1]);
            let shape_4d = Variable::new(Tensor::from_vec(vec![0.0; n * c], &[n, c, 1, 1])?);
            let t_4d = ReshapeOp::new()?.apply(&[&t_proj, &shape_4d])?;
            &h + &t_4d
        } else {
            h
        };

        // gn2 → act2 → conv2
        let h = self.branch2.apply(&h)?;

        // skip connection
        let skip = if let Some(ref mut sc) = self.skip_conv {
            sc.apply(x)?
        } else {
            x.clone()
        };

        Ok(&h + &skip)
    }

    /// 추론 경로: time embedding을 직접 받아 forward 실행.
    pub fn predict_with_t(&self, x: &dyn TensorBase, t_emb: Option<&dyn TensorBase>) -> MlResult<GlobalTensor<f32>> {
        // gn1 → act1 → conv1
        let h = self.branch1.predict(x)?.to_id()?;

        let h = if let (Some(proj), Some(t)) = (&self.t_emb_proj, t_emb) {
            let t_proj = proj.predict(t)?; // [N, C_out]
            let h_shape = h.shape();
            let (n, c) = (h_shape[0], h_shape[1]);
            let shape_4d = Tensor::from_vec(vec![0.0; n * c], &[n, c, 1, 1])?;
            let t_4d = ReshapeOp::new()?.forward(&[&t_proj, &shape_4d])?.remove(0).to_id()?;
            &h + &t_4d
        } else {
            h
        };

        // gn2 → act2 → conv2
        let h = self.branch2.predict(&h)?;

        // skip connection
        let skip: GlobalTensor<f32> = if let Some(ref sc) = self.skip_conv {
            sc.predict(x)?
        } else {
            GlobalTensor::from_vec(x.data().to_vec(), x.shape())?
        };

        let add_op = Add::new()?;
        Ok(add_op.forward(&[&h, &skip])?.remove(0))
    }
}

impl Layer for ResNetBlock {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        self.forward_with_t(input, None)
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        self.predict_with_t(input, None)
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

        // 10) softmax (axis=-1: 마지막 축 기준 row-wise)
        let softmax_axis = Variable::new(Tensor::from_vec(vec![-1.0], &[1, 1])?);
        let attn = SoftmaxOp::new()?.apply(&[&scores_scaled, &softmax_axis])?;

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
        let scaled_tensor = GlobalTensor::from_vec(scaled, scores.shape())?;

        // 10) softmax (axis=-1: 마지막 축 기준 row-wise)
        let softmax_axis = GlobalTensor::from_vec(vec![-1.0], &[1, 1])?;
        let attn = SoftmaxOp::new()?.forward(&[&scaled_tensor, &softmax_axis])?.remove(0);

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
    use crate::nn::activation::SoftmaxOp;
    use crate::tensor::operators::Function;

    // ── SelfAttentionBlock 테스트 ────────────────────────────────────────────

    #[test]
    fn self_attention_predict_shape() -> MlResult<()> {
        // C=16, num_groups=8 (16 % 8 == 0), spatial 4x4
        let block = SelfAttentionBlock::new(16, 8, "attn")?;
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

    // ── Axis-aware Softmax 테스트 ────────────────────────────────────────────

    #[test]
    fn softmax_axis_2d_row_sum() -> MlResult<()> {
        // [2, 3] 텐서에서 axis=1 → 각 row 합=1.0
        let input = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3])?;
        let axis = GlobalTensor::from_vec(vec![1.0], &[1, 1])?;
        let out = SoftmaxOp::new()?.forward(&[&input, &axis])?.remove(0);

        assert_eq!(out.shape(), &[2, 3]);
        // row 0 합 ≈ 1.0
        let row0_sum: f32 = out.data()[0..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6, "row0 sum = {}", row0_sum);
        // row 1 합 ≈ 1.0
        let row1_sum: f32 = out.data()[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-6, "row1 sum = {}", row1_sum);
        // row 1 은 uniform → 각 값 ≈ 1/3
        for &v in &out.data()[3..6] {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn softmax_axis_3d_last() -> MlResult<()> {
        // [2, 3, 4] 텐서, axis=-1 (=2) → 마지막 축 기준
        let n = 2 * 3 * 4;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let input = GlobalTensor::from_vec(data, &[2, 3, 4])?;
        let axis = GlobalTensor::from_vec(vec![-1.0], &[1, 1])?;
        let out = SoftmaxOp::new()?.forward(&[&input, &axis])?.remove(0);

        assert_eq!(out.shape(), &[2, 3, 4]);
        // 각 [2,3,:] row 의 합 ≈ 1.0
        for batch in 0..2 {
            for row in 0..3 {
                let start = batch * 12 + row * 4;
                let row_sum: f32 = out.data()[start..start + 4].iter().sum();
                assert!((row_sum - 1.0).abs() < 1e-5, "batch={} row={} sum={}", batch, row, row_sum);
            }
        }
        Ok(())
    }

    #[test]
    fn softmax_global_compat() -> MlResult<()> {
        // targets.len()==1 → 기존 전역 softmax 와 동일한 결과
        let input = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let out = SoftmaxOp::new()?.forward(&[&input])?.remove(0);
        let total: f32 = out.data().iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        // 순서 보존: out[2] > out[1] > out[0]
        assert!(out.data()[2] > out.data()[1]);
        assert!(out.data()[1] > out.data()[0]);
        Ok(())
    }

    // ── ResNetBlock forward_with_t 테스트 ────────────────────────────────────

    #[test]
    fn resnet_block_predict_no_temb() -> MlResult<()> {
        // t_emb 없이 정상 통과 확인
        let block = ResNetBlock::new(8, 8, 4, None, "res")?;
        let data: Vec<f32> = (0..(1 * 8 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[1, 8, 4, 4])?;
        let out = block.predict_with_t(&x, None)?;
        assert_eq!(out.shape(), &[1, 8, 4, 4]);
        Ok(())
    }

    #[test]
    fn resnet_block_predict_with_temb() -> MlResult<()> {
        let t_emb_dim = 16;
        let block = ResNetBlock::new(8, 8, 4, Some(t_emb_dim), "res")?;
        let data: Vec<f32> = (0..(1 * 8 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[1, 8, 4, 4])?;

        // t_emb: [1, 16]
        let t_data: Vec<f32> = (0..t_emb_dim).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;

        let out_with = block.predict_with_t(&x, Some(&t))?;
        let out_without = block.predict_with_t(&x, None)?;
        assert_eq!(out_with.shape(), &[1, 8, 4, 4]);
        // t_emb 주입 시 출력이 달라져야 함
        assert_ne!(out_with.data(), out_without.data());
        Ok(())
    }

    #[test]
    fn resnet_block_channel_change() -> MlResult<()> {
        // in_channels != out_channels → skip_conv 활성화
        let block = ResNetBlock::new(8, 16, 4, None, "res_ch")?;
        let data: Vec<f32> = (0..(1 * 8 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[1, 8, 4, 4])?;
        let out = block.predict(&x)?;
        assert_eq!(out.shape(), &[1, 16, 4, 4]);
        Ok(())
    }

    // ── DownBlock / MidBlock 테스트 ──────────────────────────────────────────

    #[test]
    fn downblock_predict_halves_spatial() -> MlResult<()> {
        // 8×8 → downsample(stride=2) → 4×4
        let t_emb_dim = 16;
        let block = DownBlock::new(8, 8, 4, Some(t_emb_dim), false, "down0")?;
        let data: Vec<f32> = (0..(1 * 8 * 8 * 8)).map(|i| i as f32 * 0.001).collect();
        let x = Tensor::from_vec(data, &[1, 8, 8, 8])?;
        let t_data: Vec<f32> = vec![0.1; t_emb_dim];
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;

        let out = block.predict_forward(&x, &t)?;
        assert_eq!(out.shape(), &[1, 8, 4, 4]);
        Ok(())
    }

    #[test]
    fn midblock_predict_preserves_shape() -> MlResult<()> {
        let t_emb_dim = 16;
        let block = MidBlock::new(8, 4, Some(t_emb_dim), "mid")?;
        let data: Vec<f32> = (0..(1 * 8 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[1, 8, 4, 4])?;
        let t_data: Vec<f32> = vec![0.1; t_emb_dim];
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;

        let out = block.predict_forward(&x, &t)?;
        assert_eq!(out.shape(), &[1, 8, 4, 4]);
        Ok(())
    }

    #[test]
    fn upblock_predict_doubles_spatial() -> MlResult<()> {
        // 4×4 → NearestUpsample(2×) + Conv → 8×8
        let t_emb_dim = 16;
        let block = UpBlock::new(8, 8, 4, Some(t_emb_dim), false, "up0")?;
        let data: Vec<f32> = (0..(1 * 8 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[1, 8, 4, 4])?;
        let t_data: Vec<f32> = vec![0.1; t_emb_dim];
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;

        let out = block.predict_forward(&x, &t)?;
        assert_eq!(out.shape(), &[1, 8, 8, 8]);
        Ok(())
    }

    // ── Tensor::randn 테스트 ────────────────────────────────────────────────

    #[test]
    fn randn_shape_and_distribution() -> MlResult<()> {
        let t = Tensor::randn(&[1000]);
        assert_eq!(t.shape(), &[1000]);

        let data = t.data();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

        // N(0,1): 평균 ≈ 0, 분산 ≈ 1 (통계적 허용 범위)
        assert!(mean.abs() < 0.15, "mean = {} (expected ≈ 0)", mean);
        assert!((variance - 1.0).abs() < 0.25, "variance = {} (expected ≈ 1)", variance);
        Ok(())
    }
}
