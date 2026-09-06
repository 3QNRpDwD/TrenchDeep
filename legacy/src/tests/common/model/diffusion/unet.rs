use super::*;

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                          U-Net for DDPM                                 ║
// ║                                                                         ║
// ║  DDPM (Ho et al., 2020) 의 노이즈 예측 네트워크 ε_θ(x_t, t).           ║
// ║                                                                         ║
// ║  ── 핵심 아이디어 ──                                                     ║
// ║  디퓨전 모델은 깨끗한 이미지 x_0 에 점진적으로 가우시안 노이즈를 추가하여 ║
// ║  x_T ≈ N(0, I) 를 만드는 forward process 와, 그 역방향으로 노이즈를      ║
// ║  제거하여 이미지를 복원하는 reverse process 로 구성됨.                ║
// ║                                                                         ║
// ║  U-Net 의 역할은 reverse process 에서 "현재 시점 t 의 이미지 x_t 에     ║
// ║  어떤 노이즈가 섞여 있는지"를 예측하는 것:                          ║
// ║                                                                         ║
// ║      ε_θ(x_t, t) ≈ ε   (실제로 추가된 노이즈)                           ║
// ║                                                                         ║
// ║  학습 목표:  L = E[‖ε - ε_θ(x_t, t)‖²]   (단순 MSE)                   ║
// ║                                                                         ║
// ║  ── U-Net 구조 ──                                                        ║
// ║                                                                         ║
// ║  입력 x_t ──→ init_conv ──→ [Down₁ → Down₂ → ... → Downₙ]             ║
// ║                                   ↓ skip connections ↓                  ║
// ║                               MidBlock (bottleneck)                     ║
// ║                                   ↓ skip concat  ↓                     ║
// ║              [Upₙ → ... → Up₂ → Up₁] ──→ final_conv ──→ ε_θ           ║
// ║                                                                         ║
// ║  time embedding t ──→ SinusoidalPE → MLP → 각 ResNetBlock에 주입        ║
// ║                                                                         ║
// ║  "skip connection" 이 U-Net의 핵심 특징.                           ║
// ║  Down path 에서 생성된 feature map을 Up path 의 대응되는 해상도에        ║
// ║  채널 방향으로 concat 하여, 고해상도 세부 정보를 보존.              ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  1. 구조체 정의 (Struct Definitions)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ── DownBlock ──────────────────────────────────────────────────────────────
///
/// U-Net 인코더(down path)의 한 단계.
///
/// ```text
///  입력 [N, C_in, H, W]
///   ├── ResNet₁(C_in → C_out) + t_emb 주입
///   ├── ResNet₂(C_out → C_out) + t_emb 주입
///   ├── (optional) SelfAttention(C_out)
///   └── Downsample: stride-2 Conv2D → [N, C_out, H/2, W/2]
/// ```
///
/// **왜 downsample 이 필요한가?**
/// 공간 해상도를 점진적으로 줄이면서 채널 수를 늘려,
/// 네트워크가 점점 더 추상적인 (global) 특징을 학습하게함.
/// 이게 바로 "coarse-to-fine" 전략임
#[derive(Debug)]
struct DownBlock {
    resnet1:    ResNetBlock,
    resnet2:    ResNetBlock,
    attn:       Option<SelfAttentionBlock>,
    downsample: Conv2D,  // stride=2 conv → 공간 해상도 1/2
}

/// ── MidBlock (Bottleneck) ──────────────────────────────────────────────────
///
/// U-Net 의 가장 깊은 지점. 가장 낮은 해상도에서 전역적(global) 정보를 처리.
///
/// ```text
///  [N, C_mid, H_min, W_min]
///   ├── ResNet₁ + t_emb
///   ├── SelfAttention  ← 낮은 해상도이므로 attention 비용이 적음
///   └── ResNet₂ + t_emb
/// ```
///
/// **왜 bottleneck 에 attention 을 두는가?**
/// 가장 작은 feature map (예: 4×4, 8×8) 에서 attention 을 수행하면
/// O(HW²) 비용이 감당 가능하고, 이미지의 전역적 구조(global structure)를
/// 포착할 수 있음.
#[derive(Debug)]
struct MidBlock {
    resnet1: ResNetBlock,
    attn:    SelfAttentionBlock,
    resnet2: ResNetBlock,
}

/// ── UpBlock ────────────────────────────────────────────────────────────────
///
/// U-Net 디코더(up path)의 한 단계.
///
/// ```text
///  입력 = Concat(skip_from_down, previous_up)   ← ★ skip connection!
///  [N, C_skip + C_prev, H, W]
///   ├── ResNet₁(C_skip + C_prev → C_out) + t_emb
///   ├── ResNet₂(C_out → C_out) + t_emb
///   ├── (optional) SelfAttention(C_out)
///   └── Upsample: NearestUpsample(2×) + Conv2D → [N, C_out, 2H, 2W]
/// ```
///
/// **skip connection 의 수학적 의미:**
/// Down path 의 feature h_down ∈ ℝ^{C×H×W} 과 Up path 의 feature h_up 을
/// 채널 축으로 concat 하면 [h_down; h_up] ∈ ℝ^{2C×H×W} 가 됨.
/// 이를 통해 인코더의 고해상도 디테일을 디코더가 직접 참조할 수 있어,
/// 노이즈 제거 시 세밀한 구조를 보존.
#[derive(Debug)]
struct UpBlock {
    resnet1:       ResNetBlock,
    resnet2:       ResNetBlock,
    attn:          Option<SelfAttentionBlock>,
    upsample_conv: Conv2D,  // upsample 후 3×3 conv (채널 보정)
}

/// ── U-Net 본체 ─────────────────────────────────────────────────────────────
///
/// DDPM 의 노이즈 예측 네트워크 ε_θ.
///
/// ## 전체 데이터 흐름 (Forward Pass)
///
/// ```text
///  (x_t, t)                       ← 입력: noisy image + timestep
///     │
///     ├── t → SinusoidalPE → MLP → t_emb ∈ ℝ^{t_emb_dim}
///     │       (시간 정보를 연속 벡터로 인코딩)
///     │
///     ├── x_t → init_conv → h₀ ∈ ℝ^{init_dim × H × W}
///     │
///     ├── ╔══ Down Path (인코더) ══════════════════════╗
///     │   ║  h₁ = Down₁(h₀, t_emb)  →  skip₁ = h₁    ║
///     │   ║  h₂ = Down₂(h₁, t_emb)  →  skip₂ = h₂    ║   skip 저장
///     │   ║  ...                                        ║
///     │   ╚═════════════════════════════════════════════╝
///     │
///     ├── h_mid = MidBlock(hₙ, t_emb)
///     │
///     ├── ╔══ Up Path (디코더) ════════════════════════╗
///     │   ║  h = Concat(h_mid, skipₙ)  →  Upₙ(h, t_emb)║  skip 사용
///     │   ║  h = Concat(h, skipₙ₋₁)   →  Upₙ₋₁(h, t_emb)║
///     │   ║  ...                                        ║
///     │   ╚═════════════════════════════════════════════╝
///     │
///     ├── h = final_res_block(Concat(h, h₀), t_emb)
///     └── ε_θ = final_conv(h) ∈ ℝ^{C × H × W}
/// ```
///
/// ## dim_mults 의 의미
///
/// `dim_mults = [1, 2, 4]` 이고 `dim = 64` 이면:
/// - Down₁: 64 → 64   (×1)
/// - Down₂: 64 → 128  (×2)
/// - Down₃: 128 → 256 (×4)   ← 여기서 Mid
/// - Up₃:   256 → 128 (×2)
/// - Up₂:   128 → 64  (×1)
/// - Up₁:   64 → 64
///
/// 즉, dim_mults 는 각 해상도 단계에서의 "채널 배수"를 지정.
#[derive(Debug)]
pub struct Unet {
    /// 입력 이미지의 채널 수 (e.g., 1 for grayscale, 3 for RGB)
    channels: usize,
    /// self-conditioning 사용 여부 (이전 예측을 입력에 concat)
    self_condition: bool,
    /// x_t 를 init_dim 채널로 변환하는 첫 번째 convolution
    init_conv: Box<dyn Layer>,
    /// timestep t → t_emb 벡터를 생성하는 MLP (embedding.rs 참조)
    time_mlp: TimeEmbeddingMLP,
    /// Down path 블록들 (해상도 감소)
    downs: Vec<DownBlock>,
    /// Bottleneck 블록
    mid: MidBlock,
    /// Up path 블록들 (해상도 복원)
    ups: Vec<UpBlock>,
    /// 출력 채널 수 (None 이면 입력과 동일)
    out_dim: Option<usize>,
    /// 마지막 skip concat 후 적용되는 ResNet
    final_res_block: ResNetBlock,
    /// 최종 출력 convolution (→ 입력과 동일한 채널 수)
    final_conv: Conv2D,
    label: String,
}

/// ── ResNetBlock ─────────────────────────────────────────────────────────────
///
/// 잔차 학습(Residual Learning) 블록 + 시간 조건부 주입.
///
/// ## 수학적 정의
///
/// ```text
///  h = Conv₂(SiLU(GN₂( Conv₁(SiLU(GN₁(x))) + proj(t_emb) )))
///  y = h + skip(x)
/// ```
///
/// 여기서:
/// - `GN`: GroupNorm — 채널을 그룹으로 나누어 정규화 (BatchNorm보다 작은 배치에 강건)
/// - `SiLU(x) = x · σ(x)`: Smooth ReLU — 0 근처에서 미분 가능한 활성화 함수
/// - `proj(t_emb)`: time embedding 을 채널 차원으로 투사하여 broadcast-add
/// - `skip(x)`: 입력을 출력에 직접 더함 (차원 불일치 시 1×1 conv)
///
/// ## 잔차 연결의 의미
///
/// `y = F(x) + x` 형태로, 네트워크가 학습하는 것은 입력과 출력의 "차이"(잔차).
/// 이 구조의 gradient:  ∂y/∂x = ∂F/∂x + I
/// 항등 행렬 I 가 더해지므로 gradient 가 최소 1 이상이 보장되어,
/// 깊은 네트워크에서도 gradient vanishing 이 크게 완화됨.
///
/// ## 시간 조건부 주입 (Time Conditioning)
///
/// DDPM에서 U-Net은 "지금이 몇 번째 노이즈 제거 단계인지"를 알아야 .
/// time embedding t_emb ∈ ℝ^D 를 Linear 로 채널 차원 C 로 변환한 후,
/// [N, C, 1, 1] 로 reshape 하여 feature map 에 broadcast-add .
/// 이로써 각 채널의 activation 이 시간 t 에 따라 조절(modulation)됨.
#[derive(Debug)]
struct ResNetBlock {
    label:      String,
    /// 첫 번째 변환: GroupNorm → SiLU → Conv2D (in_ch → out_ch)
    branch1:    Sequential,
    /// 두 번째 변환: GroupNorm → SiLU → Conv2D (out_ch → out_ch)
    branch2:    Sequential,
    /// 차원 불일치 시 1×1 conv 로 skip connection 채널을 맞춤
    skip_conv:  Option<Conv2D>,
    /// t_emb ∈ ℝ^D → ℝ^{C_out} 투사 (None이면 시간 주입 없음)
    t_emb_proj: Option<Linear>,
}

/// ── SelfAttentionBlock ─────────────────────────────────────────────────────
///
/// Scaled Dot-Product Self-Attention.
///
/// ## 수학적 정의
///
/// 입력 X ∈ ℝ^{N×(HW)×C} 에 대해:
///
/// ```text
///  Q = X · W_Q,   K = X · W_K,   V = X · W_V      (선형 투사)
///
///  Attention(Q, K, V) = softmax( Q · K^T / √C ) · V
///
///                         ↑ 스케일링 ─ dot product 의 분산이
///                           C에 비례해 커지는 것을 보정
/// ```
///
/// ## 왜 Attention 이 필요한가?
///
/// Convolution 은 지역적(local) 패턴만 포착 (커널 크기 범위).
/// Self-Attention 은 feature map 의 모든 위치 쌍 간의 관계를 계산하므로,
/// 이미지의 먼 곳에 있는 구조적 패턴(대칭, 반복 등)도 포착.
///
/// ## 구현 흐름
///
/// ```text
///  [N,C,H,W] → GN → reshape [N,HW,C] → Q,K,V
///   → scores = Q·K^T/√C → softmax → ·V
///   → out_proj → reshape [N,C,H,W] → +input (residual)
/// ```
#[derive(Debug)]
struct SelfAttentionBlock {
    label:    String,
    gn:       GroupNorm,
    query:    Linear,
    key:      Linear,
    value:    Linear,
    out_proj: Linear,
    /// 1/√C 스케일링 상수 (shape [1,1] — Mul 브로드캐스트용)
    scale:    Variable,
    channels: usize,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  2. DownBlock 구현
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl DownBlock {
    /// DownBlock 생성.
    ///
    /// * `in_ch`      - 입력 채널 (이전 단계의 출력)
    /// * `out_ch`     - 출력 채널 (이 단계의 목표 채널 수)
    /// * `num_groups` - GroupNorm 그룹 수 (C % groups == 0 필요)
    /// * `t_emb_dim`  - time embedding 벡터 차원 (각 ResNet에 주입)
    /// * `use_attn`   - 이 해상도에서 SelfAttention 을 사용할지
    fn new(
        in_ch: usize, out_ch: usize, num_groups: usize,
        t_emb_dim: Option<usize>, use_attn: bool, label: &str,
    ) -> MlResult<Self> {
        Ok(Self {
            // 첫 번째 ResNet: 채널 변환 (in_ch → out_ch)
            resnet1: ResNetBlock::new(in_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res1", label))?,
            // 두 번째 ResNet: 채널 유지 (out_ch → out_ch), 더 깊은 특징 추출
            resnet2: ResNetBlock::new(out_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res2", label))?,
            // (선택) Self-Attention — 보통 낮은 해상도에서만 사용 (비용 절감)
            attn: if use_attn {
                Some(SelfAttentionBlock::new(out_ch, num_groups, &format!("{}_attn", label))?)
            } else {
                None
            },
            // stride=2 convolution: 공간 해상도를 정확히 절반으로 축소
            // H_out = (H + 2·pad - kernel) / stride + 1 = (H + 2 - 3) / 2 + 1 = H/2
            downsample: Conv2D::new(out_ch, out_ch, (3, 3), (2, 2), (1, 1), &format!("{}_down", label))?,
        })
    }

    /// 학습 forward: ResNet×2 → (Attn) → Downsample.
    /// downsample 전의 feature 를 반환하여 skip connection 으로 사용.
    #[cfg(feature = "enableBackward")]
    fn forward(&mut self, x: &Variable, t_emb: &Variable) -> MlResult<(Variable, Variable)> {
        let mut h = self.resnet1.forward_with_t(x, Some(t_emb))?;
        h = self.resnet2.forward_with_t(&h, Some(t_emb))?;
        if let Some(ref mut attn) = self.attn {
            h = attn.apply(&h)?;
        }
        // ★ skip = downsample 전의 h (Up path 에서 concat 할 대상)
        let skip = h.clone();
        let downsampled = self.downsample.apply(&h)?;
        Ok((downsampled, skip))
    }

    /// 추론 forward: 동일 로직, GlobalTensor 반환.
    fn predict_forward(&self, x: &dyn TensorBase, t_emb: &dyn TensorBase) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>)> {
        let mut h = self.resnet1.predict_with_t(x, Some(t_emb))?;
        h = self.resnet2.predict_with_t(&h, Some(t_emb))?;
        if let Some(ref attn) = self.attn {
            h = attn.predict(&h)?;
        }
        // skip = downsample 전 feature
        let skip = GlobalTensor::from_vec(h.data().to_vec(), h.shape())?;
        let downsampled = self.downsample.predict(&h)?;
        Ok((downsampled, skip))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p = self.resnet1.params();
        p.extend(self.resnet2.params());
        if let Some(ref attn) = self.attn { p.extend(attn.params()); }
        p.extend(self.downsample.params());
        p
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  3. MidBlock 구현
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  4. UpBlock 구현
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl UpBlock {
    /// UpBlock 생성.
    ///
    /// * `pre_ch` - upsample 입력 채널 (이전 Up 블록 또는 Mid 블록의 출력)
    /// * `in_ch`  - concat 후 입력 채널 수 (= pre_ch + skip_channels)
    /// * `out_ch` - 출력 채널 수
    ///
    /// ## Upsample Conv 의 역할
    ///
    /// NearestUpsample 은 채널 수를 변경하지 않으므로,
    /// upsample 후 3×3 conv 로 (1) 채널을 유지하고 (2) 업샘플 아티팩트를 스무딩.
    /// 이 conv 의 입력 채널 = pre_ch (upsample 전의 채널 수).
    fn new(
        pre_ch: usize, in_ch: usize, out_ch: usize, num_groups: usize,
        t_emb_dim: Option<usize>, use_attn: bool, label: &str,
    ) -> MlResult<Self> {
        Ok(Self {
            // ★ in_ch = (pre_ch + skip_ch) → out_ch 로 채널 축소
            resnet1: ResNetBlock::new(in_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res1", label))?,
            resnet2: ResNetBlock::new(out_ch, out_ch, num_groups, t_emb_dim, &format!("{}_res2", label))?,
            attn: if use_attn {
                Some(SelfAttentionBlock::new(out_ch, num_groups, &format!("{}_attn", label))?)
            } else {
                None
            },
            // upsample conv: pre_ch → pre_ch (채널 유지, 스무딩 목적)
            upsample_conv: Conv2D::new(pre_ch, pre_ch, (3, 3), (1, 1), (1, 1), &format!("{}_up_conv", label))?,
        })
    }

    /// NearestUpsample(2×): 각 픽셀을 2×2 블록으로 복제 → conv 로 스무딩.
    ///
    /// 왜 transposed convolution 대신 nearest upsample 을 쓰는가?
    /// Transposed conv 는 "체커보드 아티팩트(checkerboard artifact)"를
    /// 만들기 쉽지만, nearest + conv 조합은 더 부드러운 결과를 냄.
    #[cfg(feature = "enableBackward")]
    fn upsample_apply(&mut self, x: &Variable) -> MlResult<Variable> {
        let scale_h = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1])?);
        let scale_w = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1])?);
        let upsampled = NearestUpsample2d::new()?.apply(&[x, &scale_h, &scale_w])?;
        self.upsample_conv.apply(&upsampled)
    }

    /// NearestUpsample(2×) 추론 경로.
    fn upsample_predict(&self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let scale_h = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;
        let scale_w = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;
        let upsampled = NearestUpsample2d::new()?.forward(&[x, &scale_h, &scale_w])?.remove(0);
        self.upsample_conv.predict(&upsampled)
    }

    /// 학습 forward: Upsample → ResNet×2 → (Attn).
    ///
    /// ★ 연산 순서가 중요:
    ///   1. 먼저 upsample 하여 해상도를 skip 과 맞춤
    ///   2. 외부에서 skip 과 concat (Unet.forward_with_t 에서 수행)
    ///   3. concat 된 입력으로 ResNet + Attn 처리
    ///
    /// 이 메서드는 concat 후의 입력 x 를 받아 ResNet + Attn 만 수행.
    /// upsample 은 Unet forward 에서 별도로 호출.
    #[cfg(feature = "enableBackward")]
    fn forward(&mut self, x: &Variable, t_emb: &Variable) -> MlResult<Variable> {
        let mut h = self.resnet1.forward_with_t(x, Some(t_emb))?;
        h = self.resnet2.forward_with_t(&h, Some(t_emb))?;
        if let Some(ref mut attn) = self.attn {
            h = attn.apply(&h)?;
        }
        Ok(h)
    }

    /// 추론 forward: ResNet×2 → (Attn). upsample 은 외부에서.
    fn predict_forward(&self, x: &dyn TensorBase, t_emb: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let mut h = self.resnet1.predict_with_t(x, Some(t_emb))?;
        h = self.resnet2.predict_with_t(&h, Some(t_emb))?;
        if let Some(ref attn) = self.attn {
            h = attn.predict(&h)?;
        }
        Ok(h)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut p = self.resnet1.params();
        p.extend(self.resnet2.params());
        if let Some(ref attn) = self.attn { p.extend(attn.params()); }
        p.extend(self.upsample_conv.params());
        p
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  5. Unet 핵심 구현 — 조립(new) + Forward Pass
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl Unet {
    /// U-Net 생성.
    ///
    /// ## 파라미터 설명
    ///
    /// * `dim`       - 기본 채널 수 (예: 64). 이 값에 dim_mults 를 곱해 각 단계 채널을 결정.
    /// * `init_dim`  - init_conv 출력 채널 (None 이면 dim 사용)
    /// * `out_dim`   - 최종 출력 채널 (None 이면 입력 channels 와 동일)
    /// * `dim_mults` - 각 해상도 단계의 채널 배수 (예: [1, 2, 4])
    ///                 길이 = Down/Up 블록 수
    /// * `channels`  - 입력 이미지 채널 (1=grayscale, 3=RGB)
    /// * `resnet_block_groups` - GroupNorm 그룹 수
    /// * `use_attn_at` - 각 해상도에서 attention 사용 여부 (예: [false, false, true])
    ///                   길이가 dim_mults 와 같아야 .
    ///
    /// ## 채널 흐름 예시 (dim=64, dim_mults=[1,2,4], init_dim=64)
    ///
    /// ```text
    ///  입력 [N, 1, 32, 32]
    ///    → init_conv → [N, 64, 32, 32]     (init_dim)
    ///
    ///  Down₁: 64 → 64   [N, 64, 16, 16]   (64×1 → 64×1)
    ///  Down₂: 64 → 128  [N, 128, 8, 8]    (64×1 → 64×2)
    ///  Down₃: 128 → 256 [N, 256, 4, 4]    (64×2 → 64×4)
    ///
    ///  Mid:   256        [N, 256, 4, 4]
    ///
    ///  Up₃:   256+256=512 → 128  [N, 128, 8, 8]   (concat → 64×2)
    ///  Up₂:   128+128=256 → 64   [N, 64, 16, 16]  (concat → 64×1)
    ///  Up₁:   64+64=128   → 64   [N, 64, 32, 32]  (concat → 64×1)
    ///
    ///  final: Concat(h, h₀) = 128 → final_res → final_conv → [N, 1, 32, 32]
    /// ```
    pub fn 
    new(
        dim: usize,
        init_dim: Option<usize>,
        out_dim: Option<usize>,
        dim_mults: &[usize],
        channels: usize,
        resnet_block_groups: usize,
        use_attn_at: &[bool],
    ) -> MlResult<Self> {
        let init_dim = init_dim.unwrap_or(dim);
        let out_dim_val = out_dim.unwrap_or(channels);
        let num_stages = dim_mults.len();

        // ── Step 1: Time Embedding MLP ──────────────────────────────
        //
        // timestep t (정수) → SinusoidalPE → MLP → t_emb ∈ ℝ^{t_emb_dim}
        //
        // t_emb_dim = dim × 4 (DDPM 논문 관례):
        //   time 정보를 충분히 풍부한 표현으로 변환하기 위해
        //   기본 채널의 4배 차원을 사용함.
        //
        // 구현: embedding.rs 의 TimeEmbeddingMLP 참조
        //   [N, 1] → SinusoidalPE → [N, dim] → Linear → SiLU → Linear → [N, t_emb_dim]
        let t_emb_dim = dim * 4;
        let time_mlp = TimeEmbeddingMLP::new(dim, t_emb_dim)?;

        // ── Step 2: Initial Convolution ─────────────────────────────
        //
        // 입력 이미지의 채널 수(1 또는 3)를 init_dim 으로 변환.
        // 여기가 U-Net 의 시작지점.
        let init_conv: Box<dyn Layer> = Box::new(
            Conv2D::new(channels, init_dim, (3, 3), (1, 1), (1, 1), "init_conv")?
        );

        // ── Step 3: 각 해상도 단계의 채널 수 계산 ──────────────────
        //
        // dims = [init_dim, dim×m₁, dim×m₂, ..., dim×mₙ]
        //
        // 예: dim=64, dim_mults=[1,2,4], init_dim=64
        //     dims = [64, 64, 128, 256]
        //     in_out pairs = [(64,64), (64,128), (128,256)]
        let mut dims = vec![init_dim];
        for &m in dim_mults {
            dims.push(dim * m);
        }

        // (in_ch, out_ch) 쌍: Down path 각 단계의 입출력 채널
        let in_out: Vec<(usize, usize)> = dims.windows(2)
            .map(|w| (w[0], w[1]))
            .collect();

        // ── Step 4: Down Path 블록 생성 ─────────────────────────────
        //
        // 각 단계에서:
        //   - 채널 수 변환 (in_ch → out_ch)
        //   - 공간 해상도 절반 (H×W → H/2 × W/2)
        //   - (optional) Self-Attention
        let mut downs = Vec::with_capacity(num_stages);
        for (i, &(in_ch, out_ch)) in in_out.iter().enumerate() {
            let use_attn = use_attn_at.get(i).copied().unwrap_or(false);
            downs.push(DownBlock::new(
                in_ch, out_ch, resnet_block_groups,
                Some(t_emb_dim), use_attn,
                &format!("down_{}", i),
            )?);
        }

        // ── Step 5: Mid (Bottleneck) 블록 생성 ─────────────────────
        //
        // 가장 깊은 지점: 최대 채널, 최소 해상도
        let mid_ch = *dims.last().unwrap();
        let mid = MidBlock::new(
            mid_ch, resnet_block_groups, Some(t_emb_dim), "mid",
        )?;

        // ── Step 6: Up Path 블록 생성 ───────────────────────────────
        //
        // Down path 를 역순으로 거슬러 올라감.
        //
        // ★ 핵심: UpBlock 의 입력 채널 = skip_ch + previous_up_ch
        //
        // 예: Down 이 [(64,64), (64,128), (128,256)] 이었다면
        //     Up 은 역순으로:
        //       Up₃: in = 256(mid) + 256(skip₃) = 512  → out = 128
        //       Up₂: in = 128(up₃) + 128(skip₂) = 256  → out = 64
        //       Up₁: in = 64(up₂)  + 64(skip₁)  = 128  → out = 64
        let mut ups = Vec::with_capacity(num_stages);
        for (i, &(_, out_ch)) in in_out.iter().enumerate().rev() {
            // 이전 Up 블록의 출력 채널 (첫 번째 Up 은 mid_ch)
            let prev_up_ch = if i == num_stages - 1 {
                mid_ch
            } else {
                in_out[i + 1].0  // 다음 단계의 in_ch = 이 단계의 up 출력
            };
            // skip connection 채널 = 이 단계의 Down 출력 = out_ch
            let skip_ch = out_ch;
            // upsample 후: prev_up_ch (유지), concat: prev_up_ch + skip_ch
            let concat_ch = prev_up_ch + skip_ch;
            // Up 출력 채널 = 이 단계의 Down 입력 (= 한 단계 위의 채널 수)
            let up_out_ch = in_out[i].0;

            let use_attn = use_attn_at.get(i).copied().unwrap_or(false);
            ups.push(UpBlock::new(
                prev_up_ch,  // upsample conv 의 입력 채널
                concat_ch, up_out_ch, resnet_block_groups,
                Some(t_emb_dim), use_attn,
                &format!("up_{}", i),
            )?);
        }

        // ── Step 7: Final Block ─────────────────────────────────────
        //
        // 마지막 Up 출력과 init_conv 출력(h₀)을 concat 후
        // ResNet + Conv 로 최종 출력 생성.
        //
        // 왜 마지막에도 skip concat 을 하는가?
        // init_conv 직후의 feature 에는 입력 이미지의 가장 세밀한
        // 저수준 정보(edge, texture)가 담겨 있어, 최종 노이즈 예측에 직접적으로 도움이 됨.
        let final_in_ch = init_dim + init_dim; // init_conv 출력 + 마지막 Up 출력
        let final_res_block = ResNetBlock::new(
            final_in_ch, init_dim, resnet_block_groups,
            Some(t_emb_dim), "final_res",
        )?;
        let final_conv = Conv2D::new(
            init_dim, out_dim_val, (1, 1), (1, 1), (0, 0), "final_conv",
        )?;

        Ok(Self {
            channels,
            self_condition: false,
            init_conv,
            time_mlp,
            downs,
            mid,
            ups,
            out_dim: out_dim,
            final_res_block,
            final_conv,
            label: "Unet".to_string(),
        })
    }
}

/// ── Unet Forward Pass ──────────────────────────────────────────────────────
///
/// U-Net 은 두 개의 입력을 받음:
///   1. `x_t`:  noisy image [N, C, H, W]
///   2. `t_emb`: time embedding [N, 1] (timestep 스칼라 배치)
///
/// Layer trait 의 apply/predict 는 단일 입력만 받으므로,
/// x_t 와 t_emb 를 채널 방향으로 합치거나 별도 메서드를 사용.
///
/// 여기서는 `forward_with_t` / `predict_with_t` 메서드로 구현하고,
/// Layer trait 은 time embedding 없는 fallback 을 제공.
impl Unet {
    /// 추론(inference) 경로.
    ///
    /// * `x_t`   - noisy 이미지 [N, C, H, W]
    /// * `t_raw` - timestep [N, 1] (SinusoidalPE 로 인코딩 전 raw 값)
    pub fn predict_with_t(&self, x_t: &dyn TensorBase, t_raw: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        // ── (1) Time Embedding ──────────────────────────────────────
        // t_raw [N, 1] → SinusoidalPE → MLP → t_emb [N, t_emb_dim]
        //
        // 직관: "지금 노이즈 제거의 몇 번째 단계인지"를 연속 벡터로 표현
        let t_emb = self.time_mlp.predict(t_raw)?;

        // ── (2) Init Conv ───────────────────────────────────────────
        // x_t [N, C, H, W] → h₀ [N, init_dim, H, W]
        let h0 = self.init_conv.predict(x_t)?;
        // h₀ 를 저장: 마지막에 final skip concat 에 사용
        let h0_skip = GlobalTensor::from_vec(h0.data().to_vec(), h0.shape())?;

        // ── (3) Down Path (인코더) ──────────────────────────────────
        //
        // 각 DownBlock 은:
        //   - 입력을 처리하여 더 추상적인 feature 추출
        //   - downsample 전의 feature 를 skip 으로 저장
        //   - 공간 해상도를 절반으로 축소
        //
        // 예: h₀ [8,16,16] → Down₁ → skip₁[8,16,16], h[8,8,8]
        //                   → Down₂ → skip₂[16,8,8],  h[16,4,4]
        let mut h: GlobalTensor<f32> = h0;
        let mut skips: Vec<GlobalTensor<f32>> = Vec::with_capacity(self.downs.len());

        for down in self.downs.iter() {
            let (downsampled, skip) = down.predict_forward(&h, &t_emb)?;
            skips.push(skip);
            h = downsampled;
        }

        // ── (4) Mid Block (병목) ────────────────────────────────────
        // 가장 낮은 해상도에서 전역적 정보를 attention 으로 처리
        h = self.mid.predict_forward(&h, &t_emb)?;

        // ── (5) Up Path (디코더) ────────────────────────────────────
        //
        //  순서: Upsample → Concat(skip) → ResNet + Attn
        //
        //   h 는 현재 해상도가 가장 낮음.
        //   각 Up 단계에서:
        //     1) h 를 2× upsample → skip 과 같은 해상도로 복원
        //     2) skip 과 채널 방향으로 concat
        //     3) ResNet + Attn 으로 처리
        //
        // 예: h[16,4,4] → upsample[16,8,8] → concat(skip₂[16,8,8]) → [32,8,8]
        //       → ResNet → [8,8,8]
        //     h[8,8,8] → upsample[8,16,16] → concat(skip₁[8,16,16]) → [16,16,16]
        //       → ResNet → [8,16,16]
        let concat = Concat::new()?;
        let axis_ch = GlobalTensor::from_vec(vec![1.0], &[1, 1])?; // axis=1 (채널 축)

        for up in self.ups.iter() {
            // Step 1: Upsample — 해상도 2배 복원
            let h_upsampled = up.upsample_predict(&h)?;

            // Step 2: Skip Connection (LIFO — 마지막 Down 의 skip 부터 사용)
            let skip = skips.pop().unwrap();
            let concatenated = concat.forward(&[&h_upsampled, &skip, &axis_ch])?.remove(0);

            // Step 3: ResNet×2 + (Attn) 처리
            h = up.predict_forward(&concatenated, &t_emb)?;
        }

        // ── (6) Final Block ─────────────────────────────────────────
        // 마지막 Up 출력과 init_conv 직후의 h₀ 를 concat
        // → 최초의 저수준 feature 를 최종 예측에 직접 활용
        let h_final = concat.forward(&[&h, &h0_skip, &axis_ch])?.remove(0);
        let h_final = self.final_res_block.predict_with_t(&h_final, Some(&t_emb))?;

        // 1×1 conv 로 출력 채널에 맞춤 → ε_θ (예측 노이즈)
        self.final_conv.predict(&h_final)
    }

    /// 학습(training) 경로.
    #[cfg(feature = "enableBackward")]
    pub fn forward_with_t(&mut self, x_t: &Variable, t_raw: &Variable) -> MlResult<Variable> {
        // (1) Time Embedding
        let t_emb = self.time_mlp.apply(t_raw)?;

        // (2) Init Conv
        let h0 = self.init_conv.apply(x_t)?;
        let h0_skip = h0.clone();

        // (3) Down Path — skip 저장
        let mut h = h0;
        let mut skips: Vec<Variable> = Vec::with_capacity(self.downs.len());

        for down in self.downs.iter_mut() {
            let (downsampled, skip) = down.forward(&h, &t_emb)?;
            skips.push(skip);
            h = downsampled;
        }

        // (4) Mid Block
        h = self.mid.forward(&h, &t_emb)?;

        // (5) Up Path: Upsample → Concat(skip) → ResNet + Attn
        let axis_ch = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1])?);

        for up in self.ups.iter_mut() {
            // Upsample 먼저 → skip 과 같은 해상도로
            h = up.upsample_apply(&h)?;
            // Skip concat
            let skip = skips.pop().unwrap();
            let concatenated = Concat::new()?.apply(&[&h, &skip, &axis_ch])?;
            // ResNet + Attn
            h = up.forward(&concatenated, &t_emb)?;
        }

        // (6) Final Block
        let h_final = Concat::new()?.apply(&[&h, &h0_skip, &axis_ch])?;
        let h_final = self.final_res_block.forward_with_t(&h_final, Some(&t_emb))?;
        self.final_conv.apply(&h_final)
    }
}

impl Layer for Unet {
    /// Layer trait 용 apply — t_emb 없이 호출 시 dummy timestep 0 사용.
    /// 실제 학습에서는 forward_with_t 를 직접 사용하세요.
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let n = input.tensor().shape()[0];
        let t_dummy = Variable::new(Tensor::from_vec(vec![0.0; n], &[n, 1])?);
        self.forward_with_t(input, &t_dummy)
    }

    /// Layer trait 용 predict — t_emb 없이 호출 시 dummy timestep 0 사용.
    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let n = input.shape()[0];
        let t_dummy = GlobalTensor::from_vec(vec![0.0; n], &[n, 1])?;
        self.predict_with_t(input, &t_dummy)
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  6. ResNetBlock 구현
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl ResNetBlock {
    /// ResNetBlock 생성.
    ///
    /// * `in_channels`  - 입력 채널 수
    /// * `out_channels` - 출력 채널 수
    /// * `num_groups`   - GroupNorm 그룹 수 (in/out 채널 모두 나누어 떨어져야 함)
    /// * `t_emb_dim`    - time embedding 차원 (None 이면 시간 주입 없음)
    ///
    /// ## 내부 구조
    ///
    /// ```text
    ///         x ──────────────────────────────┐ (skip)
    ///         │                                │
    ///    ┌────▼────┐                           │
    ///    │ GN₁→SiLU│                           │
    ///    │ →Conv₁  │  (in → out)               │  1×1 Conv (채널 불일치 시)
    ///    └────┬────┘                           │
    ///         │ + t_emb (broadcast-add)        │
    ///    ┌────▼────┐                           │
    ///    │ GN₂→SiLU│                           │
    ///    │ →Conv₂  │  (out → out)              │
    ///    └────┬────┘                           │
    ///         │                                │
    ///         └──────────── + ─────────────────┘
    ///                       ↓
    ///                     output
    /// ```
    fn new(
        in_channels:  usize,
        out_channels: usize,
        num_groups:   usize,
        t_emb_dim:    Option<usize>,
        label:        &str,
    ) -> MlResult<Self> {
        // branch1: 입력 채널 → 출력 채널 변환
        let branch1 = Sequential::from(vec![
            Box::new(GroupNorm::new(num_groups, in_channels,  1e-5, "norm1")?) as Box<dyn Layer>,
            Box::new(SiLU::new("act1")?),
            Box::new(Conv2D::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), "conv1")?),
        ], "branch1");

        // branch2: 출력 채널 유지 (추가 특징 추출)
        let branch2 = Sequential::from(vec![
            Box::new(GroupNorm::new(num_groups, out_channels, 1e-5, "norm2")?) as Box<dyn Layer>,
            Box::new(SiLU::new("act2")?),
            Box::new(Conv2D::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), "conv2")?),
        ], "branch2");

        // skip connection: 채널 수가 다를 때만 1×1 conv 로 맞춤
        // 채널이 같으면 항등 변환(identity) — 입력을 그대로 더
        let skip_conv = if in_channels != out_channels {
            Some(Conv2D::new(in_channels, out_channels, (1, 1), (1, 1), (0, 0), "skip_conv")?)
        } else {
            None
        };

        // time embedding 투사: t_emb ∈ ℝ^D → ℝ^{C_out}
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

    /// 학습 경로: time embedding 을 직접 받아 forward 실행.
    ///
    /// ## Time Embedding 주입 상세
    ///
    /// ```text
    /// t_emb [N, D]  →  Linear  →  t_proj [N, C_out]
    ///                                ↓ reshape
    ///                           t_4d [N, C_out, 1, 1]
    ///                                ↓ broadcast-add
    ///                h [N, C_out, H, W] + t_4d [N, C_out, 1, 1]
    ///                = h' [N, C_out, H, W]
    /// ```
    ///
    /// [N, C_out, 1, 1] 이 [N, C_out, H, W] 에 더해질 때,
    /// 1×1 이 H×W 로 broadcast 되어 각 채널 전체에 동일한 값이 더해짐.
    /// → 채널별 bias shift (시간에 따라 각 feature 의 활성도를 조절)
    #[cfg(feature = "enableBackward")]
    pub fn forward_with_t(&mut self, x: &Variable, t_emb: Option<&Variable>) -> MlResult<Variable> {
        // branch1: GN₁ → SiLU → Conv₁
        let h = self.branch1.apply(x)?;

        // time embedding 주입
        let h = if let (Some(proj), Some(t)) = (self.t_emb_proj.as_mut(), t_emb) {
            let t_proj = proj.apply(t)?; // [N, C_out]
            let h_shape = h.tensor().shape();
            let (n, c) = (h_shape[0], h_shape[1]);
            // [N, C_out] → [N, C_out, 1, 1] 로 reshape (broadcast-add 를 위해)
            let shape_4d = Variable::new(Tensor::from_vec(vec![0.0; n * c], &[n, c, 1, 1])?);
            let t_4d = ReshapeOp::new()?.apply(&[&t_proj, &shape_4d])?;
            &h + &t_4d
        } else {
            h
        };

        // branch2: GN₂ → SiLU → Conv₂
        let h = self.branch2.apply(&h)?;

        // skip connection: y = F(x) + skip(x)
        let skip = if let Some(ref mut sc) = self.skip_conv {
            sc.apply(x)?  // 1×1 conv 로 채널 맞춤
        } else {
            x.clone()     // 항등: 그대로 더함
        };

        Ok(&h + &skip)
    }

    /// 추론 경로: 학습 경로와 동일한 수학적 연산, 그래프 추적 없음.
    pub fn predict_with_t(&self, x: &dyn TensorBase, t_emb: Option<&dyn TensorBase>) -> MlResult<GlobalTensor<f32>> {
        let h = self.branch1.predict(x)?.to_id()?;

        let h = if let (Some(proj), Some(t)) = (&self.t_emb_proj, t_emb) {
            let t_proj = proj.predict(t)?;
            let h_shape = h.shape();
            let (n, c) = (h_shape[0], h_shape[1]);
            let shape_4d = Tensor::from_vec(vec![0.0; n * c], &[n, c, 1, 1])?;
            let t_4d = ReshapeOp::new()?.forward(&[&t_proj, &shape_4d])?.remove(0).to_id()?;
            &h + &t_4d
        } else {
            h
        };

        let h = self.branch2.predict(&h)?;

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  7. SelfAttentionBlock 구현
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl SelfAttentionBlock {
    /// SelfAttentionBlock 생성.
    ///
    /// * `channels`   - 입력/출력 채널 C (Q, K, V 모두 C 차원)
    /// * `num_groups` - GroupNorm 그룹 수 (C % groups == 0 필요)
    ///
    /// ## Scaled Dot-Product Attention
    ///
    /// 스케일링 상수 = 1/√C.
    ///
    /// 왜 √C 로 나누는가?
    /// Q, K 의 각 원소가 독립적이고 분산 1 이라면,
    /// dot product Q·K^T 의 분산은 C 에 비례.
    /// softmax 의 입력이 너무 크면 gradient 가 0 에 가까워지므로
    /// (softmax saturation), √C 로 나누어 분산을 1 로 정규화.
    pub fn new(channels: usize, num_groups: usize, label: &str) -> MlResult<Self> {
        let scale_val = (channels as f32).sqrt().recip(); // 1/√C
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
    /// 학습 경로의 Self-Attention.
    ///
    /// ## 단계별 Shape 변환 (예: N=2, C=8, H=4, W=4)
    ///
    /// ```text
    ///  입력:       [2, 8, 4, 4]       ← NCHW feature map
    ///  GN:         [2, 8, 4, 4]       ← 정규화 (분포 안정화)
    ///  reshape:    [2, 8, 16]          ← HW 를 하나로 합침 (spatial flatten)
    ///  transpose:  [2, 16, 8]          ← [N, HW, C] — "시퀀스" 형태
    ///  flatten:    [32, 8]             ← [N*HW, C] — Linear 입력 형식
    ///  Q, K, V:    [32, 8] 각각       ← 선형 투사
    ///  unflatten:  [2, 16, 8] 각각    ← 배치 복원
    ///  K^T:        [2, 8, 16]          ← transpose
    ///  Q·K^T:      [2, 16, 16]         ← attention scores (모든 위치 쌍)
    ///  ÷√C:        [2, 16, 16]         ← 스케일링
    ///  softmax:    [2, 16, 16]         ← 확률 분포 (각 행 합=1)
    ///  ·V:         [2, 16, 8]          ← context vectors
    ///  out_proj:   [2, 16, 8]          ← 출력 투사
    ///  reshape:    [2, 8, 4, 4]        ← NCHW 복원
    ///  +input:     [2, 8, 4, 4]        ← residual connection
    /// ```
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

        // Transpose 연산에 사용할 축 인덱스 변수
        let d1 = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1])?);
        let d2 = Variable::new(Tensor::from_vec(vec![2.0], &[1, 1])?);

        // reshape 용 더미 Variable 헬퍼 (shape 정보 전달용)
        let mk_shape = |new_shape: &[usize]| -> MlResult<Variable> {
            let sz: usize = new_shape.iter().product();
            Ok(Variable::new(Tensor::from_vec(vec![0.0; sz], new_shape)?))
        };

        // 1) GroupNorm — feature 분포를 안정화
        let h_ = self.gn.apply(input)?;

        // 2) [N, C, H, W] → [N, C, HW] — 공간 차원을 하나로 합침
        let sh_bcl = mk_shape(&[n, c, hw])?;
        let x_bcl = ReshapeOp::new()?.apply(&[&h_, &sh_bcl])?;

        // 3) transpose(1,2): [N, C, HW] → [N, HW, C] — 시퀀스 형태로
        let x_nhwc = Transpose::new()?.apply(&[&x_bcl, &d1, &d2])?;

        // 4) [N, HW, C] → [N*HW, C] — Linear 가 2D 입력만 지원하므로 flatten
        let sh_flat = mk_shape(&[n * hw, c])?;
        let x_flat = ReshapeOp::new()?.apply(&[&x_nhwc, &sh_flat])?;

        // 5) Q, K, V 선형 투사
        let q_flat = self.query.apply(&x_flat)?;
        let k_flat = self.key.apply(&x_flat)?;
        let v_flat = self.value.apply(&x_flat)?;

        // 6) [N*HW, C] → [N, HW, C] — 배치 차원 복원 (batched matmul 을 위해)
        let sh_nhwc = mk_shape(&[n, hw, c])?;
        let q = ReshapeOp::new()?.apply(&[&q_flat, &sh_nhwc])?;
        let sh_nhwc_k = mk_shape(&[n, hw, c])?;
        let k = ReshapeOp::new()?.apply(&[&k_flat, &sh_nhwc_k])?;
        let sh_nhwc_v = mk_shape(&[n, hw, c])?;
        let v = ReshapeOp::new()?.apply(&[&v_flat, &sh_nhwc_v])?;

        // 7) K^T: [N, HW, C] → [N, C, HW]
        let k_t = Transpose::new()?.apply(&[&k, &d1, &d2])?;

        // 8) Attention Scores: Q · K^T → [N, HW, HW]
        //    scores[i][j] = query position i 와 key position j 의 유사도
        let scores = Matmul::new()?.apply(&[&q, &k_t])?;

        // 9) 스케일링: scores / √C
        let scores_scaled = Mul::new()?.apply(&[&scores, &self.scale])?;

        // 10) Softmax (마지막 축): 각 query position 에 대한 확률 분포
        //     attention_weights[i] = softmax(scores[i]) → 합이 1
        let softmax_axis = Variable::new(Tensor::from_vec(vec![-1.0], &[1, 1])?);
        let attn = SoftmaxOp::new()?.apply(&[&scores_scaled, &softmax_axis])?;

        // 11) Context: attention_weights · V → [N, HW, C]
        //     각 position 의 출력 = 모든 value 의 가중합
        let ctx = Matmul::new()?.apply(&[&attn, &v])?;

        // 12) Output Projection: flatten → Linear → unflatten
        let sh_flat2 = mk_shape(&[n * hw, c])?;
        let ctx_flat = ReshapeOp::new()?.apply(&[&ctx, &sh_flat2])?;
        let proj_flat = self.out_proj.apply(&ctx_flat)?;
        let sh_nhwc2 = mk_shape(&[n, hw, c])?;
        let proj = ReshapeOp::new()?.apply(&[&proj_flat, &sh_nhwc2])?;

        // 13) [N, HW, C] → [N, C, HW] — 채널 우선으로 복원
        let proj_bcl = Transpose::new()?.apply(&[&proj, &d1, &d2])?;

        // 14) [N, C, HW] → [N, C, H, W] — 원래 공간 구조로 복원
        let sh_bchw = mk_shape(&[n, c, h, w])?;
        let out = ReshapeOp::new()?.apply(&[&proj_bcl, &sh_bchw])?;

        // 15) Residual: output = attention(x) + x
        //     Attention 이 학습하는 것은 "입력에 무엇을 더할지"를 의미
        Ok(&out + input)
    }

    /// 추론 경로: 학습 경로와 동일한 수학, 계산 그래프 없음.
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

        // 2) reshape: [N,C,H,W] → [N,C,HW]
        let x_bcl = GlobalTensor::from_vec(h_.data().to_vec(), &[n, c, hw])?;

        // 3) transpose: [N,C,HW] → [N,HW,C]
        let x_nhwc = Transpose::new()?.forward(&[&x_bcl, &d1, &d2])?.remove(0);

        // 4) flatten: [N,HW,C] → [N*HW,C]
        let x_flat = GlobalTensor::from_vec(x_nhwc.data().to_vec(), &[n * hw, c])?;

        // 5) Q, K, V
        let q_flat = self.query.predict(&x_flat)?;
        let k_flat = self.key.predict(&x_flat)?;
        let v_flat = self.value.predict(&x_flat)?;

        // 6) unflatten: [N*HW,C] → [N,HW,C]
        let q = GlobalTensor::from_vec(q_flat.data().to_vec(), &[n, hw, c])?;
        let k = GlobalTensor::from_vec(k_flat.data().to_vec(), &[n, hw, c])?;
        let v = GlobalTensor::from_vec(v_flat.data().to_vec(), &[n, hw, c])?;

        // 7) K^T
        let k_t = Transpose::new()?.forward(&[&k, &d1, &d2])?.remove(0);

        // 8) scores = Q · K^T
        let scores = Matmul::new()?.forward(&[&q, &k_t])?.remove(0);

        // 9) 스케일링: 직접 스칼라 곱 (predict 에서는 Mul 연산자 대신 직접 계산)
        let scale_val = (self.channels as f32).sqrt().recip();
        let scaled: Vec<f32> = scores.data().iter().map(|&x| x * scale_val).collect();
        let scaled_tensor = GlobalTensor::from_vec(scaled, scores.shape())?;

        // 10) softmax
        let softmax_axis = GlobalTensor::from_vec(vec![-1.0], &[1, 1])?;
        let attn = SoftmaxOp::new()?.forward(&[&scaled_tensor, &softmax_axis])?.remove(0);

        // 11) context = attention · V
        let ctx = Matmul::new()?.forward(&[&attn, &v])?.remove(0);

        // 12) out_proj
        let ctx_flat = GlobalTensor::from_vec(ctx.data().to_vec(), &[n * hw, c])?;
        let proj_flat = self.out_proj.predict(&ctx_flat)?;
        let proj = GlobalTensor::from_vec(proj_flat.data().to_vec(), &[n, hw, c])?;

        // 13) transpose: [N,HW,C] → [N,C,HW]
        let proj_bcl = Transpose::new()?.forward(&[&proj, &d1, &d2])?.remove(0);

        // 14) reshape: [N,C,HW] → [N,C,H,W]
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  8. 테스트
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::activation::SoftmaxOp;
    use crate::tensor::operators::Function;

    // ── SelfAttentionBlock 테스트 ────────────────────────────────────────────

    #[test]
    fn self_attention_predict_shape() -> MlResult<()> {
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
        let input = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3])?;
        let axis = GlobalTensor::from_vec(vec![1.0], &[1, 1])?;
        let out = SoftmaxOp::new()?.forward(&[&input, &axis])?.remove(0);

        assert_eq!(out.shape(), &[2, 3]);
        let row0_sum: f32 = out.data()[0..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6, "row0 sum = {}", row0_sum);
        let row1_sum: f32 = out.data()[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-6, "row1 sum = {}", row1_sum);
        for &v in &out.data()[3..6] {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn softmax_axis_3d_last() -> MlResult<()> {
        let n = 2 * 3 * 4;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let input = GlobalTensor::from_vec(data, &[2, 3, 4])?;
        let axis = GlobalTensor::from_vec(vec![-1.0], &[1, 1])?;
        let out = SoftmaxOp::new()?.forward(&[&input, &axis])?.remove(0);

        assert_eq!(out.shape(), &[2, 3, 4]);
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
        let input = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let out = SoftmaxOp::new()?.forward(&[&input])?.remove(0);
        let total: f32 = out.data().iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(out.data()[2] > out.data()[1]);
        assert!(out.data()[1] > out.data()[0]);
        Ok(())
    }

    // ── ResNetBlock 테스트 ───────────────────────────────────────────────────

    #[test]
    fn resnet_block_predict_no_temb() -> MlResult<()> {
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

        let t_data: Vec<f32> = (0..t_emb_dim).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;

        let out_with = block.predict_with_t(&x, Some(&t))?;
        let out_without = block.predict_with_t(&x, None)?;
        assert_eq!(out_with.shape(), &[1, 8, 4, 4]);
        assert_ne!(out_with.data(), out_without.data());
        Ok(())
    }

    #[test]
    fn resnet_block_channel_change() -> MlResult<()> {
        let block = ResNetBlock::new(8, 16, 4, None, "res_ch")?;
        let data: Vec<f32> = (0..(1 * 8 * 4 * 4)).map(|i| i as f32 * 0.01).collect();
        let x = Tensor::from_vec(data, &[1, 8, 4, 4])?;
        let out = block.predict(&x)?;
        assert_eq!(out.shape(), &[1, 16, 4, 4]);
        Ok(())
    }

    // ── DownBlock / MidBlock / UpBlock 테스트 ────────────────────────────────

    #[test]
    fn downblock_predict_halves_spatial() -> MlResult<()> {
        let t_emb_dim = 16;
        let block = DownBlock::new(8, 8, 4, Some(t_emb_dim), false, "down0")?;
        let data: Vec<f32> = (0..(1 * 8 * 8 * 8)).map(|i| i as f32 * 0.001).collect();
        let x = Tensor::from_vec(data, &[1, 8, 8, 8])?;
        let t_data: Vec<f32> = vec![0.1; t_emb_dim];
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;

        let (out, skip) = block.predict_forward(&x, &t)?;
        // downsample: 8×8 → 4×4
        assert_eq!(out.shape(), &[1, 8, 4, 4]);
        // skip: downsample 전의 feature (8×8)
        assert_eq!(skip.shape(), &[1, 8, 8, 8]);
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

    /// UpBlock 의 3단계 동작을 테스트:
    ///   1) Upsample: [1,8,4,4] → [1,8,8,8]
    ///   2) Concat with skip: [1,8,8,8] ++ [1,8,8,8] → [1,16,8,8]
    ///   3) ResNet: [1,16,8,8] → [1,8,8,8]
    #[test]
    fn upblock_predict_upsample_then_process() -> MlResult<()> {
        let t_emb_dim = 16;
        // pre_ch=8 (upsample 입력), in_ch=16 (concat 후), out_ch=8
        let block = UpBlock::new(8, 16, 8, 4, Some(t_emb_dim), false, "up0")?;

        // Step 1: upsample 4×4 → 8×8
        let h = Tensor::from_vec(vec![0.01; 1 * 8 * 4 * 4], &[1, 8, 4, 4])?;
        let h_up = block.upsample_predict(&h)?;
        assert_eq!(h_up.shape(), &[1, 8, 8, 8]);

        // Step 2: concat with skip (8 + 8 = 16 channels)
        let skip = Tensor::from_vec(vec![0.01; 1 * 8 * 8 * 8], &[1, 8, 8, 8])?;
        let axis = GlobalTensor::from_vec(vec![1.0], &[1, 1])?;
        let concatenated = Concat::new()?.forward(&[&h_up, &skip, &axis])?.remove(0);
        assert_eq!(concatenated.shape(), &[1, 16, 8, 8]);

        // Step 3: process (ResNet + Attn)
        let t_data: Vec<f32> = vec![0.1; t_emb_dim];
        let t = Tensor::from_vec(t_data, &[1, t_emb_dim])?;
        let out = block.predict_forward(&concatenated, &t)?;
        assert_eq!(out.shape(), &[1, 8, 8, 8]);
        Ok(())
    }

    // ── U-Net 전체 테스트 ───────────────────────────────────────────────────

    /// U-Net 생성 테스트: 파라미터가 올바르게 조립되는지 확인.
    #[test]
    fn unet_construction() -> MlResult<()> {
        // dim=8, dim_mults=[1,2], channels=1, groups=4
        // → Down: (8→8), (8→16)
        // → Mid: 16
        // → Up: (16+16→8), (8+8→8)
        // → final: 8+8=16 → 8 → 1
        let unet = Unet::new(
            8,                      // dim
            None,                   // init_dim = dim
            None,                   // out_dim = channels
            &[1, 2],               // dim_mults
            1,                      // channels (grayscale)
            4,                      // resnet_block_groups
            &[false, true],        // attention: 마지막 단계에서만
        )?;

        assert_eq!(unet.downs.len(), 2);
        assert_eq!(unet.ups.len(), 2);
        assert!(unet.downs[0].attn.is_none());   // use_attn=false
        assert!(unet.downs[1].attn.is_some());    // use_attn=true

        // 파라미터 수 > 0 (레이어들이 정상 생성됨)
        assert!(!unet.params().is_empty());
        Ok(())
    }

    /// U-Net predict 테스트: 입출력 shape 이 일치하는지 확인.
    ///
    /// 입력 [1, 1, 16, 16] → 출력 [1, 1, 16, 16] (노이즈 예측)
    #[test]
    fn unet_predict_shape() -> MlResult<()> {
        let unet = Unet::new(
            8, None, None,
            &[1, 2],
            1, 4,
            &[false, true],
        )?;

        // 입력: grayscale 16×16 이미지
        let x = Tensor::from_vec(vec![0.1; 1 * 1 * 16 * 16], &[1, 1, 16, 16])?;
        // timestep (raw, SinusoidalPE 인코딩 전)
        let t = Tensor::from_vec(vec![0.5], &[1, 1])?;

        let out = unet.predict_with_t(&x, &t)?;

        // 출력 shape 은 입력과 동일해야 함 (노이즈 예측)
        assert_eq!(out.shape(), &[1, 1, 16, 16]);
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

        assert!(mean.abs() < 0.15, "mean = {} (expected ≈ 0)", mean);
        assert!((variance - 1.0).abs() < 0.25, "variance = {} (expected ≈ 1)", variance);
        Ok(())
    }
}
