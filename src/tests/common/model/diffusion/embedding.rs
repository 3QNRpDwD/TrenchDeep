use super::*;

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  TimeEmbeddingMLP — 타임스텝 임베딩 모듈                                     ║
// ║                                                                           ║
// ║  DDPM 에서 U-Net 은 "지금 몇 번째 노이즈 제거 단계인지"를 알아야                ║
// ║  각 단계에 맞는 노이즈를 예측할 수 있음.                                       ║
// ║                                                                           ║
// ║  ── 변환 과정 ──                                                           ║
// ║                                                                           ║
// ║  t (스칼라)                                                                ║
// ║    ↓  SinusoidalPE                                                        ║
// ║  [sin(ω₁t), sin(ω₂t), ..., cos(ω₁t), cos(ω₂t), ...]  ∈ ℝ^dim              ║
// ║    ↓  Linear(dim → t_emb_dim)                                             ║
// ║    ↓  SiLU (비선형성)                                                      ║
// ║    ↓  Linear(t_emb_dim → t_emb_dim)                                       ║
// ║  t_emb ∈ ℝ^{t_emb_dim}                                                    ║
// ║                                                                           ║
// ║  ── 왜 MLP 가 필요한가? ──                                                  ║
// ║                                                                           ║
// ║  SinusoidalPE 만으로는 고정된 주파수 패턴만 제공함.                            ║  
// ║  MLP 를 거치면 모델이 학습 과정에서 각 ResNetBlock 에 최적화된                  ║
// ║  시간 표현을 스스로 학습할 수 있음.                                           ║
// ║                                                                           ║
// ║  ── Shape 흐름 ──                                                          ║
// ║                                                                           ║
// ║  t [N, 1] → SinusoidalPE → [N, dim]                                       ║
// ║           → Linear        → [N, t_emb_dim]                                ║
// ║           → SiLU          → [N, t_emb_dim]                                ║
// ║           → Linear        → [N, t_emb_dim]                                ║
// ║                                                                           ║
// ║  관례: t_emb_dim = dim × 4 (DDPM 논문)                                     ║
// ║  dim 대비 4배 넓은 표현 공간을 사용하여 시간 정보의 표현력을 높임.               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// 타임스텝 임베딩 MLP.
///
/// `SinusoidalPE → Linear → SiLU → Linear` 파이프라인으로
/// 스칼라 타임스텝을 풍부한 벡터 표현으로 변환함.
///
/// ## 사용처
///
/// U-Net 내부에서 각 ResNetBlock 에 시간 정보를 주입할 때 사용:
///
/// ```text
/// t_emb = TimeEmbeddingMLP(t)
/// h = ResNetBlock(x, t_emb)   ← h + MLP(t_emb)  형태로 주입
/// ```
#[derive(Debug)]
pub struct TimeEmbeddingMLP {
    /// SinusoidalPE 의 출력 차원 (= MLP 입력 차원)
    dim: usize,
    /// MLP 출력 차원 (= t_emb_dim, 보통 dim × 4)
    t_emb_dim: usize,
    /// 내부 파이프라인: SinusoidalPE → Linear → SiLU → Linear
    mlp: Sequential,
}

impl TimeEmbeddingMLP {
    /// TimeEmbeddingMLP 생성.
    ///
    /// ## 파라미터
    ///
    /// * `dim`       - SinusoidalPE 출력 차원 (U-Net 의 기본 채널 수와 동일)
    /// * `t_emb_dim` - MLP 출력 차원 (보통 `dim × 4`)
    ///
    /// ## 내부 구조
    ///
    /// ```text
    /// SinusoidalPE(dim)           [N, 1] → [N, dim]
    ///   → Linear(dim, t_emb_dim)  [N, dim] → [N, t_emb_dim]
    ///   → SiLU                    [N, t_emb_dim] (비선형 활성화)
    ///   → Linear(t_emb_dim, t_emb_dim)  [N, t_emb_dim] → [N, t_emb_dim]
    /// ```
    pub fn new(dim: usize, t_emb_dim: usize) -> MlResult<Self> {
        info!(
            "TimeEmbeddingMLP: [N,1] → SinusoidalPE → [N,{}] → Linear → [N,{}] → SiLU → Linear → [N,{}]",
            dim, t_emb_dim, t_emb_dim
        );

        let mlp = Sequential::from(vec![
            // 1) Sinusoidal Positional Encoding
            //    정수 타임스텝을 다양한 주파수의 sin/cos 로 인코딩
            //    Transformer 의 위치 인코딩과 동일한 원리
            Box::new(SinusoidalPE::new(dim, "time_sinusoidal_pe")?) as Box<dyn Layer>,

            // 2) 선형 변환: dim → t_emb_dim (차원 확장)
            Box::new(Linear::new(dim, t_emb_dim, "time_mlp_linear1")?),

            // 3) SiLU 활성화: x · σ(x)
            //    ReLU 보다 부드러운 비선형성, 디퓨전 모델에서 표준적으로 사용
            Box::new(SiLU::new("time_mlp_silu")?),

            // 4) 선형 변환: t_emb_dim → t_emb_dim (표현 정제)
            Box::new(Linear::new(t_emb_dim, t_emb_dim, "time_mlp_linear2")?),
        ], "time_mlp");

        Ok(Self { dim, t_emb_dim, mlp })
    }

    /// t_emb_dim 을 반환함.
    ///
    /// U-Net 조립 시 ResNetBlock 의 `t_emb_dim` 파라미터에 전달하기 위해 사용.
    pub fn t_emb_dim(&self) -> usize {
        self.t_emb_dim
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  테스트
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    /// TimeEmbeddingMLP 의 shape 변환 검증.
    ///
    /// [N, 1] → [N, t_emb_dim]
    #[test]
    fn time_embedding_mlp_shape() -> MlResult<()> {
        let dim = 8;
        let t_emb_dim = dim * 4; // 32
        let mlp = TimeEmbeddingMLP::new(dim, t_emb_dim)?;

        // 배치 크기 3, 타임스텝 스칼라 [3, 1]
        let t = Tensor::from_vec(vec![0.1, 0.5, 0.9], &[3, 1])?;
        let out = mlp.predict(&t)?;

        assert_eq!(out.shape(), &[3, t_emb_dim]);
        // 모든 값이 유한해야 함
        for &v in out.data() {
            assert!(v.is_finite(), "TimeEmbeddingMLP output contains non-finite: {}", v);
        }
        Ok(())
    }

    /// t_emb_dim 접근자 테스트.
    #[test]
    fn time_embedding_mlp_dim_accessor() -> MlResult<()> {
        let mlp = TimeEmbeddingMLP::new(16, 64)?;
        assert_eq!(mlp.t_emb_dim(), 64);
        Ok(())
    }

    /// 학습 경로 (Variable) shape 검증.
    #[cfg(feature = "enableBackward")]
    #[test]
    fn time_embedding_mlp_apply_shape() -> MlResult<()> {
        let dim = 8;
        let t_emb_dim = 32;
        let mut mlp = TimeEmbeddingMLP::new(dim, t_emb_dim)?;

        let t = Variable::new(Tensor::from_vec(vec![0.0, 0.5], &[2, 1])?);
        let out = mlp.apply(&t)?;

        assert_eq!(out.tensor().shape(), &[2, t_emb_dim]);
        Ok(())
    }
}
