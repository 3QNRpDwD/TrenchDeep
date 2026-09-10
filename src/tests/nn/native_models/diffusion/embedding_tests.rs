use super::*;
use crate::legacy::tests::common::logging::setup_logging;

/// TimeEmbeddingMLP 의 shape 변환 검증.
///
/// [N, 1] → [N, t_emb_dim]
#[test]
fn time_embedding_mlp_shape() -> MlResult<()> {
    setup_logging();
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
    setup_logging();
    let mlp = TimeEmbeddingMLP::new(16, 64)?;
    assert_eq!(mlp.t_emb_dim(), 64);
    Ok(())
}

/// 학습 경로 (Variable) shape 검증.
#[cfg(feature = "enableBackward")]
#[test]
fn time_embedding_mlp_apply_shape() -> MlResult<()> {
    setup_logging();
    let dim = 8;
    let t_emb_dim = 32;
    let mut mlp = TimeEmbeddingMLP::new(dim, t_emb_dim)?;

    let t = Variable::new(Tensor::from_vec(vec![0.0, 0.5], &[2, 1])?);
    let out = mlp.apply(&t)?;

    assert_eq!(out.tensor().shape(), &[2, t_emb_dim]);
    Ok(())
}
