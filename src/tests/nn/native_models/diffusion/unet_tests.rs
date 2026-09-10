use super::*;
use crate::legacy::nn::activation::SoftmaxOp;
use crate::legacy::tensor::operators::Function;

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
