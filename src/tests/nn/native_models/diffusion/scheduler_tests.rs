use super::*;

#[test]
fn linear_schedule_values() {
    let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
    assert_eq!(s.betas.len(), 1000);
    // β_0 ≈ 1e-4, β_999 ≈ 0.02
    assert!((s.betas[0] - 1e-4).abs() < 1e-6);
    assert!((s.betas[999] - 0.02).abs() < 1e-4);
    // ᾱ 는 단조감소
    for i in 1..1000 {
        assert!(s.alpha_bars[i] < s.alpha_bars[i - 1],
            "alpha_bars must decrease: t={} ({}) >= t={} ({})",
            i, s.alpha_bars[i], i - 1, s.alpha_bars[i - 1]);
    }
    // ᾱ_0 ≈ 1, ᾱ_999 ≈ 0
    assert!(s.alpha_bars[0] > 0.99);
    assert!(s.alpha_bars[999] < 0.1);
}

#[test]
fn cosine_schedule_monotonic() {
    let s = DDPMScheduler::cosine_schedule(1000, 0.008);
    assert_eq!(s.betas.len(), 1000);
    for i in 1..1000 {
        assert!(s.alpha_bars[i] <= s.alpha_bars[i - 1] + 1e-6,
            "cosine alpha_bars must decrease: t={}", i);
    }
}

#[test]
fn q_sample_shape_preserved() -> MlResult<()> {
    let s = DDPMScheduler::linear_schedule(100, 1e-4, 0.02);
    let x_0 = Tensor::from_vec(vec![1.0; 2 * 3 * 4 * 4], &[2, 3, 4, 4])?;
    let noise = Tensor::randn(&[2, 3, 4, 4]);
    let x_t = s.q_sample(&x_0, 50, &noise)?;
    assert_eq!(x_t.shape(), &[2, 3, 4, 4]);
    Ok(())
}

#[test]
fn q_sample_t0_close_to_original() -> MlResult<()> {
    let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
    let x_0 = Tensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4])?;
    let noise = Tensor::from_vec(vec![0.5; 16], &[1, 1, 4, 4])?;

    // t=0: √ᾱ_0 ≈ 1, √(1-ᾱ_0) ≈ 0 → x_t ≈ x_0
    let x_t = s.q_sample(&x_0, 0, &noise)?;
    for (&xt, &x0) in x_t.data().iter().zip(x_0.data().iter()) {
        assert!((xt - x0).abs() < 0.05, "t=0: x_t should be close to x_0");
    }
    Ok(())
}

#[test]
fn q_sample_large_t_dominated_by_noise() -> MlResult<()> {
    let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
    let x_0 = Tensor::from_vec(vec![10.0; 16], &[1, 1, 4, 4])?;
    let noise = Tensor::from_vec(vec![0.0; 16], &[1, 1, 4, 4])?;

    // t=999: ᾱ_999 ≈ 0 → x_t ≈ noise (≈0)
    let x_t = s.q_sample(&x_0, 999, &noise)?;
    let mean: f32 = x_t.data().iter().sum::<f32>() / x_t.data().len() as f32;
    assert!(mean.abs() < 2.0,
        "t=999: signal should be nearly gone, mean={}", mean);
    Ok(())
}

#[test]
fn p_sample_shape_preserved() -> MlResult<()> {
    let s = DDPMScheduler::linear_schedule(100, 1e-4, 0.02);
    let x_t = Tensor::from_vec(vec![0.5; 2 * 1 * 4 * 4], &[2, 1, 4, 4])?;
    let eps = Tensor::from_vec(vec![0.1; 2 * 1 * 4 * 4], &[2, 1, 4, 4])?;
    let x_prev = s.p_sample(&x_t, &eps, 50)?;
    assert_eq!(x_prev.shape(), &[2, 1, 4, 4]);
    Ok(())
}

#[test]
fn p_sample_loop_returns_correct_shape() -> MlResult<()> {
    let s = DDPMScheduler::linear_schedule(10, 1e-4, 0.02);
    let shape = [1, 1, 4, 4];

    let result = s.p_sample_loop(&shape, |x_t, _t| {
        // 더미 denoise: 그냥 0 반환
        GlobalTensor::from_vec(vec![0.0; x_t.data().len()], x_t.shape())
    })?;

    assert_eq!(result.shape(), &[1, 1, 4, 4]);
    Ok(())
}

#[test]
fn posterior_variance_t0_is_zero() {
    let s = DDPMScheduler::linear_schedule(1000, 1e-4, 0.02);
    // t=0: ᾱ_{t-1} = ᾱ_{-1} = 1.0 → posterior_var = β_0 * 0 / (1-ᾱ_0) ≈ 0
    assert!(s.posterior_variance[0].abs() < 1e-6,
        "posterior_variance[0] = {}", s.posterior_variance[0]);
}
