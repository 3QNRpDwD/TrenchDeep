use super::*;

// ─── GroupNorm Function ──────────────────────────────────────────────────────
//
// forward 입력 순서:
//   [0] X:          [N, C, H, W]
//   [1] γ (gamma):  [C]
//   [2] β (beta):   [C]
//   [3] num_groups: scalar
//   [4] eps:        scalar
//
// forward 반환 순서:
//   [0] Y:     [N, C, H, W]  ← 최종 출력
//   [1] x_hat: [N, C, H, W]  ← backward 재사용
//   [2] mean:  [N*G]          ← backward 재사용 (flatten)
//   [3] var:   [N*G]          ← backward 재사용 (flatten)
//
// backward targets 순서 (Layer가 조립):
//   [0] X, [1] γ, [2] β, [3] num_groups, [4] eps,
//   [5] x_hat, [6] mean, [7] var
//
// backward 반환:
//   [dX, dγ, dβ, 0, 0, 0, 0, 0]

impl Function for GroupNormOp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(GroupNormOp)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.len() < 5 {
            return Err(MlError::StringError(
                "GroupNorm::forward: [X, gamma, beta, num_groups, eps] 필요".into(),
            ));
        }
        let x     = targets[0];
        let gamma = targets[1];
        let beta  = targets[2];
        let g     = targets[3].data()[0] as usize;
        let eps   = targets[4].data()[0];

        let shape = x.shape();
        if shape.len() != 4 {
            return Err(MlError::StringError(format!(
                "GroupNorm: 입력은 4D [N, C, H, W] 이어야 합니다. 현재: {:?}",
                shape
            )));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c % g != 0 {
            return Err(MlError::StringError(format!(
                "GroupNorm: C({}) % num_groups({}) != 0",
                c, g
            )));
        }
        let cg = c / g;        // 그룹당 채널 수
        let m  = cg * h * w;   // 그룹당 원소 수

        let x_data     = x.data();
        let gamma_data = gamma.data();
        let beta_data  = beta.data();

        let mut y_data    = vec![0.0f32; n * c * h * w];
        let mut xhat_data = vec![0.0f32; n * c * h * w];
        let mut mean_data = vec![0.0f32; n * g];
        let mut var_data  = vec![0.0f32; n * g];

        for ni in 0..n {
            for gi in 0..g {
                // ── 이 그룹의 원소 인덱스 범위 ──────────────────────────
                // 채널: [gi*cg, (gi+1)*cg)
                let c_start = gi * cg;

                // ── 평균 ─────────────────────────────────────────────────
                let mut sum = 0.0f32;
                for ci in c_start..c_start + cg {
                    let base = ni * c * h * w + ci * h * w;
                    for k in 0..h * w {
                        sum += x_data[base + k];
                    }
                }
                let mu = sum / m as f32;
                mean_data[ni * g + gi] = mu;

                // ── 분산 ─────────────────────────────────────────────────
                let mut var_sum = 0.0f32;
                for ci in c_start..c_start + cg {
                    let base = ni * c * h * w + ci * h * w;
                    for k in 0..h * w {
                        let diff = x_data[base + k] - mu;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / m as f32;
                var_data[ni * g + gi] = var;

                let inv_std = 1.0 / (var + eps).sqrt();

                // ── 정규화 + 스케일/시프트 ────────────────────────────────
                for ci in c_start..c_start + cg {
                    let base = ni * c * h * w + ci * h * w;
                    for k in 0..h * w {
                        let x_hat = (x_data[base + k] - mu) * inv_std;
                        xhat_data[base + k] = x_hat;
                        y_data[base + k]    = gamma_data[ci] * x_hat + beta_data[ci];
                    }
                }
            }
        }

        #[cfg(feature = "debugging")]
        {
            let mean_preview: Vec<f32> = mean_data.iter().take(4).copied().collect();
            let var_preview:  Vec<f32> = var_data.iter().take(4).copied().collect();
            tracing::debug!(
                "[GroupNorm::forward] [{},{},{},{}] G={} cg={} m={} mean_preview={:?} var_preview={:?}",
                n, c, h, w, g, cg, m, mean_preview, var_preview
            );
            crate::tensor::operators::debug::stats_raw("  └─ y", &y_data, shape);
        }

        let out_shape = shape.to_vec();
        Ok(vec![
            GlobalTensor::from_vec(y_data,    &out_shape)?,
            GlobalTensor::from_vec(xhat_data, &out_shape)?,
            GlobalTensor::from_vec(mean_data, &[n * g])?,
            GlobalTensor::from_vec(var_data,  &[n * g])?,
        ])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(
        &self,
        targets: &[&dyn TensorBase],
        grad: &dyn TensorBase,
    ) -> MlResult<Vec<GlobalTensor<f32>>> {
        // targets: [X, γ, β, num_groups, eps, x_hat, mean, var]
        if targets.len() < 8 {
            return Err(MlError::StringError(
                "GroupNorm::backward: targets 8개 필요 [X, γ, β, G, eps, x_hat, μ, σ²]".into(),
            ));
        }
        let x     = targets[0];
        let gamma = targets[1];
        let g     = targets[3].data()[0] as usize;
        let eps   = targets[4].data()[0];
        let x_hat = targets[5];
        let mean  = targets[6];
        let var   = targets[7];

        let shape = x.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let cg = c / g;
        let m  = (cg * h * w) as f32;

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[GroupNorm::backward] [{},{},{},{}] G={} cg={} {}",
            n, c, h, w, g, cg,
            crate::tensor::operators::debug::summary("grad", grad)
        );

        let dy        = grad.data();
        let gamma_d   = gamma.data();
        let xhat_d    = x_hat.data();
        let var_d     = var.data();

        let mut dx    = vec![0.0f32; n * c * h * w];
        let mut dgamma = vec![0.0f32; c];
        let mut dbeta  = vec![0.0f32; c];

        // ── dγ, dβ ───────────────────────────────────────────────────────
        for ni in 0..n {
            for ci in 0..c {
                let base = ni * c * h * w + ci * h * w;
                for k in 0..h * w {
                    dgamma[ci] += dy[base + k] * xhat_d[base + k];
                    dbeta[ci]  += dy[base + k];
                }
            }
        }

        // ── dX ───────────────────────────────────────────────────────────
        // 표준 GroupNorm backward:
        //   dx_hat  = dy * γ
        //   dvar    = sum(dx_hat * (x - μ)) * (-1/2) * (σ²+ε)^(-3/2)
        //   dmu     = sum(dx_hat) * (-1/√(σ²+ε)) + dvar * sum(-2(x-μ))/m
        //   dx      = dx_hat/√(σ²+ε) + dvar * 2(x-μ)/m + dmu/m
        for ni in 0..n {
            for gi in 0..g {
                let c_start = gi * cg;
                let inv_std = 1.0 / (var_d[ni * g + gi] + eps).sqrt();
                let inv_std3 = inv_std * inv_std * inv_std; // (σ²+ε)^(-3/2)

                // dx_hat = dy * γ  (이 그룹 내 원소에 대해)
                // dvar   = sum over group of (dx_hat * x_hat) * (-0.5) * inv_std²
                // Note: x_hat = (x-μ)/sqrt(σ²+ε)  →  (x-μ) = x_hat * sqrt(σ²+ε)
                //       dvar = sum(dx_hat * x_hat * sqrt(σ²+ε)) * (-1/2) * inv_std³
                //            = -0.5 * inv_std * sum(dx_hat * x_hat)
                let mut sum_dxhat       = 0.0f32;
                let mut sum_dxhat_xhat  = 0.0f32;

                for ci in c_start..c_start + cg {
                    let base = ni * c * h * w + ci * h * w;
                    for k in 0..h * w {
                        let dx_hat = dy[base + k] * gamma_d[ci];
                        sum_dxhat      += dx_hat;
                        sum_dxhat_xhat += dx_hat * xhat_d[base + k];
                    }
                }

                // 실제 dx 계산
                for ci in c_start..c_start + cg {
                    let base = ni * c * h * w + ci * h * w;
                    for k in 0..h * w {
                        let dx_hat_k = dy[base + k] * gamma_d[ci];
                        // 수식: dx = (1/m) * inv_std * (m*dx_hat - sum_dxhat - x_hat*sum_dxhat_xhat)
                        dx[base + k] = (1.0 / m) * inv_std * (
                            m * dx_hat_k
                            - sum_dxhat
                            - xhat_d[base + k] * sum_dxhat_xhat
                        );
                    }
                }
            }
        }

        #[cfg(feature = "debugging")]
        {
            crate::tensor::operators::debug::stats_raw("  └─ dX",     &dx,     shape);
            crate::tensor::operators::debug::stats_raw("  └─ dgamma", &dgamma, &[c]);
            crate::tensor::operators::debug::stats_raw("  └─ dbeta",  &dbeta,  &[c]);
        }

        let zero = GlobalTensor::from_vec(vec![0.0], &[1, 1])?;
        Ok(vec![
            GlobalTensor::from_vec(dx,     shape)?,
            GlobalTensor::from_vec(dgamma, &[c])?,
            GlobalTensor::from_vec(dbeta,  &[c])?,
            zero.clone(), // num_groups scalar — grad 없음
            zero.clone(), // eps scalar — grad 없음
            zero.clone(), // x_hat — grad 흐름 불필요 (저장 목적)
            zero.clone(), // mean  — 동일
            zero,         // var   — 동일
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}

// ─── 테스트 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// GroupNorm forward를 편리하게 호출하는 헬퍼
    fn gn_forward(
        x: &dyn TensorBase,
        gamma: &dyn TensorBase,
        beta: &dyn TensorBase,
        num_groups: usize,
        eps: f32,
    ) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>, GlobalTensor<f32>, GlobalTensor<f32>)> {
        let g   = GlobalTensor::from_vec(vec![num_groups as f32], &[1, 1])?;
        let e   = GlobalTensor::from_vec(vec![eps], &[1, 1])?;
        let op  = GroupNormOp::new()?;
        let mut out = op.forward(&[x, gamma, beta, &g, &e])?;
        let var_t  = out.remove(3);
        let mean_t = out.remove(2);
        let xhat_t = out.remove(1);
        let y_t    = out.remove(0);
        Ok((y_t, xhat_t, mean_t, var_t))
    }

    #[test]
    fn test_gn_output_shape() -> MlResult<()> {
        // [2, 4, 3, 3], G=2 → 출력 shape 동일
        let x     = GlobalTensor::from_vec(vec![1.0f32; 2 * 4 * 3 * 3], &[2, 4, 3, 3])?;
        let gamma = GlobalTensor::from_vec(vec![1.0f32; 4], &[4])?;
        let beta  = GlobalTensor::from_vec(vec![0.0f32; 4], &[4])?;

        let (y, xhat, mean, var) = gn_forward(&x, &gamma, &beta, 2, 1e-5)?;

        assert_eq!(y.shape(),    &[2, 4, 3, 3]);
        assert_eq!(xhat.shape(), &[2, 4, 3, 3]);
        assert_eq!(mean.shape(), &[4]);   // N*G = 2*2
        assert_eq!(var.shape(),  &[4]);
        Ok(())
    }

    #[test]
    fn test_gn_zero_mean_unit_var() -> MlResult<()> {
        // γ=1, β=0 일 때 각 그룹 내 출력의 평균≈0, 분산≈1
        // [1, 4, 2, 2], G=1 (채널 전체가 하나의 그룹)
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x     = GlobalTensor::from_vec(data, &[1, 4, 2, 2])?;
        let gamma = GlobalTensor::from_vec(vec![1.0f32; 4], &[4])?;
        let beta  = GlobalTensor::from_vec(vec![0.0f32; 4], &[4])?;

        let (y, _, _, _) = gn_forward(&x, &gamma, &beta, 1, 1e-5)?;

        // 전체 원소 평균 ≈ 0
        let mean: f32 = y.data().iter().sum::<f32>() / y.data().len() as f32;
        assert!(mean.abs() < 1e-5, "mean={}", mean);

        // 전체 원소 분산 ≈ 1
        let var: f32 = y.data().iter().map(|&v| v * v).sum::<f32>() / y.data().len() as f32;
        assert!((var - 1.0).abs() < 1e-4, "var={}", var);
        Ok(())
    }

    #[test]
    fn test_gn_gamma_beta_effect() -> MlResult<()> {
        // γ=2, β=1 이면 출력 ≈ 2 * x_hat + 1
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x     = GlobalTensor::from_vec(data, &[1, 4, 2, 2])?;
        let gamma = GlobalTensor::from_vec(vec![2.0f32; 4], &[4])?;
        let beta  = GlobalTensor::from_vec(vec![1.0f32; 4], &[4])?;

        let (y_scaled, _, _, _) = gn_forward(&x, &gamma, &beta, 1, 1e-5)?;

        let gamma1 = GlobalTensor::from_vec(vec![1.0f32; 4], &[4])?;
        let beta0  = GlobalTensor::from_vec(vec![0.0f32; 4], &[4])?;
        let x2     = GlobalTensor::from_vec((0..16).map(|i| i as f32).collect(), &[1, 4, 2, 2])?;
        let (y_base, _, _, _) = gn_forward(&x2, &gamma1, &beta0, 1, 1e-5)?;

        // y_scaled ≈ 2 * y_base + 1
        for (&ys, &yb) in y_scaled.data().iter().zip(y_base.data().iter()) {
            let expected = 2.0 * yb + 1.0;
            assert!((ys - expected).abs() < 1e-5, "ys={}, expected={}", ys, expected);
        }
        Ok(())
    }

    #[test]
    fn test_gn_constant_input() -> MlResult<()> {
        // 모든 입력이 동일한 값이면 x_hat = 0, y = β
        let x     = GlobalTensor::from_vec(vec![5.0f32; 8], &[1, 2, 2, 2])?;
        let gamma = GlobalTensor::from_vec(vec![1.0f32; 2], &[2])?;
        let beta  = GlobalTensor::from_vec(vec![3.0f32; 2], &[2])?;

        let (y, xhat, _, _) = gn_forward(&x, &gamma, &beta, 1, 1e-5)?;

        // x_hat ≈ 0 (분산 0 → inv_std ≈ 1/sqrt(eps), 하지만 x_hat = (x-μ)/std = 0)
        for &v in xhat.data() {
            assert!(v.abs() < 1e-4, "xhat={}", v);
        }
        // y ≈ β = 3
        for &v in y.data() {
            assert!((v - 3.0).abs() < 1e-4, "y={}", v);
        }
        Ok(())
    }

    #[test]
    fn test_gn_multi_group() -> MlResult<()> {
        // G=4, C=4 → 각 채널이 독립 그룹
        // 각 채널 내 H*W 원소들만 정규화됨
        let x     = GlobalTensor::from_vec(vec![1.0f32; 1 * 4 * 2 * 2], &[1, 4, 2, 2])?;
        let gamma = GlobalTensor::from_vec(vec![1.0f32; 4], &[4])?;
        let beta  = GlobalTensor::from_vec(vec![0.0f32; 4], &[4])?;

        let (y, _, _, _) = gn_forward(&x, &gamma, &beta, 4, 1e-5)?;

        assert_eq!(y.shape(), &[1, 4, 2, 2]);
        // 모든 입력이 동일 → x_hat = 0 → y = 0
        for &v in y.data() {
            assert!(v.abs() < 1e-4, "y={}", v);
        }
        Ok(())
    }

    #[test]
    fn test_gn_invalid_groups() -> MlResult<()> {
        // C=4, G=3 → 4 % 3 != 0 → 오류
        let x     = GlobalTensor::from_vec(vec![1.0f32; 8], &[1, 4, 1, 2])?;
        let gamma = GlobalTensor::from_vec(vec![1.0f32; 4], &[4])?;
        let beta  = GlobalTensor::from_vec(vec![0.0f32; 4], &[4])?;
        let g     = GlobalTensor::from_vec(vec![3.0f32], &[1, 1])?;
        let e     = GlobalTensor::from_vec(vec![1e-5f32], &[1, 1])?;

        let op  = GroupNormOp::new()?;
        assert!(op.forward(&[&x, &gamma, &beta, &g, &e]).is_err());
        Ok(())
    }

    #[test]
    fn test_gn_batch() -> MlResult<()> {
        // N=3 배치 — 각 샘플이 독립적으로 정규화됨을 확인
        let x     = GlobalTensor::from_vec(vec![1.0f32; 3 * 2 * 2 * 2], &[3, 2, 2, 2])?;
        let gamma = GlobalTensor::from_vec(vec![1.0f32; 2], &[2])?;
        let beta  = GlobalTensor::from_vec(vec![0.0f32; 2], &[2])?;

        let (y, _, mean, _) = gn_forward(&x, &gamma, &beta, 1, 1e-5)?;

        assert_eq!(y.shape(),    &[3, 2, 2, 2]);
        assert_eq!(mean.shape(), &[3]); // N*G = 3*1
        Ok(())
    }
}
