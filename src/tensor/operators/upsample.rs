use super::*;

// ─── NearestUpsample2d ──────────────────────────────────────────────────────

/// Nearest-neighbor 2D upsampling operator.
///
/// `forward` 입력 순서:
///   `[input, scale_h, scale_w]`
///   - input:   `[N, C, H, W]`
///   - scale_h: 스칼라 `[1, 1]` — 높이 확대 배율 (정수)
///   - scale_w: 스칼라 `[1, 1]` — 너비 확대 배율 (정수)
///
/// 반환: `[Y]`  shape `[N, C, H*scale_h, W*scale_w]`
///
/// 각 입력 픽셀을 scale_h × scale_w 블록으로 복제합니다.
///
/// `backward` 반환: `[dX, 0, 0]`
///   - dX의 각 원소 = 대응하는 scale_h × scale_w 블록의 upstream gradient 합
impl Function for NearestUpsample2d {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(NearestUpsample2d)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.len() < 3 {
            return Err(MlError::StringError(
                "NearestUpsample2d::forward: [input, scale_h, scale_w] 필요".into(),
            ));
        }
        let input   = targets[0];
        let scale_h = targets[1].data()[0] as usize;
        let scale_w = targets[2].data()[0] as usize;

        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err(MlError::StringError(format!(
                "NearestUpsample2d: 입력은 4D [N, C, H, W] 이어야 합니다. 현재: {:?}", in_shape
            )));
        }
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let h_out = h * scale_h;
        let w_out = w * scale_w;

        let in_data = input.data();
        let mut y_data = vec![0.0f32; n * c * h_out * w_out];

        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let val = in_data[ni * c * h * w + ci * h * w + hi * w + wi];
                        // scale_h × scale_w 블록에 복제
                        for sh in 0..scale_h {
                            for sw in 0..scale_w {
                                let oh = hi * scale_h + sh;
                                let ow = wi * scale_w + sw;
                                y_data[ni * c * h_out * w_out + ci * h_out * w_out + oh * w_out + ow] = val;
                            }
                        }
                    }
                }
            }
        }

        let out_shape = vec![n, c, h_out, w_out];

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[NearestUpsample2d::forward] [{},{},{},{}] → [{},{},{},{}]  scale=({},{})",
            n, c, h, w, n, c, h_out, w_out, scale_h, scale_w
        );

        Ok(vec![GlobalTensor::from_vec(y_data, &out_shape)?])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(
        &self,
        targets: &[&dyn TensorBase],
        grad: &dyn TensorBase,
    ) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input   = targets[0];
        let scale_h = targets[1].data()[0] as usize;
        let scale_w = targets[2].data()[0] as usize;

        let in_shape = input.shape();
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let h_out = h * scale_h;
        let w_out = w * scale_w;

        let dy_data = grad.data();
        let mut dx = vec![0.0f32; n * c * h * w];

        // 각 scale_h × scale_w 블록의 gradient를 합산
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let mut sum = 0.0f32;
                        for sh in 0..scale_h {
                            for sw in 0..scale_w {
                                let oh = hi * scale_h + sh;
                                let ow = wi * scale_w + sw;
                                sum += dy_data[ni * c * h_out * w_out + ci * h_out * w_out + oh * w_out + ow];
                            }
                        }
                        dx[ni * c * h * w + ci * h * w + hi * w + wi] = sum;
                    }
                }
            }
        }

        #[cfg(feature = "debugging")]
        crate::tensor::operators::debug::stats_raw("  └─ dX (NearestUpsample2d)", &dx, in_shape);

        let zero = GlobalTensor::from_vec(vec![0.0], &[1, 1])?;
        Ok(vec![
            GlobalTensor::from_vec(dx, in_shape)?,
            zero.clone(),
            zero,
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}

// ─── 테스트 ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn upsample_forward(
        input: &dyn TensorBase,
        scale: (usize, usize),
    ) -> MlResult<GlobalTensor<f32>> {
        let sh = GlobalTensor::from_vec(vec![scale.0 as f32], &[1, 1])?;
        let sw = GlobalTensor::from_vec(vec![scale.1 as f32], &[1, 1])?;
        let op = NearestUpsample2d::new()?;
        Ok(op.forward(&[input, &sh, &sw])?.remove(0))
    }

    #[test]
    fn test_upsample_shape_2x() -> MlResult<()> {
        // [1, 1, 2, 2] → scale 2x2 → [1, 1, 4, 4]
        let input = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let y = upsample_forward(&input, (2, 2))?;
        assert_eq!(y.shape(), &[1, 1, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_upsample_values_2x() -> MlResult<()> {
        // [1, 1, 2, 2]:
        //  1 2
        //  3 4
        // → scale 2x2 → [1, 1, 4, 4]:
        //  1 1 2 2
        //  1 1 2 2
        //  3 3 4 4
        //  3 3 4 4
        let input = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let y = upsample_forward(&input, (2, 2))?;
        assert_eq!(y.data(), &[
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ]);
        Ok(())
    }

    #[test]
    fn test_upsample_multi_channel() -> MlResult<()> {
        // [1, 2, 2, 2] → scale 2x2 → [1, 2, 4, 4]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let input = GlobalTensor::from_vec(data, &[1, 2, 2, 2])?;
        let y = upsample_forward(&input, (2, 2))?;
        assert_eq!(y.shape(), &[1, 2, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_upsample_multi_batch() -> MlResult<()> {
        // [2, 1, 2, 2] → scale 2x2 → [2, 1, 4, 4]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let input = GlobalTensor::from_vec(data, &[2, 1, 2, 2])?;
        let y = upsample_forward(&input, (2, 2))?;
        assert_eq!(y.shape(), &[2, 1, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_upsample_scale_1x() -> MlResult<()> {
        // scale=1 → identity
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = GlobalTensor::from_vec(data.clone(), &[1, 1, 2, 2])?;
        let y = upsample_forward(&input, (1, 1))?;
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(y.data(), &data);
        Ok(())
    }

    #[cfg(feature = "enableBackward")]
    #[test]
    fn test_upsample_backward_sum() -> MlResult<()> {
        // backward: scale 2x2 → 각 입력 픽셀에 대해 2×2=4개 grad 합산
        let input = GlobalTensor::from_vec(vec![0.0; 4], &[1, 1, 2, 2])?;
        let sh = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;
        let sw = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;

        // upstream grad: 모두 1.0 → [1, 1, 4, 4]
        let grad = GlobalTensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4])?;

        let op = NearestUpsample2d::new()?;
        let dx = op.backward(&[&input, &sh, &sw], &grad)?;

        // 각 원소 = 4개 합 = 4.0
        assert_eq!(dx[0].shape(), &[1, 1, 2, 2]);
        assert!(dx[0].data().iter().all(|&v| (v - 4.0).abs() < 1e-6));
        Ok(())
    }
}
