use super::*;

// ─── MaxPool2d ───────────────────────────────────────────────────────────────

/// MaxPool2d operator
///
/// `forward` 입력 순서:
///   `[input, kH, kW, stride_h, stride_w]`
///
/// 반환: `[Y, mask]`
///   - Y:    `[N, C, H_out, W_out]`
///   - mask: `[N, C, H_out, W_out]` — 최댓값 위치의 flattened input index (f32 캐스팅)
///
/// `backward` 입력 순서 (forward 결과 포함):
///   `[input, kH, kW, stride_h, stride_w, mask]`  ← targets에 mask 추가
///
/// 반환: `[dX, 0, 0, 0, 0, 0]`
impl Function for MaxPool2d {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(MaxPool2d)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.len() < 5 {
            return Err(MlError::StringError(
                "MaxPool2d::forward: [input, kH, kW, stride_h, stride_w] 필요".into(),
            ));
        }
        let input    = targets[0];
        let kh       = targets[1].data()[0] as usize;
        let kw       = targets[2].data()[0] as usize;
        let stride_h = targets[3].data()[0] as usize;
        let stride_w = targets[4].data()[0] as usize;

        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err(MlError::StringError(
                format!("MaxPool2d: 입력은 4D [N, C, H, W] 이어야 합니다. 현재: {:?}", in_shape)
            ));
        }
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let h_out = (h - kh) / stride_h + 1;
        let w_out = (w - kw) / stride_w + 1;

        let in_data = input.data();
        let out_size = n * c * h_out * w_out;
        let mut y_data    = vec![f32::NEG_INFINITY; out_size];
        let mut mask_data = vec![0.0f32; out_size];

        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h_out {
                    for wi in 0..w_out {
                        let out_idx = ni * c * h_out * w_out
                            + ci * h_out * w_out
                            + hi * w_out
                            + wi;
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_src_idx = 0usize;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let src_h = hi * stride_h + khi;
                                let src_w = wi * stride_w + kwi;
                                let src_idx = ni * c * h * w
                                    + ci * h * w
                                    + src_h * w
                                    + src_w;
                                let val = in_data[src_idx];
                                if val > max_val {
                                    max_val = val;
                                    max_src_idx = src_idx;
                                }
                            }
                        }
                        y_data[out_idx] = max_val;
                        mask_data[out_idx] = max_src_idx as f32;
                    }
                }
            }
        }

        let out_shape = vec![n, c, h_out, w_out];

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[MaxPool2d::forward] [{},{},{},{}] → [{},{},{},{}]  kernel=({},{}) stride=({},{})",
            n, c, h, w, n, c, h_out, w_out, kh, kw, stride_h, stride_w
        );

        Ok(vec![
            GlobalTensor::from_vec(y_data,    &out_shape)?,
            GlobalTensor::from_vec(mask_data, &out_shape)?,
        ])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(
        &self,
        targets: &[&dyn TensorBase],
        grad: &dyn TensorBase,
    ) -> MlResult<Vec<GlobalTensor<f32>>> {
        // targets: [input, kH, kW, stride_h, stride_w, mask]
        let input = targets[0];
        let mask  = targets[5];

        let in_shape  = input.shape();
        let in_size: usize = in_shape.iter().product();
        let mut dx = vec![0.0f32; in_size];

        let dy_data   = grad.data();
        let mask_data = mask.data();

        for (out_idx, (&dy, &mask_idx)) in dy_data.iter().zip(mask_data.iter()).enumerate() {
            dx[mask_idx as usize] += dy;
        }

        #[cfg(feature = "debugging")]
        crate::tensor::operators::debug::stats_raw("  └─ dX (MaxPool2d)", &dx, in_shape);

        let zero = GlobalTensor::from_vec(vec![0.0], &[1, 1])?;
        Ok(vec![
            GlobalTensor::from_vec(dx, in_shape)?,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}

// ─── AvgPool2d ───────────────────────────────────────────────────────────────

/// AvgPool2d operator
///
/// `forward` 입력 순서:
///   `[input, kH, kW, stride_h, stride_w]`
///
/// 반환: `[Y]`  shape `[N, C, H_out, W_out]`
///
/// `backward` 반환: `[dX, 0, 0, 0, 0]`
impl Function for AvgPool2d {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(AvgPool2d)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.len() < 5 {
            return Err(MlError::StringError(
                "AvgPool2d::forward: [input, kH, kW, stride_h, stride_w] 필요".into(),
            ));
        }
        let input    = targets[0];
        let kh       = targets[1].data()[0] as usize;
        let kw       = targets[2].data()[0] as usize;
        let stride_h = targets[3].data()[0] as usize;
        let stride_w = targets[4].data()[0] as usize;

        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err(MlError::StringError(
                format!("AvgPool2d: 입력은 4D [N, C, H, W] 이어야 합니다. 현재: {:?}", in_shape)
            ));
        }
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let h_out = (h - kh) / stride_h + 1;
        let w_out = (w - kw) / stride_w + 1;
        let kernel_area = (kh * kw) as f32;

        let in_data = input.data();
        let mut y_data = vec![0.0f32; n * c * h_out * w_out];

        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h_out {
                    for wi in 0..w_out {
                        let out_idx = ni * c * h_out * w_out
                            + ci * h_out * w_out
                            + hi * w_out
                            + wi;
                        let mut sum = 0.0f32;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let src_h = hi * stride_h + khi;
                                let src_w = wi * stride_w + kwi;
                                let src_idx = ni * c * h * w
                                    + ci * h * w
                                    + src_h * w
                                    + src_w;
                                sum += in_data[src_idx];
                            }
                        }
                        y_data[out_idx] = sum / kernel_area;
                    }
                }
            }
        }

        let out_shape = vec![n, c, h_out, w_out];

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[AvgPool2d::forward] [{},{},{},{}] → [{},{},{},{}]  kernel=({},{}) stride=({},{})",
            n, c, h, w, n, c, h_out, w_out, kh, kw, stride_h, stride_w
        );

        Ok(vec![GlobalTensor::from_vec(y_data, &out_shape)?])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(
        &self,
        targets: &[&dyn TensorBase],
        grad: &dyn TensorBase,
    ) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input    = targets[0];
        let kh       = targets[1].data()[0] as usize;
        let kw       = targets[2].data()[0] as usize;
        let stride_h = targets[3].data()[0] as usize;
        let stride_w = targets[4].data()[0] as usize;

        let in_shape = input.shape();
        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let h_out = (h - kh) / stride_h + 1;
        let w_out = (w - kw) / stride_w + 1;
        let kernel_area = (kh * kw) as f32;

        let dy_data = grad.data();
        let mut dx = vec![0.0f32; n * c * h * w];

        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h_out {
                    for wi in 0..w_out {
                        let out_idx = ni * c * h_out * w_out
                            + ci * h_out * w_out
                            + hi * w_out
                            + wi;
                        let dy_val = dy_data[out_idx] / kernel_area;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let src_h = hi * stride_h + khi;
                                let src_w = wi * stride_w + kwi;
                                let src_idx = ni * c * h * w
                                    + ci * h * w
                                    + src_h * w
                                    + src_w;
                                dx[src_idx] += dy_val;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(feature = "debugging")]
        crate::tensor::operators::debug::stats_raw("  └─ dX (AvgPool2d)", &dx, in_shape);

        let zero = GlobalTensor::from_vec(vec![0.0], &[1, 1])?;
        Ok(vec![
            GlobalTensor::from_vec(dx, in_shape)?,
            zero.clone(),
            zero.clone(),
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

    fn maxpool_forward(
        input: &dyn TensorBase,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<(GlobalTensor<f32>, GlobalTensor<f32>)> {
        let kh = GlobalTensor::from_vec(vec![kernel.0 as f32], &[1, 1])?;
        let kw = GlobalTensor::from_vec(vec![kernel.1 as f32], &[1, 1])?;
        let sh = GlobalTensor::from_vec(vec![stride.0 as f32], &[1, 1])?;
        let sw = GlobalTensor::from_vec(vec![stride.1 as f32], &[1, 1])?;
        let op = MaxPool2d::new()?;
        let mut result = op.forward(&[input, &kh, &kw, &sh, &sw])?;
        let mask = result.remove(1);
        let y    = result.remove(0);
        Ok((y, mask))
    }

    fn avgpool_forward(
        input: &dyn TensorBase,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<GlobalTensor<f32>> {
        let kh = GlobalTensor::from_vec(vec![kernel.0 as f32], &[1, 1])?;
        let kw = GlobalTensor::from_vec(vec![kernel.1 as f32], &[1, 1])?;
        let sh = GlobalTensor::from_vec(vec![stride.0 as f32], &[1, 1])?;
        let sw = GlobalTensor::from_vec(vec![stride.1 as f32], &[1, 1])?;
        let op = AvgPool2d::new()?;
        Ok(op.forward(&[input, &kh, &kw, &sh, &sw])?.remove(0))
    }

    #[test]
    fn test_maxpool_output_shape() -> MlResult<()> {
        // [1, 1, 4, 4], 커널 2x2, stride 2 → [1, 1, 2, 2]
        let input = GlobalTensor::from_vec(vec![1.0f32; 16], &[1, 1, 4, 4])?;
        let (y, mask) = maxpool_forward(&input, (2, 2), (2, 2))?;
        assert_eq!(y.shape(),    &[1, 1, 2, 2]);
        assert_eq!(mask.shape(), &[1, 1, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_maxpool_values() -> MlResult<()> {
        // 2x2 max pool, stride 2
        // 입력:
        // 1 3 2 4
        // 5 6 1 2
        // 3 1 4 2
        // 7 8 9 0
        let data = vec![
            1.0, 3.0, 2.0, 4.0,
            5.0, 6.0, 1.0, 2.0,
            3.0, 1.0, 4.0, 2.0,
            7.0, 8.0, 9.0, 0.0,
        ];
        let input = GlobalTensor::from_vec(data, &[1, 1, 4, 4])?;
        let (y, _) = maxpool_forward(&input, (2, 2), (2, 2))?;
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        // 좌상: max(1,3,5,6)=6, 우상: max(2,4,1,2)=4
        // 좌하: max(3,1,7,8)=8, 우하: max(4,2,9,0)=9
        assert_eq!(y.data(), &[6.0, 4.0, 8.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_avgpool_output_shape() -> MlResult<()> {
        let input = GlobalTensor::from_vec(vec![1.0f32; 16], &[1, 1, 4, 4])?;
        let y = avgpool_forward(&input, (2, 2), (2, 2))?;
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_avgpool_values() -> MlResult<()> {
        // 2x2 avg pool, stride 2, 모든 값 = 4.0 → 평균 = 4.0
        let data = vec![4.0f32; 16];
        let input = GlobalTensor::from_vec(data, &[1, 1, 4, 4])?;
        let y = avgpool_forward(&input, (2, 2), (2, 2))?;
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert!(y.data().iter().all(|&v| (v - 4.0).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn test_avgpool_avg_correctness() -> MlResult<()> {
        // 2x2 avg pool, stride 2
        // 입력:
        // 1 3
        // 5 7
        // 평균 = (1+3+5+7) / 4 = 4.0
        let input = GlobalTensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], &[1, 1, 2, 2])?;
        let y = avgpool_forward(&input, (2, 2), (2, 2))?;
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert!((y.data()[0] - 4.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_maxpool_multi_channel() -> MlResult<()> {
        // [1, 2, 4, 4], 커널 2x2, stride 2 → [1, 2, 2, 2]
        let input = GlobalTensor::from_vec(vec![1.0f32; 32], &[1, 2, 4, 4])?;
        let (y, _) = maxpool_forward(&input, (2, 2), (2, 2))?;
        assert_eq!(y.shape(), &[1, 2, 2, 2]);
        Ok(())
    }
}
