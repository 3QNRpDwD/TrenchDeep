use super::*;

// ─── im2col / col2im ────────────────────────────────────────────────────────

/// 입력 텐서를 im2col 형태로 변환합니다.
///
/// 입력 shape: `[N, C_in, H, W]`
/// 출력 shape: `[N, C_in*kH*kW, H_out*W_out]` (행 우선 평탄화)
///
/// 반환: (col 데이터, N, H_out, W_out, col_rows=C_in*kH*kW)
fn im2col(
    input: &[f32],
    in_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<f32>, usize, usize, usize, usize) {
    let (n, c_in, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;
    let col_rows = c_in * kh * kw;
    let col_cols = h_out * w_out;

    let mut col = vec![0.0f32; n * col_rows * col_cols];

    for ni in 0..n {
        for ci in 0..c_in {
            for ki in 0..kh {
                for kj in 0..kw {
                    let row = ci * kh * kw + ki * kw + kj;
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            let h_in_pad = hi * sh + ki;
                            let w_in_pad = wi * sw + kj;
                            let col_idx = ni * col_rows * col_cols
                                + row * col_cols
                                + hi * w_out
                                + wi;
                            if h_in_pad < ph
                                || w_in_pad < pw
                                || h_in_pad >= h + ph
                                || w_in_pad >= w + pw
                            {
                                col[col_idx] = 0.0;
                            } else {
                                let src_h = h_in_pad - ph;
                                let src_w = w_in_pad - pw;
                                let in_idx = ni * c_in * h * w
                                    + ci * h * w
                                    + src_h * w
                                    + src_w;
                                col[col_idx] = input[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    (col, n, h_out, w_out, col_rows)
}

/// im2col 역변환: dcol → dX 누적 (gradient 역전파용)
fn col2im(
    col: &[f32],
    in_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Vec<f32> {
    let (n, c_in, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;
    let col_rows = c_in * kh * kw;
    let col_cols = h_out * w_out;

    let mut dx = vec![0.0f32; n * c_in * h * w];

    for ni in 0..n {
        for ci in 0..c_in {
            for ki in 0..kh {
                for kj in 0..kw {
                    let row = ci * kh * kw + ki * kw + kj;
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            let h_in_pad = hi * sh + ki;
                            let w_in_pad = wi * sw + kj;
                            if h_in_pad < ph
                                || w_in_pad < pw
                                || h_in_pad >= h + ph
                                || w_in_pad >= w + pw
                            {
                                continue;
                            }
                            let src_h = h_in_pad - ph;
                            let src_w = w_in_pad - pw;
                            let in_idx = ni * c_in * h * w
                                + ci * h * w
                                + src_h * w
                                + src_w;
                            let col_idx = ni * col_rows * col_cols
                                + row * col_cols
                                + hi * w_out
                                + wi;
                            dx[in_idx] += col[col_idx];
                        }
                    }
                }
            }
        }
    }
    dx
}

// ─── 행렬곱 헬퍼 ────────────────────────────────────────────────────────────

/// A[m, k] × B[k, n] → C[m, n]
fn matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_val = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_val * b[l * n + j];
            }
        }
    }
    c
}

/// A^T[k, m] × B[k, n] → C[m, n]  (A는 [m, k] 레이아웃)
fn matmul_at_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // A^T is [k, m], B is [k, n] → C[m, n]
    let mut c = vec![0.0f32; m * n];
    for l in 0..k {
        for i in 0..m {
            let at_val = a[i * k + l]; // A^T[l, i] = A[i, l]
            for j in 0..n {
                c[i * n + j] += at_val * b[l * n + j];
            }
        }
    }
    c
}

/// A[m, k] × B^T[n, k] → C[m, n]  (B는 [n, k] 레이아웃)
fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // A is [m, k], B^T is [k, n] where B is [n, k]
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l]; // B^T[l, j] = B[j, l]
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ─── Conv2d Function ────────────────────────────────────────────────────────

/// Conv2d operator
///
/// `forward` 입력 순서:
///   `[input, weight, bias, stride_h, stride_w, pad_h, pad_w]`
///
/// `backward` 반환 순서 (입력과 대응):
///   `[dX, dW, db, 0, 0, 0, 0]`
impl Function for Conv2dOp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Conv2dOp)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.len() < 7 {
            return Err(MlError::StringError(
                "Conv2dOp::forward: [input, weight, bias, stride_h, stride_w, pad_h, pad_w] 필요".into(),
            ));
        }
        let input  = targets[0];
        let weight = targets[1];
        let bias   = targets[2];
        let stride_h = targets[3].data()[0] as usize;
        let stride_w = targets[4].data()[0] as usize;
        let pad_h    = targets[5].data()[0] as usize;
        let pad_w    = targets[6].data()[0] as usize;

        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err(MlError::StringError(
                format!("Conv2d: 입력은 4D [N, C_in, H, W] 이어야 합니다. 현재: {:?}", in_shape)
            ));
        }
        let w_shape = weight.shape();
        if w_shape.len() != 4 {
            return Err(MlError::StringError(
                format!("Conv2d: 가중치는 4D [C_out, C_in, kH, kW] 이어야 합니다. 현재: {:?}", w_shape)
            ));
        }

        let (n, _c_in, _h, _w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, _c_in2, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

        let (col, _n, h_out, w_out, col_rows) = im2col(
            input.data(),
            in_shape,
            (kh, kw),
            (stride_h, stride_w),
            (pad_h, pad_w),
        );
        let col_cols = h_out * w_out;

        // W_mat: [C_out, C_in*kH*kW]
        // col:   [N, C_in*kH*kW, H_out*W_out]  → per batch slice [col_rows, col_cols]
        // Y_mat: [N, C_out, H_out*W_out]
        let mut y_data = vec![0.0f32; n * c_out * h_out * w_out];
        let w_data = weight.data();

        for ni in 0..n {
            let col_slice = &col[ni * col_rows * col_cols..(ni + 1) * col_rows * col_cols];
            // [C_out, col_rows] × [col_rows, col_cols] → [C_out, col_cols]
            let y_slice = matmul_2d(w_data, col_slice, c_out, col_rows, col_cols);
            let out_start = ni * c_out * h_out * w_out;
            y_data[out_start..out_start + c_out * h_out * w_out]
                .copy_from_slice(&y_slice);
        }

        // bias broadcast: Y[n, c, h, w] += bias[c]
        let bias_data = bias.data();
        for ni in 0..n {
            for ci in 0..c_out {
                let base = ni * c_out * h_out * w_out + ci * h_out * w_out;
                for i in 0..h_out * w_out {
                    y_data[base + i] += bias_data[ci];
                }
            }
        }

        let out_shape = vec![n, c_out, h_out, w_out];

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Conv2dOp::forward] [{},{},{},{}] → [{},{},{},{}]  kernel=({},{}) stride=({},{}) pad=({},{})",
            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
            n, c_out, h_out, w_out,
            kh, kw, stride_h, stride_w, pad_h, pad_w
        );

        Ok(vec![GlobalTensor::from_vec(y_data, &out_shape)?])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(
        &self,
        targets: &[&dyn TensorBase],
        grad: &dyn TensorBase,
    ) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input  = targets[0];
        let weight = targets[1];
        let stride_h = targets[3].data()[0] as usize;
        let stride_w = targets[4].data()[0] as usize;
        let pad_h    = targets[5].data()[0] as usize;
        let pad_w    = targets[6].data()[0] as usize;

        let in_shape = input.shape();
        let w_shape  = weight.shape();
        let (n, _c_in, _h, _w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, _c_in2, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Conv2dOp::backward] in={:?} w={:?} {}",
            in_shape, w_shape,
            crate::tensor::operators::debug::summary("grad", grad)
        );

        let (col, _n, h_out, w_out, col_rows) = im2col(
            input.data(),
            in_shape,
            (kh, kw),
            (stride_h, stride_w),
            (pad_h, pad_w),
        );
        let col_cols = h_out * w_out;

        let dy_data = grad.data(); // [N, C_out, H_out, W_out]
        let w_data  = weight.data();

        // ── dW: [C_out, C_in*kH*kW] ──────────────────────────────────────
        // dW = sum_n ( dY_mat[n] × col[n]^T )
        // dY_mat[n]: [C_out, col_cols], col[n]^T: [col_cols, col_rows]
        let mut dw_data = vec![0.0f32; c_out * col_rows];
        for ni in 0..n {
            let dy_slice = &dy_data[ni * c_out * col_cols..(ni + 1) * c_out * col_cols];
            let col_slice = &col[ni * col_rows * col_cols..(ni + 1) * col_rows * col_cols];
            // [C_out, col_cols] × [col_cols, col_rows]^T
            // = [C_out, col_cols] × col^T [col_cols->col_rows]
            // matmul_a_bt: A[m,k] × B^T[n,k] → C[m,n]
            // A=dy_slice[C_out, col_cols], B=col_slice[col_rows, col_cols] → dw[C_out, col_rows]
            let dw_n = matmul_a_bt(dy_slice, col_slice, c_out, col_cols, col_rows);
            for i in 0..c_out * col_rows {
                dw_data[i] += dw_n[i];
            }
        }

        // ── dcol → dX ────────────────────────────────────────────────────
        // dcol = W_mat^T × dY_mat  →  [col_rows, col_cols] per batch
        let mut dcol = vec![0.0f32; n * col_rows * col_cols];
        for ni in 0..n {
            let dy_slice = &dy_data[ni * c_out * col_cols..(ni + 1) * c_out * col_cols];
            // matmul_at_b: A^T[k,m] × B[k,n] → C[m,n]
            // A=w_data[C_out, col_rows], B=dy_slice[C_out, col_cols] → dcol_n[col_rows, col_cols]
            let dcol_n = matmul_at_b(w_data, dy_slice, col_rows, c_out, col_cols);
            let start = ni * col_rows * col_cols;
            dcol[start..start + col_rows * col_cols].copy_from_slice(&dcol_n);
        }
        let dx_data = col2im(&dcol, in_shape, (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

        // ── db: [C_out] ───────────────────────────────────────────────────
        let mut db_data = vec![0.0f32; c_out];
        for ni in 0..n {
            for ci in 0..c_out {
                let base = ni * c_out * col_cols + ci * col_cols;
                db_data[ci] += dy_data[base..base + col_cols].iter().sum::<f32>();
            }
        }

        #[cfg(feature = "debugging")]
        {
            crate::tensor::operators::debug::stats_raw("  └─ dX",     &dx_data, in_shape);
            crate::tensor::operators::debug::stats_raw("  └─ dW",     &dw_data, w_shape);
            crate::tensor::operators::debug::stats_raw("  └─ db",     &db_data, &[c_out]);
        }

        let zero_scalar = GlobalTensor::from_vec(vec![0.0], &[1, 1])?;
        Ok(vec![
            GlobalTensor::from_vec(dx_data, in_shape)?,
            GlobalTensor::from_vec(dw_data, w_shape)?,
            GlobalTensor::from_vec(db_data, &[c_out])?,
            zero_scalar.clone(),
            zero_scalar.clone(),
            zero_scalar.clone(),
            zero_scalar,
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}

// ─── 테스트 ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn conv2d_forward(
        input: &dyn TensorBase,
        weight: &dyn TensorBase,
        bias: &dyn TensorBase,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<GlobalTensor<f32>> {
        let sh = GlobalTensor::from_vec(vec![stride.0 as f32],   &[1, 1])?;
        let sw = GlobalTensor::from_vec(vec![stride.1 as f32],   &[1, 1])?;
        let ph = GlobalTensor::from_vec(vec![padding.0 as f32],  &[1, 1])?;
        let pw = GlobalTensor::from_vec(vec![padding.1 as f32],  &[1, 1])?;
        let op = Conv2dOp::new()?;
        let mut result = op.forward(&[input, weight, bias, &sh, &sw, &ph, &pw])?;
        Ok(result.remove(0))
    }

    #[test]
    fn test_conv2d_output_shape_no_padding() -> MlResult<()> {
        // 입력 [1, 1, 4, 4], 커널 [1, 1, 3, 3], stride=(1,1), padding=(0,0)
        // H_out = (4 - 3) / 1 + 1 = 2, W_out = 2
        let input  = GlobalTensor::from_vec(vec![1.0f32; 16], &[1, 1, 4, 4])?;
        let weight = GlobalTensor::from_vec(vec![1.0f32; 9],  &[1, 1, 3, 3])?;
        let bias   = GlobalTensor::from_vec(vec![0.0f32],     &[1])?;

        let y = conv2d_forward(&input, &weight, &bias, (1, 1), (0, 0))?;

        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        // 각 위치의 값 = 3*3 수용장 내 1.0 합산 = 9.0
        assert_eq!(y.data(), &[9.0, 9.0, 9.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_conv2d_output_shape_with_padding() -> MlResult<()> {
        // 입력 [1, 1, 4, 4], 커널 [1, 1, 3, 3], stride=(1,1), padding=(1,1)
        // H_out = (4 + 2 - 3) / 1 + 1 = 4, W_out = 4
        let input  = GlobalTensor::from_vec(vec![1.0f32; 16], &[1, 1, 4, 4])?;
        let weight = GlobalTensor::from_vec(vec![1.0f32; 9],  &[1, 1, 3, 3])?;
        let bias   = GlobalTensor::from_vec(vec![0.0f32],     &[1])?;

        let y = conv2d_forward(&input, &weight, &bias, (1, 1), (1, 1))?;

        assert_eq!(y.shape(), &[1, 1, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_conv2d_stride_2() -> MlResult<()> {
        // 입력 [1, 1, 6, 6], 커널 [1, 1, 3, 3], stride=(2,2), padding=(0,0)
        // H_out = (6 - 3) / 2 + 1 = 2, W_out = 2
        let input  = GlobalTensor::from_vec(vec![1.0f32; 36], &[1, 1, 6, 6])?;
        let weight = GlobalTensor::from_vec(vec![1.0f32; 9],  &[1, 1, 3, 3])?;
        let bias   = GlobalTensor::from_vec(vec![0.0f32],     &[1])?;

        let y = conv2d_forward(&input, &weight, &bias, (2, 2), (0, 0))?;

        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(y.data(), &[9.0, 9.0, 9.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_conv2d_multi_channel() -> MlResult<()> {
        // 입력 [1, 2, 3, 3], 커널 [4, 2, 3, 3], stride=(1,1), padding=(0,0)
        // H_out = (3 - 3) / 1 + 1 = 1, W_out = 1
        // 각 출력 채널 = sum(input * 대응 kernel)
        let input  = GlobalTensor::from_vec(vec![1.0f32; 18], &[1, 2, 3, 3])?;
        let weight = GlobalTensor::from_vec(vec![1.0f32; 72], &[4, 2, 3, 3])?;
        let bias   = GlobalTensor::from_vec(vec![0.0f32; 4],  &[4])?;

        let y = conv2d_forward(&input, &weight, &bias, (1, 1), (0, 0))?;

        assert_eq!(y.shape(), &[1, 4, 1, 1]);
        // 각 채널 = 2 * 9 = 18 (입력 채널 2개, 각 채널 9개 원소 모두 1.0)
        assert_eq!(y.data(), &[18.0, 18.0, 18.0, 18.0]);
        Ok(())
    }

    #[test]
    fn test_conv2d_bias() -> MlResult<()> {
        let input  = GlobalTensor::from_vec(vec![0.0f32; 9], &[1, 1, 3, 3])?;
        let weight = GlobalTensor::from_vec(vec![1.0f32; 9], &[1, 1, 3, 3])?;
        let bias   = GlobalTensor::from_vec(vec![5.0f32],    &[1])?;

        let y = conv2d_forward(&input, &weight, &bias, (1, 1), (0, 0))?;

        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert_eq!(y.data(), &[5.0]);
        Ok(())
    }

    #[test]
    fn test_conv2d_batch() -> MlResult<()> {
        // 배치 크기 2
        let input  = GlobalTensor::from_vec(vec![1.0f32; 32], &[2, 1, 4, 4])?;
        let weight = GlobalTensor::from_vec(vec![1.0f32; 9],  &[1, 1, 3, 3])?;
        let bias   = GlobalTensor::from_vec(vec![0.0f32],     &[1])?;

        let y = conv2d_forward(&input, &weight, &bias, (1, 1), (0, 0))?;

        assert_eq!(y.shape(), &[2, 1, 2, 2]);
        assert!(y.data().iter().all(|&v| (v - 9.0).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn test_conv2d_1x1_kernel() -> MlResult<()> {
        // 1x1 convolution: 각 위치에서 채널 간 선형 변환
        // 입력 [1, 3, 2, 2], 커널 [2, 3, 1, 1]
        // 출력 [1, 2, 2, 2]
        let input = GlobalTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0,   // channel 0
                 5.0, 6.0, 7.0, 8.0,   // channel 1
                 9.0,10.0,11.0,12.0],  // channel 2
            &[1, 3, 2, 2],
        )?;
        // 커널: 첫 번째 출력 채널 = [1, 0, 0], 두 번째 = [0, 1, 0]
        let weight = GlobalTensor::from_vec(
            vec![1.0, 0.0, 0.0,   // out_ch0: picks in_ch0
                 0.0, 1.0, 0.0],  // out_ch1: picks in_ch1
            &[2, 3, 1, 1],
        )?;
        let bias = GlobalTensor::from_vec(vec![0.0f32; 2], &[2])?;

        let y = conv2d_forward(&input, &weight, &bias, (1, 1), (0, 0))?;

        assert_eq!(y.shape(), &[1, 2, 2, 2]);
        // out_ch0 = input_ch0 = [1, 2, 3, 4]
        assert_eq!(&y.data()[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // out_ch1 = input_ch1 = [5, 6, 7, 8]
        assert_eq!(&y.data()[4..8], &[5.0, 6.0, 7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_im2col_basic() {
        // 입력 [1, 1, 3, 3], 커널 3x3, stride 1, padding 0
        // H_out = W_out = 1, col_rows = 9, col_cols = 1
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let (col, n, h_out, w_out, col_rows) =
            im2col(&input, &[1, 1, 3, 3], (3, 3), (1, 1), (0, 0));
        assert_eq!(n, 1);
        assert_eq!(h_out, 1);
        assert_eq!(w_out, 1);
        assert_eq!(col_rows, 9);
        assert_eq!(col, input);
    }
}
