use super::*;

impl Function for Conv2d {
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let input = inputs[0];
        let weight = inputs[1];
        let bias = inputs.get(2);

        let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let (oc, _, kh, kw) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;

        // 1. Input to columns
        let col = im2col(input, kh, kw, sh, sw, ph, pw)?;

        // 2. Reshape weights for matmul
        let weight_reshaped = Tensor::from_vec(weight.data().to_vec(), &[oc, c * kh * kw])?;

        // 3. Matrix multiplication
        let matmul = Matmul::new();
        let output_col = matmul.forward(&[&weight_reshaped, &col])?.remove(0); // Shape: [oc, n * oh * ow]

        // 4. Reshape and transpose output_col to output tensor
        // The custom im2col implementation results in a column layout of (oh, ow, n).
        // So we first reshape to [oc, oh, ow, n].
        let output_reshaped_data = output_col.data().to_vec();
        let mut output_transposed_data = vec![0.0; output_reshaped_data.len()];
        // Transpose from [oc, oh, ow, n] to [n, oc, oh, ow]
        for n_idx in 0..n {
            for oc_idx in 0..oc {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let src_idx = oc_idx * (oh * ow * n) + oh_idx * (ow * n) + ow_idx * n + n_idx;
                        let dest_idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + oh_idx * w + ow_idx;
                        output_transposed_data[dest_idx] = output_reshaped_data[src_idx];
                    }
                }
            }
        }
        // let mut output_tensor = Tensor::from_vec(output_transposed_data, &[n, oc, oh, ow])?;

        // 5. Add bias
        if let Some(bias_tensor) = bias {
            let bias_data = bias_tensor.data();
            for n_idx in 0..n {
                for oc_idx in 0..oc {
                    for i in 0..(oh * ow) {
                        let idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + i;
                        output_transposed_data[idx] += bias_data[oc_idx];
                    }
                }
            }
        }

        Ok(vec![PooledTensor::from_vec(output_transposed_data, &[n, oc, oh, ow])?])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = inputs[0];
        let weight = inputs[1];

        let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let (oc, _, kh, kw) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;

        // Reshape grad (dL/dY) from [n, oc, oh, ow] to grad_col [oc, n * oh * ow]
        let grad_data = grad.data();
        let mut grad_transposed_data = vec![0.0; grad_data.len()];
        for n_idx in 0..n {
            for oc_idx in 0..oc {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let src_idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + oh_idx * ow + ow_idx;
                        let dest_idx = oc_idx * (n * oh * ow) + (oh_idx * ow + ow_idx) * n + n_idx;
                        grad_transposed_data[dest_idx] = grad_data[src_idx];
                    }
                }
            }
        }
        let grad_col = Tensor::from_vec(grad_transposed_data, &[oc, n * oh * ow])?;

        // 1. Calculate dL/dW (weight gradient)
        // dL/dW_reshaped = grad_col * col^T
        let col = im2col(input, kh, kw, sh, sw, ph, pw)?;
        let T = Transpose::new((0, 1));
        let col_t = T.forward(&[&col])?.remove(0);
        let matmul = Matmul::new();
        let dw_reshaped = matmul.forward(&[&grad_col, &col_t])?.remove(0);
        let dw = PooledTensor::from_vec(dw_reshaped.data().to_vec(), &[oc, c, kh, kw])?;

        // 2. Calculate dL/dX (input gradient)
        // dL/dX_col = W^T * grad_col
        let weight_reshaped = Tensor::from_vec(weight.data().to_vec(), &[oc, c * kh * kw])?;
        let weight_reshaped_t = T.forward(&[&weight_reshaped])?.remove(0);
        let dx_col = matmul.forward(&[&weight_reshaped_t, &grad_col])?.remove(0);
        let dx_tensor = col2im(&dx_col.to_id(true)?, (n, c, h, w), kh, kw, sh, sw, ph, pw)?;

        let mut results = vec![dx_tensor, dw];

        // 3. Calculate dL/dB (bias gradient)
        if inputs.len() > 2 {
            let mut db_data = vec![0.0; oc];
            for oc_idx in 0..oc {
                for n_idx in 0..n {
                    for i in 0..(oh * ow) {
                        let idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + i;
                        db_data[oc_idx] += grad.data()[idx];
                    }
                }
            }
            results.push(PooledTensor::from_vec(db_data, &[oc])?);
        }

        Ok(results)
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &HandleId { &self.node_id }
}

// Helper functions
fn im2col(input: &dyn TensorBase, kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> MlResult<Tensor> {
    let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;

    let padded_h = h + 2 * ph;
    let padded_w = w + 2 * pw;
    let mut padded_input = vec![0.0; n * c * padded_h * padded_w];
    let input_data = input.data();

    for n_idx in 0..n {
        for c_idx in 0..c {
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let src_idx = n_idx*c*h*w + c_idx*h*w + h_idx*w + w_idx;
                    let dest_idx = n_idx*c*padded_h*padded_w + c_idx*padded_h*padded_w + (h_idx+ph)*padded_w + (w_idx+pw);
                    padded_input[dest_idx] = input_data[src_idx];
                }
            }
        }
    }

    let mut col_data = vec![0.0; c * kh * kw * n * oh * ow];
    for c_col in 0..(c * kh * kw) {
        let kw_offset = c_col % kw;
        let kh_offset = (c_col / kw) % kh;
        let c_in = c_col / (kh * kw);
        for oh_idx in 0..oh {
            for ow_idx in 0..ow {
                for n_idx in 0..n {
                    let h_pad = oh_idx * sh + kh_offset;
                    let w_pad = ow_idx * sw + kw_offset;
                    let padded_idx = n_idx*c*padded_h*padded_w + c_in*padded_h*padded_w + h_pad*padded_w + w_pad;
                    let col_idx = c_col * (n * oh * ow) + (oh_idx * ow + ow_idx) * n + n_idx;
                    col_data[col_idx] = padded_input[padded_idx];
                }
            }
        }
    }
    Tensor::from_vec(col_data, &[c * kh * kw, n * oh * ow])
}

fn col2im(col: &dyn TensorBase, input_shape: (usize, usize, usize, usize), kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> MlResult<PooledTensor> {
    let (n, c, h, w) = input_shape;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    let padded_h = h + 2 * ph;
    let padded_w = w + 2 * pw;
    let mut padded_img_data = vec![0.0; n * c * padded_h * padded_w];
    let col_data = col.data();

    for c_col in 0..(c * kh * kw) {
        let kw_offset = c_col % kw;
        let kh_offset = (c_col / kw) % kh;
        let c_in = c_col / (kh * kw);
        for oh_idx in 0..oh {
            for ow_idx in 0..ow {
                for n_idx in 0..n {
                    let h_pad = oh_idx * sh + kh_offset;
                    let w_pad = ow_idx * sw + kw_offset;
                    let padded_idx = n_idx*c*padded_h*padded_w + c_in*padded_h*padded_w + h_pad*padded_w + w_pad;
                    let col_idx = c_col * (n * oh * ow) + (oh_idx * ow + ow_idx) * n + n_idx;
                    padded_img_data[padded_idx] += col_data[col_idx];
                }
            }
        }
    }

    let mut img_data = vec![0.0; n * c * h * w];
    for n_idx in 0..n {
        for c_idx in 0..c {
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let padded_idx = n_idx*c*padded_h*padded_w + c_idx*padded_h*padded_w + (h_idx+ph)*padded_w + (w_idx+pw);
                    let img_idx = n_idx*c*h*w + c_idx*h*w + h_idx*w + w_idx;
                    img_data[img_idx] = padded_img_data[padded_idx];
                }
            }
        }
    }
    PooledTensor::from_vec(img_data, &[n, c, h, w])
}
