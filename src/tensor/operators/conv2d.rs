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
        let output_col_data = output_col.data();
        let mut output_data = vec![0.0; n * oc * oh * ow];
        for n_idx in 0..n {
            for oc_idx in 0..oc {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let src_idx = oc_idx * (n * oh * ow) + (oh_idx * ow + ow_idx) * n + n_idx;
                        let dest_idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + oh_idx * ow + ow_idx;
                        output_data[dest_idx] = output_col_data[src_idx];
                    }
                }
            }
        }

        // 5. Add bias
        if let Some(bias_tensor) = bias {
            let bias_data = bias_tensor.data();
            for n_idx in 0..n {
                for oc_idx in 0..oc {
                    for i in 0..(oh * ow) {
                        let idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + i;
                        output_data[idx] += bias_data[oc_idx];
                    }
                }
            }
        }

        Ok(vec![PooledTensor::from_vec(output_data, &[n, oc, oh, ow])?])
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
        let mut grad_col_data = vec![0.0; oc * n * oh * ow];
        for n_idx in 0..n {
            for oc_idx in 0..oc {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let src_idx = n_idx * (oc * oh * ow) + oc_idx * (oh * ow) + oh_idx * ow + ow_idx;
                        let dest_idx = oc_idx * (n * oh * ow) + (oh_idx * ow + ow_idx) * n + n_idx;
                        grad_col_data[dest_idx] = grad_data[src_idx];
                    }
                }
            }
        }
        let grad_col = Tensor::from_vec(grad_col_data, &[oc, n * oh * ow])?;

        // 1. Calculate dL/dW (weight gradient)
        // dL/dW_reshaped = grad_col * col^T
        let col = im2col(input, kh, kw, sh, sw, ph, pw)?;
        let T = Transpose::new();
        let col_t = T.forward(&[&col, &Tensor::from_vec(vec![1.0,0.0], &[6])?])?.remove(0);
        // col.transpose(&[1, 0])?;
        let matmul = Matmul::new();
        let dw_reshaped = matmul.forward(&[&grad_col, &col_t])?.remove(0);
        let dw = PooledTensor::from_vec(dw_reshaped.data().to_vec(), &[oc, c, kh, kw])?;

        // 2. Calculate dL/dX (input gradient)
        // dL/dX_col = W^T * grad_col
        let weight_reshaped = Tensor::from_vec(weight.data().to_vec(), &[oc, c * kh * kw])?;
        let weight_reshaped_t = T.forward(&[&weight_reshaped, &Tensor::from_vec(vec![1.0,0.0], &[6])? ])?.remove(0);
        let dx_col = matmul.forward(&[&weight_reshaped_t, &grad_col])?.remove(0);
        let dx_tensor = col2im(&dx_col.to_id(true)?, (n, c, h, w), kh, kw, sh, sw, ph, pw)?;

        let mut results = vec![dx_tensor, dw];

        // 3. Calculate dL/dB (bias gradient)
        if inputs.len() > 2 {
            let db_data: Vec<f32> = grad.data().chunks(oh * ow).map(|chunk| chunk.iter().sum()).collect();
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

    let mut padded_input = Tensor::new_with_padding(input, (ph, pw))?;
    let mut col_data = Vec::with_capacity(n * c * oh * ow * kh * kw);

    for i in 0..n {
        for j in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let window = padded_input.slice(i, j, y * sh, x * sw, kh, kw)?;
                    col_data.extend_from_slice(window.data());
                }
            }
        }
    }
    
    let dims_tensor = Tensor::from_vec(vec![1.0, 4.0, 5.0, 0.0, 2.0, 3.0], &[6])?;
    let T = Transpose::new();
    let R = Reshape::new();
    let col_tensor = Tensor::from_vec(col_data, &[n, c, oh, ow, kh, kw])?;
    let col_transposed = T.forward(&[&col_tensor, &dims_tensor])?.remove(0);

    let final_shape = &[c * kh * kw, n * oh * ow];
    R.forward(&[&col_transposed, &Tensor::from_vec(final_shape.iter().map(|&x| x as f32).collect(), &[2])?])?
        .remove(0)
        .to_id(true)
}

fn col2im(col: &dyn TensorBase, input_shape: (usize, usize, usize, usize), kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> MlResult<PooledTensor> {
    let (n, c, h, w) = input_shape;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (w + 2 * pw - kw) / sw + 1;
    let mut img_data = vec![0.0; n * c * h * w];
    let col_data = col.data();

    for c_in in 0..c {
        for ky in 0..kh {
            for kx in 0..kw {
                for y in 0..oh {
                    for x in 0..ow {
                        for i in 0..n {
                            let h_pad = y * sh + ky - ph;
                            let w_pad = x * sw + kx - pw;
                            if h_pad >= 0 && h_pad < h && w_pad >= 0 && w_pad < w {
                                let col_idx = (c_in * kh * kw + ky * kw + kx) * (n * oh * ow) + (y * ow + x) * n + i;
                                let img_idx = i * (c * h * w) + c_in * (h * w) + h_pad * w + w_pad;
                                img_data[img_idx] += col_data[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    PooledTensor::from_vec(img_data, &[n, c, h, w])
}

#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Conv2d, Function};
    use crate::{tensor::{Tensor, TensorBase}, MlResult};

    #[test]
    fn tensor_conv2d_operator() -> MlResult<()> {
        let input = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[1, 1, 4, 4]).unwrap();
        let weight = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let op = Conv2d::new((2, 2), (2, 2), (0, 0));
        let result = op.forward(&[&input, &weight])?.remove(0);
        assert_eq!(result.shape(), vec![1, 1, 2, 2]);
        assert_eq!(result.data(), vec![24.0, 28.0, 40.0, 44.0]);
        Ok(())
    }

    #[test]
    fn tensor_conv2d_with_bias_operator() -> MlResult<()> {
        let input = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[1, 1, 4, 4]).unwrap();
        let weight = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let bias = Tensor::from_vec(vec![10.0], &[1]).unwrap();
        let op = Conv2d::new((2, 2), (2, 2), (0, 0));
        let result = op.forward(&[&input, &weight, &bias])?.remove(0);
        assert_eq!(result.shape(), vec![1, 1, 2, 2]);
        assert_eq!(result.data(), vec![34.0, 38.0, 50.0, 54.0]);
        Ok(())
    }
}
