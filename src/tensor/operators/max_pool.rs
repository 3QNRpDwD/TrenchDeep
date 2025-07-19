use super::*;

impl Function for MaxPool {
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let input = inputs[0];
        let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;

        let mut output_data = vec![0.0; n * c * oh * ow];
        let mut indices_data = vec![0.0; n * c * oh * ow];

        for n_idx in 0..n {
            for c_idx in 0..c {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let h_start = oh_idx * sh;
                        let w_start = ow_idx * sw;

                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_idx = h_start + kh_idx;
                                let w_idx = w_start + kw_idx;

                                if h_idx >= h || w_idx >= w {
                                    continue;
                                }
                                let val = input.get(&[n_idx, c_idx, h_idx, w_idx]).cloned().unwrap_or(f32::NEG_INFINITY);
                                if val > max_val {
                                    max_val = val;
                                    max_idx = (h_idx * w + w_idx) as i32;
                                 }
                            }
                        }
                        let out_idx = n_idx * (c * oh * ow) + c_idx * (oh * ow) + oh_idx * ow + ow_idx;
                        output_data[out_idx] = max_val;
                        indices_data[out_idx] = max_idx as f32;
                    }
                }
            }
        }

        let output = PooledTensor::from_vec(output_data, &[n, c, oh, ow])?;
        let indices = PooledTensor::from_vec(indices_data, &[n, c, oh, ow])?;

        Ok(vec![output, indices])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, _inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = _inputs[0];
        let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);

        let mut input_grad = vec![0.0; n * c * h * w];

        let indices = _inputs[1];
        for i in 0..grad.data().len() {
            let max_idx = indices.data()[i] as usize;
            input_grad[max_idx] += grad.data()[i];
        }

        let input_grad_tensor = PooledTensor::from_vec(input_grad, &[n, c, h, w])?;
        Ok(vec![input_grad_tensor])
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::Parameter;
    use crate::nn::Variable;
    use crate::tensor::operators::tests::assert_tensor_eq;
    use crate::tensor::operators::{Function, MaxPool};
    use crate::tensor::AutogradFunction;
    use crate::{tensor::{Tensor, TensorBase}, MlResult};

    #[test]
    fn tensor_max_pool_operator() -> MlResult<()> {
        let tensor = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[1, 1, 4, 4]).unwrap();
        let op = MaxPool::new((2, 2), (2, 2), (0, 0));
        let result = op.forward(&[&tensor])?.remove(0);
        assert_eq!(result.shape(), vec![1, 1, 2, 2]);
        assert_eq!(result.data(), vec![6.0, 8.0, 14.0, 16.0]);
        Ok(())
    }

    #[test]
    fn test_max_pool_backward() -> MlResult<()> {
        let a = Variable::new(Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[1, 1, 4, 4])?);
        let op = MaxPool::new((2, 2), (2, 2), (0, 0));
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[1, 1, 4, 4])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}