use super::*;

impl Function for AvgPool {
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let input = inputs[0];
        let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;

        let mut output_data = vec![0.0; n * c * oh * ow];
        let pool_size = (kh * kw) as f32;

        for n_idx in 0..n {
            for c_idx in 0..c {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let h_start = oh_idx * sh;
                        let w_start = ow_idx * sw;
                        let mut sum = 0.0;

                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_idx = h_start + kh_idx;
                                let w_idx = w_start + kw_idx;
                                
                                if h_idx >= h && w_idx >= w {
                                    continue;
                                }
                                sum += input.get(&[n_idx, c_idx, h_idx, w_idx]).cloned().unwrap_or(0.0);
                            }
                        }
                        let out_idx = n_idx * (c * oh * ow) + c_idx * (oh * ow) + oh_idx * ow + ow_idx;
                        output_data[out_idx] = sum / pool_size;
                    }
                }
            }
        }

        let output = PooledTensor::from_vec(output_data, &[n, c, oh, ow])?;
        Ok(vec![output])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = inputs[0];
        let (n, c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;

        let mut input_grad = vec![0.0; n * c * h * w];
        let pool_size = (kh * kw) as f32;

        for n_idx in 0..n {
            for c_idx in 0..c {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let h_start = oh_idx * sh;
                        let w_start = ow_idx * sw;
                        let grad_val = grad.get(&[n_idx, c_idx, oh_idx, ow_idx]).cloned().unwrap_or(0.0) / pool_size;

                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let h_idx = h_start + kh_idx;
                                let w_idx = w_start + kw_idx;
                                if h_idx < h && w_idx < w {
                                    let grad_idx = n_idx * (c * h * w) + c_idx * (h * w) + h_idx * w + w_idx;
                                    input_grad[grad_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        let input_grad_tensor = PooledTensor::from_vec(input_grad, &[n, c, h, w])?;
        Ok(vec![input_grad_tensor])
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::{Parameter, Variable};
use crate::{MlResult, tensor::{TensorBase, Tensor}, variable};
    use crate::tensor::AutogradFunction;
    use crate::tensor::operators::{AvgPool, Function};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn tensor_avg_pool_operator() -> MlResult<()> {
        let tensor = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[1, 1, 4, 4]).unwrap();
        let op = AvgPool::new((2, 2), (2, 2), (0, 0));
        let result = op.forward(&[&tensor])?.remove(0);
        assert_eq!(result.shape(), vec![1, 1, 2, 2]);
        assert_eq!(result.data(), vec![3.5, 5.5, 11.5, 13.5]);
        Ok(())
    }

    #[test]
    fn test_avg_pool_backward() -> MlResult<()> {
        let a = Variable::new(Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[1, 1, 4, 4])?);
        let op = AvgPool::new((2, 2), (2, 2), (0, 0));
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], &[1, 1, 4, 4])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}