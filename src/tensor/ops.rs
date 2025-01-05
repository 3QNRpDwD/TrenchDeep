use std::ops::{Add, Div, Mul, Sub};
use crate::{MlError, MlResult};
use crate::tensor::{Tensor, BroadcastLayer, OpsLayer, TensorError, DefaultLayer};

impl<T: PartialEq> OpsLayer<T> for Tensor {
    // Tenser Calculate
    fn add(&self, other: &Self) -> MlResult<Self> where T: Add<Output = T> {
        Ok(self + other)
    } // todo: 브로드케스팅 연산 구조 변경 필요
    fn sub(&self, other: &Self) -> MlResult<Self> where T: Sub<Output = T> {
        Ok(self - other)
    } // todo: 브로드케스팅 연산 구조 변경 필요
    fn mul(&self, other: &Self) -> MlResult<Self> where T: Mul<Output = T> {
        Ok(self * other)
    } // todo: 브로드케스팅 연산 구조 변경 필요
    fn div(&self, other: &Self) -> MlResult<Self> where T: Div<Output = T> {
        Ok(self / other)
    } // todo: 브로드케스팅 연산 구조 변경 필요

    fn add_scalar(&self, scalar: f32) -> MlResult<Self>
    where
        T: Add<Output=T>,
        Self: Sized {
        let data: Vec<f32> = self.data.iter().map(|&x| x + scalar).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn sub_scalar(&self, scalar: f32) -> MlResult<Self>
    where
        T: Sub<Output=T>,
        Self: Sized {
        let data: Vec<f32> = self.data.iter().map(|&x| x - scalar).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn mul_scalar(&self, scalar: f32) -> MlResult<Self>
    where
        T: Mul<Output=T>,
        Self: Sized {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn div_scalar(&self, scalar: f32) -> MlResult<Self>
    where
        T: Div<Output=T>,
        Self: Sized {
        let data: Vec<f32> = self.data.iter().map(|&x| x / scalar).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn scalar_sub(&self, scalar: f32) -> MlResult<Self>
    where
        T: Sub<Output=T>,
        Self: Sized {
        let data: Vec<f32> = self.data.iter().map(|&x| scalar - x).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn scalar_div(&self, scalar: f32) -> MlResult<Self>
    where
        T: Div<Output=T>,
        Self: Sized {
        let data: Vec<f32> = self.data.iter().map(|&x| scalar / x).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn neg(&self) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| -x).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn exp(&self) -> MlResult<Self> {
        !todo!("거듭제곱 연산 추가하기")
    }

    fn pow(&self, power: f32) -> MlResult<Self> {
        !todo!("x 의 n 제곱을 반환해주는 함수 추가하기")
    }

    fn pow_scalar(&self, exponent: f32) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| x.powf(exponent)).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn scalar_pow(&self, scalar: f32) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| scalar.powf(x)).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn sqrt(&self) -> MlResult<Self> {
        todo!("제곱근 함수 연산 추가하기")
    }

    fn square(&self) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| x * x).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn log(&self) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| x.ln()).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn matmul(&self, other: &Tensor) -> MlResult<Self> {
        // Handle empty tensors
        if self.data.is_empty() || other.data.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let a = self.shape.len();
        let b = other.shape.len();

        match (a, b) {
            // Case 1: 1D * 1D (dot product)
            (1, 1) => {
                if self.shape[0] != other.shape[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape.clone(),
                            right_shape: other.shape.clone(),
                        },
                    ));
                }
                let sum = self
                    .data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();
                DefaultLayer::from(vec![sum], &vec![])
            }

            // Case 2: 2D * 1D or 1D * 2D
            (2, 1) => {
                if self.shape[1] != other.shape[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape.clone(),
                            right_shape: other.shape.clone(),
                        },
                    ));
                }
                let m = self.shape[0];
                let k = self.shape[1];
                let mut data = vec![0.0; m];

                for i in 0..m {
                    let mut sum = 0.0;
                    for j in 0..k {
                        sum += self.data[i * k + j] * other.data[j];
                    }
                    data[i] = sum;
                }
                DefaultLayer::from(data, &[m].to_vec())
            }

            (1, 2) => {
                if self.shape[0] != other.shape[0] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape.clone(),
                            right_shape: other.shape.clone(),
                        },
                    ));
                }
                let k = self.shape[0];
                let n = other.shape[1];
                let mut data = vec![0.0; n];

                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..k {
                        sum += self.data[i] * other.data[i * n + j];
                    }
                    data[j] = sum;
                }
                DefaultLayer::from(data, &[n].to_vec())
            }

            // Case 3: Higher dimensional tensor multiplication
            (a, b) => {
                // Get batch dimensions
                let batch_size = if a > 2 {
                    self.shape[..a - 2].iter().product()
                } else {
                    1
                };
                let m = self.shape[a - 2];
                let k = self.shape[a - 1];
                let n = other.shape[b - 1];

                if k != other.shape[b - 2] {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape.clone(),
                            right_shape: other.shape.clone(),
                        },
                    ));
                }

                // Handle broadcasting for batch dimensions
                let other_batch_size = if b > 2 {
                    other.shape[..b - 2].iter().product()
                } else {
                    1
                };

                let output_batch_size = if batch_size == 1 {
                    other_batch_size
                } else if other_batch_size == 1 {
                    batch_size
                } else if batch_size == other_batch_size {
                    batch_size
                } else {
                    return Err(MlError::TensorError(
                        TensorError::MatrixMultiplicationError {
                            left_shape: self.shape.clone(),
                            right_shape: other.shape.clone(),
                        },
                    ));
                };

                let mut data = vec![0.0; output_batch_size * m * n];

                for batch in 0..output_batch_size {
                    let batch1 = if batch_size == 1 { 0 } else { batch };
                    let batch2 = if other_batch_size == 1 { 0 } else { batch };

                    let start1 = batch1 * m * k;
                    let start2 = batch2 * k * n;
                    let result_start = batch * m * n;

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum +=
                                    self.data[start1 + i * k + l] * other.data[start2 + l * n + j];
                            }
                            data[result_start + i * n + j] = sum;
                        }
                    }
                }

                // Construct output shape
                let mut shape = Vec::new();
                if a > 2 || b > 2 {
                    if batch_size > 1 {
                        shape.extend_from_slice(&self.shape[..a - 2]);
                    } else {
                        shape.extend_from_slice(&other.shape[..b - 2]);
                    }
                }
                shape.push(m);
                shape.push(n);

                DefaultLayer::from(data, &shape)
            }
        }
    }

    fn eq_scalar(&self, scalar: f32) -> MlResult<Self> {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x == scalar) as i32 as f32)
            .collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn topk(&self, k: usize, sorted: bool) -> MlResult<(Self, Self)> {
        if k == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = self.shape.len() - 1;
        let last_dim_size = self.shape[last_dim];

        if k > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    k, last_dim_size
                ),
            }));
        }


        let slice_size = last_dim_size;
        let num_slices: usize = self.shape[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * k);
        let mut indices = Vec::with_capacity(num_slices * k);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &self.data[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..k];
            let mut selected = top_k.to_vec();
            if !sorted {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = self.shape.clone();
        new_shape[last_dim] = k;

        Ok((
            DefaultLayer::from(values, &new_shape)?,
            DefaultLayer::from(indices, &new_shape)?,
        ))
    }

    fn abs(&self) -> MlResult<Self> {
        let data: Vec<f32> = self.data.iter().map(|&x| x.abs()).collect();
        DefaultLayer::from(data, &self.shape)
    }

    fn matmax(&self, dim: Option<i32>, keepdim: bool) -> MlResult<(Tensor, Option<Tensor>)> {
        match dim {
            None => {
                // Find global maximum
                let max_val = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                Ok((DefaultLayer::from(vec![max_val], &vec![1])?, None))

            }
            Some(d) => {
                let dim = if d < 0 {
                    (self.shape.len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= self.shape.len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: self.shape.clone(),
                    }));
                }

                let mut new_shape = self.shape.clone();
                if !keepdim {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = self.shape[dim + 1..].iter().product();
                let outer_stride: usize = self.shape[dim..].iter().product();
                let outer_dims: usize = self.shape[..dim].iter().product();
                let dim_size = self.shape[dim];

                let mut max_values = Vec::with_capacity(self.data.len() / dim_size);
                let mut max_indices = Vec::with_capacity(self.data.len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = self.data[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = k;
                            }
                        }

                        max_values.push(max_val);
                        max_indices.push(max_idx as f32);
                    }
                }

                Ok((
                    DefaultLayer::from(max_values, &new_shape)?,
                    Some(DefaultLayer::from(max_indices, &new_shape)?),
                ))
            }
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        todo!()
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        todo!()
    }
}


impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        todo!()
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

