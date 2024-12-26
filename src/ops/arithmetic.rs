use crate::core::Tensor;
use crate::core::TensorLayer;
use crate::broadcast::TensorBroadcast;
use crate::ops::traits::TensorArithmetic;
use std::ops::{Add, Sub, Div, Mul};

impl<T> TensorLayer<T> for Tensor<T> {
    fn new(shape: Vec<usize>, value: T) -> Self where T: Clone {
        Tensor { data: vec![value; shape.iter().product()], shape }
    }
    fn get(&self, indices: &[usize]) -> Option<&T> {
        self.data.get(self.index(indices)?)
    }
    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        Some(
            indices
                .iter()
                .zip(&self.shape)
                .fold(0, |acc, (&i, &dim)| acc * dim + i),
        )
    }
}

impl<T> TensorBroadcast<T> for Tensor<T> {
    fn can_broadcast(&self, other: &Self) -> bool {
        if self.shape.len() != other.shape.len() {
            return false;
        }
        // 각 차원을 뒤에서부터 비교
        self.shape.iter().zip(other.shape.iter()).all(|(&a, &right)| {
            a == right || a == 1 || right == 1
        })
    }
    fn broadcast_op<F>(&self, other: &Self, op: F) -> Option<Self>
    where
        T: Clone,
        F: Fn(&T, &T) -> T,
    {
        let shape: Vec<usize> = self.shape
            .iter()
            .zip(&other.shape)
            .map(|(&left, &right)| std::cmp::max(left, right))
            .collect();
        let mut data = Vec::with_capacity(shape.iter().product());

        // TODO: 실제 브로드캐스팅 구현
        data = self.data
            .iter()
            .zip(&other.data)
            .map(|(left, right)| op(left, right))
            .collect();

        Some(Self { data, shape })
    }
    fn into_broadcast_op<F>(self, other: Self, op: F) -> Option<Self>
    where F: Fn(T, T) -> T
    {
        let shape: Vec<usize> = self.shape
            .into_iter()
            .zip(other.shape)
            .map(|(left, right)| std::cmp::max(left, right))
            .collect();

        let mut data = Vec::with_capacity(shape.iter().product());

        // TODO: 실제 브로드캐스팅 구현
        data = self.data
            .into_iter()
            .zip(other.data)
            .map(|(left, right)| op(left, right))
            .collect();

        Some(Self { data, shape })
    }
}

impl<T> TensorArithmetic<T> for Tensor<T> {
    // Tenser Calculate
    fn add(&self, other: &Self) -> Option<Self> where T: Add<Output = T> + Clone
    { self.broadcast_op(other, |left, right| left.clone() + right.clone()) }
    fn sub(&self, other: &Self) -> Option<Self> where T: Sub<Output = T> + Clone
    { self.broadcast_op(other, |left, right| left.clone() - right.clone()) }
    fn div(&self, other: &Self) -> Option<Self> where T: Div<Output = T> + Clone
    { self.broadcast_op(other, |left, right| left.clone() / right.clone()) }
    fn mul(&self, other: &Self) -> Option<Self> where T: Mul<Output = T> + Clone
    { self.broadcast_op(other, |left, right| left.clone() * right.clone()) }

    // Tenser Calculate By Into
    fn into_add(self, other: Self) -> Option<Self> where T: Add<Output = T>
    { self.into_broadcast_op(other, |left, right| left + right) }
    fn into_sub(self, other: Self) -> Option<Self> where T: Sub<Output = T>
    { self.into_broadcast_op(other, |left, right| left - right) }
    fn into_div(self, other: Self) -> Option<Self> where T: Div<Output = T>
    { self.into_broadcast_op(other, |left, right| left / right) }
    fn into_mul(self, other: Self) -> Option<Self> where T: Mul<Output = T>
    { self.into_broadcast_op(other, |left, right| left * right) }
}


