use std::ops::{Add, Div, Mul, Sub};
use crate::tensor::{Tensor, DefaultLayer, BroadcastLayer, OpsLayer};

impl<T: Sized + PartialEq + IntoIterator> DefaultLayer<T> for Tensor<T> {
    fn new(data: T) -> Self{
        Ok(Self {
            data: data.into_iter().flatten().collect(),
            shape: vec![data.len(), data[0].len()]
        })
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

impl<T: PartialEq> OpsLayer<T> for Tensor<T> {
    // Tenser Calculate
    fn add(self, other: Self) -> Option<Self> where T: Add<Output = T>
    { self.broadcast_op(other, |left, right| left + right) }
    fn sub(self, other: Self) -> Option<Self> where T: Sub<Output = T>
    { self.broadcast_op(other, |left, right| left - right) }
    fn div(self, other: Self) -> Option<Self> where T: Div<Output = T>
    { self.broadcast_op(other, |left, right| left / right) }
    fn mul(self, other: Self) -> Option<Self> where T: Mul<Output = T>
    { self.broadcast_op(other, |left, right| left * right) }

    fn add_scalar(self, other: Tensor<T>) -> Option<Self>
    where
        T: Add<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn mul_scalar(self, other: Tensor<T>) -> Option<Self>
    where
        T: Mul<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn sub_scalar(self, other: Tensor<T>) -> Option<Self>
    where
        T: Sub<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn div_scalar(self, other: Tensor<T>) -> Option<Self>
    where
        T: Div<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn scalar_sub(self, other: Tensor<T>) -> Option<Self>
    where
        T: Sub<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn scalar_div(self, other: Tensor<T>) -> Option<Self>
    where
        T: Div<Output=T>,
        Self: Sized
    {
        todo!()
    } // Todo: error[E0507]: cannot move out of a shared reference 오류 해결하기
}

impl<T: Add<Output = T> + Sized + PartialEq> Add for Tensor<T> {
    type Output = Vec<T>;

    fn add(self, other: Tensor<T>) -> Vec<T> {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect()
    }
}

impl<T: Sub<Output = T> + Sized + PartialEq> Sub for Tensor<T> {
    type Output = Vec<T>;

    fn sub(self, other: Tensor<T>) -> Vec<T> {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect()
    }
}

impl<T: Mul<Output = T> + Sized + PartialEq> Mul for Tensor<T> {
    type Output = Vec<T>;

    fn mul(self, other: Tensor<T>) -> Vec<T> {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect()
    }
}

impl<T: Div<Output = T> + Sized + PartialEq> Div for Tensor<T> {
    type Output = Vec<T>;

    fn div(self, other: Tensor<T>) -> Vec<T> {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a / b)
            .collect()
    }
}

