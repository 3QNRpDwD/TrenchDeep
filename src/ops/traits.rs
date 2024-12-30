use std::ops::{Add, Div, Mul, Sub};
use crate::core::Tensor;

pub trait TensorOps<T> {
    fn add(&self, other: &Tensor<T>)
        -> Option<Self> where T: Add<Output = T>, Self: Sized;
    fn sub(&self, other: &Tensor<T>)
        -> Option<Self> where T: Sub<Output = T>, Self: Sized;
    fn div(&self, other: &Tensor<T>)
        -> Option<Self> where T: Div<Output = T>, Self: Sized;
    fn mul(&self, other: &Tensor<T>)
        -> Option<Self> where T: Mul<Output = T>, Self: Sized;

    fn add_scalar(&self, other: &Tensor<T>)
        -> Option<Self> where T: Add<Output = T>, Self: Sized;

    fn mul_scalar(&self, other: &Tensor<T>)
        -> Option<Self> where T: Mul<Output = T>, Self: Sized;

    fn sub_scalar(&self, other: &Tensor<T>)
        -> Option<Self> where T: Sub<Output = T>, Self: Sized;
    fn scalar_sub(&self, other: &Tensor<T>)
        -> Option<Self> where T: Sub<Output = T>, Self: Sized;

    fn div_scalar(&self, other: &Tensor<T>)
        -> Option<Self> where T: Div<Output = T>, Self: Sized;
    fn scalar_div(&self, other: &Tensor<T>)
        -> Option<Self> where T: Div<Output = T>, Self: Sized;
}