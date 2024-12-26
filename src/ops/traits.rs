use std::ops::{Add, Div, Mul, Sub};

pub trait TensorArithmetic<T> {
    // Tenser Calculate
    fn add(&self, other: &Self)
           -> Option<Self> where T: Add<Output = T> + Clone, Self: Sized;
    fn sub(&self, other: &Self)
           -> Option<Self> where T: Sub<Output = T> + Clone, Self: Sized;
    fn div(&self, other: &Self)
           -> Option<Self> where T: Div<Output = T> + Clone, Self: Sized;
    fn mul(&self, other: &Self)
           -> Option<Self> where T: Mul<Output = T> + Clone, Self: Sized;

    // Tenser Calculate By Into
    fn into_add(self, other: Self)
                -> Option<Self> where T: Add<Output = T>, Self: Sized;
    fn into_sub(self, other: Self)
                -> Option<Self> where T: Sub<Output = T>, Self: Sized;
    fn into_div(self, other: Self)
                -> Option<Self> where T: Div<Output = T>, Self: Sized;
    fn into_mul(self, other: Self)
                -> Option<Self> where T: Mul<Output = T>, Self: Sized;
}