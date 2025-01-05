use std::ops::{Add, Div, Mul, Sub};
use crate::tensor::{Tensor, BroadcastLayer, OpsLayer};

impl<T: PartialEq> OpsLayer<T> for Tensor {
    // Tenser Calculate
    fn add(self, other: Self) -> Option<Self> where T: Add<Output = T>
    { self.broadcast_op(other, |left, right| left + right) } // todo: 브로드케스팅 연산 구조 변경 필요
    fn sub(self, other: Self) -> Option<Self> where T: Sub<Output = T>
    { self.broadcast_op(other, |left, right| left - right) } // todo: 브로드케스팅 연산 구조 변경 필요
    fn div(self, other: Self) -> Option<Self> where T: Div<Output = T>
    { self.broadcast_op(other, |left, right| left / right) } // todo: 브로드케스팅 연산 구조 변경 필요
    fn mul(self, other: Self) -> Option<Self> where T: Mul<Output = T>
    { self.broadcast_op(other, |left, right| left * right) } // todo: 브로드케스팅 연산 구조 변경 필요

    fn add_scalar(self, other: Tensor) -> Option<Self>
    where
        T: Add<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn mul_scalar(self, other: Tensor) -> Option<Self>
    where
        T: Mul<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn sub_scalar(self, other: Tensor) -> Option<Self>
    where
        T: Sub<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn div_scalar(self, other: Tensor) -> Option<Self>
    where
        T: Div<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn scalar_sub(self, other: Tensor) -> Option<Self>
    where
        T: Sub<Output=T>,
        Self: Sized
    {
        todo!()
    }

    fn scalar_div(self, other: Tensor) -> Option<Self>
    where
        T: Div<Output=T>,
        Self: Sized
    {
        todo!()
    } // Todo: error[E0507]: cannot move out of a shared reference 오류 해결하기
}

// impl<T: Add<Output = Vec<f32>> + Sized + PartialEq> Add for Tensor {
//     type Output = Vec<T>;
//
//     fn add(self, other: Tensor) -> Vec<f32> {
//         self.data.iter()
//             .zip(other.data.iter())
//             .map(|(&a, &b)| a + b)
//             .collect()
//     }
// }
//
// impl<T: Sub<Output = T> + Sized + PartialEq> Sub for Tensor {
//     type Output = Vec<T>;
//
//     fn sub(self, other: Tensor) -> Vec<T> {
//         self.data.iter()
//             .zip(other.data.iter())
//             .map(|(&a, &b)| a - b)
//             .collect()
//     }
// }
//
// impl<T: Mul<Output = T> + Sized + PartialEq> Mul for Tensor {
//     type Output = Vec<T>;
//
//     fn mul(self, other: Tensor) -> Vec<T> {
//         self.data.iter()
//             .zip(other.data.iter())
//             .map(|(&a, &b)| a * b)
//             .collect()
//     }
// }
//
// impl<T: Div<Output = T> + Sized + PartialEq> Div for Tensor {
//     type Output = Vec<T>;
//
//     fn div(self, other: Tensor) -> Vec<T> {
//         self.data.iter()
//             .zip(other.data.iter())
//             .map(|(&a, &b)| a / b)
//             .collect()
//     }
// }

