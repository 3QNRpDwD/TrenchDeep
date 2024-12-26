pub trait TensorBroadcast<T> {
    fn can_broadcast(&self, other: &Self) -> bool;
    fn broadcast_op<F>(&self, other: &Self, op: F) -> Option<Self>
    where
        T: Clone,
        F: Fn(&T, &T) -> T,
        Self: Sized;
    fn into_broadcast_op<F>(self, other: Self, op: F) -> Option<Self>
    where
        F: Fn(T, T) -> T,
        Self: Sized;
}