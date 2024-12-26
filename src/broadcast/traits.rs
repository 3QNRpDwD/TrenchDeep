pub trait TensorBroadcast<T> {
    fn can_broadcast(&self, other: &Self) -> bool;
    fn broadcast_shape(&self, other: &Self) -> Vec<usize>;
    fn broadcast_op<F>(&self, other: &Self, op: F) -> Option<Self>
    where
        F: Fn(&T, &T) -> T,
        Self: Sized;
    fn into_broadcast_op<F>(self, other: Self, op: F) -> Option<Self>
    where
        F: Fn(&T, &T) -> T,
        Self: Sized;
    fn calculate_broadcast_indices(&self, other: &Self, idx: usize, shape: &[usize]) -> Option<(usize, usize)>;
}