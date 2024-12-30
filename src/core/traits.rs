pub trait TensorLayer<T> where T: Sized{
    fn new(data: T) -> Self;
    fn get(&self, indices: &[usize]) -> Option<&T>;
    fn index(&self, indices: &[usize]) -> Option<usize>;
}