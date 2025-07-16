use super::*;

impl Function for Div {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![PooledTensor::from_vec(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape()).unwrap()])
        }
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: HandleId) -> MlResult<Vec<Tensor>> {
        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::with_id(self.backend().div(targets[0].data(), targets[1].data()), targets[0].shape(), node_id).unwrap()])
        }
    }
    
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let x1 = targets[1];

        Ok(vec![
            self.forward(&[grad, x1])?.remove(0), // grad / x2
            grad * &self.forward(&[&-targets[0], &(x1 * x1)])?.remove(0) // grad * (-x0 / x1^2)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = PooledTensor;

    fn div(self, other: Tensor) -> Self::Output {
        let op = Div::new();
        op.forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = PooledTensor;

    fn div(self, other: &Tensor) -> Self::Output {
        let op = Div::new();
        op.forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor> for &Tensor {
    type Output = PooledTensor;

    fn div(self, other: &Tensor) -> Self::Output {
        let op = Div::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<Tensor> for &Tensor {
    type Output = PooledTensor;

    fn div(self, other: Tensor) -> Self::Output {
        let op = Div::new();
        op.forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&dyn TensorBase> for &dyn TensorBase {
    type Output = PooledTensor;

    fn div(self, other: &dyn TensorBase) -> Self::Output {
        let op = Div::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, other: Tensor) {
        let op = Div::new();
        op.assign_forward(&[self, &other], self.0).unwrap();
    }
}

impl std::ops::DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, other: &Tensor) {
        let op = Div::new();
        op.assign_forward(&[self, other], self.0).unwrap();
    }
}
