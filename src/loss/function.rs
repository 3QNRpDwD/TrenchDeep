use super::*;

impl Function for MeanSquaredError {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(MeanSquaredError)
    }
    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let diff = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| p - t);
        let squared_error = diff.map(|d| d * d).sum::<f32>();

        Ok(vec![scalar!(squared_error / n)])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];

        let grad_pred_data: Vec<f32> = pred.data()
            .iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| grad_val * 2.0 * (p - t) / n)
            .collect();

        let grad_target_data: Vec<f32> = grad_pred_data.iter().map(|&g| -g).collect();

        let grad_pred = Tensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = Tensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}


impl Function for MeanAbsoluteError {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(MeanAbsoluteError)
    }
    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let abs_error = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| (p - t).abs()).sum::<f32>();

        Ok(vec![scalar!(abs_error / n)])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];

        let grad_pred_data: Vec<f32> = pred.data()
            .iter()
            .zip(target.data().iter())
            .map(|(&p, &t)| grad_val * (p - t).signum() / n)
            .collect();

        let grad_target_data: Vec<f32> = grad_pred_data.iter().map(|&g| -g).collect();

        let grad_pred = Tensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = Tensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}


impl Function for HuberLoss {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "HuberLoss";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Arc::new(HuberLoss { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), delta: 1.0 })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }

    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let delta = self.delta;

        let huber_error = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let diff = (p - t).abs();
            if diff <= delta {
                0.5 * diff.powi(2)
            } else {
                delta * (diff - 0.5 * delta)
            }
        }).sum::<f32>();

        Ok(vec![scalar!(huber_error / n)])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];
        let delta = self.delta;

        let grad_pred_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let diff = p - t;
            let grad_elem = if diff.abs() <= delta {
                diff
            } else {
                delta * diff.signum()
            };
            grad_val * grad_elem / n
        }).collect();

        let grad_target_data: Vec<f32> = grad_pred_data.iter().map(|&g| -g).collect();

        let grad_pred = Tensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = Tensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

impl Function for BinaryCrossEntropyLoss {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(BinaryCrossEntropyLoss)
    }

    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        let n = pred.data().len() as f32;
        let bce_loss = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            - (t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln())
        }).sum::<f32>();

        Ok(vec![scalar!(bce_loss / n)])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];
        let n = pred.data().len() as f32;
        let grad_val = grad.data()[0];

        let grad_pred_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            grad_val * ((p_clipped - t) / (p_clipped * (1.0 - p_clipped))) / n
        }).collect();

        let grad_target_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON).min(1.0 - EPSILON);
            grad_val * -((1.0 - p_clipped).ln() - p_clipped.ln()) / n
        }).collect();

        let grad_pred = Tensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = Tensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

// ===== CategoricalCrossEntropy =====

impl Function for CrossEntropyLoss {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(CrossEntropyLoss)
    }

    fn forward(&self, targets: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1]; // Assumes target is one-hot encoded

        if pred.shape() != target.shape() {
            return Err(LossError::InvalidShape {
                expected: pred.shape().to_vec(),
                got: target.shape().to_vec(),
            }.into());
        }

        // Assuming shape is [batch_size, num_classes] or just [num_classes]
        let batch_size = if pred.shape().len() > 1 { pred.shape()[0] } else { 1 } as f32;

        let cce_loss = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON);
            - t * p_clipped.ln()
        }).sum::<f32>();

        Ok(vec![scalar!(cce_loss / batch_size)])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        let pred = targets[0];
        let target = targets[1];
        let grad_val = grad.data()[0];
        let batch_size = if pred.shape().len() > 1 { pred.shape()[0] } else { 1 } as f32;

        let grad_pred_data: Vec<f32> = pred.data().iter().zip(target.data().iter()).map(|(&p, &t)| {
            let p_clipped = p.max(EPSILON);
            grad_val * (-t / p_clipped) / batch_size
        }).collect();

        // Gradient for target is rarely used, but for completeness:
        let grad_target_data: Vec<f32> = pred.data().iter().map(|&p| {
            let p_clipped = p.max(EPSILON);
            grad_val * -p_clipped.ln() / batch_size
        }).collect();

        let grad_pred = Tensor::from_vec(grad_pred_data, pred.shape())?;
        let grad_target = Tensor::from_vec(grad_target_data, target.shape())?;

        Ok(vec![grad_pred, grad_target])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}