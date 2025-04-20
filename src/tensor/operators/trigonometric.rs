use super::*;

/// `Sin` 함수는 입력 텐서의 각 요소에 사인 함수를 적용합니다.
/// 입력 텐서는 각도로, 출력 텐서는 해당 각도의 사인 값을 포함합니다.
impl Function<f32> for Sin {
    /// 새로운 `Sin` 인스턴스를 생성합니다.
    /// CPU 백엔드를 사용합니다.
    fn new() -> MlResult<Self> {
        Ok(Self { backend: Arc::new(CpuBackend::new()? )} )
    }

    /// 입력 텐서에 사인 함수를 요소별로 적용하여, 동일한 모양을 가진 새로운 텐서를 반환합니다.
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|x| x.sin()).collect(), targets[0].shape())?])
    }

    /// 사인 함수의 기울기를 계산합니다.
    /// 반환되는 기울기 텐서는 입력 텐서와 동일한 모양을 가집니다.
    /// 각 요소는 입력 텐서의 해당 요소의 코사인 값과 다음 계층의 기울기 값의 곱입니다.
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradient = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target)|  target.cos() * grad_data)
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradient, targets[0].shape())?])
    }

    /// 연산에 사용되는 백엔드 객체의 참조를 반환
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

/// `Cos` 함수는 입력 텐서의 각 요소에 코사인 함수를 적용합니다.
/// 입력 텐서는 각도로, 출력 텐서는 해당 각도의 코사인 값을 포함합니다.
impl Function<f32> for Cos {
    /// 새로운 `Cos` 인스턴스를 생성합니다.
    /// CPU 백엔드를 사용합니다.
    fn new() -> MlResult<Self> {
        Ok(Self { backend: Arc::new(CpuBackend::new()? )} )
    }

    /// 입력 텐서에 코사인 함수를 요소별로 적용하여, 동일한 모양을 가진 새로운 텐서를 반환합니다.
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|x| x.cos()).collect(), targets[0].shape())?])
    }

    /// 코사인 함수의 기울기를 계산합니다.
    /// 반환되는 기울기 텐서는 입력 텐서와 동일한 모양을 가집니다.
    /// 각 요소는 입력 텐서의 해당 요소의 음수 사인 값과 다음 계층의 기울기 값의 곱입니다.
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradient = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target)|  -target.sin() * grad_data)
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradient, targets[0].shape())?])
    }

    /// 연산에 사용되는 백엔드 객체의 참조를 반환
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for ApproxSin {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
            threshold: 0.0001
        })
    }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let x = targets[0];
        let x_data = x.data();
        let mut result = x_data.to_vec(); // Start with x (first term of series)

        // Calculate powers for the series approximation
        let mut term_sign = -1.0;
        let mut current_power = 3;
        let mut x_power = self.backend.multiply(x_data, x_data); // x²
        x_power = self.backend.multiply(&x_power, x_data); // x³
        let mut factorial = 6.0; // 3!

        // Continue adding terms until desired threshold
        while current_power <= 15 { // Limiting to reasonable number of terms

            // Calculate term: x^n/n! with alternating sign
            let term_value = self.backend.div(&x_power, &vec![factorial; x_power.len()]);

            // Apply sign and add/subtract from result
            let term = self.backend.multiply(&term_value, &vec![term_sign; term_value.len()]);
            result = self.backend.add(&result, &term);

            // Prepare for next iteration
            term_sign *= -1.0;

            // Update power: x^n -> x^(n+2)
            x_power = self.backend.multiply(&x_power, x_data);
            x_power = self.backend.multiply(&x_power, x_data);

            // Update factorial: n! -> (n+2)!
            factorial *= (current_power + 1) as f32 * (current_power + 2) as f32;
            current_power += 2;
        }

        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        // The derivative of sin(x) is cos(x)
        // We can use the ApproxCos implementation for this
        let cos = ApproxCos {
            backend: Arc::clone(&self.backend),
            threshold: self.threshold,
        };

        let cos_output = cos.forward(targets)?;

        // Multiply the cos result with the incoming gradient
        let x = targets[0];
        cos_output[0].chk_shape(grad)?;

        let grad_data = grad.data();
        let cos_data = cos_output[0].data();
        let result = self.backend.multiply(cos_data, grad_data);
        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for ApproxCos {
    fn new() -> MlResult<Self> {
        Ok(Self {
            backend: Arc::new(CpuBackend::new()?),
            threshold: 0.0001
        })
    }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let x = targets[0];
        let x_data = x.data();
        let mut result = vec![1.0; x_data.len()];

        // Calculate x²
        let x_squared = self.backend.multiply(x_data, x_data);
        let mut term_sign = -1.0;
        let mut current_power = 2;
        let mut x_power = x_squared.clone(); // Start with x²
        let mut factorial = 2.0; // 2!

        // Continue adding terms until desired threshold
        while current_power <= 14 { // Limiting to reasonable number of terms
            // Calculate term: x^n/n! with alternating sign
            let term_value = self.backend.div(&x_power, &vec![factorial; x_power.len()]);

            // Apply sign and add/subtract from result
            let term = self.backend.multiply(&term_value, &vec![term_sign; term_value.len()]);
            result = self.backend.add(&result, &term);

            // Prepare for next iteration
            term_sign *= -1.0;

            // Update power: x^n -> x^(n+2)
            x_power = self.backend.multiply(&x_power, x_squared.as_slice());

            // Update factorial: n! -> (n+2)!
            factorial *= (current_power + 1) as f32 * (current_power + 2) as f32;
            current_power += 2;
        }

        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        // The derivative of cos(x) is -sin(x)
        // We can use the ApproxSin implementation for this
        let sin = ApproxSin {
            backend: Arc::clone(&self.backend),
            threshold: self.threshold,
        };

        let sin_output = sin.forward(targets)?;
        // Multiply the -sin result with the incoming gradient
        let x = targets[0];
        sin_output[0].chk_shape(grad)?;

        let grad_data = grad.data();
        let sin_data = sin_output[0].data();

        // Apply negative sign to sin result
        let neg_sin = self.backend.multiply(sin_data, &vec![-1.0; sin_data.len()]);
        // Then multiply by gradient
        let result = self.backend.multiply(&neg_sin, grad_data);

        Ok(vec![Tensor::from_vec(result, x.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}