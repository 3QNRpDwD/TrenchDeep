use super::*;

pub struct MnistDataset {
    pub x_train: Vec<Variable>,
    pub t_train: Vec<Variable>,
    pub x_test: Vec<Variable>,
    pub t_test: Vec<Variable>,
}

impl MnistDataset {
    pub fn new() -> Self {
        Self {
            x_train: Vec::new(),
            t_train: Vec::new(),
            x_test: Vec::new(),
            t_test: Vec::new(),
        }
    }
    
    pub fn x_train(&self) -> Vec<&Variable> {
        self.x_train.iter().collect::<Vec<_>>()
    }
    
    pub fn t_train(&self) -> Vec<&Variable>{
        self.t_train.iter().collect::<Vec<_>>()
    }
    
    pub fn x_test(&self) -> Vec<&Variable> {
        self.x_test.iter().collect::<Vec<_>>()
    }
    
    pub fn t_test(&self) -> Vec<&Variable> {
        self.t_test.iter().collect::<Vec<_>>()
    }
    pub fn load_and_prepare_data(
        n_train: u32,
        n_val: u32,
        n_features: usize,
        n_classes: usize,
    ) -> MlResult<MnistDataset> {
        info!("Loading MNIST dataset... (Train: {}, Test: {})", n_train, n_val);
        let mnist_data = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(n_train)
            .test_set_length(n_val)
            .finalize();

        info!("Converting data to model input format...");
        // 셔플을 위해 mut로 변경
        let (x_train, t_train) = Self::convert_to_variable_dataset(&mnist_data.trn_img, &mnist_data.trn_lbl, n_train as usize, n_features, n_classes)?;
        let (x_test, t_test) = Self::convert_to_variable_dataset(&mnist_data.tst_img, &mnist_data.tst_lbl, n_val as usize, n_features, n_classes)?;

        info!("Data preparation complete.");

        Ok(MnistDataset { x_train, t_train, x_test, t_test })
    }

    fn convert_to_variable_dataset(
        images: &[u8],
        labels: &[u8],
        num_items: usize,
        num_features: usize,
        num_classes: usize,
    ) -> MlResult<(Vec<Variable>, Vec<Variable>)> {
        let mut x_set = Vec::with_capacity(num_items);
        let mut t_set = Vec::with_capacity(num_items);

        let normalized_images: Vec<f32> = images.iter().map(|&pixel| pixel as f32 / 255.0).collect();
        let f32_labels: Vec<f32> = labels.iter().map(|&label| label as f32).collect();

        for i in 0..num_items {
            let image_slice = &normalized_images[i * num_features..(i + 1) * num_features];
            let x = var_input!(Tensor::from_vec(image_slice.to_vec(), &[num_features, 1])?);
            x_set.push(x);

            let label_slice = &f32_labels[i * num_classes..(i + 1) * num_classes];
            let t = var_with_label!(Tensor::from_vec(label_slice.to_vec(), &[num_classes, 1])?, "target");
            t_set.push(t);
        }

        Ok((x_set, t_set))
    }
}