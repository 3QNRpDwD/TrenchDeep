impl TimeEmbeddingMLP {
    pub fn comparison_named_parameters(&self) -> Vec<(String, &dyn Parameter)> {
        let mut result = Vec::new();
        for layer in self.mlp.comparison_layers() {
            let prefix = match layer.label() {
                "time_mlp_linear1" => "time1",
                "time_mlp_linear2" => "time2",
                _ => {
                    assert!(layer.params().is_empty());
                    continue;
                }
            };
            result.extend(crate::nn::comparison_layer_parameters(
                layer.as_ref(),
                prefix,
            ));
        }
        result
    }
}
