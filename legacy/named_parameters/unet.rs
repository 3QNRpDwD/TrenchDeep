impl ResNetBlock {
    fn comparison_named_parameters(&self, prefix: &str) -> Vec<(String, &dyn Parameter)> {
        let mut result = Vec::new();
        for branch in [&self.branch1, &self.branch2] {
            for layer in branch.comparison_layers() {
                result.extend(crate::nn::comparison_layer_parameters(
                    layer.as_ref(),
                    &format!("{prefix}.{}", layer.label()),
                ));
            }
        }
        if let Some(layer) = &self.t_emb_proj {
            result.extend(crate::nn::comparison_layer_parameters(
                layer,
                &format!("{prefix}.time"),
            ));
        }
        if let Some(layer) = &self.skip_conv {
            result.extend(crate::nn::comparison_layer_parameters(
                layer,
                &format!("{prefix}.skip"),
            ));
        }
        result
    }
}
impl SelfAttentionBlock {
    fn comparison_named_parameters(&self, prefix: &str) -> Vec<(String, &dyn Parameter)> {
        let mut result =
            crate::nn::comparison_layer_parameters(&self.gn, &format!("{prefix}.norm"));
        for (name, layer) in [
            ("q", &self.query),
            ("k", &self.key),
            ("v", &self.value),
            ("out", &self.out_proj),
        ] {
            result.extend(crate::nn::comparison_layer_parameters(
                layer,
                &format!("{prefix}.{name}"),
            ));
        }
        result
    }
}
impl Unet {
    pub fn comparison_named_parameters(&self) -> Vec<(String, &dyn Parameter)> {
        let mut result = crate::nn::comparison_layer_parameters(self.init_conv.as_ref(), "initial");
        result.extend(self.time_mlp.comparison_named_parameters());
        for (i, stage) in self.downs.iter().enumerate() {
            result.extend(
                stage
                    .resnet1
                    .comparison_named_parameters(&format!("down.{i}.first")),
            );
            result.extend(
                stage
                    .resnet2
                    .comparison_named_parameters(&format!("down.{i}.second")),
            );
            if let Some(layer) = &stage.attn {
                result.extend(layer.comparison_named_parameters(&format!("down.{i}.attention")));
            }
            result.extend(crate::nn::comparison_layer_parameters(
                &stage.downsample,
                &format!("down.{i}.resize"),
            ));
        }
        result.extend(self.mid.resnet1.comparison_named_parameters("middle1"));
        result.extend(
            self.mid
                .attn
                .comparison_named_parameters("middle_attention"),
        );
        result.extend(self.mid.resnet2.comparison_named_parameters("middle2"));
        for (i, stage) in self.ups.iter().enumerate() {
            result.extend(
                stage
                    .resnet1
                    .comparison_named_parameters(&format!("up.{i}.first")),
            );
            result.extend(
                stage
                    .resnet2
                    .comparison_named_parameters(&format!("up.{i}.second")),
            );
            if let Some(layer) = &stage.attn {
                result.extend(layer.comparison_named_parameters(&format!("up.{i}.attention")));
            }
            result.extend(crate::nn::comparison_layer_parameters(
                &stage.upsample_conv,
                &format!("up.{i}.resize"),
            ));
        }
        result.extend(
            self.final_res_block
                .comparison_named_parameters("final_residual"),
        );
        result.extend(crate::nn::comparison_layer_parameters(
            &self.final_conv,
            "final_conv",
        ));
        result
    }
}
