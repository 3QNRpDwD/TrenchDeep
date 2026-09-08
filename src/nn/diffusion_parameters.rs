use super::*;

fn conv<'a>(layer: &'a Conv2D, prefix: &str) -> Vec<(String, &'a Parameter)> {
    vec![
        (format!("{prefix}.weight"), layer.weight()),
        (format!("{prefix}.bias"), layer.bias()),
    ]
}
fn linear<'a>(layer: &'a Linear, prefix: &str) -> Vec<(String, &'a Parameter)> {
    vec![
        (format!("{prefix}.weight"), layer.weight()),
        (format!("{prefix}.bias"), layer.bias()),
    ]
}
fn norm<'a>(layer: &'a GroupNorm, prefix: &str) -> Vec<(String, &'a Parameter)> {
    vec![
        (format!("{prefix}.gamma"), layer.gamma()),
        (format!("{prefix}.beta"), layer.beta()),
    ]
}
impl Residual {
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut result = norm(&self.norm1, &format!("{prefix}.norm1"));
        result.extend(conv(&self.conv1, &format!("{prefix}.conv1")));
        result.extend(norm(&self.norm2, &format!("{prefix}.norm2")));
        result.extend(conv(&self.conv2, &format!("{prefix}.conv2")));
        result.extend(linear(&self.time, &format!("{prefix}.time")));
        if let Some(layer) = &self.skip {
            result.extend(conv(layer, &format!("{prefix}.skip")));
        }
        result
    }
}
impl Attention {
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut result = norm(&self.norm, &format!("{prefix}.norm"));
        for (name, layer) in [
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("out", &self.out),
        ] {
            result.extend(linear(layer, &format!("{prefix}.{name}")));
        }
        result
    }
}
impl Stage {
    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter)> {
        let mut result = self.first.named_parameters(&format!("{prefix}.first"));
        result.extend(self.second.named_parameters(&format!("{prefix}.second")));
        if let Some(layer) = &self.attention {
            result.extend(layer.named_parameters(&format!("{prefix}.attention")));
        }
        result.extend(conv(&self.resize, &format!("{prefix}.resize")));
        result
    }
}
impl Unet {
    /// Structural parameter paths, independent of labels and enumeration order.
    /// Repeated parameter identities are retained to expose weight sharing.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter)> {
        let mut result = conv(&self.initial, "initial");
        result.extend(linear(&self.time1, "time1"));
        result.extend(linear(&self.time2, "time2"));
        for (i, stage) in self.down.iter().enumerate() {
            result.extend(stage.named_parameters(&format!("down.{i}")));
        }
        result.extend(self.middle1.named_parameters("middle1"));
        result.extend(self.middle_attention.named_parameters("middle_attention"));
        result.extend(self.middle2.named_parameters("middle2"));
        for (i, stage) in self.up.iter().enumerate() {
            result.extend(stage.named_parameters(&format!("up.{i}")));
        }
        result.extend(self.final_residual.named_parameters("final_residual"));
        result.extend(conv(&self.final_conv, "final_conv"));
        result
    }
}
