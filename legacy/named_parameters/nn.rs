// Additive read-only access in the comparison build, never the preserved source.
impl Sequential {
    pub fn comparison_layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }
}
pub fn comparison_layer_parameters<'a>(
    layer: &'a dyn Layer,
    prefix: &str,
) -> Vec<(String, &'a dyn Parameter)> {
    // save_state supplies semantic field names from the original layer macro.
    let state = layer.save_state();
    let params = layer.params();
    assert_eq!(
        state.params.len(),
        params.len(),
        "unnamed comparison layer {prefix}"
    );
    state
        .params
        .iter()
        .zip(params)
        .map(|(field, p)| (format!("{prefix}.{}", field.name), p))
        .collect()
}
