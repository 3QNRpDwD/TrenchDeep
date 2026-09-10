use super::*;

#[test]
#[cfg(all(feature = "enableBackward"))]
fn context_linear_tracks_parameters_and_predicts_without_a_graph() -> MlResult<()> {
    let context = ExecutionContext::new();
    let layer = Linear::new(&context, 2, 3, "linear")?;
    let input = context.input(vec![1.0, -2.0, 0.5, 3.0], &[2, 2])?;
    let output = layer.apply(&input)?;
    assert_eq!(output.tensor().shape()?, vec![2, 3]);
    context.sum_variable(&output)?.backward()?;
    assert!(layer.weight().grad()?.is_some());
    assert!(layer.bias().grad()?.is_some());
    context.clear_graph()?;
    let prediction = layer.predict(input.tensor())?;
    assert_eq!(prediction.shape()?, vec![2, 3]);
    assert_eq!(context.graph_stats()?.graph_nodes, 0);
    Ok(())
}

#[test]
fn context_linear_rejects_foreign_inputs() -> MlResult<()> {
    let context = ExecutionContext::new();
    let foreign = ExecutionContext::new();
    let layer = Linear::new(&context, 2, 1, "linear")?;
    let input = foreign.input(vec![1.0, 2.0], &[1, 2])?;
    assert!(matches!(
        layer.apply(&input),
        Err(crate::MlError::ContextError(crate::ContextError::Mismatch))
    ));
    Ok(())
}

#[test]
#[cfg(all(feature = "enableBackward"))]
fn parameter_updates_are_fallible_shared_and_graph_free() -> MlResult<()> {
    let context = ExecutionContext::new();
    let layer = Linear::new(&context, 2, 1, "linear")?;
    let detached = layer.weight().variable().detach()?;
    let before = layer.weight().tensor().to_vec()?;
    context.sub_assign(
        layer.weight().variable(),
        &TensorBuffer::from_vec(vec![0.25, -0.5], &[2, 1])?,
    )?;
    let after = layer.weight().tensor().to_vec()?;
    assert_eq!(after, vec![before[0] - 0.25, before[1] + 0.5]);
    assert_eq!(detached.tensor().to_vec()?, after);
    assert_eq!(context.graph_stats()?.graph_nodes, 0);
    assert!(
        context
            .add_assign(
                layer.weight().variable(),
                &TensorBuffer::from_vec(vec![1.0], &[1])?,
            )
            .is_err()
    );
    Ok(())
}

#[test]
#[cfg(all(feature = "enableBackward"))]
fn convolution_group_norm_and_pooling_form_a_context_graph() -> MlResult<()> {
    let context = ExecutionContext::new();
    let convolution = Conv2D::new(&context, 1, 2, (3, 3), (1, 1), (1, 1), "conv")?;
    let normalization = GroupNorm::new(&context, 1, 2, 1e-5, "norm")?;
    let activation = Activation::new(&context, ActivationKind::ReLU, "relu");
    let pooling = Pooling::average(&context, (2, 2), (2, 2), "pool");
    let mut model = Sequential::new(&context, "cnn");
    model.push(Box::new(convolution))?;
    model.push(Box::new(normalization))?;
    model.push(Box::new(activation))?;
    model.push(Box::new(pooling))?;
    assert_eq!(model.parameters().len(), 4);
    let input = context.input(vec![1.0; 16], &[1, 1, 4, 4])?;
    let output = model.apply(&input)?;
    assert_eq!(output.tensor().shape()?, vec![1, 2, 2, 2]);
    context.sum_variable(&output)?.backward()?;
    assert!(
        model
            .parameters()
            .iter()
            .all(|parameter| parameter.grad().is_ok())
    );
    Ok(())
}

#[test]
fn sequential_rejects_foreign_layers_and_inputs() -> MlResult<()> {
    let context = ExecutionContext::new();
    let foreign = ExecutionContext::new();
    let mut model = Sequential::new(&context, "model");
    assert!(
        model
            .push(Box::new(Activation::new(
                &foreign,
                ActivationKind::Tanh,
                "foreign",
            )))
            .is_err()
    );
    model.push(Box::new(Linear::new(&context, 2, 2, "linear")?))?;
    let foreign_input = foreign.input(vec![1.0, 2.0], &[1, 2])?;
    assert!(model.apply(&foreign_input).is_err());
    Ok(())
}

#[test]
#[cfg(all(feature = "enableBackward"))]
fn reshape_and_upsample_layers_preserve_context_autograd() -> MlResult<()> {
    let context = ExecutionContext::new();
    let reshape = Reshape::new(&context, &[0, -1], "flatten")?;
    let upsample = Upsample2D::nearest(&context, (2, 3), "upsample")?;

    let image = context.variable(
        vec![1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        crate::RequiresGrad::Yes,
    )?;
    let enlarged = upsample.apply(&image)?;
    assert_eq!(enlarged.tensor().shape()?, vec![1, 1, 4, 6]);
    let flattened = reshape.apply(&enlarged)?;
    assert_eq!(flattened.tensor().shape()?, vec![1, 24]);
    context.sum_variable(&flattened)?.backward()?;
    assert_eq!(image.grad()?.expect("image gradient").data, vec![6.0; 4]);

    let prediction = upsample.predict(image.tensor())?;
    assert_eq!(prediction.shape()?, vec![1, 1, 4, 6]);
    Ok(())
}

#[test]
fn sequential_checkpoint_round_trip_uses_existing_format() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = Sequential::new(&context, "model");
    model.push(Box::new(Linear::new(&context, 2, 2, "linear")?))?;
    let original = model.parameters()[0].tensor().to_vec()?;
    let state = model.save_state()?;

    context.add_assign(
        model.parameters()[0].variable(),
        &TensorBuffer::from_vec(vec![1.0; 4], &[2, 2])?,
    )?;
    assert_ne!(model.parameters()[0].tensor().to_vec()?, original);
    model.load_state(&state)?;
    assert_eq!(model.parameters()[0].tensor().to_vec()?, original);
    Ok(())
}
