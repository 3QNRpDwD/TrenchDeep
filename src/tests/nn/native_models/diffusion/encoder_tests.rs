use super::*;

#[test]
fn sinusoidal_pe_predict_shape() -> MlResult<()> {
    let mut pe = SinusoidalPE::new(4, "pe")?;
    let t = Tensor::from_vec(vec![0.0, 1.0], &[2, 1])?;
    let out = pe.predict(&t)?;
    assert_eq!(out.shape(), &[2, 4]);
    // 앞 half = sin(args), 뒤 half = cos(args).  t=0 → sin=0, cos=1.
    assert!((out.data()[0] - 0.0).abs() < 1e-6);
    assert!((out.data()[1] - 0.0).abs() < 1e-6);
    assert!((out.data()[2] - 1.0).abs() < 1e-6);
    assert!((out.data()[3] - 1.0).abs() < 1e-6);
    Ok(())
}

#[cfg(feature = "enableBackward")]
#[test]
fn sinusoidal_pe_apply_shape() -> MlResult<()> {
    let mut pe = SinusoidalPE::new(8, "pe")?;
    let t = Variable::new(Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3, 1])?);
    let out = pe.apply(&t)?;
    assert_eq!(out.tensor().shape(), &[3, 8]);
    Ok(())
}
