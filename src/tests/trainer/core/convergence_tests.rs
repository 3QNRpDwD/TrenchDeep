use super::*;

#[test]
fn from_tolerance_maps_zero_and_negative_to_off() {
    assert_eq!(Convergence::from_tolerance(0.0), Convergence::Off);
    assert_eq!(Convergence::from_tolerance(-1e-6), Convergence::Off);
    assert_eq!(Convergence::from_tolerance(f32::NAN), Convergence::Off);
}

#[test]
fn from_tolerance_maps_positive_to_absdelta() {
    assert_eq!(
        Convergence::from_tolerance(1e-6),
        Convergence::AbsDelta(1e-6)
    );
}

#[test]
fn tolerance_roundtrip() {
    assert_eq!(Convergence::Off.tolerance(), 0.0);
    assert_eq!(Convergence::AbsDelta(3.25e-4).tolerance(), 3.25e-4);
}

#[test]
fn should_stop_first_epoch_never_converges() {
    let c = Convergence::AbsDelta(1.0);
    assert!(!c.should_stop(f32::INFINITY, 0.5));
}

#[test]
fn should_stop_converges_when_delta_small() {
    let c = Convergence::AbsDelta(1e-3);
    assert!(c.should_stop(0.50000, 0.50005));
    assert!(!c.should_stop(0.50, 0.60));
}

#[test]
fn off_never_stops() {
    assert!(!Convergence::Off.should_stop(0.5, 0.5));
}
