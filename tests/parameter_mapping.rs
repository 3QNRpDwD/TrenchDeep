#[path = "../examples/support/reference_parameters.rs"]
mod mapping;
use mapping::{describe, validate};

#[test]
fn names_and_sharing_are_independent_of_order_and_local_ids() {
    let a = describe(vec![
        ("b".into(), vec![2], 1),
        ("a".into(), vec![2], 1),
        ("c".into(), vec![2], 2),
    ])
    .unwrap();
    let b = describe(vec![
        ("c".into(), vec![2], 90),
        ("a".into(), vec![2], 80),
        ("b".into(), vec![2], 80),
    ])
    .unwrap();
    validate(&a, &b).unwrap();
    let unshared = describe(vec![
        ("a".into(), vec![2], 1),
        ("b".into(), vec![2], 2),
        ("c".into(), vec![2], 3),
    ])
    .unwrap();
    assert!(validate(&a, &unshared).is_err());
    let mut wrong = b.clone();
    wrong[2].shape = vec![1, 2];
    assert!(validate(&a, &wrong).is_err());
    wrong = b.clone();
    wrong[2].name = "d".into();
    wrong[2].shared_with = "d".into();
    assert!(validate(&a, &wrong).is_err());
    wrong = b.clone();
    wrong.push(b[0].clone());
    assert!(validate(&a, &wrong).is_err());
    assert!(describe(vec![("a".into(), vec![2], 1), ("a".into(), vec![2], 2)]).is_err());
}

#[test]
fn invalid_shared_references_are_rejected() {
    let valid = describe(vec![("a".into(), vec![2], 1), ("b".into(), vec![2], 1)]).unwrap();
    let mut invalid = valid.clone();
    invalid[1].shared_with = "missing".into();
    assert!(validate(&valid, &invalid).is_err());
    invalid = valid.clone();
    invalid[0].shared_with = "b".into();
    assert!(validate(&valid, &invalid).is_err());
    assert!(describe(vec![("a".into(), vec![2], 1), ("b".into(), vec![3], 1)]).is_err());
}
