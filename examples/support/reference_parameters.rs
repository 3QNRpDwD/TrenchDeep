use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Descriptor {
    pub name: String,
    pub shape: Vec<usize>,
    /// Lexicographically first structural path referring to the same parameter.
    pub shared_with: String,
}

pub fn describe<I: Eq>(entries: Vec<(String, Vec<usize>, I)>) -> Result<Vec<Descriptor>, String> {
    let mut sorted = BTreeMap::new();
    for (name, shape, identity) in entries {
        if name.is_empty() || sorted.insert(name.clone(), (shape, identity)).is_some() {
            return Err(format!("empty or duplicate parameter name: {name}"));
        }
    }
    let mut result = Vec::new();
    for (name, (shape, identity)) in &sorted {
        let (canonical, (canonical_shape, _)) =
            sorted.iter().find(|(_, (_, id))| id == identity).unwrap();
        if canonical_shape != shape {
            return Err(format!("shared parameter shape mismatch: {name}"));
        }
        result.push(Descriptor {
            name: name.clone(),
            shape: shape.clone(),
            shared_with: canonical.clone(),
        });
    }
    Ok(result)
}

pub fn validate(actual: &[Descriptor], expected: &[Descriptor]) -> Result<(), String> {
    fn index(items: &[Descriptor]) -> Result<BTreeMap<&str, &Descriptor>, String> {
        let mut result = BTreeMap::new();
        for item in items {
            if item.name.is_empty() || result.insert(item.name.as_str(), item).is_some() {
                return Err(format!("empty or duplicate parameter name: {}", item.name));
            }
        }
        for item in items {
            let canonical = result
                .get(item.shared_with.as_str())
                .ok_or_else(|| format!("missing shared parameter: {}", item.name))?;
            if canonical.shared_with != canonical.name
                || canonical.shape != item.shape
                || canonical.name > item.name
            {
                return Err(format!("invalid shared parameter: {}", item.name));
            }
        }
        Ok(result)
    }
    let actual = index(actual)?;
    let expected = index(expected)?;
    if actual.keys().ne(expected.keys()) {
        return Err("parameter name set mismatch".into());
    }
    for (name, actual) in actual {
        if actual != expected[name] {
            return Err(format!("parameter shape/sharing mismatch: {name}"));
        }
    }
    Ok(())
}
