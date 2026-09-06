use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphvizFailureKind {
    NotInstalled,
    Io,
    NonZeroExit,
}

#[derive(Debug)]
pub struct GraphvizFailure {
    pub kind: GraphvizFailureKind,
    pub message: String,
}

pub(crate) fn render(program: &Path, dot: &str, output: &Path) -> Result<(), GraphvizFailure> {
    let mut child = Command::new(program)
        .arg("-Tsvg")
        .arg("-o")
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| GraphvizFailure {
            kind: if error.kind() == std::io::ErrorKind::NotFound {
                GraphvizFailureKind::NotInstalled
            } else {
                GraphvizFailureKind::Io
            },
            message: format!("failed to start Graphviz '{}': {error}", program.display()),
        })?;
    child
        .stdin
        .take()
        .ok_or_else(|| GraphvizFailure {
            kind: GraphvizFailureKind::Io,
            message: "Graphviz stdin was unavailable".to_string(),
        })?
        .write_all(dot.as_bytes())
        .map_err(|error| GraphvizFailure {
            kind: GraphvizFailureKind::Io,
            message: format!("failed to write DOT to Graphviz: {error}"),
        })?;
    let output = child.wait_with_output().map_err(|error| GraphvizFailure {
        kind: GraphvizFailureKind::Io,
        message: format!("Graphviz wait failed: {error}"),
    })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(GraphvizFailure {
            kind: GraphvizFailureKind::NonZeroExit,
            message: format!(
                "Graphviz failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_graphviz_preserves_the_dot_node_id_in_svg() {
        let output = std::env::temp_dir().join(format!(
            "trench-deep-graphviz-id-{}-{:?}.svg",
            std::process::id(),
            std::thread::current().id(),
        ));
        let dot = "digraph G { \"node-7\" [id=\"node-7\"]; }";
        match render(Path::new("dot"), dot, &output) {
            Err(GraphvizFailure {
                kind: GraphvizFailureKind::NotInstalled,
                ..
            }) => return,
            Err(error) => panic!("Graphviz rendering failed: {error:?}"),
            Ok(()) => {}
        }
        let svg = std::fs::read_to_string(&output).unwrap();
        assert!(svg.contains("id=\"node-7\"") || svg.contains("id=\"node&#45;7\""));
        let _ = std::fs::remove_file(output);
    }
}
