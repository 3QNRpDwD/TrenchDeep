use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

pub(crate) fn render(program: &Path, dot: &str, output: &Path) -> Result<(), String> {
    let mut child = Command::new(program)
        .arg("-Tsvg")
        .arg("-o")
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start Graphviz '{}': {error}", program.display()))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "Graphviz stdin was unavailable".to_string())?
        .write_all(dot.as_bytes())
        .map_err(|error| format!("failed to write DOT to Graphviz: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Graphviz wait failed: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "Graphviz failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}
