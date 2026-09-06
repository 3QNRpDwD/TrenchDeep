use super::{DotEncoder, DotProfile, GraphSnapshot, VisualizationError, graphviz};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WriteWarning {
    pub artifact: PathBuf,
    pub kind: graphviz::GraphvizFailureKind,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct WriteReport {
    pub artifacts: Vec<PathBuf>,
    pub warnings: Vec<WriteWarning>,
}

pub trait SnapshotWriter: Send {
    fn write(
        &mut self,
        snapshot: &GraphSnapshot,
        stem: &str,
    ) -> Result<WriteReport, VisualizationError>;
}

pub struct FileSnapshotWriterBuilder {
    output_directory: PathBuf,
    render_svg: bool,
    graphviz_program: PathBuf,
    dot_profile: DotProfile,
}

impl FileSnapshotWriterBuilder {
    pub fn render_svg(mut self, render: bool) -> Self {
        self.render_svg = render;
        self
    }

    pub fn graphviz_program<P: AsRef<Path>>(mut self, program: P) -> Self {
        self.graphviz_program = program.as_ref().to_path_buf();
        self
    }

    pub fn dot_profile(mut self, profile: DotProfile) -> Self {
        self.dot_profile = profile;
        self
    }

    pub fn build(self) -> Result<FileSnapshotWriter, VisualizationError> {
        if self.output_directory.as_os_str().is_empty() {
            return Err(VisualizationError::InvalidArtifactStem(
                "empty output directory".into(),
            ));
        }
        Ok(FileSnapshotWriter {
            output_directory: self.output_directory,
            render_svg: self.render_svg,
            graphviz_program: self.graphviz_program,
            dot_profile: self.dot_profile,
        })
    }
}

pub struct FileSnapshotWriter {
    output_directory: PathBuf,
    render_svg: bool,
    graphviz_program: PathBuf,
    dot_profile: DotProfile,
}

impl FileSnapshotWriter {
    pub fn builder<P: AsRef<Path>>(output_directory: P) -> FileSnapshotWriterBuilder {
        FileSnapshotWriterBuilder {
            output_directory: output_directory.as_ref().to_path_buf(),
            render_svg: false,
            graphviz_program: PathBuf::from("dot"),
            dot_profile: DotProfile::Auto,
        }
    }
}

impl SnapshotWriter for FileSnapshotWriter {
    fn write(
        &mut self,
        snapshot: &GraphSnapshot,
        stem: &str,
    ) -> Result<WriteReport, VisualizationError> {
        validate_stem(stem)?;
        std::fs::create_dir_all(&self.output_directory)
            .map_err(|error| VisualizationError::io(&self.output_directory, error))?;

        let dot_text = DotEncoder::encode_with_profile(snapshot, self.dot_profile);
        let dot_path = self.output_directory.join(format!("{stem}.dot"));
        let json_path = self.output_directory.join(format!("{stem}.json"));
        std::fs::write(&dot_path, &dot_text)
            .map_err(|error| VisualizationError::io(&dot_path, error))?;
        let json = serde_json::to_vec_pretty(snapshot)?;
        std::fs::write(&json_path, json)
            .map_err(|error| VisualizationError::io(&json_path, error))?;

        let mut report = WriteReport {
            artifacts: vec![dot_path, json_path],
            warnings: Vec::new(),
        };
        if self.render_svg {
            let svg_path = self.output_directory.join(format!("{stem}.svg"));
            match graphviz::render(&self.graphviz_program, &dot_text, &svg_path) {
                Ok(()) => report.artifacts.push(svg_path),
                Err(error) => report.warnings.push(WriteWarning {
                    artifact: svg_path,
                    kind: error.kind,
                    message: error.message,
                }),
            }
        }
        Ok(report)
    }
}

fn validate_stem(stem: &str) -> Result<(), VisualizationError> {
    let valid = !stem.is_empty()
        && stem != "."
        && stem != ".."
        && stem.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        });
    valid
        .then_some(())
        .ok_or_else(|| VisualizationError::InvalidArtifactStem(stem.into()))
}
