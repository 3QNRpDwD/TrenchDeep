use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use crate::{MlError, MlResult};

use super::{DataError, InMemoryDataset};

/// A decoded record together with its source location.
pub struct LocatedRecord<R> {
    pub record: R,
    pub path: Option<PathBuf>,
    pub line: usize,
}

/// Eager source of raw records.
pub trait RecordSource {
    type Record;

    fn read(self) -> MlResult<Vec<Self::Record>>;

    fn read_located(self) -> MlResult<Vec<LocatedRecord<Self::Record>>>
    where
        Self: Sized,
    {
        Ok(self
            .read()?
            .into_iter()
            .enumerate()
            .map(|(index, record)| LocatedRecord {
                record,
                path: None,
                line: index + 1,
            })
            .collect())
    }
}

pub struct MemorySource<R> {
    records: Vec<R>,
}

impl<R> MemorySource<R> {
    pub fn new(records: Vec<R>) -> Self {
        Self { records }
    }
}

impl<R> From<Vec<R>> for MemorySource<R> {
    fn from(records: Vec<R>) -> Self {
        Self::new(records)
    }
}

impl<R> RecordSource for MemorySource<R> {
    type Record = R;

    fn read(self) -> MlResult<Vec<R>> {
        Ok(self.records)
    }
}

/// An owned CSV row with name- and index-based access.
#[derive(Debug, Clone)]
pub struct CsvRecord {
    headers: Vec<String>,
    values: Vec<String>,
}

impl CsvRecord {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .position(|header| header == name)
            .and_then(|index| self.get_index(index))
    }

    pub fn get_index(&self, index: usize) -> Option<&str> {
        self.values.get(index).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn as_map(&self) -> BTreeMap<&str, &str> {
        self.headers
            .iter()
            .zip(&self.values)
            .map(|(header, value)| (header.as_str(), value.as_str()))
            .collect()
    }
}

pub struct CsvSource {
    path: PathBuf,
    has_headers: bool,
    delimiter: u8,
}

impl CsvSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            has_headers: true,
            delimiter: b',',
        }
    }

    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }
}

impl RecordSource for CsvSource {
    type Record = CsvRecord;

    fn read(self) -> MlResult<Vec<Self::Record>> {
        Ok(self
            .read_located()?
            .into_iter()
            .map(|located| located.record)
            .collect())
    }

    fn read_located(self) -> MlResult<Vec<LocatedRecord<Self::Record>>> {
        let file = File::open(&self.path).map_err(|error| DataError::Io {
            path: self.path.clone(),
            message: error.to_string(),
        })?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(self.has_headers)
            .delimiter(self.delimiter)
            .from_reader(file);

        let headers = if self.has_headers {
            reader
                .headers()
                .map_err(|error| DataError::Decode {
                    path: self.path.clone(),
                    line: 1,
                    message: error.to_string(),
                })?
                .iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut output = Vec::new();
        for (record_index, result) in reader.records().enumerate() {
            let line = record_index + if self.has_headers { 2 } else { 1 };
            let record = result.map_err(|error| DataError::Decode {
                path: self.path.clone(),
                line: error
                    .position()
                    .map(|position| position.line() as usize)
                    .unwrap_or(line),
                message: error.to_string(),
            })?;
            let row_headers = if self.has_headers {
                headers.clone()
            } else {
                (0..record.len()).map(|index| index.to_string()).collect()
            };
            output.push(LocatedRecord {
                record: CsvRecord {
                    headers: row_headers,
                    values: record.iter().map(str::to_owned).collect(),
                },
                path: Some(self.path.clone()),
                line,
            });
        }
        Ok(output)
    }
}

pub type JsonRecord = serde_json::Value;

pub struct JsonLinesSource {
    path: PathBuf,
}

impl JsonLinesSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl RecordSource for JsonLinesSource {
    type Record = JsonRecord;

    fn read(self) -> MlResult<Vec<Self::Record>> {
        Ok(self
            .read_located()?
            .into_iter()
            .map(|located| located.record)
            .collect())
    }

    fn read_located(self) -> MlResult<Vec<LocatedRecord<Self::Record>>> {
        let file = File::open(&self.path).map_err(|error| DataError::Io {
            path: self.path.clone(),
            message: error.to_string(),
        })?;
        let reader = BufReader::new(file);
        let mut output = Vec::new();
        for (index, result) in reader.lines().enumerate() {
            let line_number = index + 1;
            let line = result.map_err(|error| DataError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str(&line).map_err(|error| DataError::Decode {
                path: self.path.clone(),
                line: line_number,
                message: error.to_string(),
            })?;
            output.push(LocatedRecord {
                record,
                path: Some(self.path.clone()),
                line: line_number,
            });
        }
        Ok(output)
    }
}

/// Converts one decoded record into one model-independent dataset sample.
pub trait Transform<R> {
    type Sample;

    fn transform(&mut self, record: R) -> MlResult<Self::Sample>;
}

impl<R, Sample, F> Transform<R> for F
where
    F: FnMut(R) -> MlResult<Sample>,
{
    type Sample = Sample;

    fn transform(&mut self, record: R) -> MlResult<Self::Sample> {
        self(record)
    }
}

pub struct DatasetBuilder<S, F = ()> {
    source: S,
    transform: F,
}

impl<S> DatasetBuilder<S, ()> {
    pub fn from_source(source: S) -> Self {
        Self {
            source,
            transform: (),
        }
    }

    pub fn map<F>(self, transform: F) -> DatasetBuilder<S, F> {
        DatasetBuilder {
            source: self.source,
            transform,
        }
    }
}

impl<S, F> DatasetBuilder<S, F>
where
    S: RecordSource,
    F: Transform<S::Record>,
{
    pub fn build(mut self) -> MlResult<InMemoryDataset<F::Sample>> {
        let records = self.source.read_located()?;
        if records.is_empty() {
            return Err(DataError::EmptyDataset.into());
        }
        let mut samples = Vec::with_capacity(records.len());
        for located in records {
            let location = match located.path {
                Some(path) => format!(" at {}:{}", path.display(), located.line),
                None => format!(" at record {}", located.line),
            };
            let sample = self
                .transform
                .transform(located.record)
                .map_err(|error| DataError::Transform {
                    location,
                    message: error.to_string(),
                })?;
            samples.push(sample);
        }
        InMemoryDataset::new(samples).map_err(MlError::from)
    }
}
