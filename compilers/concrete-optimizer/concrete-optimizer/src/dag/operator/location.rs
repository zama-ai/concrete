use std::{fmt::Display, path::PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    Unknown,
    File(PathBuf),
    Line(PathBuf, usize),
    LineColumn(PathBuf, usize, usize),
}

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "unknown"),
            Self::File(file) => write!(f, "{}", file.file_name().unwrap().to_str().unwrap()),
            Self::Line(file, line) => {
                write!(f, "{}:{line}", file.file_name().unwrap().to_str().unwrap())
            }
            Self::LineColumn(file, line, column) => {
                write!(
                    f,
                    "{}:{line}:{column}",
                    file.file_name().unwrap().to_str().unwrap()
                )
            }
        }
    }
}
