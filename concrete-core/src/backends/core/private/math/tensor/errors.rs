use std::fmt;

/// A type that represents an error when saving a tensor to a file.
#[derive(Debug)]
pub enum SaveError {
    /// The error occured when creating the file. Probably a wrong path.
    CreatingFile {
        filename: String,
        source: std::io::Error,
    },
    /// The error occured when writing the file.
    WritingFile {
        filename: String,
        source: bincode::Error,
    },
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreatingFile { filename, source } => {
                write!(f, "Failed to open file {}: {}", filename, source)
            }
            Self::WritingFile { filename, source } => {
                write!(f, "Failed to write file {}: {}", filename, source)
            }
        }
    }
}

impl std::error::Error for SaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreatingFile { source, .. } => Some(source),
            Self::WritingFile { source, .. } => Some(source),
        }
    }
}

/// A type representing an error when loading a tensor from a file.
#[derive(Debug)]
pub enum LoadError {
    /// The error occurred when opening the file. Probably a wrong path.
    OpeningFile {
        filename: String,
        source: std::io::Error,
    },
    /// The error occurred when reading the file.
    ReadingFile {
        filename: String,
        source: bincode::Error,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpeningFile { filename, source } => {
                write!(f, "Failed to open file {}: {}", filename, source)
            }
            Self::ReadingFile { filename, source } => {
                write!(f, "Failed to read file {}: {}", filename, source)
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpeningFile { source, .. } => Some(source),
            Self::ReadingFile { source, .. } => Some(source),
        }
    }
}
