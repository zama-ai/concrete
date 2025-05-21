use std::hash::{Hash, Hasher};
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

pub struct FastPathHasher {
    path: PathBuf,
    ctime: i64,
    mtime: i64,
}

impl FastPathHasher {
    pub fn from_pathbuf(path: &PathBuf) -> FastPathHasher {
        let path = path.canonicalize().unwrap();
        let metadata = path.metadata().unwrap();
        FastPathHasher {
            ctime: metadata.ctime(),
            mtime: metadata.mtime(),
            path,
        }
    }
}

impl Hash for FastPathHasher {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.path.hash(state);
        self.ctime.hash(state);
        self.mtime.hash(state);
    }
}
