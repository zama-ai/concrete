use libc::{flock, LOCK_EX, LOCK_UN};
use std::{
    fs::File,
    os::fd::{AsRawFd, RawFd},
};

#[cfg(unix)]
pub struct FileLock(RawFd);

impl FileLock {
    pub fn acquire(file: &File) -> Result<FileLock, std::io::Error> {
        match unsafe { flock(file.as_raw_fd(), LOCK_EX) } {
            0 => Ok(FileLock(file.as_raw_fd())),
            _ => Err(std::io::Error::last_os_error()),
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        if unsafe { flock(self.0, LOCK_UN) } < 0 {
            eprintln!(
                "Failed to release filelock {}: {}",
                self.0,
                std::io::Error::last_os_error()
            );
        }
    }
}
