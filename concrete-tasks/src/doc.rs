use crate::cmd;
use std::io::Error;

pub fn test_book_boolean() -> Result<(), Error> {
    cmd!("cargo build -p concrete-boolean --release")?;
    cmd!("mdbook test concrete-boolean/book -L target/release/deps")
}
