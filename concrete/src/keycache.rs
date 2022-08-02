use crate::{generate_keys, ClientKey, Config, ServerKey};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

/// KeyCacher
///
/// This struct is a very simple "key cache"
/// meant to speedup execution time by not having to generate
/// the key for the given config each time.
///
/// # Limitations
///
/// For now this cacher is extremely basic, if you were
/// to change the config between runs and still use the same path,
/// the cacher wouldn't detect that you'd end up with wrong keys.
///
/// # Example
///
/// with `bincode` in your cargo dependencies:
///
///
/// ```
/// const KEY_PATH: &str = "../keys/fheuint3and2.bin";
///
/// fn main() {
///     use concrete::{ConfigBuilder, KeyCacher};
///     let config = ConfigBuilder::all_disabled()
///         .enable_default_uint2()
///         .enable_default_uint3()
///         .build();
///
///     // The key will be generated on the first run and saved to the
///     // the filepath `KEY_PATH` so later runs will read this file
///     // and avoid regenerating keys.
///     let (client_key, server_key) = KeyCacher::new(
///         KEY_PATH,
///         config,
///         bincode::serialize_into,
///         bincode::deserialize_from,
///     )
///     .get();
/// }
/// ```
#[cfg_attr(doc, cfg(feature = "serde"))]
pub struct KeyCacher<SF, DF> {
    path: PathBuf,
    config: Config,
    serialize_func: SF,
    deserialize_func: DF,
}

impl<SerFn, DeFn> KeyCacher<SerFn, DeFn> {
    pub fn new<P: Into<PathBuf>>(
        path: P,
        config: Config,
        serialize_func: SerFn,
        deserialize_func: DeFn,
    ) -> Self {
        Self {
            path: path.into(),
            config,
            serialize_func,
            deserialize_func,
        }
    }
}

impl<SerFn, DeFn, E> KeyCacher<SerFn, DeFn>
where
    E: std::fmt::Debug,
    SerFn: Fn(BufWriter<File>, &(ClientKey, ServerKey)) -> Result<(), E>,
    DeFn: Fn(BufReader<File>) -> Result<(ClientKey, ServerKey), E>,
{
    pub fn get(&self) -> (ClientKey, ServerKey) {
        let maybe_keys = if self.path.exists() {
            File::open(&self.path)
                .ok()
                .map(BufReader::new)
                .and_then(|file| (self.deserialize_func)(file).ok())
        } else {
            None
        };

        match maybe_keys {
            Some(keys) => keys,
            None => {
                let keys = generate_keys(self.config.clone());

                let output_file = File::create(&self.path).map(BufWriter::new).unwrap();
                (self.serialize_func)(output_file, &keys).unwrap();

                keys
            }
        }
    }
}
