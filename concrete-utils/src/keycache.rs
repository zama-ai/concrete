use once_cell::sync::OnceCell;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::RwLock;

pub trait PersistentStorage<P, K> {
    fn load(&self, param: P) -> Option<K>;
    fn store(&self, param: P, key: &K);
}

pub trait NamedParam {
    fn name(&self) -> String;
}

#[macro_export]
macro_rules! named_params_impl(
  ( $thing:ident == ( $($const_param:ident),* $(,)? )) => {
      named_params_impl!({ *$thing } == ( $($const_param),* ))
  };

  ( { $thing:expr } == ( $($const_param:ident),* $(,)? )) => {
      $(
        if $thing == $const_param {
            return stringify!($const_param).to_string();
        }
      )*

      panic!("Unnamed parameters");
  }
);

pub struct FileStorage {
    prefix: String,
}

impl FileStorage {
    pub fn new(prefix: String) -> Self {
        Self { prefix }
    }
}

impl<P, K> PersistentStorage<P, K> for FileStorage
where
    P: NamedParam + DeserializeOwned + Serialize + PartialEq,
    K: DeserializeOwned + Serialize,
{
    fn load(&self, param: P) -> Option<K> {
        let mut path_buf = PathBuf::with_capacity(256);
        path_buf.push(&self.prefix);
        path_buf.push(param.name());
        path_buf.set_extension("bin");

        if path_buf.exists() {
            let file = BufReader::new(File::open(&path_buf).unwrap());
            bincode::deserialize_from::<_, (P, K)>(file)
                .ok()
                .and_then(|(p, k)| if p == param { Some(k) } else { None })
        } else {
            None
        }
    }

    fn store(&self, param: P, key: &K) {
        let mut path_buf = PathBuf::with_capacity(256);
        path_buf.push(&self.prefix);
        std::fs::create_dir_all(&path_buf).unwrap();
        path_buf.push(param.name());
        path_buf.set_extension("bin");

        let file = BufWriter::new(File::create(&path_buf).unwrap());
        bincode::serialize_into(file, &(param, key)).unwrap();
    }
}

pub struct KeyCache<P, K: 'static, S> {
    // Where the keys will be stored persistently
    // So they are not generated between each run
    persistent_storage: S,
    // Temporary memory storage to avoid querying the persistent storage each time
    memory_storage: RwLock<Vec<(P, OnceCell<&'static K>)>>,
}

impl<P, K, S> KeyCache<P, K, S> {
    pub fn new(storage: S) -> Self {
        Self {
            persistent_storage: storage,
            memory_storage: RwLock::new(vec![]),
        }
    }
}

impl<P, K, S> KeyCache<P, K, S>
where
    P: Copy + PartialEq,
    S: PersistentStorage<P, K>,
    K: From<P> + Clone,
{
    pub fn get(&self, param: P) -> &'static K {
        self.with_key(param, |k| k)
    }

    pub fn with_key<F, R>(&self, param: P, f: F) -> R
    where
        F: FnOnce(&'static K) -> R,
    {
        let load_from_persistent_storage = || {
            // we check if we can load the key from persistent storage
            let persistent_storage = &self.persistent_storage;
            let maybe_key = persistent_storage.load(param);
            match maybe_key {
                Some(key) => key,
                None => {
                    let key = K::from(param.clone());
                    persistent_storage.store(param.clone(), &key);
                    key
                }
            }
        };

        let load_from_memory_and_init_with = |f: F| {
            // we only hold a read lock for a short duration to find the key
            let memory_storage = self.memory_storage.read().unwrap();
            let maybe_param_key = memory_storage
                .iter()
                .find(|(p, _)| *p == param)
                .map(|param_key| param_key.1.clone());
            drop(memory_storage);

            if let Some(key) = maybe_param_key {
                let key = key.get_or_init(|| Box::leak(Box::new(load_from_persistent_storage())));
                Ok(f(key))
            } else {
                Err(f)
            }
        };

        match load_from_memory_and_init_with(f) {
            Ok(result) => result,
            Err(f) => {
                {
                    // we only hold a write lock for a short duration to push the lazily evaluated key
                    // without actually evaluating the key
                    let mut memory_storage = self.memory_storage.write().unwrap();
                    if memory_storage.iter().find(|(p, _)| *p == param).is_none() {
                        memory_storage.push((param.clone(), OnceCell::new()));
                    }
                }
                load_from_memory_and_init_with(f).ok().unwrap()
            }
        }
    }
}
