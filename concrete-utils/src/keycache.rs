use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::Mutex;

pub trait PersistentStorage<P, K> {
    fn load(&mut self, param: P) -> Option<K>;

    fn store(&mut self, param: P, key: &K);
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
    path_buf: PathBuf,
}

impl FileStorage {
    pub fn new(prefix: String) -> Self {
        let path_buf = PathBuf::with_capacity(255);
        Self { prefix, path_buf }
    }
}

impl<P, K> PersistentStorage<P, K> for FileStorage
where
    P: NamedParam + DeserializeOwned + Serialize + PartialEq,
    K: DeserializeOwned + Serialize,
{
    fn load(&mut self, param: P) -> Option<K> {
        self.path_buf.clear();
        self.path_buf.push(&self.prefix);
        self.path_buf.push(param.name());
        self.path_buf.set_extension("bin");

        if self.path_buf.exists() {
            let file = BufReader::new(File::open(&self.path_buf).unwrap());
            bincode::deserialize_from::<_, (P, K)>(file)
                .ok()
                .and_then(|(p, k)| if p == param { Some(k) } else { None })
        } else {
            None
        }
    }

    fn store(&mut self, param: P, key: &K) {
        self.path_buf.clear();
        self.path_buf.push(&self.prefix);
        std::fs::create_dir_all(&self.path_buf).unwrap();
        self.path_buf.push(param.name());
        self.path_buf.set_extension("bin");

        let file = BufWriter::new(File::create(&self.path_buf).unwrap());
        bincode::serialize_into(file, &(param, key)).unwrap();
    }
}

pub struct KeyCache<P, K, S> {
    // Where the keys will be stored persistently
    // So they are not generated between each run
    persistent_storage: Mutex<S>,
    // Temporary memory storage to avoid querying the persistent storage each time
    memory_storage: Mutex<Vec<(P, K)>>,
}

impl<P, K, S> KeyCache<P, K, S> {
    pub fn new(storage: S) -> Self {
        Self {
            persistent_storage: Mutex::new(storage),
            memory_storage: Mutex::new(vec![]),
        }
    }
}

impl<P, K, S> KeyCache<P, K, S>
where
    P: Copy + PartialEq,
    S: PersistentStorage<P, K>,
    K: From<P> + Clone,
{
    // TODO
    // Making a function that returns &K is not easily possible without an external dep
    // https://stackoverflow.com/questions/40095383/how-to-return-a-reference-to-a-sub-value-of-a-value-that-is-under-a-mutex
    // But that may be useful to avoid cloning too much and being easier on the memory
    //
    // But since we are returning ref to objects in a Vec that we push, the ref may get invalidated
    // so its more complex thant that, we me require `Arc` or something like that
    pub fn get(&self, param: P) -> K {
        self.with_key(param, Clone::clone)
    }

    pub fn with_key<F, R>(&self, param: P, f: F) -> R
    where
        F: FnOnce(&K) -> R,
    {
        let mut memory_storage = self.memory_storage.lock().unwrap();

        let key = if let Some((_, key)) = memory_storage.iter().find(|(p, _)| *p == param) {
            key
        } else {
            let mut persistent_storage = self.persistent_storage.lock().unwrap();

            let maybe_key = persistent_storage.load(param);
            let key = match maybe_key {
                None => {
                    let key = K::from(param);
                    persistent_storage.store(param, &key);
                    key
                }
                Some(key) => key,
            };

            memory_storage.push((param, key));
            &memory_storage.last().unwrap().1
        };
        f(key)
    }
}
