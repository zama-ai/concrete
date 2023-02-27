use std::borrow::BorrowMut;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::os::unix::prelude::MetadataExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, RwLockWriteGuard};
use std::time::Instant;

use file_lock::{FileLock, FileOptions};

use super::ephemeral;
use super::ephemeral::{EphemeralCache, KeyValueFunction};
use super::read_only::{Map, ReadOnlyCache};

const SHOW_DISK_ACCESS: bool = false;
const DISABLE_CACHE: bool = false;

/* PersistentCache is compatible with multi-threading */
pub struct PersistentCache<ROC>
where
    ROC: ReadOnlyCache,
{
    // path on disk
    path: String,
    // version to invalidate no longer valid cache
    version: u64,
    // content: the HashMap is read-only, but it can be a new HashMap
    content: RwLock<Arc<ROC>>, // the HashMap is read once, never modified and shared
    content_changed: AtomicBool, // true if the content changed since loading from disk
    function: KeyValueFunction<ROC::K, ROC::V>,
}

impl<ROC> Drop for PersistentCache<ROC>
where
    ROC: ReadOnlyCache,
{
    fn drop(&mut self) {
        if DISABLE_CACHE {
            return;
        }
        self.sync_to_disk();
    }
}

impl<ROC> PersistentCache<ROC>
where
    ROC: ReadOnlyCache,
{
    pub fn new(
        path: &str,
        version: u64,
        function: impl 'static + Send + Sync + Fn(ROC::K) -> ROC::V,
    ) -> Self {
        let cache = Self::new_no_read(path, version, function);
        cache.read();
        cache
    }

    pub fn new_no_read(
        path: &str,
        version: u64,
        function: impl 'static + Send + Sync + Fn(ROC::K) -> ROC::V,
    ) -> Self {
        let path = path.into();
        let content = RwLock::new(Arc::new(ROC::default()));
        let content_changed = AtomicBool::new(false);
        Self {
            path,
            content,
            content_changed,
            version,
            function: Arc::new(function),
        }
    }

    pub fn read(&self) {
        let t0 = Instant::now();
        let content = Self::read_from_disk(&self.path, self.version).unwrap_or_default();
        if SHOW_DISK_ACCESS {
            println!(
                "PersistentCache: {}, reading time {} msec, {} entries",
                self.path,
                t0.elapsed().as_millis(),
                content.len()
            );
        }
        self.update_with(|_| content);
        self.content_changed.store(false, Ordering::Relaxed);
    }

    pub fn cache(&self) -> ephemeral::Cache<ROC> {
        let initial_content = self.content.read().unwrap().clone();
        ephemeral::Cache::<ROC>::new(initial_content, self.function.clone())
    }

    pub fn backport(&self, cache: ephemeral::Cache<ROC>) {
        if DISABLE_CACHE {
            return;
        }
        let new_entries = ephemeral::Cache::<ROC>::own_new_entries(cache);
        if new_entries.is_empty() {
            return;
        }
        self.update_with(|content| ROC::extend(content, new_entries));
    }

    #[allow(clippy::nursery)]
    fn update_with<F>(&self, update: F)
    where
        F: FnOnce(ROC) -> ROC,
    {
        let (mut lock, content) = self.own_or_clone_content();
        let len_before = content.len();
        let content = update(content);
        let len_after = content.len();
        *lock = Arc::new(content);
        if len_before != len_after {
            self.content_changed.store(true, Ordering::Relaxed);
        }
    }

    fn own_or_clone_content(&self) -> (RwLockWriteGuard<Arc<ROC>>, ROC) {
        let mut lock = self.content.write().unwrap();
        // let's take the map ownership if possible, we need to have only one local copy
        let arc_map = std::mem::take(&mut *lock);
        let content = match Arc::try_unwrap(arc_map) {
            // we own the map
            Ok(content_map) => content_map,
            // the map is shared elsewhere, let's clone it
            Err(arc_map) => arc_map.as_ref().clone(),
        };
        (lock, content)
    }

    fn read_from_disk(path: &str, version: u64) -> Option<ROC> {
        if DISABLE_CACHE {
            return None;
        }
        if !std::path::Path::new(path).exists() {
            // avoid creating the file if it does not exists
            return None;
        }
        let options = FileOptions::new().read(true).write(true).create(true);
        let is_blocking = true;
        let filelock = match FileLock::lock(path, is_blocking, options) {
            Ok(lock) => lock,
            Err(error) => {
                println!("PersistentCache::read_from_disk: Cannot lock cache file {path}: {error}");
                return None;
            }
        };
        Self::read_given_lock(&filelock, path, version)
    }

    pub fn sync_to_disk(&self) {
        if !self.content_changed.load(Ordering::Relaxed) {
            if SHOW_DISK_ACCESS {
                println!("PersistentCache: skip sync to disk, {}", self.path);
            }
            return;
        }
        if SHOW_DISK_ACCESS {
            println!("PersistentCache: sync to disk, {}", self.path);
        }
        match std::fs::create_dir_all(std::path::Path::new(&self.path).parent().unwrap()) {
            Ok(()) => (),
            Err(err) => {
                let path = &self.path;
                println!("PersistentCache::sync_to_disk: Cannot create directory {path}, {err}");
                return;
            }
        };
        let options = FileOptions::new().read(true).write(true).create(true);
        let is_blocking = true;
        let mut filelock = match FileLock::lock(&self.path, is_blocking, options) {
            Ok(lock) => lock,
            Err(_err) => {
                println!(
                    "PersistentCache::sync_to_disk: Cannot lock cache file {}",
                    self.path
                );
                return;
            }
        };
        let maybe_disk_content = Self::read_given_lock(&filelock, &self.path, self.version);
        if let Some(disk_content) = maybe_disk_content {
            self.update_with(|content| ROC::merge(content, disk_content));
        }
        self.write_given_lock(
            &mut filelock,
            &self.content.read().unwrap().as_ref().clone(),
        );
        drop(filelock.file.flush());
        drop(filelock.file.sync_all());
        drop(filelock);
        self.content_changed.store(false, Ordering::Relaxed);
    }

    fn read_given_lock(filelock: &FileLock, path: &str, version: u64) -> Option<ROC> {
        match filelock.file.metadata() {
            Ok(metadata) => {
                if metadata.size() == 0 {
                    return None;
                }
            }
            Err(err) => {
                println!("PersistentCache::read_from_disk: cannot read size {path} {err}");
                return None;
            }
        };
        let mut buf = BufReader::new(&filelock.file);

        let disk_version: Result<u64, _> = bincode::deserialize_from(buf.borrow_mut());

        match disk_version {
            Ok(disk_version) => {
                if disk_version != version {
                    println!("PersistentCache:: Invalid version {path}: cleaning");
                    Self::clear_file(path);
                    return None;
                }
            }
            Err(error) => {
                println!("PersistentCache::read_given_lock: Cannot read version {path}: {error}");
                Self::clear_file(path);
                return None;
            }
        }
        match bincode::deserialize_from(buf.borrow_mut()) {
            Ok(content) => Some(content),
            Err(error) => {
                println!("PersistentCache::read_given_lock: Cannot read hashmap {path}: {error}");
                Self::clear_file(path);
                None
            }
        }
    }

    fn write_given_lock(&self, filelock: &mut FileLock, content: &ROC) {
        if SHOW_DISK_ACCESS {
            println!(
                "PersistentCache::write: to disk {}: {} records",
                self.path,
                content.len()
            );
        }
        if let Err(err) = filelock.file.rewind() {
            println!(
                "PersistentCache::write: cannot rewind file: {}, {err}",
                self.path
            );
            return;
        }
        if let Err(err) = filelock.file.set_len(0) {
            println!(
                "PersistentCache::write: cannot truncate file: {}, {err}",
                self.path
            );
        }
        let file = &mut filelock.file;
        let mut buf = BufWriter::new(file);

        bincode::serialize_into(&mut buf, &self.version).unwrap();
        bincode::serialize_into(&mut buf, content).unwrap();
    }

    pub fn clear_file(path: &str) {
        if !std::path::Path::new(path).exists() {
            return;
        }
        let options = FileOptions::new().write(true).create(true).truncate(true);
        let is_blocking = true;
        let filelock = match FileLock::lock(path, is_blocking, options) {
            Ok(lock) => lock,
            Err(_err) => {
                println!("PersistentCache::clear: Cannot lock cache file {path}");
                return;
            }
        };
        drop(filelock.file.sync_all());
        drop(filelock.unlock());
        drop(std::fs::remove_file(path));
    }
}

pub type PersistentCacheHashMap<K, V> = PersistentCache<Map<K, V>>;

pub fn default_cache_dir() -> String {
    let mut cache_dir = std::env::temp_dir();
    cache_dir.push("optimizer");
    cache_dir.push("cache");
    cache_dir.to_str().expect("Invalid tmp dir").into()
}

#[cfg(test)]
mod tests {
    use super::super::ephemeral::CacheHashMap;
    use super::*;

    #[test]
    fn test_life_cycle() {
        type K = (u64, u64);
        type V = [u64; 2];
        type PCache = PersistentCacheHashMap<K, V>;
        let path = "/tmp/optimizer/tests/test_life_cycle";
        let key1 = (1, 2);
        let value1 = [3, 4];
        let f = move |_key| value1;
        {
            PCache::clear_file(path);
            let disk_cache = PCache::new(path, 0, f);
            let mut mem_cache = disk_cache.cache();
            let res = mem_cache.get(key1);
            assert_eq!(res, &value1);
            disk_cache.backport(mem_cache);
        }
        {
            let cache_later = PCache::new(path, 0, f);
            let mut mem_cache = cache_later.cache();
            let res = mem_cache.get(key1);
            assert_eq!(res, &value1);
            assert!(CacheHashMap::own_new_entries(mem_cache).is_empty());
        }
        {
            let cache_later = PCache::new(path, 1, f);
            let mut mem_cache = cache_later.cache();
            let res = mem_cache.get(key1);
            assert_eq!(res, &value1);
            assert!(!CacheHashMap::own_new_entries(mem_cache).is_empty());
        }
    }
}
