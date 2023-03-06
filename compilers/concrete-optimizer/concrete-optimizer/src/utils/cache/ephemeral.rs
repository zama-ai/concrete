use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use super::read_only::{Map, ReadOnlyCache};

pub type KeyValueFunction<K, V> = Arc<dyn Send + Sync + Fn(K) -> V>;

pub trait EphemeralCache {
    type K;
    type V;
    type ROC: ReadOnlyCache<K = Self::K, V = Self::V>;

    fn new(initial_content: Arc<Self::ROC>, function: KeyValueFunction<Self::K, Self::V>) -> Self;
    fn own_new_entries(cache: Self) -> Map<Self::K, Self::V>;
    fn get(&mut self, k: Self::K) -> &Self::V;
}

/* Cache is mono-thread obtained from a PersistentCache */
pub struct Cache<ROC>
where
    ROC: ReadOnlyCache,
{
    initial_content: Arc<ROC>,            // shared and read-only
    updated_content: Map<ROC::K, ROC::V>, // private mutable
    function: KeyValueFunction<ROC::K, ROC::V>,
}

impl<ROC> EphemeralCache for Cache<ROC>
where
    ROC: ReadOnlyCache,
    ROC::K: Hash + std::cmp::Eq + Copy,
{
    type K = ROC::K;
    type V = ROC::V;
    type ROC = ROC;

    fn new(initial_content: Arc<Self::ROC>, function: KeyValueFunction<ROC::K, ROC::V>) -> Self {
        Self {
            initial_content,
            updated_content: HashMap::default(),
            function,
        }
    }

    fn own_new_entries(cache: Self) -> Map<ROC::K, ROC::V> {
        cache.updated_content
    }

    fn get(&mut self, k: ROC::K) -> &ROC::V {
        let value = self.initial_content.get(k);
        if let Some(value) = value {
            return value;
        }
        self.updated_content
            .entry(k)
            .or_insert_with(|| (self.function)(k))
    }
}

pub type CacheHashMap<K, V> = Cache<Map<K, V>>;
