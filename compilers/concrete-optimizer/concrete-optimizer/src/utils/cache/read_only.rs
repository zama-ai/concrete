use crate::utils::hasher_builder::FxRandomState;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::hash::Hash;

pub type Map<K, V> = HashMap<K, V, FxRandomState>;

#[allow(clippy::len_without_is_empty)]
pub trait ReadOnlyCache: Clone + Serialize + DeserializeOwned + Default {
    type K: Copy + Hash + Eq;
    type V: Clone;

    fn get(&self, k: Self::K) -> Option<&Self::V>;
    fn merge(new_cache: Self, other_cache: Self) -> Self;
    fn extend(roc: Self, ext: Map<Self::K, Self::V>) -> Self;
    fn len(&self) -> usize;
}

#[allow(clippy::implicit_hasher)]
impl<K, V> ReadOnlyCache for Map<K, V>
where
    K: Hash + Eq + Copy + Serialize + DeserializeOwned,
    V: Clone + Serialize + DeserializeOwned,
{
    type K = K;
    type V = V;

    #[allow(clippy::only_used_in_recursion)] // clippy false positive
    fn get(&self, k: K) -> Option<&V> {
        self.get(&k)
    }

    fn merge(new_cache: Self, other_cache: Self) -> Self {
        let mut new_cache = new_cache;
        if new_cache.len() < other_cache.len() {
            return Self::merge(other_cache, new_cache);
        }
        for (k, v) in other_cache {
            let _unused = new_cache.insert(k, v);
        }
        new_cache
    }

    fn extend(roc: Self, ext: Map<Self::K, Self::V>) -> Self {
        Self::merge(roc, ext)
    }

    fn len(&self) -> usize {
        self.len()
    }
}
