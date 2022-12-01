use rand::Rng;
use rustc_hash::FxHasher;
use std::cell::Cell;
use std::hash::{BuildHasher, Hasher};

// Randomized hasher builder to avoid the stable hashmap trap
// see https://morestina.net/blog/1843/the-stable-hashmap-trap
#[derive(Copy, Clone, Debug)]
pub struct FxRandomState(usize);

impl BuildHasher for FxRandomState {
    type Hasher = FxHasher;

    fn build_hasher(&self) -> FxHasher {
        let mut hasher = FxHasher::default();
        hasher.write_usize(self.0);
        hasher
    }
}

impl Default for FxRandomState {
    fn default() -> Self {
        thread_local! {
            static SEED: Cell<usize>  = Cell::new(rand::thread_rng().gen())
        }
        let seed = SEED.with(|seed| {
            let n = seed.get();
            seed.set(n.wrapping_add(1));
            n
        });
        Self(seed)
    }
}
