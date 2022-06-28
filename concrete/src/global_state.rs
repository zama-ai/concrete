//! In this module, we store the hidden (to the end-user) internal state/keys that are needed to
//! perform operations.
use crate::errors::{UninitializedServerKey, UnwrapResultExt};
use std::cell::RefCell;

use crate::keys::ServerKey;

/// We store the internal keys as thread local, meaning each thread has its own set of keys.
///
/// This means that the user can do computations in multiple threads
/// (eg a web server that processes multiple requests in multiple threads).
/// The user however, has to initialize the internal keys each time it starts a thread.
///
/// We could have made the internal keys non thread local, this would have meant that
/// the internal keys would have been behind a mutex.
/// However, that means we would effectively have a kind of "Global interpreter lock" like Python,
/// meaning the user could start multiple threads (without needing to initialize the keys in each
/// threads) however each operation (+, -, *, etc) could happen in one thread at a time so no real
/// multi-threading.
thread_local! {
    static INTERNAL_KEYS: RefCell<ServerKey> = RefCell::new(ServerKey::default());
}

/// The function used to initialize internal keys.
///
/// As each thread has its own set of keys,
/// this function must be called at least once on each thread to initialize its keys.
///
///
/// # Example
///
/// Only working in the `main` thread
///
/// ```
/// use concrete;
///
/// # let config = concrete::ConfigBuilder::all_disabled().build();
/// let (client_key, server_key) = concrete::generate_keys(config);
///
/// concrete::set_server_key(server_key);
/// // Now we can do operations on homomorphic types
/// ```
///
///
/// Working with multiple threads
///
/// ```
/// use concrete;
/// use concrete::ConfigBuilder;
/// use std::thread;
///
/// # let config = concrete::ConfigBuilder::all_disabled().build();
/// let (client_key, server_key) = concrete::generate_keys(config);
/// let server_key_2 = server_key.clone();
///
/// let th1 = thread::spawn(move || {
///     concrete::set_server_key(server_key);
///     // Now, this thread we can do operations on homomorphic types
/// });
///
/// let th2 = thread::spawn(move || {
///     concrete::set_server_key(server_key_2);
///     // Now, this thread we can do operations on homomorphic types
/// });
///
/// th2.join();
/// th1.join();
/// ```
pub fn set_server_key(keys: ServerKey) {
    INTERNAL_KEYS.with(|internal_keys| internal_keys.replace_with(|_old| keys));
}

/// Convenience function that allows to write functions that needs to access the internal keys.
#[cfg(any(feature = "shortints", feature = "integers", feature = "booleans"))]
#[inline]
pub(crate) fn with_internal_keys_mut<T, F>(func: F) -> T
where
    F: FnOnce(&mut ServerKey) -> T,
{
    // Should use `with_borrow_mut` when its stabilized
    INTERNAL_KEYS.with(|keys| {
        let key = &mut *keys.borrow_mut();
        func(key)
    })
}

/// Helper macro to help reduce boiler plate
/// needed to implement `WithGlobalKey` since for
/// our keys, the implementation is the same, only a few things change.
///
/// It expects:
/// - The  `name` of the key type for which the trait will be implemented.
/// - The identifier (or identifier chain) that points to the member in the `ServerKey` that hols
///   the key for which the trait is implemented.
/// - Type Variant used to identify the type at runtime (see `error.rs`)
#[cfg(any(feature = "shortints", feature = "integers", feature = "booleans"))]
macro_rules! impl_with_global_key {
    (
        for $key_type:ty {
            keychain_member: $($member:ident).*,
            type_variant: $enum_variant:expr,
        }
    ) => {
        impl crate::global_state::WithGlobalKey for $key_type {
            fn with_global_mut<R, F>(func: F) -> Result<R, UninitializedServerKey>
            where
                F: FnOnce(&mut Self) -> R,
            {
                crate::global_state::with_internal_keys_mut(|keys| {
                    keys$(.$member)*
                        .as_mut()
                        .map(func)
                        .ok_or(UninitializedServerKey($enum_variant))
                })
            }
        }
    }
}

/// Global key access trait
///
/// Each type we will expose to the user is going to need to have some internal keys.
/// This trait is there to make each of these internal keys have a convenience function that gives
/// access to the internal keys of its type.
///
/// Typically, the implementation of the trait will be on the 'internal' key type
/// and will call [with_internal_keys_mut] and select the right member of the [ServerKey] type.
pub trait WithGlobalKey {
    fn with_global_mut<R, F>(func: F) -> Result<R, UninitializedServerKey>
    where
        F: FnOnce(&mut Self) -> R;

    #[track_caller]
    fn with_unwrapped_global_mut<R, F>(func: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        Self::with_global_mut(func).unwrap_display()
    }
}
