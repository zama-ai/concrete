pub mod bootstrap;
#[cfg(feature = "csprng")]
pub mod csprng;
pub mod encrypt;
pub mod fft;
pub mod fheint;
pub mod keyswitch;
pub mod linear_op;
pub mod secret_key;
pub mod types;
pub mod wop_pbs;
pub mod wop_pbs_simulation;

mod utils {

    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use tfhe::named::Named;
    use tfhe::{Unversionize, Versionize};

    #[inline]
    pub fn nounwind<R>(f: impl FnOnce() -> R) -> R {
        struct AbortOnDrop;

        impl Drop for AbortOnDrop {
            #[inline]
            fn drop(&mut self) {
                panic!();
            }
        }

        let abort = AbortOnDrop;
        let val = f();
        core::mem::forget(abort);
        val
    }

    const __ASSERT_USIZE_SAME_AS_SIZE_T: () = {
        let _: libc::size_t = 0_usize;
    };

    // Serialize a tfhe-rs versionable value into a buffer, returns 0 if any error
    // TODO: Better error management
    pub unsafe fn safe_serialize<T: Serialize + Versionize + Named>(
        value: &T,
        buffer: *mut u8,
        buffer_len: usize,
    ) -> usize {
        let writer = core::slice::from_raw_parts_mut(buffer, buffer_len);
        let size = match tfhe::safe_serialization::safe_serialized_size(value) {
            Ok(size) => {
                if size > buffer_len as u64 {
                    return 0;
                }
                size as usize
            }
            Err(_e) => return 0,
        };

        match tfhe::safe_serialization::safe_serialize(value, writer, buffer_len as u64) {
            Ok(_) => size,
            Err(_e) => 0,
        }
    }

    // Deserialize a tfhe-rs versionable value from a buffer, panic if any error
    // TODO: Better error management
    pub unsafe fn safe_deserialize<T: DeserializeOwned + Unversionize + Named>(
        buffer: *const u8,
        buffer_len: usize,
    ) -> T {
        let reader = core::slice::from_raw_parts(buffer, buffer_len);
        // TODO: Fix approximation when is fixed in TFHE-rs
        tfhe::safe_serialization::safe_deserialize(reader, (buffer_len + 1000) as u64).unwrap()
    }

    // Serialize a tfhe-rs NON-versionable value into a buffer, returns 0 if any error.
    // TODO: Remove me when safe_serialization by thfe-rs is implemented for all object.
    pub unsafe fn unsafe_serialize<T: Serialize>(
        value: &T,
        buffer: *mut u8,
        buffer_len: usize,
    ) -> usize {
        let serialized_size: usize = match bincode::serialized_size(value) {
            Ok(size) if size <= buffer_len as u64 => size as usize,
            _ => return 0,
        };

        let writer: &mut [u8] = core::slice::from_raw_parts_mut(buffer, buffer_len);
        match bincode::serialize_into(&mut writer[..], value) {
            Ok(_) => serialized_size,
            Err(_) => 0,
        }
    }

    // Deserialize a tfhe-rs NON-versionable value into a buffer, panic if any error
    // TODO: Remove me when safe_serialization by thfe-rs is implemented for all object.
    pub unsafe fn unsafe_deserialize<T: DeserializeOwned>(
        buffer: *const u8,
        buffer_len: usize,
    ) -> T {
        let reader = core::slice::from_raw_parts(buffer, buffer_len);
        bincode::deserialize_from(reader).unwrap()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_unwind() {
            // can't test caught panics
            // so we just test the successful path
            assert_eq!(nounwind(|| 1), 1);
        }
    }
}
