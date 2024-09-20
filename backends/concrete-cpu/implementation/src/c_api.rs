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

    pub unsafe fn serialize<T>(value: &T, out_buffer: *mut u8, out_buffer_len: usize) -> usize
    where
        T: serde::ser::Serialize,
    {
        let serialized_size: usize = match bincode::serialized_size(value) {
            Ok(size) => {
                if size > out_buffer_len as u64 {
                    return 0;
                }
                size as usize
            }
            Err(_) => return 0,
        };

        let write_buff: &mut [u8] = core::slice::from_raw_parts_mut(out_buffer, out_buffer_len);
        match bincode::serialize_into(&mut write_buff[..], value) {
            Ok(_) => serialized_size,
            Err(_) => 0,
        }
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
