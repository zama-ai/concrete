pub mod bootstrap;
pub mod compact_public_key;
#[cfg(feature = "csprng")]
pub mod csprng;
pub mod encrypt;
pub mod fft;
pub mod keyswitch;
pub mod linear_op;
pub mod public_key;
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
