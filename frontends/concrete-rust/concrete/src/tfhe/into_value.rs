use crate::ffi::{Tensor, Value};
use crate::utils::into_value::IntoValue;
use cxx::UniquePtr;
use std::ptr::copy_nonoverlapping;
use tfhe::integer::IntegerCiphertext;
use tfhe::{
    FheInt10, FheInt12, FheInt14, FheInt16, FheInt2, FheInt4, FheInt6, FheInt8, FheUint10,
    FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8,
};

macro_rules! impl_into_value {
    ($($type:ty),*) => {
        $(
            impl IntoValue for $type {
                fn into_value(self) -> UniquePtr<Value> {
                    let (radix, _, _) = self.into_raw_parts();
                    let n_cts = radix.blocks().len();
                    let lwe_size = radix.blocks()[0].ct.lwe_size().0;
                    let mut vals: Vec<u64> = Vec::with_capacity(n_cts * lwe_size);

                    // SAFETY: We are setting the length of the vector to match its capacity.
                    // This is safe because the vector was allocated with exactly `n_cts * lwe_size` capacity,
                    // and we will initialize all elements before using them.
                    unsafe { vals.set_len(n_cts * lwe_size) };

                    for (i, block) in radix.blocks().iter().enumerate() {
                        unsafe {
                            // SAFETY:
                            // 1. `block.ct.as_view().into_container().as_ptr()` points to valid memory
                            //    because `block.ct` is a valid ciphertext.
                            // 2. `vals.as_mut_ptr().add(i * lwe_size)` points to a valid, non-overlapping
                            //    region of memory within the allocated vector because `vals` was allocated
                            //    with sufficient capacity.
                            // 3. `lwe_size` elements are copied, which matches the size of the source and
                            //    destination regions.
                            // 4. `block` and `vals` are non overlapping, because `block` is allocated before
                            //    the function, and `vals` is allocated inside the scope of the function.
                            copy_nonoverlapping(
                                block.ct.as_view().into_container().as_ptr(),
                                vals.as_mut_ptr().add(i * lwe_size),
                                lwe_size,
                            );
                        }
                    }

                    Value::from_tensor(Tensor::<u64>::new(vals, vec![n_cts, lwe_size]))
                }
            }
        )*
    };
}

impl_into_value!(
    FheInt2, FheInt4, FheInt6, FheInt8, FheInt10, FheInt12, FheInt14, FheInt16, FheUint2, FheUint4,
    FheUint6, FheUint8, FheUint10, FheUint12, FheUint14, FheUint16
);

#[cfg(test)]
mod tests {
    use super::*;
    use tfhe::prelude::*;
    use tfhe::{generate_keys, ConfigBuilder, FheUint8};

    #[test]
    fn test_into_value() {
        let config = ConfigBuilder::default().build();
        let (client_key, _) = generate_keys(config);
        let clear_a = 27u8;
        let a = FheUint8::encrypt(clear_a, &client_key);
        let val = a.into_value();
        assert!(val._has_element_type_u64());
        assert_eq!(val.get_dimensions(), [4, 2049]);
    }
}
