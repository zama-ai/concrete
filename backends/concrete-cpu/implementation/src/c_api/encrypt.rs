use crate::c_api::csprng::CONCRETE_CSPRNG_VTABLE;
use crate::c_api::types::Csprng;
use crate::implementation::encrypt::fill_with_random_gaussian;
use crate::implementation::types::CsprngMut;
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_fill_with_random_gaussian(
    buffer: *mut u64,
    size: usize,
    variance: f64,
    csprng: *mut Csprng,
) {
    unsafe {
        let buff: &mut [u64] = slice::from_raw_parts_mut(buffer, size);
        let csprng_mut: CsprngMut<'_, '_> = CsprngMut::new(csprng, &CONCRETE_CSPRNG_VTABLE);
        fill_with_random_gaussian(buff, variance, csprng_mut)
    }
}
