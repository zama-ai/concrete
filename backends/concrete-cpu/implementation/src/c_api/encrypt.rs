use crate::c_api::types::Csprng;
use concrete_csprng::generators::SoftwareRandomGenerator;
use std::slice;
use tfhe::core_crypto::commons::math::random::RandomGenerator;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_fill_with_random_gaussian(
    buffer: *mut u64,
    size: usize,
    variance: f64,
    csprng: *mut Csprng,
) {
    unsafe {
        let buff: &mut [u64] = slice::from_raw_parts_mut(buffer, size);
        let csprng = &mut *(csprng as *mut RandomGenerator<SoftwareRandomGenerator>);
        csprng.fill_slice_with_random_gaussian(buff, 0.0, variance)
    }
}
