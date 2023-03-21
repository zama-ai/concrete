use crate::implementation::fft::Fft;

#[no_mangle]
pub static CONCRETE_FFT_SIZE: usize = core::mem::size_of::<Fft>();

#[no_mangle]
pub static CONCRETE_FFT_ALIGN: usize = core::mem::align_of::<Fft>();

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_construct_concrete_fft(
    mem: *mut Fft,
    polynomial_size: usize,
) {
    mem.write(Fft::new(polynomial_size));
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_destroy_concrete_fft(mem: *mut Fft) {
    core::ptr::drop_in_place(mem);
}
