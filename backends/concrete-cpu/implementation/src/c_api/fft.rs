use tfhe::core_crypto::commons::parameters::PolynomialSize;

type FftImpl = tfhe::core_crypto::fft_impl::fft64::math::fft::Fft;

pub struct Fft {
    _private: (),
}

#[no_mangle]
pub static CONCRETE_FFT_SIZE: usize = core::mem::size_of::<FftImpl>();

#[no_mangle]
pub static CONCRETE_FFT_ALIGN: usize = core::mem::align_of::<FftImpl>();

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_construct_concrete_fft(
    mem: *mut Fft,
    polynomial_size: usize,
) {
    let mem = mem as *mut FftImpl;
    mem.write(FftImpl::new(PolynomialSize(polynomial_size)));
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_destroy_concrete_fft(mem: *mut Fft) {
    core::ptr::drop_in_place(mem);
}
