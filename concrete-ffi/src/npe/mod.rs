use concrete_npe::add_ciphertexts;

pub use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::GlweDimension;

#[no_mangle]
pub extern "C" fn npe_add_ciphertexts_variance_variance(dispersion_ct1: Variance, dispersion_ct2:
Variance) ->
                                                                                              Variance {
    return add_ciphertexts::<Variance, Variance>(dispersion_ct1, dispersion_ct2);
}
