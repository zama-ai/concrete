use super::add_ciphertexts;
use concrete_commons::dispersion::{Variance, DispersionParameter};

#[no_mangle]
pub extern "C" fn add_ciphertexts_variance_variance(dispersion_ct1: Variance, dispersion_ct2:
Variance) ->
                                                                                              Variance {
    return add_ciphertexts::<Variance, Variance>(dispersion_ct1, dispersion_ct2);
}
