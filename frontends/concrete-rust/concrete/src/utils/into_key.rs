use cxx::UniquePtr;
use crate::ffi::LweSecretKey;

pub trait IntoLweSecretKey{
    fn into_lwe_secret_key(&self) -> UniquePtr<LweSecretKey>;
}
