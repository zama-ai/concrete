use tfhe::ClientKey;

use crate::protocol::{KeyType, LweSecretKeyInfo, LweSecretKeyParams};
use crate::utils::into_key::IntoLweSecretKey;

impl IntoLweSecretKey for ClientKey {
    fn into_lwe_secret_key(&self) -> cxx::UniquePtr<crate::ffi::LweSecretKey> {
        let (integer_ck, _, _, _, _) = self.clone().into_raw_parts();
        let shortint_ck = integer_ck.into_raw_parts();
        let (glwe_secret_key, _, _) = shortint_ck.into_raw_parts();
        let lwe_secret_key = glwe_secret_key.into_lwe_secret_key();
        let buffer = lwe_secret_key.as_view().into_container();
        let info = LweSecretKeyInfo {
            id: 0,
            params: LweSecretKeyParams {
                lweDimension: buffer.len() as u32,
                integerPrecision: 64,
                keyType: KeyType::binary,
            },
        };
        crate::ffi::_lwe_secret_key_from_buffer_and_info(buffer, &serde_json::to_string(&info).unwrap())
    }
}
