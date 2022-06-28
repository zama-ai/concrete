use crate::parameters::parameters_wopbs::*;
use crate::parameters::parameters_wopbs_message_carry::*;
use crate::parameters::*;
use crate::wopbs::WopbsKey;
use crate::{ClientKey, ServerKey};
use concrete_utils::keycache::{FileStorage, KeyCache as TKeyCache, NamedParam};
use concrete_utils::named_params_impl;
use lazy_static::*;

impl NamedParam for Parameters {
    fn name(&self) -> String {
        named_params_impl!(
            self == (
                PARAM_MESSAGE_1_CARRY_1,
                PARAM_MESSAGE_1_CARRY_2,
                PARAM_MESSAGE_1_CARRY_3,
                PARAM_MESSAGE_1_CARRY_4,
                PARAM_MESSAGE_1_CARRY_5,
                PARAM_MESSAGE_1_CARRY_6,
                PARAM_MESSAGE_1_CARRY_7,
                PARAM_MESSAGE_2_CARRY_1,
                PARAM_MESSAGE_2_CARRY_2,
                PARAM_MESSAGE_2_CARRY_3,
                PARAM_MESSAGE_2_CARRY_4,
                PARAM_MESSAGE_2_CARRY_5,
                PARAM_MESSAGE_2_CARRY_6,
                PARAM_MESSAGE_3_CARRY_1,
                PARAM_MESSAGE_3_CARRY_2,
                PARAM_MESSAGE_3_CARRY_3,
                PARAM_MESSAGE_3_CARRY_4,
                PARAM_MESSAGE_3_CARRY_5,
                PARAM_MESSAGE_4_CARRY_1,
                PARAM_MESSAGE_4_CARRY_2,
                PARAM_MESSAGE_4_CARRY_3,
                PARAM_MESSAGE_4_CARRY_4,
                PARAM_MESSAGE_5_CARRY_1,
                PARAM_MESSAGE_5_CARRY_2,
                PARAM_MESSAGE_5_CARRY_3,
                PARAM_MESSAGE_6_CARRY_1,
                PARAM_MESSAGE_6_CARRY_2,
                PARAM_MESSAGE_7_CARRY_1,
                WOPBS_PARAM_MESSAGE_1_NORM2_2,
                WOPBS_PARAM_MESSAGE_1_NORM2_4,
                WOPBS_PARAM_MESSAGE_1_NORM2_6,
                WOPBS_PARAM_MESSAGE_1_NORM2_8,
                WOPBS_PARAM_MESSAGE_2_NORM2_2,
                WOPBS_PARAM_MESSAGE_2_NORM2_4,
                WOPBS_PARAM_MESSAGE_2_NORM2_6,
                WOPBS_PARAM_MESSAGE_2_NORM2_8,
                WOPBS_PARAM_MESSAGE_3_NORM2_2,
                WOPBS_PARAM_MESSAGE_3_NORM2_4,
                WOPBS_PARAM_MESSAGE_3_NORM2_6,
                WOPBS_PARAM_MESSAGE_3_NORM2_8,
                WOPBS_PARAM_MESSAGE_4_NORM2_2,
                WOPBS_PARAM_MESSAGE_4_NORM2_4,
                WOPBS_PARAM_MESSAGE_4_NORM2_6,
                WOPBS_PARAM_MESSAGE_4_NORM2_8,
                WOPBS_PARAM_MESSAGE_5_NORM2_2,
                WOPBS_PARAM_MESSAGE_5_NORM2_4,
                WOPBS_PARAM_MESSAGE_5_NORM2_6,
                WOPBS_PARAM_MESSAGE_5_NORM2_8,
                WOPBS_PARAM_MESSAGE_6_NORM2_2,
                WOPBS_PARAM_MESSAGE_6_NORM2_4,
                WOPBS_PARAM_MESSAGE_6_NORM2_6,
                WOPBS_PARAM_MESSAGE_6_NORM2_8,
                WOPBS_PARAM_MESSAGE_7_NORM2_2,
                WOPBS_PARAM_MESSAGE_7_NORM2_4,
                WOPBS_PARAM_MESSAGE_7_NORM2_6,
                WOPBS_PARAM_MESSAGE_7_NORM2_8,
                WOPBS_PARAM_MESSAGE_8_NORM2_2,
                WOPBS_PARAM_MESSAGE_8_NORM2_4,
                WOPBS_PARAM_MESSAGE_8_NORM2_6,
                WOPBS_PARAM_MESSAGE_1_CARRY_0,
                WOPBS_PARAM_MESSAGE_1_CARRY_1,
                WOPBS_PARAM_MESSAGE_1_CARRY_2,
                WOPBS_PARAM_MESSAGE_1_CARRY_3,
                WOPBS_PARAM_MESSAGE_1_CARRY_4,
                WOPBS_PARAM_MESSAGE_1_CARRY_5,
                WOPBS_PARAM_MESSAGE_1_CARRY_6,
                WOPBS_PARAM_MESSAGE_1_CARRY_7,
                WOPBS_PARAM_MESSAGE_2_CARRY_0,
                WOPBS_PARAM_MESSAGE_2_CARRY_1,
                WOPBS_PARAM_MESSAGE_2_CARRY_2,
                WOPBS_PARAM_MESSAGE_2_CARRY_3,
                WOPBS_PARAM_MESSAGE_2_CARRY_4,
                WOPBS_PARAM_MESSAGE_2_CARRY_5,
                WOPBS_PARAM_MESSAGE_2_CARRY_6,
                WOPBS_PARAM_MESSAGE_3_CARRY_0,
                WOPBS_PARAM_MESSAGE_3_CARRY_1,
                WOPBS_PARAM_MESSAGE_3_CARRY_2,
                WOPBS_PARAM_MESSAGE_3_CARRY_3,
                WOPBS_PARAM_MESSAGE_3_CARRY_4,
                WOPBS_PARAM_MESSAGE_3_CARRY_5,
                WOPBS_PARAM_MESSAGE_4_CARRY_0,
                WOPBS_PARAM_MESSAGE_4_CARRY_1,
                WOPBS_PARAM_MESSAGE_4_CARRY_2,
                WOPBS_PARAM_MESSAGE_4_CARRY_3,
                WOPBS_PARAM_MESSAGE_4_CARRY_4,
                WOPBS_PARAM_MESSAGE_5_CARRY_0,
                WOPBS_PARAM_MESSAGE_5_CARRY_1,
                WOPBS_PARAM_MESSAGE_5_CARRY_2,
                WOPBS_PARAM_MESSAGE_5_CARRY_3,
                WOPBS_PARAM_MESSAGE_6_CARRY_0,
                WOPBS_PARAM_MESSAGE_6_CARRY_1,
                WOPBS_PARAM_MESSAGE_6_CARRY_2,
                WOPBS_PARAM_MESSAGE_7_CARRY_0,
                WOPBS_PARAM_MESSAGE_7_CARRY_1,
                WOPBS_PARAM_MESSAGE_8_CARRY_0
            )
        );
    }
}

impl From<Parameters> for (ClientKey, ServerKey) {
    fn from(param: Parameters) -> Self {
        let cks = ClientKey::new(param);
        let sks = ServerKey::new(&cks);
        (cks, sks)
    }
}

pub struct Keycache {
    inner: TKeyCache<Parameters, (ClientKey, ServerKey), FileStorage>,
}

impl Default for Keycache {
    fn default() -> Self {
        Self {
            inner: TKeyCache::new(FileStorage::new(
                "../keys/shortint/client_server".to_string(),
            )),
        }
    }
}

impl Keycache {
    pub fn get_from_param(&self, param: Parameters) -> (ClientKey, ServerKey) {
        self.inner.get(param)
    }
}

impl From<Parameters> for WopbsKey {
    fn from(param: Parameters) -> Self {
        // use with_key to avoid doing a temporary cloning
        KEY_CACHE
            .inner
            .with_key(param, |(cks, sks)| WopbsKey::new_wopbs_key(cks, sks))
    }
}

pub struct KeycacheWopbsV0 {
    inner: TKeyCache<Parameters, WopbsKey, FileStorage>,
}

impl Default for KeycacheWopbsV0 {
    fn default() -> Self {
        Self {
            inner: TKeyCache::new(FileStorage::new("../keys/shortint/wopbs_v0".to_string())),
        }
    }
}

impl KeycacheWopbsV0 {
    pub fn get_from_param(&self, param: Parameters) -> (ClientKey, ServerKey, WopbsKey) {
        let (cks, sks) = KEY_CACHE.get_from_param(param);
        let wk = self.inner.get(param);
        (cks, sks, wk)
    }
}

lazy_static! {
    pub static ref KEY_CACHE: Keycache = Keycache::default();
}

lazy_static! {
    pub static ref KEY_CACHE_WOPBS: KeycacheWopbsV0 = KeycacheWopbsV0::default();
}
