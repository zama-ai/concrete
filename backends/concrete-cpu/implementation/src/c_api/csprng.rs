use std::io::Read;

use super::types::{Csprng, EncCsprng, SecCsprng, Uint128};
use concrete_csprng::generators::SoftwareRandomGenerator;
use concrete_csprng::seeders::Seed;
use libc::c_int;
use tfhe::core_crypto::commons::math::random::RandomGenerator;
use tfhe::core_crypto::prelude::{EncryptionRandomGenerator, SecretRandomGenerator};
use tfhe::core_crypto::seeders::Seeder;

pub struct DynamicSeeder;

impl Seeder for DynamicSeeder {
    fn seed(&mut self) -> Seed {
        let mut u128 = Uint128 {
            little_endian_bytes: [0; 16],
        };
        unsafe {
            concrete_cpu_crypto_secure_random_128(std::ptr::addr_of_mut!(u128));
        }
        Seed(u128::from_ne_bytes(u128.little_endian_bytes))
    }

    fn is_available() -> bool {
        true
    }
}

pub fn new_dyn_seeder() -> Box<dyn Seeder> {
    Box::new(DynamicSeeder)
}

#[no_mangle]
pub static CSPRNG_SIZE: usize = core::mem::size_of::<RandomGenerator<SoftwareRandomGenerator>>();

#[no_mangle]
pub static CSPRNG_ALIGN: usize = core::mem::align_of::<RandomGenerator<SoftwareRandomGenerator>>();

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_construct_csprng(mem: *mut Csprng, seed: Uint128) {
    let mem = mem as *mut RandomGenerator<SoftwareRandomGenerator>;
    let seed = Seed(u128::from_le_bytes(seed.little_endian_bytes));
    mem.write(RandomGenerator::new(seed));
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_destroy_csprng(mem: *mut Csprng) {
    core::ptr::drop_in_place(mem as *mut RandomGenerator<SoftwareRandomGenerator>);
}

#[no_mangle]
pub static SECRET_CSPRNG_SIZE: usize =
    core::mem::size_of::<SecretRandomGenerator<SoftwareRandomGenerator>>();

#[no_mangle]
pub static SECRET_CSPRNG_ALIGN: usize =
    core::mem::align_of::<SecretRandomGenerator<SoftwareRandomGenerator>>();

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_construct_secret_csprng(mem: *mut SecCsprng, seed: Uint128) {
    let mem = mem as *mut SecretRandomGenerator<SoftwareRandomGenerator>;
    let seed = Seed(u128::from_le_bytes(seed.little_endian_bytes));
    mem.write(SecretRandomGenerator::new(seed));
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_destroy_secret_csprng(mem: *mut SecCsprng) {
    core::ptr::drop_in_place(mem as *mut SecretRandomGenerator<SoftwareRandomGenerator>);
}

#[no_mangle]
pub static ENCRYPTION_CSPRNG_SIZE: usize =
    core::mem::size_of::<EncryptionRandomGenerator<SoftwareRandomGenerator>>();

#[no_mangle]
pub static ENCRYPTION_CSPRNG_ALIGN: usize =
    core::mem::align_of::<EncryptionRandomGenerator<SoftwareRandomGenerator>>();

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_construct_encryption_csprng(
    mem: *mut EncCsprng,
    seed: Uint128,
) {
    let mem = mem as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>;
    let seed = Seed(u128::from_le_bytes(seed.little_endian_bytes));
    let mut boxed_seeder = new_dyn_seeder();
    let seeder = boxed_seeder.as_mut();
    mem.write(EncryptionRandomGenerator::new(seed, seeder));
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_destroy_encryption_csprng(mem: *mut EncCsprng) {
    core::ptr::drop_in_place(mem as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>);
}

// Randomly fill a uint128.
// Returns 1 if the random is crypto secure, -1 if it not secure, 0 if fail.
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_crypto_secure_random_128(u128: *mut Uint128) -> c_int {
    let buf = &mut (*u128).little_endian_bytes[0..16];

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if is_x86_feature_detected!("rdseed") {
        let mut rand: u64 = 0;
        loop {
            if core::arch::x86_64::_rdseed64_step(&mut rand) == 1 {
                buf[0..8].copy_from_slice(&rand.to_ne_bytes());
                break;
            }
        }
        loop {
            if core::arch::x86_64::_rdseed64_step(&mut rand) == 1 {
                buf[8..16].copy_from_slice(&rand.to_ne_bytes());
                break;
            }
        }
        return 1;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // SecRandomCopyBytes is available starting with Mac OS 10.7
        // https://developer.apple.com/documentation/security/1399291-secrandomcopybytes?language=objc
        // M1 processors started with Mac OS Big Sur 11
        pub enum __SecRandom {}
        pub type SecRandomRef = *const __SecRandom;

        #[link(name = "Security", kind = "framework")]
        extern "C" {
            pub static kSecRandomDefault: SecRandomRef;

            pub fn SecRandomCopyBytes(
                rnd: SecRandomRef,
                count: usize,
                bytes: *mut libc::c_void,
            ) -> c_int;
        }
        unsafe {
            let err = SecRandomCopyBytes(kSecRandomDefault, 16, buf.as_ptr() as *mut libc::c_void);
            if err == 0 {
                return 1;
            }
        }
    }
    if let Ok(mut random) = std::fs::File::open("/dev/random") {
        if let Ok(16) = random.read(buf) {
            return -1;
        }
    }

    0
}
