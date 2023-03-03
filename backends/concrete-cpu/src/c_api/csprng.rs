use std::io::Read;

use super::types::{Csprng, CsprngVtable, Uint128};
use concrete_csprng::{
    generators::{RandomGenerator, SoftwareRandomGenerator},
    seeders::Seed,
};
use libc::c_int;

type Generator = SoftwareRandomGenerator;

#[no_mangle]
pub static CONCRETE_CSPRNG_VTABLE: CsprngVtable = CsprngVtable {
    remaining_bytes: {
        unsafe extern "C" fn remaining_bytes(csprng: *const Csprng) -> Uint128 {
            let csprng = &*(csprng as *const Generator);
            Uint128 {
                little_endian_bytes: csprng.remaining_bytes().0.to_le_bytes(),
            }
        }

        remaining_bytes
    },
    next_bytes: {
        unsafe extern "C" fn next_bytes(
            csprng: *mut Csprng,
            byte_array: *mut u8,
            byte_count: usize,
        ) -> usize {
            let csprng = &mut *(csprng as *mut Generator);
            let mut count = 0;

            while count < byte_count {
                if let Some(byte) = csprng.next() {
                    *byte_array.add(count) = byte;
                    count += 1;
                } else {
                    break;
                };
            }

            count
        }

        next_bytes
    },
};

#[no_mangle]
pub static CONCRETE_CSPRNG_SIZE: usize = core::mem::size_of::<Generator>();

#[no_mangle]
pub static CONCRETE_CSPRNG_ALIGN: usize = core::mem::align_of::<Generator>();

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_construct_concrete_csprng(mem: *mut Csprng, seed: Uint128) {
    let mem = mem as *mut Generator;
    let seed = Seed(u128::from_le_bytes(seed.little_endian_bytes));
    mem.write(Generator::new(seed));
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_destroy_concrete_csprng(mem: *mut Csprng) {
    core::ptr::drop_in_place(mem as *mut Generator);
}

// Randomly fill a uint128.
// Returns 1 if the random is crypto secure, -1 if it not secure, 0 if fail.
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_crypto_secure_random_128(u128: *mut Uint128) -> c_int {
    if is_x86_feature_detected!("rdseed") {
        let mut rand: u64 = 0;
        loop {
            if core::arch::x86_64::_rdseed64_step(&mut rand) == 1 {
                (*u128).little_endian_bytes[0..8].copy_from_slice(&rand.to_ne_bytes());
                break;
            }
        }
        loop {
            if core::arch::x86_64::_rdseed64_step(&mut rand) == 1 {
                (*u128).little_endian_bytes[8..16].copy_from_slice(&rand.to_ne_bytes());
                break;
            }
        }
        return 1;
    }
    let buf = &mut (*u128).little_endian_bytes[0..16];
    if let Ok(mut random) = std::fs::File::open("/dev/random") {
        if let Ok(16) = random.read(buf) {
            return -1;
        }
    }

    0
}
