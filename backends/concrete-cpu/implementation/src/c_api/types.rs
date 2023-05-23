#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uint128 {
    pub little_endian_bytes: [u8; 16],
}

impl core::fmt::Debug for Uint128 {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        u128::from_le_bytes(self.little_endian_bytes).fmt(f)
    }
}

pub struct Csprng {
    __private: (),
}

pub struct SecCsprng {
    __private: (),
}

pub struct EncCsprng {
    __private: (),
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum ScratchStatus {
    Valid = 0,
    SizeOverflow = 1,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum Parallelism {
    No = 0,
    Rayon = 1,
}
