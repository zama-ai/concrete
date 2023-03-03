use crate::c_api::types::{Csprng, CsprngVtable};
use core::marker::PhantomData;

pub struct CsprngMut<'value, 'vtable: 'value> {
    ptr: *mut Csprng,
    vtable: &'vtable CsprngVtable,
    __marker: PhantomData<&'value mut ()>,
}

impl<'value, 'vtable: 'value> CsprngMut<'value, 'vtable> {
    #[inline]
    pub unsafe fn new(ptr: *mut Csprng, vtable: *const CsprngVtable) -> Self {
        Self {
            ptr,
            vtable: &*vtable,
            __marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_mut<'this>(&'this mut self) -> CsprngMut<'this, 'vtable> {
        Self {
            ptr: self.ptr,
            vtable: self.vtable,
            __marker: PhantomData,
        }
    }

    #[inline]
    pub fn next_bytes(&self, slice: &mut [u8]) -> usize {
        let byte_count = slice.len();
        unsafe { (self.vtable.next_bytes)(self.ptr, slice.as_mut_ptr(), byte_count) }
    }
}
