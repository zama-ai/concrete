mod u32;
mod u64;
mod variance;

#[repr(C)]
pub struct FfiArray<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> FfiArray<T>
where
    T: Copy,
{
    fn allocate(val: T, size: usize) -> FfiArray<T> {
        // We create a new vec.
        let inner_vec = vec![val; size];
        // We prevent the vec from being dropped automatically.
        let mut inner_vec = std::mem::ManuallyDrop::new(inner_vec);
        // We retrieve the pointer data
        let ptr = inner_vec.as_mut_ptr();
        let len = inner_vec.len();
        let cap = inner_vec.capacity();
        FfiArray { ptr, len, cap }
    }

    fn free(self) {
        // We retrieve the original array to drop it.
        let FfiArray { ptr, len, cap } = self;
        let _vec = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        // drop the newly created vec.
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
