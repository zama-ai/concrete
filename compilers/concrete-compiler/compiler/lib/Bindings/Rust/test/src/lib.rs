use concrete_macro::from_concrete_python_export_zip;

from_concrete_python_export_zip!("src/test.zip");

#[cfg(test)]
mod test {

    #[test]
    fn test() {
        let a = unsafe{super::concrete_dec(8 as *mut u8 as *mut std::ffi::c_void)};
    }
}
