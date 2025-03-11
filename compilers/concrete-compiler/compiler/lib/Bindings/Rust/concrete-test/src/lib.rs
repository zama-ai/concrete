use concrete_macro::from_concrete_python_export_zip;

from_concrete_python_export_zip!("src/test.zip");

#[cfg(test)]
mod test {

    #[test]
    fn test() {
        let a = unsafe{super::my_c_function(3)};
    }
}
