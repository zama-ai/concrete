use concrete_macro::{from_concrete_python_export_zip, testtt};

// from_concrete_python_export_zip!("test.zip");
testtt!();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
