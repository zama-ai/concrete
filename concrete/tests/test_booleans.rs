#![cfg(feature = "booleans")]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::assign_op_pattern)]
use concrete::prelude::*;
use concrete::{generate_keys, set_server_key, ConfigBuilder, FheBool};

#[test]
fn test_and() {
    let config = ConfigBuilder::all_disabled().enable_default_bool().build();
    let (my_keys, server_keys) = generate_keys(config);
    set_server_key(server_keys);

    let a = FheBool::encrypt(true, &my_keys);
    let b = FheBool::encrypt(false, &my_keys);

    let c = a & b;
    let clear_res = c.decrypt(&my_keys);
    assert_eq!(clear_res, false);
}

#[cfg(feature = "experimental_syntax_sugar")]
mod branch_macro_tests {
    use concrete::prelude::*;
    use concrete::{branch, condition, generate_keys, set_server_key, ConfigBuilder, FheBool};
    use std::thread;

    #[test]
    fn test_branch_macro_1() {
        let config = ConfigBuilder::all_disabled().enable_default_bool().build();
        let (my_keys, server_keys) = generate_keys(config);
        set_server_key(server_keys);

        let a = FheBool::encrypt(true, &my_keys);
        let b = FheBool::encrypt(false, &my_keys);
        let c = FheBool::encrypt(true, &my_keys);

        let res = branch! {
            if a {
                let d = FheBool::encrypt(true, &my_keys);
                b & d
            } else {
                let d = FheBool::encrypt(false, &my_keys);
                c | d
            }
        };

        let clear_res = res.decrypt(&my_keys);

        assert_eq!(clear_res, false);
    }

    #[test]
    fn test_branch_macro_2() {
        let config = ConfigBuilder::all_disabled().enable_default_bool().build();
        let (my_keys, server_keys) = generate_keys(config);
        set_server_key(server_keys);

        let a = FheBool::encrypt(true, &my_keys);
        let b = FheBool::encrypt(false, &my_keys);
        let c = FheBool::encrypt(true, &my_keys);
        let d = b.clone();

        let res = branch!(if a == b { d } else { c });

        let clear_res = res.decrypt(&my_keys);
        assert_eq!(clear_res, true);
    }

    #[test]
    fn test_branch_macro_3() {
        let config = ConfigBuilder::all_disabled().enable_default_bool().build();
        let (my_keys, server_keys) = generate_keys(config);
        set_server_key(server_keys);

        let a = FheBool::encrypt(true, &my_keys);
        let b = FheBool::encrypt(false, &my_keys);
        let c = FheBool::encrypt(true, &my_keys);

        let res = branch! {
            if a {
                b
            } else {
                c
            }
        };
        let clear_res = res.decrypt(&my_keys);
        assert_eq!(clear_res, false);
    }

    #[test]
    fn test_branch_macro_4() {
        let config = ConfigBuilder::all_disabled().enable_default_bool().build();
        let (my_keys, server_keys) = generate_keys(config);
        set_server_key(server_keys);

        let a = FheBool::encrypt(true, &my_keys);
        let b = FheBool::encrypt(false, &my_keys);
        let c = FheBool::encrypt(true, &my_keys);
        let d = FheBool::encrypt(true, &my_keys);
        let e = FheBool::encrypt(false, &my_keys);

        let res = branch! {
            if (a == b) && c {
                d
            } else {
                e
            }
        };
        assert_eq!(res.decrypt(&my_keys), false);
    }

    #[test]
    fn test_threads() {
        fn handle(r: thread::Result<()>) {
            match r {
                Ok(r) => println!("All is well! {:?}", r),
                Err(e) => {
                    if let Some(e) = e.downcast_ref::<&'static str>() {
                        println!("Got an error: {}", e);
                    } else {
                        println!("Got an unknown error: {:?}", e);
                    }
                }
            }
        }

        // We could have use thead::spawn for shortness,
        // but we wanted to give the threads name
        let th1 = thread::Builder::new()
            .name("Computation 1".to_string())
            .spawn(move || {
                let config = ConfigBuilder::all_disabled().enable_default_bool().build();
                let (my_keys, server_keys) = generate_keys(config);
                set_server_key(server_keys);

                let a = FheBool::encrypt(true, &my_keys);
                let b = FheBool::encrypt(false, &my_keys);
                let c = FheBool::encrypt(true, &my_keys);

                let res = branch! {
                    if a {
                        let d = FheBool::encrypt(true, &my_keys);
                        b & d
                    } else {
                        let d = FheBool::encrypt(false, &my_keys);
                        c & d
                    }
                };

                let clear_res = res.decrypt(&my_keys);

                assert_eq!(clear_res, false);
            })
            .expect("Failed to start a new thread");

        let th2 = thread::Builder::new()
            .name("Computation 2".to_string())
            .spawn(move || {
                let config = ConfigBuilder::all_disabled().enable_default_bool().build();
                let (my_keys, server_keys) = generate_keys(config);
                set_server_key(server_keys);

                let a = FheBool::encrypt(true, &my_keys);
                let b = FheBool::encrypt(false, &my_keys);
                let c = FheBool::encrypt(true, &my_keys);
                let d = b.clone();

                let res = branch!(if a == b { d } else { c });

                let clear_res = res.decrypt(&my_keys);
                assert_eq!(clear_res, true);
            })
            .expect("Failed to start a new thread");

        let res1 = th1.join();
        let res2 = th2.join();

        handle(res1);
        handle(res2);
    }
}
