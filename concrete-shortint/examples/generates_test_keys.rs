// use concrete_shortint::gen_keys;
// use concrete_shortint::parameters::ALL_PARAMETER_VEC;
// use std::fs::File;
// use std::io::BufWriter;
// use std::thread;
// use concrete_shortint::parameters::parameters_wopbs_message_carry::ALL_PARAMETER_VEC_WOPBS;
// use concrete_shortint::server_key::keycache::KEY_CACHE_WOPBS;
// use concrete_shortint::wopbs::{WopbsKey, WopbsKey_v0};
//
// pub fn server_key() {
//     let mut th_vec = Vec::with_capacity(ALL_PARAMETER_VEC.len());
//
//     let start = 0;
//     let end = ALL_PARAMETER_VEC.len() - 1;
//     for (i, params) in ALL_PARAMETER_VEC[start..end].iter().copied().enumerate() {
//         let th = thread::spawn(move || {
//             let i = start + i;
//             let keys = gen_keys(params);
//             let filepath = format!("tests/keys/keys_{}.bin", i);
//             let file = BufWriter::new(File::create(filepath).unwrap());
//             bincode::serialize_into(file, &keys).unwrap();
//         });
//         th_vec.push(th);
//     }
//     for (i, th) in th_vec.into_iter().enumerate() {
//         let res = th.join();
//         println!("Thread for key number {} terminated", i);
//         match res {
//             Ok(_) => println!("All is well!"),
//             Err(e) => {
//                 if let Some(e) = e.downcast_ref::<&'static str>() {
//                     println!("Got an error: {}", e);
//                 } else {
//                     println!("Got an unknown error: {:?}", e);
//                 }
//             }
//         }
//     }
// }
//
// pub fn wopbs_key() {
//     let mut th_vec = Vec::with_capacity(ALL_PARAMETER_VEC_WOPBS.len());
//
//     let start = 0;
//     let end = ALL_PARAMETER_VEC_WOPBS.len() - 1;
//     for (i, params) in ALL_PARAMETER_VEC_WOPBS[start..end].iter().copied().enumerate() {
//         let th = thread::spawn(move || {
//             let i = start + i;
//             let (cks, sks) = KEY_CACHE_WOPBS.get_client_and_server_key_wopbs(i);
//             let wopbs_keys = WopbsKey_v0::new_wopbs_key(&cks, &sks);
//             let filepath = format!("tests/wopbskeys/wopbskeys_{}.bin", i);
//             let file = BufWriter::new(File::create(filepath).unwrap());
//             bincode::serialize_into(file, &wopbs_keys).unwrap();
//         });
//         th_vec.push(th);
//     }
//     for (i, th) in th_vec.into_iter().enumerate() {
//         let res = th.join();
//         println!("Thread for key number {} terminated", i);
//         match res {
//             Ok(_) => println!("All is well!"),
//             Err(e) => {
//                 if let Some(e) = e.downcast_ref::<&'static str>() {
//                     println!("Got an error: {}", e);
//                 } else {
//                     println!("Got an unknown error: {:?}", e);
//                 }
//             }
//         }
//     }
// }
//
// pub fn server_key_wopbs() {
//     let mut th_vec = Vec::with_capacity(ALL_PARAMETER_VEC_WOPBS.len());
//
//     let start = 0;
//     let end = ALL_PARAMETER_VEC_WOPBS.len() - 1;
//     for (i, params) in ALL_PARAMETER_VEC_WOPBS[start..end].iter().copied().enumerate() {
//         let th = thread::spawn(move || {
//             let i = start + i;
//             let keys = gen_keys(params);
//             let filepath = format!("tests/wopbskeys/keys_{}.bin", i);
//             let file = BufWriter::new(File::create(filepath).unwrap());
//             bincode::serialize_into(file, &keys).unwrap();
//         });
//         th_vec.push(th);
//     }
//     for (i, th) in th_vec.into_iter().enumerate() {
//         let res = th.join();
//         println!("Thread for key number {} terminated", i);
//         match res {
//             Ok(_) => println!("All is well!"),
//             Err(e) => {
//                 if let Some(e) = e.downcast_ref::<&'static str>() {
//                     println!("Got an error: {}", e);
//                 } else {
//                     println!("Got an unknown error: {:?}", e);
//                 }
//             }
//         }
//     }
// }
//
//
// fn main() {
//     //server_key();
//     //server_key_wopbs();
//     wopbs_key();
// }

fn main() {}
