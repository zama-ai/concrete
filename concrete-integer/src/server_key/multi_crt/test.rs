use crate::ciphertext::KeyId;
use crate::client_key::multi_crt::gen_key_id;
use crate::keycache::KEY_CACHE;
use crate::{CrtMultiClientKey, CrtMultiServerKey};
use concrete_shortint::parameters::get_parameters_from_message_and_carry;
use rand::Rng;

/// Generates automatically the appropriate keys regarding the input basis.
/// # Example
///
/// ```rust
/// use concrete_integer::crt::gen_keys_from_basis_and_carry_space;
/// let basis: Vec<u64> = vec![2, 3, 5];
/// let carry_space = vec![4, 4, 4];
/// let (cks, sks) = gen_keys_from_basis_and_carry_space(&basis, &carry_space);
/// ```
pub fn gen_keys_from_basis_and_carry_space(
    basis: &[u64],
    carry: &[u64],
) -> (CrtMultiClientKey, CrtMultiServerKey, Vec<KeyId>) {
    let mut vec_param = vec![];
    let mut vec_id = vec![];

    for ((i, base), car) in basis.iter().enumerate().zip(carry) {
        let tmp_param = get_parameters_from_message_and_carry(*base as usize, *car as usize);
        let tmp_param_exists = vec_param.iter().find(|&&x| x == tmp_param);
        if tmp_param_exists != None {
            vec_id.push(vec_param.iter().position(|&x| x == tmp_param).unwrap());
        } else {
            vec_param.push(get_parameters_from_message_and_carry(
                *base as usize,
                *car as usize,
            ));
            vec_id.push(i);
        }
    }
    let vec_key_id = gen_key_id(&vec_id);

    let mut vec_sks = vec![];
    let mut vec_cks = vec![];
    for param in vec_param.iter() {
        let (cks_shortint, sks_shortint) = KEY_CACHE.get_shortint_from_params(*param);
        vec_sks.push(sks_shortint);
        vec_cks.push(cks_shortint);
    }

    (
        CrtMultiClientKey::from(vec_cks),
        CrtMultiServerKey::from(vec_sks),
        vec_key_id,
    )
}

#[test]
#[ignore]
// see internal issue #527
pub fn test_crt() {
    let basis: Vec<u64> = vec![2, 3, 5];
    let carry_space = vec![4, 4, 4];

    let (cks, sks, vec_key_id) = gen_keys_from_basis_and_carry_space(&basis, &carry_space);

    let mut rng = rand::thread_rng();

    // Define the cleartexts
    let mut msg_space = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let clear1 = rng.gen::<u64>() % msg_space as u64;
    let clear2 = rng.gen::<u64>() % msg_space as u64;

    // Encrypt the integers

    let mut ctxt_1 = cks.encrypt(&clear1, &basis, &vec_key_id);
    let mut ctxt_2 = cks.encrypt(&clear2, &basis, &vec_key_id);

    println!(
        "(Input Size {}; Carry_Space {:?}, Message_Space {},):  \
                    Unchecked Mul\
                     + \
                    Full \
                Propagate ",
        msg_space, carry_space, msg_space,
    );

    sks.unchecked_mul_crt_many_keys_assign(&mut ctxt_1, &mut ctxt_2);
    assert_eq!(cks.decrypt(&ctxt_1), clear1 * clear2 % msg_space);
}
