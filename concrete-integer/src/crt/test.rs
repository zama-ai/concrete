use crate::crt::gen_keys_from_basis_and_carry_space;
use rand::Rng;

#[test]
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

    let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear1, &basis, &vec_key_id);
    let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear2, &basis, &vec_key_id);

    println!(
        "(Input Size {}; Carry_Space {:?}, Message_Space {},):  \
                    Unchecked Mul\
                     + \
                    Full \
                Propagate ",
        msg_space, carry_space, msg_space,
    );

    sks.unchecked_mul_crt_many_keys_assign(&mut ctxt_1, &mut ctxt_2);
    assert_eq!(
        cks.decrypt_crt_several_keys(&ctxt_1),
        clear1 * clear2 % msg_space
    );
}
