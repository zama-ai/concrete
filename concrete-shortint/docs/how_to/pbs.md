# The programmable bootstrapping

In `concrete-shortint`, the user can evaluate any function on an encrypted ciphertext. To do so the user must first
create a look-up table through the `generate_accumulator` method and use its output together with the ciphertext 
as parameters for the `programmable bootstrapping` (PBS).


## Univariate functions

For univariate function the PBS can be applied directly to a ciphertext in order to get the encrypted evaluation of
a function.

```rust
use concrete_shortint::gen_keys;
use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;

fn main() {
    // Generate the client key and the server key:
    let (cks, sks) = gen_keys(PARAM_MESSAGE_2_CARRY_2);

    let msg: u64 = 1;
    let ct = cks.encrypt(msg);
    let modulus = cks.parameters.message_modulus.0 as u64;

    // Generate the accumulator for the function f: x -> x^3 mod 2^2
    let acc = sks.generate_accumulator(|x| (x * x * x) % modulus);
    let ct_res = sks.keyswitch_programmable_bootstrap(&ct, &acc);

    let dec = cks.decrypt(&ct_res);

    assert_eq!(dec, (msg * msg * msg) % modulus);
}
```

## Bivariate functions

### PBS
To evaluate bivariate functions, the PBS is applied on a ciphertext that is the concatenation of two ciphertexts.

```rust
use concrete_shortint::gen_keys;
use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
use concrete_shortint::ciphertext::Degree;

fn main() {
    // Generate the client key and the server key:
    let (cks, sks) = gen_keys(PARAM_MESSAGE_2_CARRY_2);

    let msg_left: u64 = 1;
    let msg_right: u64 = 2;
    let mut ct_left = cks.encrypt(msg_left);
    let ct_right = cks.encrypt(msg_right);
    
    let modulus = (ct_right.degree.0 + 1) as u64;
    let deg = ct_left.degree.0 * (ct_right.degree.0+1);

    // Message left is shifted to the carry bits
    sks.unchecked_scalar_mul_assign(&mut ct_left, modulus as u8);

    // Message right is placed in the message bits
    sks.unchecked_add_assign(&mut ct_left, &ct_right);

    // Generate the accumulator for the multiplication
    let acc = sks.generate_accumulator(|x| {
        (x / modulus) * ((x % modulus)  + 1)
    });

    sks.keyswitch_programmable_bootstrap_assign(&mut ct_left, &acc);

    ct_left.degree = Degree(deg);

    let dec = cks.decrypt(&ct_left);

    assert_eq!(dec, msg_left * (msg_right+1));
}
```