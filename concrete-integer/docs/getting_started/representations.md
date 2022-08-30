# How Integers are represented


In `concrete-integer`, the encrypted data is split amongst many ciphertexts
encrypted using the `concrete-shortint` library.

This crate implements two ways to represent an integer:
  - the Radix representation
  - the CRT (Chinese Reminder Theorem) representation

## Radix based Integers
The first possibility to represent a large integer is to use a radix-based decomposition on the 
plaintexts. Let $$B \in \mathbb{N}$$ be a basis such that the size of $$B$$ is smaller (or equal)
to four bits. 
Then, an integer $$m \in \mathbb{N}$$ can be written as $$m = m_0 + m_1*B + m_2*B^2 + ... $$, where 
each $$m_i$$ is strictly smaller than $$B$$. Each $$m_i$$ is then independently encrypted. In 
the end, an Integer ciphertext is defined as a set of Shortint ciphertexts. 

In practice, the definition of an Integer requires the basis and the number of blocks. This is 
done at the key creation step.
```rust
use concrete_integer::gen_keys;
use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let num_block = 4;
    let (client_key, server_key) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, num_block);
}
```

In this example, the keys are dedicated to Integers decomposed as four blocks using the basis 
$$B=2^2$$. Otherwise said, they allow to work on Integers modulus $$(2^2)^4 = 2^8$$. 


In this representation, the correctness of operations requires to propagate the carries 
between the ciphertext. This operation is costly since it relies on the computation of many 
programmable bootstrapping over Shortints.

```rust
use concrete_integer::gen_keys;
use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let num_block = 4;
    let (client_key, server_key) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, num_block);

    let msg1 = 128;
    let msg2 = 13;

    // message_modulus^vec_length
    let modulus = client_key.parameters().message_modulus.0.pow(num_block as u32) as u64;
    
    // We use the client key to encrypt two messages:
    let ct_1 = client_key.encrypt(msg1);
    let ct_2 = client_key.encrypt(msg2);
    
    // We use the server public key to execute an integer circuit:
    let ct_3 = server_key.unchecked_add(&ct_1, &ct_2);
    
    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_3);

    assert_eq!(output, (msg1 + msg2) % modulus);
}
```


## CRT based Integers
The second approach to represent large integers is based on the Chinese Remainder Theorem. 
In this cases, the basis $$B$$ is composed of several integers $$b_i$$, such that there are 
pairwise coprime, and each b_i has a size smaller than four bits. Then, the Integer will be 
defined modulus $$\prod b_i$$. For an integer $$m$$, its CRT decomposition is simply defined as 
$$m % b_0, m % b_1, ...$$. Each part is then encrypted as a Shortint ciphertext. In
the end, an Integer ciphertext is defined as a set of Shortint ciphertexts.

An example of such a basis 
could be $$B = [2, 3, 5]$$. This means that the Integer is defined modulus $$2*3*5 = 30$$. 

This representation has many advantages: no carry propagation is required, so that only cleaning 
the carry buffer of each ciphertexts is enough. This implies that operations can easily be 
parallelized. Moreover, it allows to efficiently compute PBS in the case where the function is 
CRT compliant. 

A variant of the CRT is proposed, where each block might be associated to a different key couple. 
In the end, a keychain is required to the computations, but performance might be improved.

```rust
use concrete_integer::crt::{gen_key_id, gen_several_keys};
use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1};

fn main() {
    // Generate the client key and the server key:
    let (cks, sks) = gen_several_keys(&vec![PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1]);
    
    let clear_1 = 14;
    let clear_2 = 11;
    let basis = vec![2, 3, 7];
    let keys_id = gen_key_id(&vec![0, 0, 1]);
    // Encrypt two messages
    let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    
    // Compute homomorphically a multiplication
    sks.unchecked_add_crt_many_keys_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    // Decrypt
    let res = cks.decrypt_crt_several_keys(&ctxt_1);
    assert_eq!((clear_1 + clear_2) % 30, res);
}
```


## Differences between the two representations

Because of the propagation of the carries, the radix representation is generally slower than the CRT representation.
However the CRT representation is limited in precision by the prime numbers chosen while the radix can offer ever larger 
precision by adding blocks.
