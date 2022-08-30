# Types of operations

Much like `concrete-shortint`, the operations available via a `ServerKey` may come in different variants:

- operations that take their inputs as encrypted values.
- scalar operations take at least one non-encrypted value as input.

For example, the addition has both variants:

- `ServerKey::unchecked_add` which takes two encrypted values and adds them.
- `ServerKey::unchecked_scalar_add` which takes an encrypted value and a clear value (the
  so-called scalar) and adds them.

Each operation may come in different 'flavors':

- `unchecked`: Always does the operation, without checking if the result may exceed the capacity of
  the plaintext space.
- `checked`: Checks are done before computing the operation, returning an error if operation
  cannot be done safely.
- `smart`: Always does the operation, if the operation cannot be computed safely, the smart operation
  will propagate the carry buffer to make the operation possible.

Not all operations have these 3 flavors, as some of them are implemented in a way that the operation
is always possible without ever exceeding the plaintext space capacity.


# List of available operations

`concrete-integer` comes with a set of already implemented functions:

## Additions
- addition between two ciphertexts
- addition between a ciphertext and an unencrypted scalar

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

## Multiplications
- multiplication of a ciphertext by an unencrypted scalar
- multiplication between two ciphertexts

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
    let ct_3 = server_key.unchecked_mul(&ct_1, &ct_2);
    
    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_3);

    assert_eq!(output, (msg1 * msg2) % modulus);
}
```

## Bitwise operations
- bitwise shift `<<`, `>>`
- bitwise and, or and xor

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
    let ct_3 = server_key.unchecked_bitand(&ct_1, &ct_2);
    
    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_3);

    assert_eq!(output, msg1 & msg2);
}
```

## Negation/Subtractions
- negation of a ciphertext
- subtraction of a ciphertext by another ciphertext
- subtraction of a ciphertext by an unencrypted scalar

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
    let ct_3 = server_key.unchecked_sub(&ct_1, &ct_2);
    
    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_3);

    assert_eq!(output, (msg1 - msg2) % modulus);
}
```


