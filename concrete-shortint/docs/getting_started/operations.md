# Types of operations

The operations available via a `ServerKey` may come in different variants:

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
  will clear the carry modulus to make the operation possible.

Not all operations have these 3 flavors, as some of them are implemented in a way that the operation
is always possible without ever exceeding the plaintext's space capacity.


# List of available operations

`concrete-shortint` comes with a set of already implemented functions:

## Additions
- addition between two ciphertexts
- addition between a ciphertext and an unencrypted scalar

```rust
use concrete_shortint::{gen_keys, Parameters};

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let (client_key, server_key) = gen_keys(Parameters::default());

    let msg1 = 1;
    let msg2 = 0;
    
    let modulus = client_key.parameters.message_modulus.0;

    // We use the client key to encrypt two messages:
    let ct_1 = client_key.encrypt(msg1);
    let ct_2 = client_key.encrypt(msg2);
    
    // We use the server public key to execute an integer circuit:
    let ct_3 = server_key.unchecked_add(&ct_1, &ct_2);
    
    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_3);
    assert_eq!(output, (msg1 + msg2) % modulus as u64);
}
```

## Comparisons
- comparisons `<`, `<=`, `>`, `>=`, `==` between a ciphertext and an unencrypted scalar 
- comparisons `<`, `<=`, `>`, `>=`, `==` between two ciphertexts (*)

```rust
use concrete_shortint::{gen_keys, Parameters};

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let (client_key, server_key) = gen_keys(Parameters::default());

    let msg1 = 1;
    let msg2 = 0;
    
    let modulus = client_key.parameters.message_modulus.0;

    // We use the client key to encrypt two messages:
    let ct_1 = client_key.encrypt(msg1);
    let ct_2 = client_key.encrypt(msg2);
    
    // We use the server public key to execute an integer circuit:
    let ct_3 = server_key.unchecked_greater(&ct_1, &ct_2);
    
    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_3);
    assert_eq!(output, (msg1 + msg2) % modulus as u64);
}
```

## Divisions
- division of a ciphertext by an unencrypted scalar
- division between two ciphertexts (*)

```rust
use concrete_shortint::{gen_keys, Parameters};

fn main() {
    use concrete_shortint::{gen_keys, Parameters};
    
    // Generate the client key and the server key
    let (cks, sks) = gen_keys(Parameters::default());
    
    let clear_1 = 1;
    let clear_2 = 2;
    
    // Encrypt two messages
    let ct_1 = cks.encrypt(clear_1);
    let ct_2 = cks.encrypt(clear_2);
    
    // Compute homomorphically a multiplication
    let ct_res = sks.unchecked_div(&ct_1, &ct_2);
    
    // Decrypt
    let res = cks.decrypt(&ct_res);
    assert_eq!(clear_1 / clear_2, res);
}
```

## Multiplications
- LSB multiplication between two ciphertexts returning the result truncated to fit in the `message buffer`
- multiplication of a ciphertext by an unencrypted scalar
- MSB multiplication between two ciphertexts returning the part overflowing the `message buffer` (*)

```rust
use concrete_shortint::{gen_keys, Parameters};

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let (cks, sks) = gen_keys(Parameters::default());

    let clear_1 = 1;
    let clear_2 = 2;
    
    // Encrypt two messages
    let ct_1 = cks.encrypt(clear_1);
    let ct_2 = cks.encrypt(clear_2);
    
    // Compute homomorphically a multiplication
    let ct_res = sks.unchecked_mul_lsb(&ct_1, &ct_2);
    
    // Decrypt
    let res = cks.decrypt(&ct_res);
    let modulus = cks.parameters.message_modulus.0 as u64;
    assert_eq!((clear_1 * clear_2) % modulus, res);
}
```

## Bitwise operations
- bitwise and, or and xor (*)
- bitwise shift `<<`, `>>`

```rust
use concrete_shortint::{gen_keys, Parameters};

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let (cks, sks) = gen_keys(Parameters::default());

    let clear_1 = 2;
    let clear_2 = 1;
    
    let ct_1 = cks.encrypt(clear_1);
    let ct_2 = cks.encrypt(clear_2);
    
    let ct_res = sks.unchecked_bitand(&ct_1, &ct_2);
    
    let res = cks.decrypt(&ct_res);
    assert_eq!(clear_1 & clear_2, res);
}
```

## Negation/Subtractions
- negation of a ciphertext
- subtraction of a ciphertext by another ciphertext
- subtraction of a ciphertext by an unencrypted scalar

```rust
use concrete_shortint::{gen_keys, Parameters};

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let (cks, sks) = gen_keys(Parameters::default());

    let msg_1 = 2;
    let msg_2 = 1;
    
    // Encrypt two messages:
    let ct_1 = cks.encrypt(msg_1);
    let ct_2 = cks.encrypt(msg_2);
    
    // Compute homomorphically a subtraction:
    let ct_res = sks.unchecked_sub(&ct_1, &ct_2);
    
    // Decrypt:
    let modulus = cks.parameters.message_modulus.0 as u64;
    assert_eq!(cks.decrypt(&ct_res), msg_1 - msg_2);
}
```


{% hint style="warning" %}

Currently, certain operations can only be used if the parameter set chosen is compatible with the
bivariate programmable bootstrapping, meaning the carry buffer is larger or equal than the
message buffer. These operations are marked with a star (*).

{% endhint %}


