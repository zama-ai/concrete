# Concrete Boolean

This library makes it possible to execute boolean gates over encrypted bits. 
It allows to execute a boolean circuit on an untrusted server because both circuit inputs and outputs are kept private.
Data are indeed encrypted on the client side, before being sent to the server.
On the server side every computation is performed on ciphertexts.
The server however has to know the boolean circuit to be evaluated.
At the end of the computation, the server returns the encryption of the result to the user.

## Quick Example
 The following piece of code shows how to generate keys and run a small Boolean circuit
 homomorphically.

 ```Rust
use concrete_boolean::prelude::*;

fn main() {
    // We generate a set of client/server keys, using the default parameters:
    let (mut client_key, mut server_key) = gen_keys();

    // We use the client secret key to encrypt two messages:
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(false);

    // We use the server public key to execute a boolean circuit:
    // if ((NOT ct_2) NAND (ct_1 AND ct_2)) then (NOT ct_2) else (ct_1 AND ct_2)
    let ct_3 = server_key.not(&ct_2);
    let ct_4 = server_key.and(&ct_1, &ct_2);
    let ct_5 = server_key.nand(&ct_3, &ct_4);
    let ct_6 = server_key.mux(&ct_5, &ct_3, &ct_4);

    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_6);
    assert_eq!(output, true)
}
 ```

## Links

- [documentation](https://docs.zama.ai/concrete/boolean-lib)
- [TFHE's gate bootstrapping](https://eprint.iacr.org/2018/421.pdf)

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions,
please contact us at `hello@zama.ai`.
