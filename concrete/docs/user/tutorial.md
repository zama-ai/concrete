# Tutorial

This tutorial will guide you through the steps necessary to correctly use
``concrete``.

## 1. Enabling the cargo features you will need

The first step happens in the ``Cargo.toml`` file.

As explained in the {ref}`choosing-your-features` you need to add ``concrete``
as a dependency in your project's ``Cargo.toml`` and enable the features which exposes the type kind you may need.


## 2. Configuring and creating keys

The second steps happens in your Rust code, you will need to create a configuration.

The configuration is used to declare which type you will use or not use, as well as enabling you
to use custom crypto parameters for these types for more advanced usage / testing.

Creating a configuration is done using the [ConfigBuilder] type.

Let's say we are interested in using 8-bit unsigned integers with default parameters.
As per the table in the getting started page, we need to enable the ``integers`` feature.


```toml
concrete = { version = "0.1.0", features = ["integers"]}
```

Then, in our code we create a config by first creating a builder with all types deactivated,
and then we enable the `uint8` type with default parameters.

```rust
use concrete::{ConfigBuilder, generate_keys};

fn main() {
    let config = ConfigBuilder::all_disabled()
        .enable_default_uint8()
        .build();
    
    let (_client_key, _server_key) = generate_keys(config);
}
```


The [generate_keys] returns a client key and a server key.

As the names try to convey, the ``client_key`` is meant to stay private and not leave the client whereas
the ``server_key`` can be made public and sent to a server for it to be able to do FHE computations.


[ConfigBuilder]: https://zama.ai
[generate_keys]: https://zama.ai


## 3. Setting the server key

The next step is to call [set_server_key]

This function will __move__ the server key to an internal state of the crate.
Allowing us to manage the details to be able to give you, the user, a simpler interface.


```{code-block} Rust
 :emphasize-lines: 1, 8, 10

use concrete::{ConfigBuilder, generate_keys, set_server_key};

fn main() {
    let config = ConfigBuilder::all_disabled()
        .enable_default_uint8()
        .build();

    let (_client_key, server_key) = generate_keys(config);
    
    set_server_key(server_key);
}
```


[set_server_key]: https://zama.ai



## 4. Encrypting, Computing, Decrypting

We are now ready to start creating some values and doing homomorphic computations on them.

``concrete`` uses ``traits`` to have a consistent API for creating FHE types and enable users
to write generic functions. However, to be able to use associated functions and methods of a trait,
the trait has to be in scope.

To make it easier for users we use the ``prelude`` 'pattern', that is,
all ``concrete`` important traits are in a `prelude` module that
you __glob import__. With this, no need to remember or know the traits to import.

```{code-block} Rust
    :caption: The prelude glob-import

use concrete::prelude::*;
```

```{code-block} Rust
    :emphasize-lines: 1, 2, 9

use concrete::{ConfigBuilder, generate_keys, set_server_key, FheUint8};
use concrete::prelude::*;

fn main() {
    let config = ConfigBuilder::all_disabled()
        .enable_default_uint8()
        .build();

    let (client_key, server_key) = generate_keys(config);
    
    set_server_key(server_key);
    
    let clear_a = 27u8;
    let clear_b = 128u8;
    
    let a = FheUint8::new(clear_a, &mut client_key);
    let b = FheUint8::new(clear_b, &mut client_key);
    
    let result = a + b;
    
    let decrypted_result = result.to(&mut client_key);
    let clear_result = clear_a + clear_b;
    
    assert_eq!(decrypted_result, clear_result);
}
```
