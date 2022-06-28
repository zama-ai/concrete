# How tos

(how-to-create-dynamic-types)=
## Create Dynamic Types

Creating a dynamic type is done by using the `add_*` methods of the [ConfigBuilder].

| Kind      | Builder method       |
|-----------|----------------------|
| booleans  | [add_bool_type]      |
| shortints | [add_short_int_type] |
| integers  | [add_integer_type]   |

These methods return and `instanciator`, which is the object you'll need
to use to create values of your new type.

Types created dynamically still benefits from overloaded operators

### Example

Creating a 10-bits integer by combining 5 2-bits shortints

```rust
// This requires both the integers and dynamic features enabled
#[cfg(feature = "integers")]
fn main() {
    use concrete::prelude::*;
    use concrete::{
        generate_keys, set_server_key, ConfigBuilder, DynIntegerParameters,
        FheUint2Parameters,
    };

    let mut config = ConfigBuilder::all_disabled();
    let uint10_type = config.add_integer_type(DynIntegerParameters {
        block_parameters: FheUint2Parameters::default().into(),
        num_block: 5,
    });

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let a = uint10_type.encrypt(177, &client_key);
    let b = uint10_type.encrypt(100, &client_key);

    let c: u64 = (a + b).decrypt(&client_key);
    assert_eq!(c, 277);
}
```

[ConfigBuilder]: https://zama.ai
[add_bool_type]: https://zama.ai
[add_short_int_type]: https://zama.ai
[add_integer_type]: https://zama.ai


## Write generic function that uses operators

In the {ref}`overloaded-operators` section, we explained that operators
are overloaded to work with references and non references.

If you wish to write generic functions which uses operators
with mixed reference and non reference it might get tricky at first to specify the trait [bounds],
this page should serve as a _cook book_ to help you.

| operation   | trait bound                            |
|-------------|----------------------------------------|
| `T $op T`   | `T: $Op<T, Output=T>`                  |
| `T $op &T`  | `T: for<'a> $Op<&'a T, Output=T>`      |
| `&T $op T`  | `for<'a> &'a T:  $Op<T, Output=T>`     |
| `&T $op &T` | `for<'a> &'a T:  $Op<&'a T, Output=T>` |

```{note}
The `for<'a>` syntax is something called [Higher-Rank Trait Bounds], often shortened as __HRTB__
```

### Example:

```rust
use core::ops::Add;

/// This function can be called with both FHE types and native types
fn compute_stuff<T>(a: T, b: T) -> T
where T: Add<T, Output=T>,
      T: for <'a> Add<&'a T, Output=T>,
      for <'a> &'a T: Add<T, Output=T> + Add<&'a T, Output=T>
{
    let c = &a + &b;
    
    c + &b
}


fn main() {
    let result = compute_stuff(0u32, 1u32);
    println!("result: {}", result);
}
```

[bounds]: https://doc.rust-lang.org/rust-by-example/generics/bounds.html
[Higher-Rank Trait Bounds]: https://doc.rust-lang.org/nomicon/hrtb.html


