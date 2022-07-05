# Limitations

## Control flow / branching

Due to their nature, branching operations like `if else` statements are not possible.

## Rust move semantic and ownership.

The different types exposed by this crate overloads operators.

There are two types of operators, **binary** operators and **unary** operators. Binary operators like `+` work with two values, while unary operators like `!` work with one value.

As FHE types exposed by this crate are not [Copy](https://doc.rust-lang.org/std/marker/trait.Copy.html) (since they are bigger than native types and contain data on the heap), the operators (whether binary or unary) are overloaded on both owned values (`T`) and references / borrowed values (`&T`) to eliminate the need to `clone` the values each time they are used.

```Rust
let a: FheUint2 = ..
let b: FheUint2 = ..

&a + &b // works, no values moved
&a + b // works, a not moved, b moved
a + &b // works, a moved, b not moved
a + b // works, a and b moved
```

In other words:

```rust
use concrete::FheUint2;

fn compute_stuff(a: FheUint2, b: FheUint2) {
    // Since FheUint2 is not `Copy`
    // every time we use an operator using 'owned values',
    // they are moved thus destroyed and unusable in later computations.

    // This first addition will work,
    // however the subtraction won't compile
    // as `a` and `b` were moved when using `+`
    // let c = a + b;
    // let c2 = a - b;


    // We could `clone` these values each time, but that adds some inefficiency.
    let c = a.clone() + b.clone();
    let c2 = a.clone() - b.clone();

    // Or we can use references and avoid cloning
    let c: FheUint2 = &a + &b;
    let c2: FheUint2 = &a - &b;
}
```