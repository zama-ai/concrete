(overloaded-operators)=
# Overloaded Operators

The different types exposed by this crate overloads operators.

There are two types of operators, __binary__ operators and __unary__ operators.
Binary operators like `+` work with two values, while unary operators like `!` work with
one value.

As FHE types exposed by this crate are not [Copy] (since they are bigger than native types
and contain data on the heap) the operators overloaded (whether binary or unary) are overloaded
on both owned values (`T`) and references / borrowed valued (`&T`) to allow avoiding to `clone` the
values each time they are used.

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
    // they are moved thus destroyed and un-usable in later computations.
    
    // This won't work, as `a` and `b` where moved when using `+`
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

## Booleans

| name     | symbol | type     |
|----------|--------|----------|
| [BitAnd] | `&`    | Binary   |
| [BitOr]  | `\| `  | Binary   |
| [BitXor] | `^`    | Binary   |
| [Neg]    | `!`    | Unary    |

## ShortInts and Integers

| name       | symbol | type   |
|------------|--------|--------|
| [Add]      | `+`    | Binary |
| [Sub]      | `-`    | Binary |
| [Mul]      | `*`    | Binary |
| `Mul<u8>`  | `*`    | Binary |

[Copy]: https://doc.rust-lang.org/std/marker/trait.Copy.html

[BitAnd]: https://doc.rust-lang.org/std/ops/trait.BitAnd.html
[BitOr]: https://doc.rust-lang.org/std/ops/trait.BitOr.html
[BitXor]: https://doc.rust-lang.org/std/ops/trait.BitXor.html
[Add]: https://doc.rust-lang.org/std/ops/trait.Add.html
[Sub]: https://doc.rust-lang.org/std/ops/trait.Sub.html
[Mul]: https://doc.rust-lang.org/std/ops/trait.Mul.html
[Neg]: https://doc.rust-lang.org/std/ops/trait.Neg.html



