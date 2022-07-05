# Generic Bounds

In the [Limitations](../getting\_started/limitations.md)
section, we 
explained that operators are overloaded to work with references and non-references.

If you wish to write generic functions which use operators with mixed reference and non-reference, it might get tricky at first to specify the trait [bounds](https://doc.rust-lang.org/rust-by-example/generics/bounds.html). This page should serve as a _cookbook_ to help you.

| operation   | trait bound                           |
| ----------- | ------------------------------------- |
| `T $op T`   | `T: $Op<T, Output=T>`                 |
| `T $op &T`  | `T: for<'a> $Op<&'a T, Output=T>`     |
| `&T $op T`  | `for<'a> &'a T: $Op<T, Output=T>`     |
| `&T $op &T` | `for<'a> &'a T: $Op<&'a T, Output=T>` |

{% hint style="info" %}
The `for<'a>` syntax is something called [Higher-Rank Trait Bounds](https://doc.rust-lang.org/nomicon/hrtb.html), often shortened as __HRTB__
{% endhint %}

## Example

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
