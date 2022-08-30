pub mod cache;
use std::ops::Mul;

pub fn square<V>(v: V) -> V
where
    V: Mul<Output = V> + Copy,
{
    v * v
}

pub fn square_ref<V>(v: &V) -> V
where
    V: Mul<Output = V> + Copy,
{
    square(*v)
}
