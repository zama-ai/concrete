pub mod cache;
pub mod hasher_builder;
pub mod max;

pub fn square<V>(v: V) -> V
where
    V: std::ops::Mul<Output = V> + Copy,
{
    v * v
}

pub fn square_ref<V>(v: &V) -> V
where
    V: std::ops::Mul<Output = V> + Copy,
{
    square(*v)
}
