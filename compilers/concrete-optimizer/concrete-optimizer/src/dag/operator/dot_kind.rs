use super::{ClearTensor, Shape};

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum DotKind {
    // inputs = [x,y,z], weights = [a,b,c], = x*a + y*b + z*c
    Simple,
    // inputs = [[x, y, z]], weights = [a,b,c], = same
    Tensor,
    // inputs = [[x], [y], [z]], weights = [[a],[b],[c]], = same
    CompatibleTensor,
    // inputs = [[x, y, z], [x, y, z]], weights = [[a,b,c]], = [same, same]
    // inputs = [[x, y, z], [u, v, w]], weights = [a, b], [x*a + u*b, y*a + v*b, z*c + w*c]
    // inputs = [[x, y, z]], weights = [a], [x*a, y*a, z*a]
    Broadcast { shape: Shape },
    Unsupported,
}

#[allow(clippy::if_same_then_else)]
pub fn dot_kind<W>(nb_inputs: u64, input_shape: &Shape, weights: &ClearTensor<W>) -> DotKind {
    let inputs_shape = Shape::duplicated(nb_inputs, input_shape);
    if input_shape.is_number() && inputs_shape == weights.shape {
        DotKind::Simple
    } else if nb_inputs == 1 && *input_shape == weights.shape {
        DotKind::Tensor
    } else if inputs_shape == weights.shape {
        DotKind::CompatibleTensor
    } else if nb_inputs == 1 && input_shape.erase_first_dim() == weights.shape {
        DotKind::Broadcast {
            shape: Shape::vector(input_shape.first_dim_size()),
        }
    } else if weights.shape.is_vector() && weights.shape.flat_size() == nb_inputs {
        // Same as simple but with tensor inputs
        DotKind::Broadcast {
            shape: input_shape.clone(),
        }
    } else if weights.shape.is_number() && nb_inputs == 1 {
        // Any input multiply by one number
        DotKind::Broadcast {
            shape: input_shape.clone(),
        }
    } else {
        DotKind::Unsupported
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::operator::{Shape, Weights};

    #[test]
    fn test_simple() {
        assert_eq!(
            dot_kind(2, &Shape::number(), &Weights::vector([1, 2])),
            DotKind::Simple
        );
    }

    #[test]
    fn test_tensor() {
        assert_eq!(
            dot_kind(1, &Shape::vector(2), &Weights::vector([1, 2])),
            DotKind::Tensor
        );
    }

    #[test]
    fn test_broadcast() {
        let s2x2 = Shape {
            dimensions_size: vec![2, 2],
        };
        assert_eq!(
            dot_kind(1, &s2x2, &Weights::vector([1, 2])),
            DotKind::Broadcast {
                shape: Shape::vector(2)
            }
        );
    }

    #[test]
    fn test_broadcast_scalar_mul() {
        let s2x2 = Shape {
            dimensions_size: vec![2, 2],
        };
        assert_eq!(
            dot_kind(1, &s2x2, &Weights::number(1)),
            DotKind::Broadcast {
                shape: s2x2.clone()
            }
        );
    }

    #[test]
    fn test_compatible() {
        let weights = ClearTensor {
            shape: Shape {
                dimensions_size: vec![2, 1],
            },
            values: vec![1, 2],
        };
        assert_eq!(
            dot_kind(2, &Shape::vector(1), &weights),
            DotKind::CompatibleTensor
        );
    }

    #[test]
    fn test_unsupported() {
        assert_eq!(
            dot_kind(3, &Shape::number(), &Weights::vector([1, 2])),
            DotKind::Unsupported
        );
    }
}
