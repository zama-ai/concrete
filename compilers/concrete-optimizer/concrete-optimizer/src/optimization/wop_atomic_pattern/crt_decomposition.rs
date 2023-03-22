use crate::dag::operator::Precision;

// Default heuristic to split in several word
pub fn default_coprimes(precision: Precision) -> Result<Vec<u64>, String> {
    Ok(match precision {
        1 => vec![2],                         // 1 bit
        2 => vec![4],                         // 2 bit
        3 => vec![8],                         // 3 bit
        4 | 5 => vec![2, 3, 7],               // 1,2,3 bits
        6 => vec![2, 5, 7],                   // 1,3,3 bits
        7 => vec![3, 7, 8],                   // 2,3,3 bits
        8 => vec![5, 7, 8],                   // 3,3,3 bits
        9 => vec![5, 7, 16],                  // 3,3,4 bits
        10 => vec![7, 15, 16],                // 3,4,4 bits
        11 => vec![13, 15, 16],               // 4,4,4 bits
        12 => vec![7, 13, 15, 16],            // 3,4,4,4 bits
        13 | 14 | 15 => vec![11, 13, 15, 16], // 4,4,4,4 bits
        16 => vec![7, 8, 9, 11, 13],          // 4,4,4,4,4 bits
        0 => return Err("Precision cannot be zero".into()),
        _ => return Err("Precision is limited to 16-bits".into()),
    })
}

fn bitwidth(v: u64) -> f64 {
    assert!(v > 0);
    (v as f64).log2()
}

pub fn fractional_precisions_from_coprimes(coprimes: &[u64]) -> Vec<f64> {
    coprimes.iter().map(|&coprime| bitwidth(coprime)).collect()
}

#[allow(clippy::cast_sign_loss)]
pub fn precisions_from_coprimes(coprimes: &[u64]) -> Vec<u64> {
    coprimes
        .iter()
        .map(|&coprime| bitwidth(coprime).ceil() as u64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coprimes() {
        for precision in 1..=16 {
            let coprimes = default_coprimes(precision);
            assert!(coprimes.is_ok());
            let prod: u64 = coprimes.unwrap().iter().product();
            println!("{precision} {prod}");
            assert!((1 << precision) <= prod);
        }
        assert!(default_coprimes(17).is_err());
    }
}
