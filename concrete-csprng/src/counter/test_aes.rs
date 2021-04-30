use super::*;
use crate::counter::AesCtr;

#[test]
fn test_soft_hard_eq() {
    // Checks that both the software and hardware prng outputs the same values.
    let mut soft = SoftAesCtrGenerator::new(
        Some(AesKey(0)),
        Some(State::from_aes_counter(AesCtr(0))),
        None,
    );
    let mut hard = HardAesCtrGenerator::new(
        Some(AesKey(0)),
        Some(State::from_aes_counter(AesCtr(0))),
        None,
    );
    for _ in 0..1000 {
        assert_eq!(soft.generate_next(), hard.generate_next());
    }
}
