use super::*;
use crate::AesKey;
use rand::Rng;

#[test]
fn test_gen_byte_incr() {
    // Checks that the byte counter is correctly incremented.
    for _ in 0..1000 {
        let state = rand::thread_rng().gen::<u128>();
        let mut a = SoftAesCtrGenerator::new(
            Some(AesKey(0)),
            Some(State::from_aes_counter(AesCtr(state))),
            None,
        );
        assert_eq!(
            *a.get_state(),
            State {
                aes_ctr: AesCtr(state),
                byte_ctr: ByteCtr(0),
            }
        );
        a.generate_next();
        assert_eq!(
            *a.get_state(),
            State {
                aes_ctr: AesCtr(state),
                byte_ctr: ByteCtr(1),
            }
        );
    }
}

#[test]
fn test_gen_aes_incr() {
    // Checks that the aes counter is correctly incremented.
    for _ in 0..1000 {
        let state = rand::thread_rng().gen::<u128>();
        let mut a = SoftAesCtrGenerator::new(
            Some(AesKey(0)),
            Some(State::from_aes_counter(AesCtr(state))),
            None,
        );
        assert_eq!(
            *a.get_state(),
            State {
                aes_ctr: AesCtr(state),
                byte_ctr: ByteCtr(0),
            }
        );
        for _ in 0..127 {
            a.generate_next();
        }
        assert_eq!(
            *a.get_state(),
            State {
                aes_ctr: AesCtr(state),
                byte_ctr: ByteCtr(127),
            }
        );
        a.generate_next();
        assert_eq!(
            *a.get_state(),
            State {
                aes_ctr: AesCtr(state.wrapping_add(8)),
                byte_ctr: ByteCtr(0),
            }
        );
    }
}

#[test]
fn test_state_fork_initial_batch() {
    // Checks that forking the prng into children that spawns the initial batch gives the
    // correct states.
    let state = State::from_aes_counter(AesCtr(0));
    let mut generator = SoftAesCtrGenerator::new(None, Some(state), None);
    assert_eq!(
        *generator.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(0)
        }
    );
    let children: Vec<_> = generator
        .try_fork(ChildCount(2), BytesPerChild(3))
        .unwrap()
        .collect();
    assert_eq!(
        *generator.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(6)
        }
    );
    let mut first = children.get(0).unwrap().clone();
    assert_eq!(
        *first.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(0)
        }
    );
    assert_eq!(
        *first.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    let out_first: Vec<_> = first
        .try_fork(ChildCount(3), BytesPerChild(1))
        .unwrap()
        .collect();
    assert_eq!(
        *first.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    assert_eq!(
        *first.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    let first_first = out_first.get(0).unwrap();
    assert_eq!(
        *first_first.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(0)
        }
    );
    assert_eq!(
        *first_first.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(1)
        }
    );
    let first_second = out_first.get(1).unwrap();
    assert_eq!(
        *first_second.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(1)
        }
    );
    assert_eq!(
        *first_second.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(2)
        }
    );
    let first_third = out_first.get(2).unwrap();
    assert_eq!(
        *first_third.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(2)
        }
    );
    assert_eq!(
        *first_third.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    let second = children.get(1).unwrap();
    assert_eq!(
        *second.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    assert_eq!(
        *second.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(6)
        }
    );
}

#[test]
fn test_state_fork_next_batch() {
    // Checks that forking the prng into children that spawns the next batch gives the
    // correct states.
    let state = State::from_aes_counter(AesCtr(0));
    let mut generator = SoftAesCtrGenerator::new(None, Some(state), None);
    assert_eq!(
        *generator.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(0)
        }
    );
    let out: Vec<_> = generator
        .try_fork(ChildCount(4), BytesPerChild(127))
        .unwrap()
        .collect();
    assert_eq!(
        *generator.get_state(),
        State {
            aes_ctr: AesCtr(24),
            byte_ctr: ByteCtr(124)
        }
    );
    let mut first = out.get(0).unwrap().clone();
    assert_eq!(
        *first.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(0)
        }
    );
    assert_eq!(
        *first.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(127)
        }
    );
    let out_first: Vec<_> = first
        .try_fork(ChildCount(3), BytesPerChild(1))
        .unwrap()
        .collect();
    assert_eq!(
        *first.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    assert_eq!(
        *first.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(127)
        }
    );
    let first_first = out_first.get(0).unwrap();
    assert_eq!(
        *first_first.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(0)
        }
    );
    assert_eq!(
        *first_first.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(1)
        }
    );
    let first_second = out_first.get(1).unwrap();
    assert_eq!(
        *first_second.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(1)
        }
    );
    assert_eq!(
        *first_second.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(2)
        }
    );
    let first_third = out_first.get(2).unwrap();
    assert_eq!(
        *first_third.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(2)
        }
    );
    assert_eq!(
        *first_third.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(3)
        }
    );
    let second = out.get(1).unwrap();
    assert_eq!(
        *second.get_state(),
        State {
            aes_ctr: AesCtr(0),
            byte_ctr: ByteCtr(127)
        }
    );
    assert_eq!(
        *second.get_bound().unwrap(),
        State {
            aes_ctr: AesCtr(8),
            byte_ctr: ByteCtr(126)
        }
    );
}

#[test]
fn test_state_ordering() {
    // Checks that state ordering is correct.
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    let second = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    assert_eq!(first, second);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    let second = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(55),
    };
    assert!(first > second);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    let second = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(57),
    };
    assert!(first < second);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(127),
    };
    let second = State {
        aes_ctr: AesCtr(9),
        byte_ctr: ByteCtr(0),
    };
    assert!(first < second);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    let second = State {
        aes_ctr: AesCtr(4),
        byte_ctr: ByteCtr(8),
    };
    assert_eq!(first, second);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    let second = State {
        aes_ctr: AesCtr(4),
        byte_ctr: ByteCtr(7),
    };
    assert!(second < first);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(57),
    };
    let second = State {
        aes_ctr: AesCtr(4),
        byte_ctr: ByteCtr(8),
    };
    assert!(second < first);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(55),
    };
    let second = State {
        aes_ctr: AesCtr(4),
        byte_ctr: ByteCtr(8),
    };
    assert!(second > first);
    let first = State {
        aes_ctr: AesCtr(1),
        byte_ctr: ByteCtr(56),
    };
    let second = State {
        aes_ctr: AesCtr(4),
        byte_ctr: ByteCtr(9),
    };
    assert!(second > first);
}

#[test]
fn test_randomized_fork_generation() {
    // Checks that whatever the fork, whatever the state, children generate the same outputs
    // sequence as parent, and that parent recover at the proper position.
    for _ in 0..100 {
        let state = State::from_aes_counter(AesCtr(rand::thread_rng().gen()));
        let n_child = ChildCount(rand::thread_rng().gen::<usize>() % 200);
        let bytes_child = BytesPerChild(rand::thread_rng().gen::<usize>() % 200);
        let key = AesKey(rand::thread_rng().gen());
        let mut generator = SoftAesCtrGenerator::new(Some(key), Some(state.clone()), None);
        let n_to_gen = n_child.0 * bytes_child.0;
        let initial_output: Vec<u8> = (0..n_to_gen).map(|_| generator.generate_next()).collect();
        let mut forking_generator = SoftAesCtrGenerator::new(Some(key), Some(state), None);
        let children_output: Vec<u8> = forking_generator
            .try_fork(n_child, bytes_child)
            .unwrap()
            .flat_map(|mut child| (0..bytes_child.0).map(move |_| child.generate_next()))
            .collect();
        assert_eq!(initial_output, children_output);
        assert_eq!(forking_generator.generate_next(), generator.generate_next());
    }
}

#[test]
fn test_randomized_remaining_bytes() {
    for _ in 0..1000 {
        let state = State::from_aes_counter(AesCtr(rand::thread_rng().gen()));
        let n_child = ChildCount(rand::thread_rng().gen::<usize>() % 200);
        let bytes_child = BytesPerChild(rand::thread_rng().gen::<usize>() % 200);
        let key = AesKey(rand::thread_rng().gen());
        let mut forking_generator = SoftAesCtrGenerator::new(Some(key), Some(state), None);
        forking_generator
            .try_fork(n_child, bytes_child)
            .unwrap()
            .for_each(|child| assert_eq!(child.remaining_bytes(), Some(bytes_child.0)));
        assert_eq!(forking_generator.remaining_bytes(), None);
    }
}
