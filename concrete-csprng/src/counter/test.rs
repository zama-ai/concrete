use super::*;
use crate::AesKey;
use rand::{thread_rng, Rng};

const REPEATS: usize = 1_000_000;

fn any_table_index() -> impl Iterator<Item = TableIndex> {
    std::iter::repeat_with(|| {
        TableIndex::new(
            AesIndex(thread_rng().gen()),
            ByteIndex(thread_rng().gen::<usize>() % 16),
        )
    })
}

fn any_usize() -> impl Iterator<Item = usize> {
    std::iter::repeat_with(|| thread_rng().gen())
}

fn any_children_count() -> impl Iterator<Item = ChildrenCount> {
    std::iter::repeat_with(|| ChildrenCount(thread_rng().gen::<usize>() % 2048 + 1))
}

fn any_bytes_per_child() -> impl Iterator<Item = BytesPerChild> {
    std::iter::repeat_with(|| BytesPerChild(thread_rng().gen::<usize>() % 2048 + 1))
}

fn any_key() -> impl Iterator<Item = AesKey> {
    std::iter::repeat_with(|| AesKey(thread_rng().gen()))
}

#[test]
fn prop_fork_first_state_table_index() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let original_generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        let mut forked_generator = original_generator.clone();
        let first_child = forked_generator.try_fork(nc, nb).unwrap().next().unwrap();
        assert_eq!(
            original_generator.last_table_index(),
            first_child.last_table_index()
        );
    }
}

#[test]
fn prop_fork_last_bound_table_index() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let mut parent_generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        let last_child = parent_generator.try_fork(nc, nb).unwrap().last().unwrap();
        assert_eq!(
            parent_generator.last_table_index().incremented(),
            last_child.get_bound()
        );
    }
}

#[test]
fn prop_fork_parent_bound_table_index() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let original_generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        let mut forked_generator = original_generator.clone();
        forked_generator.try_fork(nc, nb).unwrap().last().unwrap();
        assert_eq!(original_generator.get_bound(), forked_generator.get_bound());
    }
}

#[test]
fn prop_fork_parent_state_table_index() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let original_generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        let mut forked_generator = original_generator.clone();
        forked_generator.try_fork(nc, nb).unwrap().last().unwrap();
        assert!(original_generator.last_table_index() < forked_generator.last_table_index());
    }
}

#[test]
fn prop_fork() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let bytes_to_go = nc.0 * nb.0;
        let original_generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        let mut forked_generator = original_generator.clone();
        let initial_output: Vec<u8> = original_generator.take(bytes_to_go as usize).collect();
        let forked_output: Vec<u8> = forked_generator
            .try_fork(nc, nb)
            .unwrap()
            .flat_map(|child| child.collect::<Vec<_>>())
            .collect();
        assert_eq!(initial_output, forked_output);
    }
}

#[test]
fn prop_fork_children_remaining_bytes() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let mut generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        assert!(generator
            .try_fork(nc, nb)
            .unwrap()
            .all(|c| c.remaining_bytes().0 == nb.0 as u128));
    }
}

#[test]
fn prop_fork_parent_remaining_bytes() {
    for _ in 0..REPEATS {
        let ((((t, nc), nb), k), i) = any_table_index()
            .zip(any_children_count())
            .zip(any_bytes_per_child())
            .zip(any_key())
            .zip(any_usize())
            .find(|((((t, nc), nb), _k), i)| {
                TableIndex::distance(&TableIndex::LAST, t).unwrap().0 > (nc.0 * nb.0 + i) as u128
            })
            .unwrap();
        let bytes_to_go = nc.0 * nb.0;
        let mut generator =
            SoftAesCtrGenerator::new(Some(k), Some(t), Some(t.increased(nc.0 * nb.0 + i)));
        let before_remaining_bytes = generator.remaining_bytes();
        let _ = generator.try_fork(nc, nb).unwrap();
        let after_remaining_bytes = generator.remaining_bytes();
        assert_eq!(
            before_remaining_bytes.0 - after_remaining_bytes.0,
            bytes_to_go as u128
        );
    }
}
