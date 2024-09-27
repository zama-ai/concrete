use super::partitions::PartitionIndex;
use std::{collections::HashMap, fmt::Display};

/// A map associating symbols with values.
///
/// By default all symbols are assumed to be associated with the default value
/// of the type T. In practice, only associations with non-default values are
/// stored in the map.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolMap<T: Default + Clone + PartialEq>(HashMap<Symbol, T>);

impl<T: Default + Clone + PartialEq> SymbolMap<T> {
    /// Returns an empty symbol map.
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Update a symbol's value.
    pub fn update<F: FnOnce(T) -> T>(&mut self, sym: Symbol, f: F) {
        let val = f(self.get(sym));
        if val != T::default() {
            let _ = self.0.insert(sym, val);
        } else {
            let _ = self.0.remove(&sym);
        }
    }

    /// Sets a symbol's value.
    pub fn set(&mut self, sym: Symbol, val: T) {
        self.update(sym, |_| val)
    }

    /// Returns the value associated with the symbol.
    pub fn get(&self, sym: Symbol) -> T {
        self.0.get(&sym).cloned().unwrap_or_default()
    }

    /// Returns an iterator over associations with non-default values.
    pub fn iter(&self) -> impl Iterator<Item = (Symbol, T)> {
        self.0
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Consumes the symbol map and return an iterator.
    pub fn into_iter(self) -> impl Iterator<Item = (Symbol, T)> {
        self.0.into_iter()
    }

    /// Reset all associations to the default value.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[allow(unused)]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: Default + Clone + PartialEq + Display> SymbolMap<T> {
    /// Formats the symbol map with a given separator and symbol prefix.
    pub fn fmt_with(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        separator: &str,
        sym_prefix: &str,
    ) -> std::fmt::Result {
        let mut terms = self.iter().collect::<Vec<_>>();
        terms.sort_by_key(|t| t.0);
        let mut terms = terms.into_iter();
        match terms.next() {
            Some((sym, val)) => write!(f, "{val}{sym_prefix}{sym}")?,
            None => return write!(f, "∅"),
        }
        for (sym, val) in terms {
            write!(f, " {separator} {val}{sym_prefix}{sym}")?;
        }
        Ok(())
    }
}

/// A symbol related to an fhe operation.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Symbol {
    Input(PartitionIndex),
    Bootstrap(PartitionIndex),
    Keyswitch(PartitionIndex, PartitionIndex),
    FastKeyswitch(PartitionIndex, PartitionIndex),
    ModulusSwitch(PartitionIndex),
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::Keyswitch(from, to) if from == to => write!(f, "K[{from}]"),
            Symbol::Keyswitch(from, to) => write!(f, "K[{from}→{to}]"),
            Symbol::FastKeyswitch(from, to) => write!(f, "FK[{from}→{to}]"),
            Symbol::Bootstrap(p) => write!(f, "Br[{p}]"),
            Symbol::Input(p) => write!(f, "In[{p}]"),
            Symbol::ModulusSwitch(p) => write!(f, "M[{p}]"),
        }
    }
}

/// Returns an input symbol.
#[allow(unused)]
pub fn input(partition: PartitionIndex) -> Symbol {
    Symbol::Input(partition)
}

/// Returns an keyswitch symbol.
#[allow(unused)]
pub fn keyswitch(from: PartitionIndex, to: PartitionIndex) -> Symbol {
    Symbol::Keyswitch(from, to)
}

/// Returns a fast keyswitch symbol.
#[allow(unused)]
pub fn fast_keyswitch(from: PartitionIndex, to: PartitionIndex) -> Symbol {
    Symbol::FastKeyswitch(from, to)
}

/// Returns a pbs symbol.
#[allow(unused)]
pub fn bootstrap(partition: PartitionIndex) -> Symbol {
    Symbol::Bootstrap(partition)
}

/// Returns a modulus switch symbol.
#[allow(unused)]
pub fn modulus_switching(partition: PartitionIndex) -> Symbol {
    Symbol::Bootstrap(partition)
}
