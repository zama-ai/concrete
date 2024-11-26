use super::partitions::PartitionIndex;
use std::{collections::HashMap, fmt::Display};

/// A flexible and slow map associating values with symbols.
///
/// By default all symbols are assumed to be associated with the default value
/// of the type T. In practice, only associations with non-default values are
/// stored in the map.
///
/// Note:
/// -----
/// This map is flexible but slow to lookup. Hence it is mostly suited to the
/// analysis part of the optimizer.
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

impl<T: Default + Clone + PartialEq> Default for SymbolMap<T> {
    fn default() -> Self {
        Self::new()
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

/// An indexing scheme for symbol arrays.
///
/// Returns the linear index in a symbol array.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolScheme(usize);

impl SymbolScheme {
    /// Creates a new symbol scheme for a given number of partitions.
    pub fn new(n_partitions: usize) -> Self {
        SymbolScheme(n_partitions)
    }

    /// Checks if a symbol is valid.
    fn has_symbol(&self, sym: &Symbol) -> bool {
        match sym {
            Symbol::Input(i) => i.0 < self.0,
            Symbol::Bootstrap(i) => i.0 < self.0,
            Symbol::ModulusSwitch(i) => i.0 < self.0,
            Symbol::Keyswitch(i, j) => i.0 < self.0 && j.0 < self.0,
            Symbol::FastKeyswitch(i, j) => i.0 < self.0 && j.0 < self.0,
        }
    }

    /// Returns the linear index for a given symbol
    pub fn get_symbol_index(&self, sym: &Symbol) -> usize {
        debug_assert!(self.has_symbol(sym));
        match sym {
            Symbol::Input(i) => i.0,
            Symbol::Bootstrap(i) => self.0 + i.0,
            Symbol::ModulusSwitch(i) => self.0 * 2 + i.0,
            Symbol::Keyswitch(i, j) => self.0 * 3 + i.0 * self.0 + j.0,
            Symbol::FastKeyswitch(i, j) => self.0 * (3 + self.0) + i.0 * self.0 + j.0,
        }
    }

    /// Returns the number of symbols in the scheme.
    pub fn len(&self) -> usize {
        self.0 * (3 + 2 * self.0)
    }

    /// Returns an iterator over valid symbols.
    pub fn iter(&self) -> impl Iterator<Item = Symbol> + '_ {
        (0..self.len()).map(|i| {
            if i < self.len() {
                Symbol::Input(PartitionIndex(i))
            } else if i < 2 * self.0 {
                Symbol::Bootstrap(PartitionIndex(i - self.len()))
            } else if i < 3 * self.0 {
                Symbol::ModulusSwitch(PartitionIndex(i - 2 * self.len()))
            } else if i < self.0 * (3 + self.0) {
                let a = i - 3 * self.0;
                Symbol::Keyswitch(PartitionIndex(a / self.0), PartitionIndex(a % self.0))
            } else {
                let a = i - (3 + self.0) * self.0;
                Symbol::FastKeyswitch(PartitionIndex(a / self.0), PartitionIndex(a % self.0))
            }
        })
    }
}

/// A rigid and fast map associating values with symbols.
///
/// Stores all the possible values for a circuit with a given number of partitions.
///
/// Note:
/// -----
/// This map is rigid but allows fast lookup and iteration. Hence it is mostly suited to
/// the optimization part of the optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolArray<T: Default + Clone + PartialEq> {
    pub(super) scheme: SymbolScheme,
    pub(super) values: Vec<T>,
}

impl<T: Default + Clone + PartialEq> SymbolArray<T> {
    /// Creates a new Symbol array from a scheme.
    pub fn from_scheme(scheme: &SymbolScheme) -> SymbolArray<T> {
        SymbolArray {
            scheme: scheme.to_owned(),
            values: vec![T::default(); scheme.len()],
        }
    }

    pub fn from_scheme_and_map(scheme: &SymbolScheme, map: &SymbolMap<T>) -> SymbolArray<T> {
        let mut output = Self::from_scheme(scheme);
        map.iter().for_each(|(sym, v)| output.set(&sym, v));
        output
    }

    /// Sets the value associated with a given symbol.
    pub fn set(&mut self, sym: &Symbol, val: T) {
        self.values[self.scheme.get_symbol_index(sym)] = val;
    }

    /// Returns the value associated with a given symbol.
    pub fn get<'a>(&'a self, sym: &Symbol) -> &'a T {
        &self.values[self.scheme.get_symbol_index(sym)]
    }

    /// Returns the scheme used for this array.
    pub fn scheme(&self) -> &SymbolScheme {
        &self.scheme
    }

    /// Returns an iterator over value refs.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    /// Returns an iterator over values and associated symbols.
    pub fn iter_with_sym(&self) -> impl Iterator<Item = (Symbol, &T)> {
        self.scheme.iter().zip(self.values.iter())
    }
}

impl<T: Default + Clone + PartialEq + Display> SymbolArray<T> {
    /// Formats the symbol array with a given separator and symbol prefix.
    pub fn fmt_with(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        separator: &str,
        sym_prefix: &str,
    ) -> std::fmt::Result {
        let mut terms = self.iter_with_sym();
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
    Symbol::ModulusSwitch(partition)
}
