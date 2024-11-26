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

    pub fn scheme(&self) -> SymbolScheme {
        SymbolScheme(self.iter().map(|(s, _)| s).collect())
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

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolScheme(Vec<Symbol>);

impl SymbolScheme {
    pub fn new() -> Self {
        SymbolScheme(vec![])
    }

    pub fn contains_symbol(&self, sym: &Symbol) -> bool {
        self.0.iter().any(|s| s == sym)
    }

    pub fn add_symbol(&mut self, sym: Symbol) {
        if !self.contains_symbol(&sym) {
            self.0.push(sym);
        }
    }

    pub fn get_symbol_index(&self, sym: &Symbol) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find_map(|(i, s)| (s == sym).then_some(i))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SymbolArray<T: Default + Clone + PartialEq> {
    pub(super) scheme: SymbolScheme,
    pub(super) values: Vec<T>,
}

impl<T: Default + Clone + PartialEq> SymbolArray<T> {
    pub fn from_scheme(scheme: &SymbolScheme) -> SymbolArray<T> {
        SymbolArray {
            scheme: scheme.to_owned(),
            values: vec![T::default(); scheme.len()],
        }
    }

    // pub fn from_scheme_and_vals(scheme: &SymbolScheme, values: Vec<T>) -> SymbolArray<'_, T> {
    //     debug_assert_eq!(scheme.len(), values.len());
    //     SymbolArray { scheme, values }
    // }

    pub fn from_scheme_and_map(scheme: &SymbolScheme, map: &SymbolMap<T>) -> SymbolArray<T> {
        let mut output = Self::from_scheme(scheme);
        map.iter().for_each(|(sym, v)| output.set(&sym, v));
        output
    }

    pub fn set(&mut self, sym: &Symbol, val: T) {
        self.scheme
            .get_symbol_index(sym)
            .map(|i| self.values[i] = val)
            .expect(&format!("Failed to set {sym}"));
    }

    pub fn get<'a>(&'a self, sym: &Symbol) -> &'a T {
        self.scheme
            .get_symbol_index(sym)
            .and_then(|i| self.values.get(i))
            .unwrap()
    }

    pub fn scheme(&self) -> &SymbolScheme {
        &self.scheme
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.values.iter_mut()
    }

    pub fn iter_with_sym(&self) -> impl Iterator<Item = (&Symbol, &T)> {
        self.scheme.0.iter().zip(self.values.iter())
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
