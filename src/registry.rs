//! `VarRegistry` — frozen-after-construction ordered set of variable names.

use std::fmt;
use indexmap::IndexSet;

/// Insertion-ordered set of variable names backing every named AD wrapper.
///
/// `VarRegistry` owns an `IndexSet<String>` with three load-bearing properties:
///
/// - **O(1) hash lookup by name:** [`index_of`](Self::index_of) returns the
///   positional slot used by the inner positional AD type.
/// - **Stable insertion-ordered iteration:** [`iter`](Self::iter) yields
///   names in the order they were registered — deterministic across runs.
/// - **Index-by-position:** [`name`](Self::name) maps a slot back to its
///   label for gradient readback.
///
/// # Frozen-after-construction contract
///
/// Once a `VarRegistry` is shared via `Arc`, it must not be mutated. The
/// public API encourages one-shot construction via
/// [`from_names`](Self::from_names) precisely so the shared form has no
/// mutation surface.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use xad_rs::VarRegistry;
///
/// let registry = Arc::new(VarRegistry::from_names(["spot", "strike", "vol"]));
/// assert_eq!(registry.index_of("spot"), Some(0));
/// assert_eq!(registry.index_of("strike"), Some(1));
/// assert_eq!(registry.len(), 3);
/// ```
#[derive(Clone, Default)]
pub struct VarRegistry {
    names: IndexSet<String>,
}

impl VarRegistry {
    /// Create an empty registry. Typically followed by repeated
    /// [`insert`](Self::insert) calls before the registry is wrapped in `Arc`.
    #[inline]
    pub fn new() -> Self {
        Self { names: IndexSet::new() }
    }

    /// Build a registry in one shot from an iterator of names.
    ///
    /// Duplicate names are idempotent — first insertion wins, the existing
    /// index is preserved, and subsequent duplicates are silently ignored.
    pub fn from_names<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut set = IndexSet::new();
        for n in names {
            set.insert(n.into());
        }
        Self { names: set }
    }

    /// Insert a new name, returning its positional index.
    pub fn insert(&mut self, name: impl Into<String>) -> usize {
        let (idx, _is_new) = self.names.insert_full(name.into());
        idx
    }

    /// Look up a name's positional index. Returns `None` if the name is not
    /// present.
    #[inline]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.get_index_of(name)
    }

    /// Look up a name by its positional index. Returns `None` if out of range.
    #[inline]
    pub fn name(&self, idx: usize) -> Option<&str> {
        self.names.get_index(idx).map(String::as_str)
    }

    /// Number of registered names.
    #[inline]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// True if no names are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Iterate over names in insertion order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }
}

impl fmt::Debug for VarRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarRegistry")
            .field("len", &self.names.len())
            .field("names", &self.names.iter().collect::<Vec<_>>())
            .finish()
    }
}
