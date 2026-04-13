//! Unit tests for `VarRegistry`.
//!
//! Coverage: empty, single-element, multi-element, iteration order,
//! duplicate handling, Arc::ptr_eq identity, builder-phase insert.


use std::sync::Arc;
use xad_rs::VarRegistry;

#[test]
fn test_empty_registry() {
    let r = VarRegistry::new();
    assert_eq!(r.len(), 0);
    assert!(r.is_empty());
    assert_eq!(r.index_of("x"), None);
    assert_eq!(r.name(0), None);
    let names: Vec<&str> = r.iter().collect();
    assert!(names.is_empty());
}

#[test]
fn test_default_is_empty() {
    let r = VarRegistry::default();
    assert_eq!(r.len(), 0);
    assert!(r.is_empty());
}

#[test]
fn test_single_element() {
    let r = VarRegistry::from_names(["spot"]);
    assert_eq!(r.len(), 1);
    assert!(!r.is_empty());
    assert_eq!(r.index_of("spot"), Some(0));
    assert_eq!(r.index_of("missing"), None);
    assert_eq!(r.name(0), Some("spot"));
    assert_eq!(r.name(1), None);
}

#[test]
fn test_multi_element_lookup() {
    let r = VarRegistry::from_names(["x", "y", "z"]);
    assert_eq!(r.len(), 3);
    assert_eq!(r.index_of("x"), Some(0));
    assert_eq!(r.index_of("y"), Some(1));
    assert_eq!(r.index_of("z"), Some(2));
    assert_eq!(r.name(0), Some("x"));
    assert_eq!(r.name(1), Some("y"));
    assert_eq!(r.name(2), Some("z"));
    assert_eq!(r.name(3), None);
}

#[test]
fn test_iteration_order_is_insertion_not_alphabetical() {
    // If this were alphabetical, order would be ["a", "b", "c"].
    // Insertion order must preserve ["c", "a", "b"].
    let r = VarRegistry::from_names(["c", "a", "b"]);
    let names: Vec<&str> = r.iter().collect();
    assert_eq!(names, ["c", "a", "b"]);
}

#[test]
fn test_duplicate_name_idempotent_first_wins() {
    // Duplicates return the original index, don't shift others.
    let r = VarRegistry::from_names(["x", "y", "x", "z"]);
    assert_eq!(r.len(), 3); // "x" deduped
    assert_eq!(r.index_of("x"), Some(0)); // original position preserved
    assert_eq!(r.index_of("y"), Some(1));
    assert_eq!(r.index_of("z"), Some(2));
    assert_eq!(r.name(0), Some("x"));
    assert_eq!(r.name(1), Some("y"));
    assert_eq!(r.name(2), Some("z"));
}

#[test]
fn test_insert_builder_phase_returns_index() {
    let mut r = VarRegistry::new();
    assert_eq!(r.insert("x"), 0);
    assert_eq!(r.insert("y"), 1);
    assert_eq!(r.insert("x"), 0); // duplicate returns existing index
    assert_eq!(r.len(), 2);
    assert_eq!(r.index_of("x"), Some(0));
    assert_eq!(r.index_of("y"), Some(1));
}

#[test]
fn test_arc_ptr_eq_identity_same_content_different_alloc() {
    // Two registries with identical content but in different Arc allocations
    // are NOT identity-equal. This is the load-bearing property for the
    // cross-registry check: identity is by allocation, not by content.
    let r1 = Arc::new(VarRegistry::from_names(["x", "y"]));
    let r2 = Arc::clone(&r1);
    let r3 = Arc::new(VarRegistry::from_names(["x", "y"]));
    assert!(Arc::ptr_eq(&r1, &r2), "clones of the same Arc must be ptr_eq");
    assert!(!Arc::ptr_eq(&r1, &r3), "different Arcs must not be ptr_eq even with identical content");
}

#[test]
fn test_empty_iter_and_nonempty_iter() {
    let r = VarRegistry::from_names(["a", "b"]);
    let collected: Vec<&str> = r.iter().collect();
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0], "a");
    assert_eq!(collected[1], "b");
}

#[test]
fn test_clone_registry_is_independent() {
    // Cloning a VarRegistry by value clones the IndexSet — it is an
    // independent allocation. (In practice users share via Arc::clone,
    // but the Clone impl is derived and must work.)
    let r1 = VarRegistry::from_names(["x"]);
    let r2 = r1.clone();
    assert_eq!(r1.len(), r2.len());
    assert_eq!(r1.index_of("x"), r2.index_of("x"));
}
