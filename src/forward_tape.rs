//! `NamedForwardTape` — setup-and-freeze scope for named forward-mode values.

use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexSet;

use crate::registry::VarRegistry;

// === TLS generation counter (debug-only) ===

#[cfg(debug_assertions)]
static NEXT_GEN: AtomicU64 = AtomicU64::new(1);

thread_local! {
    #[cfg(debug_assertions)]
    static ACTIVE_GEN: Cell<u64> = const { Cell::new(0) };

    static ACTIVE_REGISTRY: Cell<*const VarRegistry> = const { Cell::new(std::ptr::null()) };
}

pub(crate) fn with_active_registry<R>(f: impl FnOnce(Option<&VarRegistry>) -> R) -> R {
    ACTIVE_REGISTRY.with(|c| {
        let ptr = c.get();
        let reg_ref: Option<&VarRegistry> = if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr })
        };
        f(reg_ref)
    })
}

#[cfg(debug_assertions)]
#[inline(always)]
pub(crate) fn current_gen() -> u64 {
    ACTIVE_GEN.with(|c| c.get())
}

#[cfg(debug_assertions)]
#[inline(always)]
pub(crate) fn check_gen(lhs: u64, rhs: u64) {
    assert_eq!(
        lhs, rhs,
        "xad_rs::named: cross-registry forward-mode op detected (lhs tape generation = {lhs}, rhs tape generation = {rhs}). \
         Both operands must come from the same NamedForwardTape scope."
    );
}

#[cfg(not(debug_assertions))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn check_gen(_lhs: (), _rhs: ()) {}

// === NamedForwardTape ===

pub struct NamedForwardTape {
    builder: IndexSet<String>,
    registry: Option<Arc<VarRegistry>>,
    pending_dual: Vec<(String, f64)>,
    pending_dual2_f64: Vec<(String, f64)>,
    prev_registry: *const VarRegistry,
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    gen_id: u64,
    #[cfg(debug_assertions)]
    prev_gen: u64,
    frozen: bool,
    _not_send: PhantomData<*const ()>,
}

impl NamedForwardTape {
    pub fn new() -> Self {
        #[cfg(debug_assertions)]
        let new_gen = NEXT_GEN.fetch_add(1, Ordering::Relaxed);
        #[cfg(debug_assertions)]
        let prev_gen = ACTIVE_GEN.with(|c| {
            let p = c.get();
            c.set(new_gen);
            p
        });
        Self {
            builder: IndexSet::new(),
            registry: None,
            pending_dual: Vec::new(),
            pending_dual2_f64: Vec::new(),
            prev_registry: std::ptr::null(),
            #[cfg(debug_assertions)]
            gen_id: new_gen,
            #[cfg(debug_assertions)]
            prev_gen,
            frozen: false,
            _not_send: PhantomData,
        }
    }

    pub fn input_freal<T: crate::scalar::Scalar>(
        &mut self,
        name: &str,
        value: T,
    ) -> crate::forward::freal::NamedFReal<T> {
        assert!(
            !self.frozen,
            "NamedForwardTape::input_freal({:?}) called after freeze(); add all inputs before forward pass",
            name
        );
        if !self.builder.contains(name) {
            self.builder.insert(name.to_string());
        }
        crate::forward::freal::NamedFReal::<T>::__from_inner(crate::forward::freal::FReal::<T>::new(
            value,
            T::one(),
        ))
    }

    pub fn constant_freal<T: crate::scalar::Scalar>(
        &self,
        value: T,
    ) -> crate::forward::freal::NamedFReal<T> {
        crate::forward::freal::NamedFReal::<T>::__from_inner(crate::forward::freal::FReal::<T>::constant(value))
    }

    pub fn freeze(&mut self) -> Arc<VarRegistry> {
        assert!(!self.frozen, "NamedForwardTape::freeze() called twice");
        let reg = Arc::new(VarRegistry::from_names(self.builder.iter().cloned()));
        self.registry = Some(Arc::clone(&reg));
        let new_ptr: *const VarRegistry = Arc::as_ptr(self.registry.as_ref().unwrap());
        ACTIVE_REGISTRY.with(|c| {
            self.prev_registry = c.get();
            c.set(new_ptr);
        });
        self.frozen = true;
        reg
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    #[inline]
    pub fn registry(&self) -> Option<&Arc<VarRegistry>> {
        self.registry.as_ref()
    }

    pub fn deactivate_all() {
        ACTIVE_REGISTRY.with(|c| c.set(std::ptr::null()));
        #[cfg(debug_assertions)]
        ACTIVE_GEN.with(|c| c.set(0));
    }

    pub fn declare_dual(&mut self, name: &str, value: f64) -> DualHandle {
        assert!(
            !self.frozen,
            "NamedForwardTape::declare_dual({:?}) called after freeze",
            name
        );
        if !self.builder.contains(name) {
            self.builder.insert(name.to_string());
        }
        let idx = self.pending_dual.len();
        self.pending_dual.push((name.to_string(), value));
        DualHandle { idx }
    }

    pub fn declare_dual2_f64(&mut self, name: &str, value: f64) -> Dual2Handle {
        assert!(
            !self.frozen,
            "NamedForwardTape::declare_dual2_f64({:?}) called after freeze",
            name
        );
        if !self.builder.contains(name) {
            self.builder.insert(name.to_string());
        }
        let idx = self.pending_dual2_f64.len();
        self.pending_dual2_f64.push((name.to_string(), value));
        Dual2Handle { idx }
    }

    pub fn freeze_dual(mut self) -> NamedForwardScope {
        assert!(
            !self.frozen,
            "NamedForwardTape::freeze_dual called after freeze"
        );
        let reg = Arc::new(VarRegistry::from_names(self.builder.iter().cloned()));
        self.registry = Some(Arc::clone(&reg));

        let new_ptr: *const VarRegistry = Arc::as_ptr(self.registry.as_ref().unwrap());
        ACTIVE_REGISTRY.with(|c| {
            self.prev_registry = c.get();
            c.set(new_ptr);
        });
        self.frozen = true;

        let n_dual = self.pending_dual.len();
        let mut duals: Vec<crate::forward::dual::NamedDual> = Vec::with_capacity(n_dual);
        for (i, (name, value)) in self.pending_dual.iter().enumerate() {
            let reg_idx = reg
                .index_of(name)
                .expect("declared name missing from frozen registry");
            let _ = i;
            let inner = crate::forward::dual::Dual::variable(*value, reg_idx, n_dual);
            duals.push(crate::forward::dual::NamedDual::__from_inner(inner));
        }

        let n_dual2 = self.pending_dual2_f64.len();
        let mut dual2s_f64: Vec<crate::forward::dual2::NamedDual2<f64>> = Vec::with_capacity(n_dual2);
        for (name, value) in self.pending_dual2_f64.iter() {
            let reg_idx = reg
                .index_of(name)
                .expect("declared name missing from frozen registry");
            let inner = crate::forward::dual2::Dual2::<f64>::variable(*value);
            dual2s_f64.push(crate::forward::dual2::NamedDual2::<f64>::__from_parts(
                inner,
                Some(reg_idx),
            ));
        }

        let prev_registry = self.prev_registry;
        #[cfg(debug_assertions)]
        let prev_gen = self.prev_gen;

        self.frozen = false;
        self.prev_registry = std::ptr::null();
        #[cfg(debug_assertions)]
        {
            self.prev_gen = ACTIVE_GEN.with(|c| c.get());
        }

        NamedForwardScope {
            registry: reg,
            duals,
            dual2s_f64,
            prev_registry,
            #[cfg(debug_assertions)]
            prev_gen,
            _not_send: PhantomData,
        }
    }
}

impl Default for NamedForwardTape {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DualHandle {
    idx: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dual2Handle {
    idx: usize,
}

pub struct NamedForwardScope {
    registry: Arc<VarRegistry>,
    duals: Vec<crate::forward::dual::NamedDual>,
    dual2s_f64: Vec<crate::forward::dual2::NamedDual2<f64>>,
    prev_registry: *const VarRegistry,
    #[cfg(debug_assertions)]
    prev_gen: u64,
    _not_send: PhantomData<*const ()>,
}

impl NamedForwardScope {
    #[inline]
    pub fn dual(&self, handle: DualHandle) -> &crate::forward::dual::NamedDual {
        &self.duals[handle.idx]
    }

    #[inline]
    pub fn dual2(&self, handle: Dual2Handle) -> &crate::forward::dual2::NamedDual2<f64> {
        &self.dual2s_f64[handle.idx]
    }

    #[inline]
    pub fn registry(&self) -> &Arc<VarRegistry> {
        &self.registry
    }

    #[inline]
    pub fn constant_dual(&self, value: f64) -> crate::forward::dual::NamedDual {
        let inner = crate::forward::dual::Dual::constant(value, self.registry.len());
        crate::forward::dual::NamedDual::__from_inner(inner)
    }

    #[inline]
    pub fn constant_dual2_f64(&self, value: f64) -> crate::forward::dual2::NamedDual2<f64> {
        let inner = crate::forward::dual2::Dual2::<f64>::constant(value);
        crate::forward::dual2::NamedDual2::<f64>::__from_parts(inner, None)
    }
}

impl fmt::Debug for NamedForwardScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedForwardScope")
            .field("registry_len", &self.registry.len())
            .field("duals", &self.duals.len())
            .field("dual2s_f64", &self.dual2s_f64.len())
            .finish()
    }
}

impl Drop for NamedForwardScope {
    fn drop(&mut self) {
        ACTIVE_REGISTRY.with(|c| c.set(self.prev_registry));
        #[cfg(debug_assertions)]
        ACTIVE_GEN.with(|c| c.set(self.prev_gen));
    }
}

impl fmt::Debug for NamedForwardTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedForwardTape")
            .field("frozen", &self.frozen)
            .field("inputs", &self.builder.len())
            .field(
                "registry_len",
                &self.registry.as_ref().map(|r: &Arc<VarRegistry>| r.len()),
            )
            .finish()
    }
}

impl Drop for NamedForwardTape {
    fn drop(&mut self) {
        if self.frozen {
            ACTIVE_REGISTRY.with(|c| c.set(self.prev_registry));
        }
        #[cfg(debug_assertions)]
        ACTIVE_GEN.with(|c| c.set(self.prev_gen));
    }
}
