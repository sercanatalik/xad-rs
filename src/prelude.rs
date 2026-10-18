//! Convenience re-exports — `use xad_rs::prelude::*;` brings the
//! mode-agnostic trait [`Real`], its copyable-mode refinement
//! [`CopyableReal`], the passive bound [`Passive`], and the most-used
//! concrete types into scope.
//!
//! Deliberately small. Excludes `Jet2Vec`, `JetK`, and the free functions
//! in `math::*` / `ops::*` — reach for those via their full paths.
//!
//! ```
//! use xad_rs::prelude::*;
//! fn poly<R: Real>(x: &R) -> R { x.clone() * x.clone() }
//! assert_eq!(poly(&3.0_f64), 9.0);
//! ```

pub use crate::forward::{Jet1, Jet2};
pub use crate::passive::Passive;
pub use crate::real::{CopyableReal, Real};
pub use crate::reverse::AReal;
pub use crate::tape::{Tape, TapeStorage};
