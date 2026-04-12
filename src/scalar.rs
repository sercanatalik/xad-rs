//! Type traits and marker types for the AD system.

use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display};

/// Trait bound for scalar types usable in the AD system.
pub trait Scalar:
    Float + NumAssign + FromPrimitive + Debug + Display + Default + Send + Sync + 'static
{
}

impl Scalar for f32 {}
impl Scalar for f64 {}
