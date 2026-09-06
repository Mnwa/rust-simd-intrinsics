//! Original educational SIMD candidates with explicit scalar oracles.
//! No benchmark result is implied by the choice of instructions.
#![cfg_attr(feature = "portable", feature(portable_simd))]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod arch;
pub mod math;
pub mod scalar;
#[cfg(feature = "fearless")]
pub mod fearless;
#[cfg(feature = "portable")]
pub mod portable;
#[cfg(feature = "wide")]
pub mod wide_examples;
