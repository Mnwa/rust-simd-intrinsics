#![cfg_attr(feature = "portable-simd-example", feature(portable_simd))]
#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unused_must_use)]

//! Runnable templates accompanying the `rust-simd-intrinsics` agent skill.
//!
//! The modules deliberately keep scalar reference behavior visible and isolate
//! architecture-specific unsafe code behind safe runtime-dispatched wrappers.

pub mod autovec;
pub mod std_arch;

#[cfg(feature = "fearless-example")]
pub mod fearless_impl;

#[cfg(feature = "portable-simd-example")]
pub mod portable_simd;

#[cfg(feature = "wide-example")]
pub mod wide_impl;
