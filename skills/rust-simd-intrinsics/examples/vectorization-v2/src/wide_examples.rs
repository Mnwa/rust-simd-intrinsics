//! Concrete f32x8 API, not a promise about every `wide` type.
use wide::f32x8;

/// Reassociated floating-point sum; not a sequential-fold replacement.
pub fn sum_reassociated(input: &[f32]) -> f32 {
    let mut acc = f32x8::splat(0.0);
    let mut chunks = input.chunks_exact(8);
    for c in &mut chunks { acc += f32x8::from(<[f32; 8]>::try_from(c).unwrap()); }
    chunks.remainder().iter().copied().fold(acc.reduce_add(), |a, b| a + b)
}
/// Reassociated product with identity 1, including its scalar tail.
pub fn product_reassociated(input: &[f32]) -> f32 {
    let mut acc = f32x8::splat(1.0);
    let mut chunks = input.chunks_exact(8);
    for c in &mut chunks { acc *= f32x8::from(<[f32; 8]>::try_from(c).unwrap()); }
    chunks.remainder().iter().copied().fold(acc.reduce_mul(), |a, b| a * b)
}
