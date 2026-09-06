//! Stable fixed-width vectors with the `wide` crate.

use wide::f32x8;

const LANES: usize = 8;

/// Adds `f32` slices in full eight-lane chunks, then uses a scalar tail.
pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len(), "out and a must have equal lengths");
    assert_eq!(out.len(), b.len(), "out and b must have equal lengths");

    let mut out_chunks = out.chunks_exact_mut(LANES);
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    for ((dst, x), y) in out_chunks
        .by_ref()
        .zip(a_chunks.by_ref())
        .zip(b_chunks.by_ref())
    {
        let x = f32x8::new(x.try_into().expect("chunks_exact returned 8 lanes"));
        let y = f32x8::new(y.try_into().expect("chunks_exact returned 8 lanes"));
        dst.copy_from_slice(&(x + y).to_array());
    }

    for ((dst, &x), &y) in out_chunks
        .into_remainder()
        .iter_mut()
        .zip(a_chunks.remainder())
        .zip(b_chunks.remainder())
    {
        *dst = x + y;
    }
}

/// Returns one bit per lane, with bit 0 corresponding to lane 0.
///
/// `to_bitmask` is valid here because `simd_eq` creates a proper all-zero or
/// all-one mask.
pub fn equality_mask(a: [f32; LANES], b: [f32; LANES]) -> u32 {
    f32x8::new(a).simd_eq(f32x8::new(b)).to_bitmask()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_scalar_for_tails() {
        for len in 0..50 {
            let a: Vec<f32> = (0..len).map(|i| i as f32 + 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| -(i as f32) * 0.25).collect();
            let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
            let mut out = vec![0.0; len];
            add_f32(&mut out, &a, &b);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn bitmask_lane_order_is_explicit() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0];
        assert_eq!(equality_mask(a, b), 0b0101_0101);
    }
}
