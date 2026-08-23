//! Nightly portable SIMD (`std::simd`).

use std::simd::Simd;

const LANES: usize = 8;
type F32x8 = Simd<f32, LANES>;

/// Adds a vectorized prefix and finishes with a scalar tail.
pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len(), "out and a must have equal lengths");
    assert_eq!(out.len(), b.len(), "out and b must have equal lengths");

    let vector_len = out.len() / LANES * LANES;
    let (out_head, out_tail) = out.split_at_mut(vector_len);
    let (a_head, a_tail) = a.split_at(vector_len);
    let (b_head, b_tail) = b.split_at(vector_len);

    for offset in (0..vector_len).step_by(LANES) {
        let x = F32x8::from_slice(&a_head[offset..]);
        let y = F32x8::from_slice(&b_head[offset..]);
        (x + y).copy_to_slice(&mut out_head[offset..]);
    }

    for ((dst, &x), &y) in out_tail.iter_mut().zip(a_tail).zip(b_tail) {
        *dst = x + y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_scalar_for_all_short_lengths() {
        for len in 0..50 {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 1.25).collect();
            let b: Vec<f32> = (0..len).map(|i| 10.0 - i as f32).collect();
            let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
            let mut out = vec![0.0; len];
            add_f32(&mut out, &a, &b);
            assert_eq!(out, expected);
        }
    }
}
