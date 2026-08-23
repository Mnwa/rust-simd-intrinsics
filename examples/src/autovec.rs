//! Scalar Rust patterns intentionally shaped for LLVM auto-vectorization.

/// Element-wise floating-point addition.
///
/// Each lane performs exactly one `f32` addition, so vectorization does not
/// require reassociating a reduction.
pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len(), "out and a must have equal lengths");
    assert_eq!(out.len(), b.len(), "out and b must have equal lengths");

    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x + y;
    }
}

/// Element-wise wrapping integer addition with identical debug/release
/// semantics.
pub fn add_u32_wrapping(out: &mut [u32], a: &[u32], b: &[u32]) {
    assert_eq!(out.len(), a.len(), "out and a must have equal lengths");
    assert_eq!(out.len(), b.len(), "out and b must have equal lengths");

    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x.wrapping_add(y);
    }
}

/// A simple branch that LLVM can often if-convert into a vector mask/select.
pub fn threshold_u8_in_place(values: &mut [u8], threshold: u8, replacement: u8) {
    for value in values {
        if *value > threshold {
            *value = replacement;
        }
    }
}

/// Scalar oracle for byte-prefix matching.
pub fn common_prefix_scalar(a: &[u8], b: &[u8]) -> usize {
    let limit = a.len().min(b.len());
    a[..limit]
        .iter()
        .zip(&b[..limit])
        .position(|(&x, &y)| x != y)
        .unwrap_or(limit)
}

/// Scalar oracle for byte search.
pub fn find_byte_scalar(bytes: &[u8], needle: u8) -> Option<usize> {
    bytes.iter().position(|&value| value == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f32_handles_boundaries() {
        for len in 0..70 {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 - 3.0).collect();
            let b: Vec<f32> = (0..len).map(|i| i as f32 * -0.5 + 1.0).collect();
            let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
            let mut out = vec![0.0; len];
            add_f32(&mut out, &a, &b);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn wrapping_add_handles_overflow() {
        let a = [u32::MAX, 0, 7, u32::MAX - 1];
        let b = [1, u32::MAX, u32::MAX, 3];
        let mut out = [0; 4];
        add_u32_wrapping(&mut out, &a, &b);
        assert_eq!(out, [0, u32::MAX, 6, 1]);
    }

    #[test]
    fn threshold_matches_contract() {
        let mut values = [0, 10, 11, 200, 255];
        threshold_u8_in_place(&mut values, 10, 42);
        assert_eq!(values, [0, 10, 42, 42, 42]);
    }

    #[test]
    fn prefix_and_find_oracles() {
        assert_eq!(common_prefix_scalar(b"abcX", b"abcY"), 3);
        assert_eq!(common_prefix_scalar(b"abc", b"abcdef"), 3);
        assert_eq!(find_byte_scalar(b"alpha,beta", b','), Some(5));
        assert_eq!(find_byte_scalar(b"alpha", b','), None);
    }
}
