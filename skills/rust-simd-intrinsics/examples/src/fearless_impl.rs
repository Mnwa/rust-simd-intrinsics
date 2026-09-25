//! Runtime multiversioning and native-width explicit vectors with
//! `fearless_simd`.

use fearless_simd::{Level, dispatch, prelude::*};
use fearless_simd_macros::simd;

/// Without the optional macro, inline the kernel into the dispatch context.
#[inline(always)]
fn double_u32s_kernel<S: Simd>(simd: S, values: &mut [u32]) {
    let mut chunks = values.chunks_exact_mut(S::u32s::LEN);

    for chunk in &mut chunks {
        let value = S::u32s::from_slice(simd, chunk);
        (value * 2).store_slice(chunk);
    }

    for value in chunks.into_remainder() {
        *value = value.wrapping_mul(2);
    }
}

/// Detects the best supported level once, then dispatches the complete slice.
pub fn double_u32s(values: &mut [u32]) {
    let level = Level::new();
    dispatch!(level, simd => double_u32s_kernel(simd, values));
}

/// Scalar-looking loop that `fearless_simd` multiversions for LLVM
/// auto-vectorization.
#[simd]
fn xor_with_key_kernel<S: Simd>(_: S, bytes: &mut [u8], key: u8) {
    for byte in bytes {
        *byte ^= key;
    }
}

pub fn xor_with_key(bytes: &mut [u8], key: u8) {
    let level = Level::new();
    dispatch!(level, simd => xor_with_key_kernel(simd, bytes, key));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_width_kernel_handles_tail_and_overflow() {
        let mut values = [0, 1, 2, 3, u32::MAX, u32::MAX - 1, 9];
        double_u32s(&mut values);
        assert_eq!(values, [0, 2, 4, 6, u32::MAX - 1, u32::MAX - 3, 18]);
    }

    #[test]
    fn auto_vectorized_kernel_matches_scalar() {
        let mut values = *b"fearless simd";
        let expected = values.map(|byte| byte ^ 0xa5);
        xor_with_key(&mut values, 0xa5);
        assert_eq!(values, expected);
    }
}
