//! Architecture-specific SIMD with safe runtime-dispatched public wrappers.
//!
//! Covered paths:
//! - x86/x86_64: SSE for `f32`, SSE2 for `u32` and byte search,
//!   AVX for `f32`, AVX2 for `u32` and byte search.
//! - AArch64: NEON for all three examples.
//! - Every other target: scalar fallback.

use crate::autovec::{add_f32 as add_f32_scalar, find_byte_scalar};

/// Adds two `f32` slices using AVX, SSE, NEON, or scalar Rust.
pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len(), "out and a must have equal lengths");
    assert_eq!(out.len(), b.len(), "out and b must have equal lengths");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            // SAFETY: runtime detection proves AVX. Equal-length slices and the
            // kernel's loop bounds prove every load/store is in bounds.
            unsafe { x86_impl::add_f32_avx(out, a, b) };
            return;
        }
        if std::arch::is_x86_feature_detected!("sse") {
            // SAFETY: runtime detection proves SSE; memory invariants are as
            // above.
            unsafe { x86_impl::add_f32_sse(out, a, b) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime detection proves NEON; the kernel accesses only
            // complete four-lane chunks and an in-bounds scalar tail.
            unsafe { aarch64_impl::add_f32_neon(out, a, b) };
            return;
        }
    }

    add_f32_scalar(out, a, b);
}

/// Adds two `u32` slices with wrapping semantics using AVX2, SSE2, NEON, or
/// scalar Rust.
pub fn add_u32_wrapping(out: &mut [u32], a: &[u32], b: &[u32]) {
    assert_eq!(out.len(), a.len(), "out and a must have equal lengths");
    assert_eq!(out.len(), b.len(), "out and b must have equal lengths");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection proves AVX2. The vector loop operates
            // only on complete eight-lane chunks.
            unsafe { x86_impl::add_u32_avx2(out, a, b) };
            return;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: runtime detection proves SSE2. The vector loop operates
            // only on complete four-lane chunks.
            unsafe { x86_impl::add_u32_sse2(out, a, b) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime detection proves NEON; all memory accesses are
            // bounded by complete four-lane chunks.
            unsafe { aarch64_impl::add_u32_neon(out, a, b) };
            return;
        }
    }

    add_u32_scalar(out, a, b);
}

/// Finds the first occurrence of `needle` using AVX2, SSE2, NEON, or scalar
/// search.
///
/// The x86 paths use equality comparison plus movemask. The NEON path uses a
/// horizontal maximum to skip non-matching blocks, then stores only the first
/// matching mask block to recover the lane index.
pub fn find_byte(bytes: &[u8], needle: u8) -> Option<usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection proves AVX2. The loop loads only full
            // 32-byte blocks inside the slice.
            return unsafe { x86_impl::find_byte_avx2(bytes, needle) };
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: runtime detection proves SSE2. The loop loads only full
            // 16-byte blocks inside the slice.
            return unsafe { x86_impl::find_byte_sse2(bytes, needle) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime detection proves NEON; the loop loads only full
            // 16-byte blocks and scans a scalar tail.
            return unsafe { aarch64_impl::find_byte_neon(bytes, needle) };
        }
    }

    find_byte_scalar(bytes, needle)
}

#[inline]
fn add_u32_scalar(out: &mut [u32], a: &[u32], b: &[u32]) {
    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x.wrapping_add(y);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_impl {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn add_f32_avx(out: &mut [f32], a: &[f32], b: &[f32]) {
        const LANES: usize = 8;
        let mut i = 0;

        while i + LANES <= out.len() {
            // SAFETY: `i + LANES <= len`; all slices have equal lengths. The
            // unaligned load/store intrinsics impose no extra alignment.
            let (x, y) = unsafe {
                (
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                )
            };
            let sum = _mm256_add_ps(x, y);
            // SAFETY: the complete eight-lane destination is in bounds.
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), sum) };
            i += LANES;
        }

        while i < out.len() {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "sse")]
    pub(super) unsafe fn add_f32_sse(out: &mut [f32], a: &[f32], b: &[f32]) {
        const LANES: usize = 4;
        let mut i = 0;

        while i + LANES <= out.len() {
            // SAFETY: `i + LANES <= len`; all slices have equal lengths.
            let (x, y) = unsafe {
                (
                    _mm_loadu_ps(a.as_ptr().add(i)),
                    _mm_loadu_ps(b.as_ptr().add(i)),
                )
            };
            let sum = _mm_add_ps(x, y);
            // SAFETY: the complete four-lane destination is in bounds.
            unsafe { _mm_storeu_ps(out.as_mut_ptr().add(i), sum) };
            i += LANES;
        }

        while i < out.len() {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_u32_avx2(out: &mut [u32], a: &[u32], b: &[u32]) {
        const LANES: usize = 8;
        let mut i = 0;

        while i + LANES <= out.len() {
            // SAFETY: each pointer covers eight in-bounds initialized `u32`s.
            // The load is explicitly unaligned.
            let (x, y) = unsafe {
                (
                    _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()),
                    _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()),
                )
            };
            let sum = _mm256_add_epi32(x, y);
            // SAFETY: the destination covers eight in-bounds `u32`s.
            unsafe {
                _mm256_storeu_si256(out.as_mut_ptr().add(i).cast::<__m256i>(), sum);
            }
            i += LANES;
        }

        while i < out.len() {
            out[i] = a[i].wrapping_add(b[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn add_u32_sse2(out: &mut [u32], a: &[u32], b: &[u32]) {
        const LANES: usize = 4;
        let mut i = 0;

        while i + LANES <= out.len() {
            // SAFETY: each pointer covers four in-bounds initialized `u32`s.
            let (x, y) = unsafe {
                (
                    _mm_loadu_si128(a.as_ptr().add(i).cast::<__m128i>()),
                    _mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>()),
                )
            };
            let sum = _mm_add_epi32(x, y);
            // SAFETY: the destination covers four in-bounds `u32`s.
            unsafe { _mm_storeu_si128(out.as_mut_ptr().add(i).cast::<__m128i>(), sum) };
            i += LANES;
        }

        while i < out.len() {
            out[i] = a[i].wrapping_add(b[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_byte_avx2(bytes: &[u8], needle: u8) -> Option<usize> {
        const LANES: usize = 32;
        let needle_byte = needle;
        let needle = _mm256_set1_epi8(needle_byte as i8);
        let mut i = 0;

        while i + LANES <= bytes.len() {
            // SAFETY: the loop condition proves a complete 32-byte load is in
            // bounds. `_mm256_loadu_si256` permits unaligned addresses.
            let block = unsafe {
                _mm256_loadu_si256(bytes.as_ptr().add(i).cast::<__m256i>())
            };
            let equal = _mm256_cmpeq_epi8(block, needle);
            let mask = _mm256_movemask_epi8(equal) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += LANES;
        }

        while i < bytes.len() {
            if bytes[i] == needle_byte {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn find_byte_sse2(bytes: &[u8], needle: u8) -> Option<usize> {
        const LANES: usize = 16;
        let needle_byte = needle;
        let needle = _mm_set1_epi8(needle_byte as i8);
        let mut i = 0;

        while i + LANES <= bytes.len() {
            // SAFETY: the loop condition proves a complete 16-byte load is in
            // bounds. `_mm_loadu_si128` permits unaligned addresses.
            let block = unsafe { _mm_loadu_si128(bytes.as_ptr().add(i).cast::<__m128i>()) };
            let equal = _mm_cmpeq_epi8(block, needle);
            let mask = _mm_movemask_epi8(equal) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += LANES;
        }

        while i < bytes.len() {
            if bytes[i] == needle_byte {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_impl {
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_f32_neon(out: &mut [f32], a: &[f32], b: &[f32]) {
        const LANES: usize = 4;
        let mut i = 0;

        while i + LANES <= out.len() {
            // SAFETY: each pointer covers four in-bounds initialized `f32`s.
            let (x, y) = unsafe {
                (
                    vld1q_f32(a.as_ptr().add(i)),
                    vld1q_f32(b.as_ptr().add(i)),
                )
            };
            let sum = vaddq_f32(x, y);
            // SAFETY: the destination covers four in-bounds `f32`s.
            unsafe { vst1q_f32(out.as_mut_ptr().add(i), sum) };
            i += LANES;
        }

        while i < out.len() {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_u32_neon(out: &mut [u32], a: &[u32], b: &[u32]) {
        const LANES: usize = 4;
        let mut i = 0;

        while i + LANES <= out.len() {
            // SAFETY: each pointer covers four in-bounds initialized `u32`s.
            let (x, y) = unsafe {
                (
                    vld1q_u32(a.as_ptr().add(i)),
                    vld1q_u32(b.as_ptr().add(i)),
                )
            };
            let sum = vaddq_u32(x, y);
            // SAFETY: the destination covers four in-bounds `u32`s.
            unsafe { vst1q_u32(out.as_mut_ptr().add(i), sum) };
            i += LANES;
        }

        while i < out.len() {
            out[i] = a[i].wrapping_add(b[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn find_byte_neon(bytes: &[u8], needle: u8) -> Option<usize> {
        const LANES: usize = 16;
        let needle_vector = vdupq_n_u8(needle);
        let mut i = 0;

        while i + LANES <= bytes.len() {
            // SAFETY: the loop condition proves a complete 16-byte load is in
            // bounds.
            let block = unsafe { vld1q_u8(bytes.as_ptr().add(i)) };
            let equal = vceqq_u8(block, needle_vector);

            if vmaxvq_u8(equal) != 0 {
                let mut lanes = [0_u8; LANES];
                // SAFETY: `lanes` has exactly 16 writable bytes.
                unsafe { vst1q_u8(lanes.as_mut_ptr(), equal) };
                let lane = lanes
                    .iter()
                    .position(|&value| value != 0)
                    .expect("horizontal max proved a matching lane");
                return Some(i + lane);
            }
            i += LANES;
        }

        while i < bytes.len() {
            if bytes[i] == needle {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_dispatch_matches_scalar_for_offsets_and_tails() {
        for offset in 0..5 {
            for len in 0..70 {
                let storage_a: Vec<f32> = (0..len + offset)
                    .map(|i| i as f32 * 0.5 - 7.0)
                    .collect();
                let storage_b: Vec<f32> = (0..len + offset)
                    .map(|i| i as f32 * -0.125 + 2.0)
                    .collect();
                let a = &storage_a[offset..];
                let b = &storage_b[offset..];
                let expected: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x + y).collect();
                let mut out_storage = vec![0.0; len + offset];
                let out = &mut out_storage[offset..];
                add_f32(out, a, b);
                assert_eq!(out, expected);
            }
        }
    }

    #[test]
    fn u32_dispatch_preserves_wrapping() {
        for len in 0..70 {
            let a: Vec<u32> = (0..len)
                .map(|i| if i % 3 == 0 { u32::MAX } else { i as u32 })
                .collect();
            let b: Vec<u32> = (0..len)
                .map(|i| if i % 5 == 0 { 7 } else { u32::MAX - i as u32 })
                .collect();
            let expected: Vec<u32> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.wrapping_add(y))
                .collect();
            let mut out = vec![0; len];
            add_u32_wrapping(&mut out, &a, &b);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn byte_search_matches_scalar_at_every_position() {
        for len in 0..100 {
            let mut bytes = vec![0x11; len];
            assert_eq!(find_byte(&bytes, 0xa5), None);

            for position in 0..len {
                bytes[position] = 0xa5;
                assert_eq!(find_byte(&bytes, 0xa5), Some(position));
                bytes[position] = 0x11;
            }
        }
    }
}
