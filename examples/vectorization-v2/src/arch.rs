//! Small raw-ISA helpers; every load/store is inside a full chunk.
//! AVX2 scan uses the same 128-bit SSE2 scan; NEON search uses the scalar oracle.
use crate::scalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend { Scalar, Sse2, Avx2, Neon }

pub fn supported_backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut result = vec![Backend::Scalar];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse2") { result.push(Backend::Sse2); }
        if std::is_x86_feature_detected!("avx2") { result.push(Backend::Avx2); }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") { result.push(Backend::Neon); }
    result
}
pub fn is_supported(backend: Backend) -> bool {
    match backend {
        Backend::Scalar => true,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => std::is_x86_feature_detected!("sse2"),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => std::is_x86_feature_detected!("avx2"),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => std::arch::is_aarch64_feature_detected!("neon"),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}
pub fn detected_backend() -> Backend {
    // No heap allocation in public-call dispatch. Enumeration allocates only in tests.
    for backend in [Backend::Avx2, Backend::Sse2, Backend::Neon] {
        if is_supported(backend) { return backend; }
    }
    Backend::Scalar
}
fn require_supported(backend: Backend) {
    assert!(is_supported(backend), "unsupported SIMD backend: {backend:?}");
}
pub fn sum_wrapping(input: &[u32]) -> u32 { sum_with(detected_backend(), input) }
pub fn sum_with(backend: Backend, input: &[u32]) -> u32 {
    require_supported(backend);
    match backend {
        Backend::Scalar => scalar::sum_wrapping(input),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => unsafe { x86::sum_sse2(input) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => unsafe { x86::sum_avx2(input) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon::sum(input) },
        #[allow(unreachable_patterns)]
        _ => unreachable!("support checked before dispatch"),
    }
}
pub fn sum_u8_widened(input: &[u8]) -> u64 { sum_u8_with(detected_backend(), input) }
pub fn sum_u8_with(backend: Backend, input: &[u8]) -> u64 {
    require_supported(backend);
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 | Backend::Avx2 => unsafe { x86::sum_u8_sse2(input) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon::sum_u8(input) },
        _ => scalar::sum_u8_widened(input),
    }
}
pub fn prefix_sum(input: &[u32]) -> Vec<u32> { prefix_with(detected_backend(), input) }
pub fn prefix_with(backend: Backend, input: &[u32]) -> Vec<u32> {
    require_supported(backend);
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 | Backend::Avx2 => unsafe { x86::prefix_sse2(input) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { neon::prefix(input) },
        _ => scalar::prefix_sum(input),
    }
}
pub fn find_byte(input: &[u8], needle: u8) -> Option<usize> {
    find_with(detected_backend(), input, needle)
}
pub fn find_with(backend: Backend, input: &[u8], needle: u8) -> Option<usize> {
    require_supported(backend);
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => unsafe { x86::find_sse2(input, needle) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => unsafe { x86::find_avx2(input, needle) },
        _ => scalar::find_byte(input, needle),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")] use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")] use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sum_sse2(input: &[u32]) -> u32 {
        // SAFETY: caller checked SSE2; each exact chunk supplies 16 readable bytes.
        // The final store targets four initialized u32s; unaligned intrinsics used.
        unsafe {
            let mut acc = _mm_setzero_si128();
            let mut chunks = input.chunks_exact(4);
            for c in &mut chunks { acc = _mm_add_epi32(acc, _mm_loadu_si128(c.as_ptr().cast())); }
            let mut lanes = [0u32; 4];
            _mm_storeu_si128(lanes.as_mut_ptr().cast(), acc);
            let sum = lanes.into_iter().fold(0, u32::wrapping_add);
            chunks.remainder().iter().copied().fold(sum, u32::wrapping_add)
        }
    }
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sum_avx2(input: &[u32]) -> u32 {
        // SAFETY: AVX2 checked by caller; all loads/stores cover full exact chunks.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let mut chunks = input.chunks_exact(8);
            for c in &mut chunks { acc = _mm256_add_epi32(acc, _mm256_loadu_si256(c.as_ptr().cast())); }
            let mut lanes = [0u32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
            let sum = lanes.into_iter().fold(0, u32::wrapping_add);
            chunks.remainder().iter().copied().fold(sum, u32::wrapping_add)
        }
    }
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sum_u8_sse2(input: &[u8]) -> u64 {
        // SAFETY: SSE2 checked, exact 16-byte loads and a two-u64 output store.
        unsafe {
            let zero = _mm_setzero_si128();
            let mut acc = zero;
            let mut chunks = input.chunks_exact(16);
            for c in &mut chunks {
                let v = _mm_loadu_si128(c.as_ptr().cast());
                acc = _mm_add_epi64(acc, _mm_sad_epu8(v, zero));
            }
            let mut lanes = [0u64; 2];
            _mm_storeu_si128(lanes.as_mut_ptr().cast(), acc);
            chunks.remainder().iter().fold(lanes[0].wrapping_add(lanes[1]),
                |a, &x| a.wrapping_add(u64::from(x)))
        }
    }
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn prefix_sse2(input: &[u32]) -> Vec<u32> {
        let mut output = vec![0u32; input.len()];
        let mut carry = 0u32;
        let full = input.len() / 4 * 4;
        // SAFETY: SSE2 checked; disjoint output has the same length as input.
        // Every vector access is bounded by a full four-element chunk.
        unsafe {
            for (src, dst) in input[..full].chunks_exact(4).zip(output[..full].chunks_exact_mut(4)) {
                let mut v = _mm_loadu_si128(src.as_ptr().cast());
                v = _mm_add_epi32(v, _mm_slli_si128::<4>(v));
                v = _mm_add_epi32(v, _mm_slli_si128::<8>(v));
                v = _mm_add_epi32(v, _mm_set1_epi32(carry as i32));
                _mm_storeu_si128(dst.as_mut_ptr().cast(), v);
                carry = dst[3];
            }
        }
        for (&x, dst) in input[full..].iter().zip(&mut output[full..]) {
            carry = carry.wrapping_add(x); *dst = carry;
        }
        output
    }
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn find_sse2(input: &[u8], needle: u8) -> Option<usize> {
        // SAFETY: SSE2 checked; only complete 16-byte chunks are loaded.
        unsafe {
            let target = _mm_set1_epi8(needle as i8);
            let mut chunks = input.chunks_exact(16);
            for (i, c) in (&mut chunks).enumerate() {
                let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(c.as_ptr().cast()), target)) as u32;
                if mask != 0 { return Some(i * 16 + mask.trailing_zeros() as usize); }
            }
            let full = input.len() - chunks.remainder().len();
            chunks.remainder().iter().position(|&x| x == needle).map(|i| full + i)
        }
    }
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_avx2(input: &[u8], needle: u8) -> Option<usize> {
        // SAFETY: AVX2 checked; only complete 32-byte chunks are loaded.
        unsafe {
            let target = _mm256_set1_epi8(needle as i8);
            let mut chunks = input.chunks_exact(32);
            for (i, c) in (&mut chunks).enumerate() {
                let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(c.as_ptr().cast()), target)) as u32;
                if mask != 0 { return Some(i * 32 + mask.trailing_zeros() as usize); }
            }
            let full = input.len() - chunks.remainder().len();
            chunks.remainder().iter().position(|&x| x == needle).map(|i| full + i)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sum(input: &[u32]) -> u32 {
        // SAFETY: NEON checked by caller; vld1q loads full four-element chunks.
        unsafe {
            let mut acc = vdupq_n_u32(0);
            let mut chunks = input.chunks_exact(4);
            for c in &mut chunks { acc = vaddq_u32(acc, vld1q_u32(c.as_ptr())); }
            chunks.remainder().iter().copied().fold(vaddvq_u32(acc), u32::wrapping_add)
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sum_u8(input: &[u8]) -> u64 {
        // SAFETY: NEON checked; successive pairwise widenings cannot overflow
        // within a block, and the u64 accumulator follows modulo-2^64 semantics.
        unsafe {
            let mut acc = vdupq_n_u64(0);
            let mut chunks = input.chunks_exact(16);
            for c in &mut chunks {
                let v = vld1q_u8(c.as_ptr());
                acc = vaddq_u64(acc, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(v))));
            }
            chunks.remainder().iter().fold(vaddvq_u64(acc), |a, &x| a.wrapping_add(u64::from(x)))
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn prefix(input: &[u32]) -> Vec<u32> {
        let mut output = vec![0u32; input.len()];
        let full = input.len() / 4 * 4;
        let mut carry = 0u32;
        // SAFETY: NEON checked; matching exact input/output chunks are in bounds.
        unsafe {
            let zero = vdupq_n_u32(0);
            for (src, dst) in input[..full].chunks_exact(4).zip(output[..full].chunks_exact_mut(4)) {
                let mut v = vld1q_u32(src.as_ptr());
                v = vaddq_u32(v, vextq_u32::<3>(zero, v));
                v = vaddq_u32(v, vextq_u32::<2>(zero, v));
                v = vaddq_u32(v, vdupq_n_u32(carry));
                vst1q_u32(dst.as_mut_ptr(), v);
                carry = vgetq_lane_u32::<3>(v);
            }
        }
        for (&x, dst) in input[full..].iter().zip(&mut output[full..]) {
            carry = carry.wrapping_add(x); *dst = carry;
        }
        output
    }
}
