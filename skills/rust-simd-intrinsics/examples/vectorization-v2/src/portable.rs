//! Nightly portable SIMD recipes. Each API/contract has a scalar oracle in tests.
use std::simd::{prelude::*, simd_swizzle, Mask, Simd};
use crate::{math, scalar};
type U32x4 = Simd<u32, 4>;
type U8x16 = Simd<u8, 16>;
type F64x4 = Simd<f64, 4>;

pub fn sum_wrapping(input: &[u32]) -> u32 {
    let mut acc = U32x4::splat(0);
    let mut chunks = input.chunks_exact(4);
    for c in &mut chunks { acc += U32x4::from_slice(c); }
    chunks.remainder().iter().copied().fold(acc.reduce_sum(), u32::wrapping_add)
}
pub fn product_wrapping(input: &[u32]) -> u32 {
    let mut acc = U32x4::splat(1);
    let mut chunks = input.chunks_exact(4);
    for c in &mut chunks { acc *= U32x4::from_slice(c); }
    chunks.remainder().iter().copied().fold(acc.reduce_product(), u32::wrapping_mul)
}
pub fn sum_u8_widened(input: &[u8]) -> u64 {
    let mut acc = Simd::<u64, 16>::splat(0);
    let mut chunks = input.chunks_exact(16);
    for c in &mut chunks { acc += U8x16::from_slice(c).cast::<u64>(); }
    chunks.remainder().iter().fold(acc.reduce_sum(), |s, &x| s.wrapping_add(u64::from(x)))
}
pub fn prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut output = vec![0u32; input.len()];
    let full = input.len() / 4 * 4;
    let zero = U32x4::splat(0);
    let mut carry = 0u32;
    for (src, dst) in input[..full].chunks_exact(4).zip(output[..full].chunks_exact_mut(4)) {
        let mut v = U32x4::from_slice(src);
        v += simd_swizzle!(v, zero, [4, 0, 1, 2]);
        v += simd_swizzle!(v, zero, [4, 5, 0, 1]);
        v += U32x4::splat(carry);
        dst.copy_from_slice(&v.to_array());
        carry = v[3];
    }
    for (&x, dst) in input[full..].iter().zip(&mut output[full..]) {
        carry = carry.wrapping_add(x); *dst = carry;
    }
    output
}

fn arg_extreme(input: &[u32], minimum: bool) -> Option<(u32, usize)> {
    if input.len() < 4 { return if minimum { scalar::argmin(input) } else { scalar::argmax(input) }; }
    let full = input.len() / 4 * 4;
    let mut values = U32x4::from_slice(&input[..4]);
    let mut indices = Simd::<usize, 4>::from_array([0, 1, 2, 3]);
    for base in (4..full).step_by(4) {
        let v = U32x4::from_slice(&input[base..base+4]);
        let better = if minimum { v.simd_lt(values) } else { v.simd_gt(values) };
        values = better.select(v, values);
        // usize indices avoid truncation on very long 64-bit slices.
        let index_mask = Mask::<isize, 4>::from_array(better.to_array());
        indices = index_mask.select(Simd::from_array([base, base+1, base+2, base+3]), indices);
    }
    let v = values.to_array();
    let ix = indices.to_array();
    let mut best = (v[0], ix[0]);
    for lane in 1..4 {
        let candidate = (v[lane], ix[lane]);
        let better = if minimum { candidate.0 < best.0 } else { candidate.0 > best.0 };
        if better || (candidate.0 == best.0 && candidate.1 < best.1) { best = candidate; }
    }
    for (offset, &value) in input[full..].iter().enumerate() {
        let better = if minimum { value < best.0 } else { value > best.0 };
        if better { best = (value, full + offset); }
    }
    Some(best)
}
pub fn argmin(input: &[u32]) -> Option<(u32, usize)> { arg_extreme(input, true) }
pub fn argmax(input: &[u32]) -> Option<(u32, usize)> { arg_extreme(input, false) }

pub fn find_byte(input: &[u8], needle: u8) -> Option<usize> {
    let mut chunks = input.chunks_exact(16);
    for (i, c) in (&mut chunks).enumerate() {
        let bits = U8x16::from_slice(c).simd_eq(U8x16::splat(needle)).to_bitmask();
        if bits != 0 { return Some(i*16 + bits.trailing_zeros() as usize); }
    }
    let full = input.len() - chunks.remainder().len();
    scalar::find_byte(chunks.remainder(), needle).map(|i| full + i)
}
pub fn count_byte(input: &[u8], needle: u8) -> usize {
    let mut count = 0usize;
    let mut chunks = input.chunks_exact(16);
    for c in &mut chunks {
        count += U8x16::from_slice(c).simd_eq(U8x16::splat(needle)).to_bitmask().count_ones() as usize;
    }
    count + scalar::count_byte(chunks.remainder(), needle)
}
pub fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let full = n / 16 * 16;
    for base in (0..full).step_by(16) {
        let unequal = U8x16::from_slice(&a[base..base+16])
            .simd_ne(U8x16::from_slice(&b[base..base+16])).to_bitmask();
        if unequal != 0 { return base + unequal.trailing_zeros() as usize; }
    }
    full + scalar::common_prefix(&a[full..n], &b[full..n])
}
/// SIMD classification, scalar stable compaction. Not a hardware-compress claim.
pub fn filter_gt(input: &[u32], threshold: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut chunks = input.chunks_exact(4);
    for c in &mut chunks {
        let mut bits = U32x4::from_slice(c).simd_gt(U32x4::splat(threshold)).to_bitmask();
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            out.push(c[lane]);
            bits &= bits - 1;
        }
    }
    out.extend(chunks.remainder().iter().copied().filter(|&v| v > threshold));
    out
}
pub fn hamming(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let full = a.len() / 16 * 16;
    let mut sum = Simd::<u64, 16>::splat(0);
    for base in (0..full).step_by(16) {
        let mut x = U8x16::from_slice(&a[base..base+16]) ^ U8x16::from_slice(&b[base..base+16]);
        x -= (x >> 1) & U8x16::splat(0x55);
        x = (x & U8x16::splat(0x33)) + ((x >> 2) & U8x16::splat(0x33));
        x = (x + (x >> 4)) & U8x16::splat(0x0f);
        sum += x.cast::<u64>();
    }
    sum.reduce_sum().wrapping_add(scalar::hamming(&a[full..], &b[full..]))
}
pub fn classify_hex(input: &[u8]) -> Vec<u8> {
    let mut output = vec![255u8; input.len()];
    let full = input.len() / 16 * 16;
    for (src, dst) in input[..full].chunks_exact(16).zip(output[..full].chunks_exact_mut(16)) {
        let v = U8x16::from_slice(src);
        let lower = v | U8x16::splat(0x20);
        let digit = v.simd_ge(U8x16::splat(b'0')) & v.simd_le(U8x16::splat(b'9'));
        let letter = lower.simd_ge(U8x16::splat(b'a')) & lower.simd_le(U8x16::splat(b'f'));
        let decoded = digit.select(v - U8x16::splat(b'0'),
            letter.select(lower - U8x16::splat(b'a') + U8x16::splat(10), U8x16::splat(255)));
        dst.copy_from_slice(&decoded.to_array());
    }
    output[full..].copy_from_slice(&scalar::classify_hex(&input[full..]));
    output
}
pub fn ascii_uppercase(input: &[u8]) -> Vec<u8> {
    let mut output = input.to_vec();
    let full = input.len() / 16 * 16;
    for (src, dst) in input[..full].chunks_exact(16).zip(output[..full].chunks_exact_mut(16)) {
        let v = U8x16::from_slice(src);
        let lowercase = v.simd_ge(U8x16::splat(b'a')) & v.simd_le(U8x16::splat(b'z'));
        dst.copy_from_slice(&lowercase.select(v - U8x16::splat(32), v).to_array());
    }
    for x in &mut output[full..] { *x = x.to_ascii_uppercase(); }
    output
}

pub fn sort4(input: [u32; 4]) -> [u32; 4] {
    let mut v = U32x4::from_array(input);
    let p = simd_swizzle!(v, [1, 0, 3, 2]);
    v = Mask::<i32, 4>::from_array([true, false, true, false]).select(v.simd_min(p), v.simd_max(p));
    let p = simd_swizzle!(v, [2, 3, 0, 1]);
    v = Mask::<i32, 4>::from_array([true, true, false, false]).select(v.simd_min(p), v.simd_max(p));
    let p = simd_swizzle!(v, [0, 2, 1, 3]);
    Mask::<i32, 4>::from_array([true, true, false, false]).select(v.simd_min(p), v.simd_max(p)).to_array()
}
pub fn adler32(input: &[u8]) -> u32 {
    let (mut s1, mut s2) = (1u32, 0u32);
    let weights = Simd::<u32, 16>::from_array([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]);
    let mut chunks = input.chunks_exact(16);
    for c in &mut chunks {
        let v = U8x16::from_slice(c).cast::<u32>();
        // Bound documented in vectorization-patterns.md; old s1 is required here.
        s2 = (s2 + 16 * s1 + (v * weights).reduce_sum()) % 65521;
        s1 = (s1 + v.reduce_sum()) % 65521;
    }
    for &x in chunks.remainder() { s1 = (s1 + u32::from(x)) % 65521; s2 = (s2 + s1) % 65521; }
    (s2 << 16) | s1
}
pub fn batched_dot4(x: &[f32], rows: &[[f32; 4]]) -> [f32; 4] {
    assert_eq!(x.len(), rows.len());
    let mut acc = Simd::<f32, 4>::splat(0.0);
    for (&v, row) in x.iter().zip(rows) { acc += Simd::splat(v) * Simd::from_array(*row); }
    acc.to_array()
}
pub fn convolve3(input: &[f32], weights: [f32; 3]) -> Vec<f32> {
    let n = input.len().saturating_sub(2);
    let full = n / 4 * 4;
    let mut out = vec![0.0f32; n];
    for base in (0..full).step_by(4) {
        let a = Simd::<f32,4>::from_slice(&input[base..base+4]);
        let b = Simd::<f32,4>::from_slice(&input[base+1..base+5]);
        let c = Simd::<f32,4>::from_slice(&input[base+2..base+6]);
        let value = (a * Simd::splat(weights[0]) + b * Simd::splat(weights[1])) + c * Simd::splat(weights[2]);
        out[base..base+4].copy_from_slice(&value.to_array());
    }
    for i in full..n { out[i] = (input[i]*weights[0] + input[i+1]*weights[1]) + input[i+2]*weights[2]; }
    out
}
pub fn statistics_u32(input: &[u32]) -> scalar::Statistics {
    let mut sum = Simd::<u64,4>::splat(0);
    let mut squares = sum;
    let mut low = U32x4::splat(u32::MAX);
    let mut high = U32x4::splat(0);
    let mut chunks = input.chunks_exact(4);
    for c in &mut chunks {
        let v = U32x4::from_slice(c);
        let w = v.cast::<u64>();
        sum += w; squares += w * w;
        low = low.simd_min(v); high = high.simd_max(v);
    }
    let mut result = scalar::statistics_u32(chunks.remainder());
    result.sum = result.sum.wrapping_add(sum.reduce_sum());
    result.sum_squares = result.sum_squares.wrapping_add(squares.reduce_sum());
    if input.len() >= 4 {
        let lo = low.reduce_min(); let hi = high.reduce_max();
        result.min = Some(result.min.map_or(lo, |x| x.min(lo)));
        result.max = Some(result.max.map_or(hi, |x| x.max(hi)));
    }
    result
}

fn log2_counts(counts: [u32;4]) -> F64x4 {
    // Scalar exponent/mantissa preparation is visible and must be costed.
    let parts = counts.map(math::decompose_count);
    let e = F64x4::from_array(std::array::from_fn(|i| parts[i].0));
    let m = F64x4::from_array(std::array::from_fn(|i| parts[i].1));
    let z = (m - F64x4::splat(1.0)) / (m + F64x4::splat(1.0));
    let q = z * z;
    let mut p = F64x4::splat(1.0/19.0);
    for k in (0..9).rev() { p = p * q + F64x4::splat(1.0/((2*k+1) as f64)); }
    e + F64x4::splat(2.0/std::f64::consts::LN_2) * z * p
}
/// Approximate score only. This is not a certified equivalent threshold decision.
pub fn histogram_score(counts: &[u32], depths: &[u8], constant: f64, table: &math::LogTable) -> f64 {
    assert_eq!(counts.len(), depths.len());
    let full = counts.len()/4*4;
    let mut acc = F64x4::splat(0.0);
    for base in (0..full).step_by(4) {
        let c = U32x4::from_slice(&counts[base..base+4]);
        let values = c.to_array();
        let small = c.simd_lt(U32x4::splat(256));
        let logs = if small.all() {
            F64x4::from_array(values.map(|x| table[x as usize]))
        } else if (!small).all() {
            log2_counts(values)
        } else {
            let approximate = log2_counts(values);
            let lookup = F64x4::from_array(values.map(|x| if x < 256 { table[x as usize] } else { 0.0 }));
            Mask::<i64,4>::from_array(small.to_array()).select(lookup, approximate)
        };
        let d = F64x4::from_array(std::array::from_fn(|i| f64::from(depths[base+i])));
        acc += c.cast::<f64>() * (d + logs);
    }
    let mut total = acc.reduce_sum();
    for (&c, &d) in counts[full..].iter().zip(&depths[full..]) {
        let log = if c < 256 { table[c as usize] } else { math::log2_count_approx(c) };
        total += f64::from(c)*(f64::from(d)+log);
    }
    constant - total
}
