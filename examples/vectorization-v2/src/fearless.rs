//! Numeric reductions composed from Fearless 0.7 primitives.
use fearless_simd::{dispatch, prelude::*, f32x4, u32x4, Level};

#[inline(always)]
pub fn reduce_sum4<S: Simd>(v: u32x4<S>) -> u32 {
    let p = v + v.rotate_elements_left::<2>();
    (p + p.rotate_elements_left::<1>())[0]
}
#[inline(always)]
pub fn reduce_product4<S: Simd>(v: u32x4<S>) -> u32 {
    let p = v * v.rotate_elements_left::<2>();
    (p * p.rotate_elements_left::<1>())[0]
}
#[inline(always)]
pub fn reduce_and4<S: Simd>(v: u32x4<S>) -> u32 {
    let p = v & v.rotate_elements_left::<2>();
    (p & p.rotate_elements_left::<1>())[0]
}
#[inline(always)]
pub fn reduce_or4<S: Simd>(v: u32x4<S>) -> u32 {
    let p = v | v.rotate_elements_left::<2>();
    (p | p.rotate_elements_left::<1>())[0]
}
#[inline(always)]
pub fn reduce_xor4<S: Simd>(v: u32x4<S>) -> u32 {
    let p = v ^ v.rotate_elements_left::<2>();
    (p ^ p.rotate_elements_left::<1>())[0]
}
/// A final lane-array epilogue is also a valid missing-primitive implementation.
#[inline(always)]
pub fn reduce_min4<S: Simd>(v: u32x4<S>) -> u32 {
    v.as_slice().iter().copied().min().expect("four lanes")
}
#[inline(always)]
pub fn reduce_max4<S: Simd>(v: u32x4<S>) -> u32 {
    v.as_slice().iter().copied().max().expect("four lanes")
}
/// Reassociated FP tree; does not preserve a left-to-right scalar fold.
#[inline(always)]
pub fn reduce_sum_f32x4<S: Simd>(v: f32x4<S>) -> f32 {
    let p = v + v.rotate_elements_left::<2>();
    (p + p.rotate_elements_left::<1>())[0]
}
/// Reassociated FP tree, including its overflow/underflow order.
#[inline(always)]
pub fn reduce_product_f32x4<S: Simd>(v: f32x4<S>) -> f32 {
    let p = v * v.rotate_elements_left::<2>();
    (p * p.rotate_elements_left::<1>())[0]
}
#[inline(always)]
fn double_kernel<S: Simd>(simd: S, input: &mut [u32]) {
    let mut chunks = input.chunks_exact_mut(S::u32s::N);
    for c in &mut chunks {
        let v = S::u32s::from_slice(simd, c);
        (v + v).store_slice(c);
    }
    for x in chunks.into_remainder() { *x = x.wrapping_mul(2); }
}
pub fn double_at(level: Level, input: &mut [u32]) {
    dispatch!(level, simd => double_kernel(simd, input))
}
#[inline(always)]
fn sum4_kernel<S: Simd>(simd: S, input: &[u32]) -> u32 {
    let mut acc = u32x4::splat(simd, 0);
    let mut chunks = input.chunks_exact(4);
    for c in &mut chunks { acc += u32x4::from_slice(simd, c); }
    chunks.remainder().iter().copied().fold(reduce_sum4(acc), u32::wrapping_add)
}
#[inline(always)]
fn product_kernel<S: Simd>(simd: S, input: &[u32]) -> u32 {
    let mut acc = u32x4::splat(simd, 1);
    let mut chunks = input.chunks_exact(4);
    for c in &mut chunks { acc *= u32x4::from_slice(simd, c); }
    chunks.remainder().iter().copied().fold(reduce_product4(acc), u32::wrapping_mul)
}
#[inline(always)]
fn native_sum_kernel<S: Simd, const A: usize>(simd: S, input: &[u32]) -> u32 {
    assert!(matches!(A, 1 | 2 | 4));
    let lanes = S::u32s::N;
    let mut acc = [S::u32s::splat(simd, 0); A];
    let mut groups = input.chunks_exact(lanes * A);
    for group in &mut groups {
        for (a, c) in acc.iter_mut().zip(group.chunks_exact(lanes)) {
            *a += S::u32s::from_slice(simd, c);
        }
    }
    let mut blocks = groups.remainder().chunks_exact(lanes);
    for c in &mut blocks { acc[0] += S::u32s::from_slice(simd, c); }
    let mut total = acc[0];
    for &a in &acc[1..] { total += a; }
    // Exactly one scalar-lane epilogue, not one extraction per input block.
    let sum = total.as_slice().iter().copied().fold(0, u32::wrapping_add);
    blocks.remainder().iter().copied().fold(sum, u32::wrapping_add)
}
#[inline(always)]
fn width<S: Simd>(_: S) -> usize { S::u32s::N }
pub fn native_width(level: Level) -> usize { dispatch!(level, simd => width(simd)) }
pub fn sum4_at(level: Level, input: &[u32]) -> u32 {
    dispatch!(level, simd => sum4_kernel(simd, input))
}
pub fn product_at(level: Level, input: &[u32]) -> u32 {
    dispatch!(level, simd => product_kernel(simd, input))
}
pub fn sum_at<const A: usize>(level: Level, input: &[u32]) -> u32 {
    dispatch!(level, simd => native_sum_kernel::<_, A>(simd, input))
}
pub fn sum_wrapping(input: &[u32]) -> u32 { sum_at::<1>(Level::new(), input) }
pub fn product_wrapping(input: &[u32]) -> u32 { product_at(Level::new(), input) }

#[inline(always)]
fn dots<S: Simd>(simd: S, x: &[f32], rows: &[[f32; 4]]) -> [f32; 4] {
    let mut acc = f32x4::splat(simd, 0.0);
    for (&v, row) in x.iter().zip(rows) {
        // Explicit multiply then add, not mul_add: one output per lane.
        acc += f32x4::splat(simd, v) * f32x4::from_slice(simd, row);
    }
    [acc[0], acc[1], acc[2], acc[3]]
}
pub fn batched_dot4(x: &[f32], rows: &[[f32; 4]]) -> [f32; 4] {
    assert_eq!(x.len(), rows.len());
    let level = Level::new();
    dispatch!(level, simd => dots(simd, x, rows))
}

#[inline(always)]
fn log2_counts<S: Simd>(simd: S, counts: [u32; 4]) -> fearless_simd::f64x4<S> {
    use fearless_simd::f64x4;
    let parts = counts.map(crate::math::decompose_count);
    // Scalar preparation is intentional and included in the kernel's cost.
    let e = f64x4::from_fn(simd, |i| parts[i].0);
    let m = f64x4::from_fn(simd, |i| parts[i].1);
    let z = (m - f64x4::splat(simd, 1.0)) / (m + f64x4::splat(simd, 1.0));
    let q = z * z;
    let mut p = f64x4::splat(simd, 1.0 / 19.0);
    for k in (0..9).rev() { p = p * q + f64x4::splat(simd, 1.0 / ((2*k+1) as f64)); }
    e + f64x4::splat(simd, 2.0 / std::f64::consts::LN_2) * z * p
}
#[inline(always)]
fn score_kernel<S: Simd>(simd: S, counts: &[u32], depths: &[u8], constant: f64,
    table: &crate::math::LogTable) -> f64 {
    use fearless_simd::f64x4;
    let full = counts.len()/4*4;
    let mut acc = f64x4::splat(simd, 0.0);
    for base in (0..full).step_by(4) {
        let c = u32x4::from_slice(simd, &counts[base..base+4]);
        let values = [c[0], c[1], c[2], c[3]];
        let small = c.simd_lt(u32x4::splat(simd, 256));
        let logs = if small.all_true() {
            f64x4::from_fn(simd, |i| table[values[i] as usize])
        } else {
            let mut approximate = log2_counts(simd, values);
            if small.any_true() {
                // Mixed-block scalar repair; not a hardware-gather claim.
                for i in 0..4 {
                    if values[i] < 256 { approximate[i] = table[values[i] as usize]; }
                }
            }
            approximate
        };
        let cf = f64x4::from_fn(simd, |i| f64::from(values[i]));
        let d = f64x4::from_fn(simd, |i| f64::from(depths[base+i]));
        acc += cf * (d + logs);
    }
    let mut sum = acc.as_slice().iter().copied().sum::<f64>();
    for (&c, &d) in counts[full..].iter().zip(&depths[full..]) {
        let log = if c < 256 { table[c as usize] } else { crate::math::log2_count_approx(c) };
        sum += f64::from(c) * (f64::from(d) + log);
    }
    constant - sum
}
/// Approximate score, not a certified equivalent sign decision.
pub fn histogram_score_at(level: Level, counts: &[u32], depths: &[u8], constant: f64,
    table: &crate::math::LogTable) -> f64 {
    assert_eq!(counts.len(), depths.len());
    dispatch!(level, simd => score_kernel(simd, counts, depths, constant, table))
}
pub fn histogram_score(counts: &[u32], depths: &[u8], constant: f64,
    table: &crate::math::LogTable) -> f64 {
    histogram_score_at(Level::new(), counts, depths, constant, table)
}
